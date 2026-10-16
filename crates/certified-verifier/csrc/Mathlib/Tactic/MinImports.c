// Lean compiler output
// Module: Mathlib.Tactic.MinImports
// Imports: public import Init public meta import Init public meta import Lean.Elab.DefView public meta import Lean.Util.CollectAxioms public meta import ImportGraph.Imports.Redundant public meta import ImportGraph.Imports.RequiredModules public meta import Mathlib.Tactic.Linter.Header public import Lean.Elab.DeclModifiers
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
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Environment_header(lean_object*);
extern lean_object* l_Lean_instInhabitedEffectiveImport_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instHashableExtraModUse_hash___boxed(lean_object*);
lean_object* l_Lean_instBEqExtraModUse_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_extraModUses;
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableExtraModUse_hash(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqExtraModUse_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
extern lean_object* l_Lean_inheritedTraceOptions;
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Std_HashMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
extern lean_object* l_Lean_indirectModUseExt;
uint8_t l_Lean_isMarkedMeta(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
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
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getSepArgs(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* l_Lean_InternalExceptionId_getName(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
uint8_t l_Lean_Elab_isAbortExceptionId(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_toAttributeKind___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
lean_object* l_Lean_mkPrivateName(lean_object*, lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_privateToUserName(lean_object*);
lean_object* l_Lean_Elab_expandMacroImpl_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ResolveName_resolveNamespace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_expandMacros(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getAttributeImpl(lean_object*, lean_object*);
extern lean_object* l_Lean_regularInitAttr;
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_lt(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint64_t lean_uint64_of_nat(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdx_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_mkDefViewOfInstance(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
extern lean_object* l_Lean_linter_redundantVisibility;
lean_object* l_Lean_Syntax_getHeadInfo(lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_extractMacroScopes(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Name_replacePrefix(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MacroScopesView_review(lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lp_importGraph_Lean_Name_transitivelyUsedConstants___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_NameSet_append(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_maxView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_Environment_findRedundantImports(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prevn(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pos_prev_x3f(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
lean_object* l_String_Slice_toNat_x3f(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_isInitImport___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Init"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_isInitImport___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_isInitImport___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_isInitImport(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_isInitImport___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getSyntaxNodeKinds(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getVisited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getVisited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getId___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instance"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 156, 84, 218, 244, 57, 142, 153)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getId___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Command_MinImports_getId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Command_MinImports_getId___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Command_MinImports_getId___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getId___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 2, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIds(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "attributes"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(66, 184, 196, 169, 25, 125, 40, 35)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Command_MinImports_getAttrNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAttrs_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_previousInstName(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqExtraModUse_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__0_value;
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableExtraModUse_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__3 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__3_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__4_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " extra mod use "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__5 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " of "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__7 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__10 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__11 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recording "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__13 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__15 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "regular"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__17 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__17_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__18 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__18_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__19 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__19_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__20 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__0 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__1 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2;
static const lean_array_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__3 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 158, .m_capacity = 158, .m_length = 157, .m_data = "maximum recursion depth has been reached\nuse `set_option maxRecDepth <num>` to increase limit\nuse `set_option diagnostics true` to get diagnostic information"};
static const lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Cannot use attribute `["};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "]`: module `"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = "` is loaded for IR only (reached as a private `meta` dependency). Add an import of `"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Unknown attribute `["};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "]`"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(107, 67, 254, 234, 65, 174, 209, 53)}};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Unknown attribute"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 150, 238, 148, 228, 221, 116, 224)}};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "internal exception: "};
static const lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "; the modifier has no effect"};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "`public` is the default visibility"};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = " inside a `public section`"};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__19_value),LEAN_SCALAR_PTR_LITERAL(213, 248, 16, 228, 25, 227, 72, 143)}};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__20_value),LEAN_SCALAR_PTR_LITERAL(99, 134, 241, 204, 211, 206, 124, 144)}};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "unexpected visibility modifier"};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8;
static const lean_string_object lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 115, .m_capacity = 115, .m_length = 114, .m_data = "`private` has no effect in a `module` file outside `public section`; declarations are already `private` by default"};
static const lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "partial"};
static const lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 175, 198, 167, 172, 79, 14, 207)}};
static const lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__18_value),LEAN_SCALAR_PTR_LITERAL(124, 247, 59, 43, 44, 177, 111, 66)}};
static const lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__2_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllDependencies___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Command_MinImports_getAllImports_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__0 = (const lean_object*)&lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__0_value;
static const lean_string_object lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__1_value;
static const lean_string_object lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_getAllImports___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getAllImports___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "public import "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "MinImports"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "minImpsStx"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 223, 110, 91, 39, 178, 94, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(100, 2, 79, 24, 141, 33, 118, 28)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#min_imports"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__11_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsStx = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "command#min_importsIn_"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 223, 110, 91, 39, 178, 94, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 125, 11, 21, 239, 102, 188, 43)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn__ = (const lean_object*)&lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__minImpsStx__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__minImpsStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__command_x23min__importsIn____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__command_x23min__importsIn____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_isInitImport(lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 1)
{
lean_object* v_pre_3_; 
v_pre_3_ = lean_ctor_get(v_x_2_, 0);
if (lean_obj_tag(v_pre_3_) == 0)
{
lean_object* v_str_4_; lean_object* v___x_5_; uint8_t v___x_6_; 
v_str_4_ = lean_ctor_get(v_x_2_, 1);
v___x_5_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_isInitImport___closed__0));
v___x_6_ = lean_string_dec_eq(v_str_4_, v___x_5_);
if (v___x_6_ == 0)
{
v_x_2_ = v_pre_3_;
goto _start;
}
else
{
return v___x_6_;
}
}
else
{
v_x_2_ = v_pre_3_;
goto _start;
}
}
else
{
uint8_t v___x_9_; 
v___x_9_ = 0;
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_isInitImport___boxed(lean_object* v_x_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Mathlib_Command_MinImports_isInitImport(v_x_10_);
lean_dec(v_x_10_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(lean_object* v_as_13_, size_t v_i_14_, size_t v_stop_15_, lean_object* v_b_16_){
_start:
{
uint8_t v___x_17_; 
v___x_17_ = lean_usize_dec_eq(v_i_14_, v_stop_15_);
if (v___x_17_ == 0)
{
lean_object* v___x_18_; lean_object* v___x_19_; size_t v___x_20_; size_t v___x_21_; 
v___x_18_ = lean_array_uget_borrowed(v_as_13_, v_i_14_);
lean_inc(v___x_18_);
v___x_19_ = l_Lean_NameSet_append(v_b_16_, v___x_18_);
v___x_20_ = ((size_t)1ULL);
v___x_21_ = lean_usize_add(v_i_14_, v___x_20_);
v_i_14_ = v___x_21_;
v_b_16_ = v___x_19_;
goto _start;
}
else
{
return v_b_16_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1___boxed(lean_object* v_as_23_, lean_object* v_i_24_, lean_object* v_stop_25_, lean_object* v_b_26_){
_start:
{
size_t v_i_boxed_27_; size_t v_stop_boxed_28_; lean_object* v_res_29_; 
v_i_boxed_27_ = lean_unbox_usize(v_i_24_);
lean_dec(v_i_24_);
v_stop_boxed_28_ = lean_unbox_usize(v_stop_25_);
lean_dec(v_stop_25_);
v_res_29_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(v_as_23_, v_i_boxed_27_, v_stop_boxed_28_, v_b_26_);
lean_dec_ref(v_as_23_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getSyntaxNodeKinds(lean_object* v_x_30_){
_start:
{
switch(lean_obj_tag(v_x_30_))
{
case 1:
{
lean_object* v_kind_31_; lean_object* v_args_32_; lean_object* v___x_33_; size_t v_sz_34_; size_t v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; uint8_t v___x_39_; 
v_kind_31_ = lean_ctor_get(v_x_30_, 1);
lean_inc(v_kind_31_);
v_args_32_ = lean_ctor_get(v_x_30_, 2);
lean_inc_ref(v_args_32_);
lean_dec_ref_known(v_x_30_, 3);
v___x_33_ = l_Lean_NameSet_empty;
v_sz_34_ = lean_array_size(v_args_32_);
v___x_35_ = ((size_t)0ULL);
v___x_36_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0(v_sz_34_, v___x_35_, v_args_32_);
v___x_37_ = lean_unsigned_to_nat(0u);
v___x_38_ = lean_array_get_size(v___x_36_);
v___x_39_ = lean_nat_dec_lt(v___x_37_, v___x_38_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; 
lean_dec_ref(v___x_36_);
v___x_40_ = l_Lean_NameSet_insert(v___x_33_, v_kind_31_);
return v___x_40_;
}
else
{
uint8_t v___x_41_; 
v___x_41_ = lean_nat_dec_le(v___x_38_, v___x_38_);
if (v___x_41_ == 0)
{
if (v___x_39_ == 0)
{
lean_object* v___x_42_; 
lean_dec_ref(v___x_36_);
v___x_42_ = l_Lean_NameSet_insert(v___x_33_, v_kind_31_);
return v___x_42_;
}
else
{
size_t v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_43_ = lean_usize_of_nat(v___x_38_);
v___x_44_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(v___x_36_, v___x_35_, v___x_43_, v___x_33_);
lean_dec_ref(v___x_36_);
v___x_45_ = l_Lean_NameSet_insert(v___x_44_, v_kind_31_);
return v___x_45_;
}
}
else
{
size_t v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_usize_of_nat(v___x_38_);
v___x_47_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(v___x_36_, v___x_35_, v___x_46_, v___x_33_);
lean_dec_ref(v___x_36_);
v___x_48_ = l_Lean_NameSet_insert(v___x_47_, v_kind_31_);
return v___x_48_;
}
}
}
case 3:
{
lean_object* v_val_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v_val_49_ = lean_ctor_get(v_x_30_, 2);
lean_inc(v_val_49_);
lean_dec_ref_known(v_x_30_, 4);
v___x_50_ = l_Lean_NameSet_empty;
v___x_51_ = l_Lean_NameSet_insert(v___x_50_, v_val_49_);
return v___x_51_;
}
default: 
{
lean_object* v___x_52_; 
lean_dec(v_x_30_);
v___x_52_ = l_Lean_NameSet_empty;
return v___x_52_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0(size_t v_sz_53_, size_t v_i_54_, lean_object* v_bs_55_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lean_usize_dec_lt(v_i_54_, v_sz_53_);
if (v___x_56_ == 0)
{
return v_bs_55_;
}
else
{
lean_object* v_v_57_; lean_object* v___x_58_; lean_object* v_bs_x27_59_; lean_object* v___x_60_; size_t v___x_61_; size_t v___x_62_; lean_object* v___x_63_; 
v_v_57_ = lean_array_uget(v_bs_55_, v_i_54_);
v___x_58_ = lean_unsigned_to_nat(0u);
v_bs_x27_59_ = lean_array_uset(v_bs_55_, v_i_54_, v___x_58_);
v___x_60_ = lp_mathlib_Mathlib_Command_MinImports_getSyntaxNodeKinds(v_v_57_);
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_54_, v___x_61_);
v___x_63_ = lean_array_uset(v_bs_x27_59_, v_i_54_, v___x_60_);
v_i_54_ = v___x_62_;
v_bs_55_ = v___x_63_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0___boxed(lean_object* v_sz_65_, lean_object* v_i_66_, lean_object* v_bs_67_){
_start:
{
size_t v_sz_boxed_68_; size_t v_i_boxed_69_; lean_object* v_res_70_; 
v_sz_boxed_68_ = lean_unbox_usize(v_sz_65_);
lean_dec(v_sz_65_);
v_i_boxed_69_ = lean_unbox_usize(v_i_66_);
lean_dec(v_i_66_);
v_res_70_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__0(v_sz_boxed_68_, v_i_boxed_69_, v_bs_67_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg(lean_object* v_constName_71_, uint8_t v_skipRealize_72_, lean_object* v___y_73_){
_start:
{
lean_object* v___x_75_; lean_object* v_env_76_; uint8_t v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_75_ = lean_st_ref_get(v___y_73_);
v_env_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc_ref(v_env_76_);
lean_dec(v___x_75_);
v___x_77_ = l_Lean_Environment_contains(v_env_76_, v_constName_71_, v_skipRealize_72_);
v___x_78_ = lean_box(v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg___boxed(lean_object* v_constName_80_, lean_object* v_skipRealize_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
uint8_t v_skipRealize_boxed_84_; lean_object* v_res_85_; 
v_skipRealize_boxed_84_ = lean_unbox(v_skipRealize_81_);
v_res_85_ = lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg(v_constName_80_, v_skipRealize_boxed_84_, v___y_82_);
lean_dec(v___y_82_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0(lean_object* v_constName_86_, uint8_t v_skipRealize_87_, lean_object* v___y_88_, lean_object* v___y_89_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg(v_constName_86_, v_skipRealize_87_, v___y_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___boxed(lean_object* v_constName_92_, lean_object* v_skipRealize_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_){
_start:
{
uint8_t v_skipRealize_boxed_97_; lean_object* v_res_98_; 
v_skipRealize_boxed_97_ = lean_unbox(v_skipRealize_93_);
v_res_98_ = lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0(v_constName_92_, v_skipRealize_boxed_97_, v___y_94_, v___y_95_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getVisited(lean_object* v_decl_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
uint8_t v___x_103_; lean_object* v___x_104_; lean_object* v_a_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_126_; 
v___x_103_ = 1;
lean_inc(v_decl_99_);
v___x_104_ = lp_mathlib_Lean_hasConst___at___00Mathlib_Command_MinImports_getVisited_spec__0___redArg(v_decl_99_, v___x_103_, v_a_101_);
v_a_105_ = lean_ctor_get(v___x_104_, 0);
v_isSharedCheck_126_ = !lean_is_exclusive(v___x_104_);
if (v_isSharedCheck_126_ == 0)
{
v___x_107_ = v___x_104_;
v_isShared_108_ = v_isSharedCheck_126_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_a_105_);
lean_dec(v___x_104_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_126_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
uint8_t v___x_109_; 
v___x_109_ = lean_unbox(v_a_105_);
lean_dec(v_a_105_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; lean_object* v___x_112_; 
lean_dec(v_decl_99_);
v___x_110_ = l_Lean_NameSet_empty;
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 0, v___x_110_);
v___x_112_ = v___x_107_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v___x_110_);
v___x_112_ = v_reuseFailAlloc_113_;
goto v_reusejp_111_;
}
v_reusejp_111_:
{
return v___x_112_;
}
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
lean_del_object(v___x_107_);
v___x_114_ = lean_st_ref_get(v_a_101_);
v___x_115_ = lean_alloc_closure((void*)(lp_importGraph_Lean_Name_transitivelyUsedConstants___boxed), 4, 1);
lean_closure_set(v___x_115_, 0, v_decl_99_);
v___x_116_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_115_, v_a_100_, v_a_101_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_object* v_a_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_125_; 
v_a_117_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_125_ == 0)
{
v___x_119_ = v___x_116_;
v_isShared_120_ = v_isSharedCheck_125_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_a_117_);
lean_dec(v___x_116_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_125_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_121_; lean_object* v___x_123_; 
v___x_121_ = lean_st_ref_set(v_a_101_, v___x_114_);
if (v_isShared_120_ == 0)
{
v___x_123_ = v___x_119_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_a_117_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
else
{
lean_dec(v___x_114_);
return v___x_116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getVisited___boxed(lean_object* v_decl_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Mathlib_Command_MinImports_getVisited(v_decl_127_, v_a_128_, v_a_129_);
lean_dec(v_a_129_);
lean_dec_ref(v_a_128_);
return v_res_131_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getId___lam__0(lean_object* v_x_141_){
_start:
{
lean_object* v___x_142_; uint8_t v___x_143_; 
v___x_142_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___closed__4));
v___x_143_ = l_Lean_Syntax_isOfKind(v_x_141_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__0___boxed(lean_object* v_x_144_){
_start:
{
uint8_t v_res_145_; lean_object* v_r_146_; 
v_res_145_ = lp_mathlib_Mathlib_Command_MinImports_getId___lam__0(v_x_144_);
v_r_146_ = lean_box(v_res_145_);
return v_r_146_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getId___lam__1(lean_object* v_x_153_){
_start:
{
lean_object* v___x_154_; uint8_t v___x_155_; 
v___x_154_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___closed__1));
v___x_155_ = l_Lean_Syntax_isOfKind(v_x_153_, v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___lam__1___boxed(lean_object* v_x_156_){
_start:
{
uint8_t v_res_157_; lean_object* v_r_158_; 
v_res_157_ = lp_mathlib_Mathlib_Command_MinImports_getId___lam__1(v_x_156_);
v_r_158_ = lean_box(v_res_157_);
return v_r_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId(lean_object* v_stx_171_, lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
lean_object* v___f_175_; lean_object* v___x_176_; 
v___f_175_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___closed__0));
lean_inc(v_stx_171_);
v___x_176_ = l_Lean_Syntax_find_x3f(v_stx_171_, v___f_175_);
if (lean_obj_tag(v___x_176_) == 0)
{
lean_object* v___f_177_; lean_object* v___x_178_; 
v___f_177_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___closed__1));
v___x_178_ = l_Lean_Syntax_find_x3f(v_stx_171_, v___f_177_);
if (lean_obj_tag(v___x_178_) == 0)
{
lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_179_ = lean_box(0);
v___x_180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
return v___x_180_;
}
else
{
lean_object* v_val_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v_val_181_ = lean_ctor_get(v___x_178_, 0);
lean_inc(v_val_181_);
lean_dec_ref_known(v___x_178_, 1);
v___x_182_ = lean_unsigned_to_nat(0u);
v___x_183_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___closed__3));
v___x_184_ = l_Lean_Elab_Command_mkDefViewOfInstance(v___x_183_, v_val_181_, v_a_172_, v_a_173_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_194_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_194_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_194_ == 0)
{
v___x_187_ = v___x_184_;
v_isShared_188_ = v_isSharedCheck_194_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_a_185_);
lean_dec(v___x_184_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_194_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v_declId_189_; lean_object* v___x_190_; lean_object* v___x_192_; 
v_declId_189_ = lean_ctor_get(v_a_185_, 3);
lean_inc(v_declId_189_);
lean_dec(v_a_185_);
v___x_190_ = l_Lean_Syntax_getArg(v_declId_189_, v___x_182_);
lean_dec(v_declId_189_);
if (v_isShared_188_ == 0)
{
lean_ctor_set(v___x_187_, 0, v___x_190_);
v___x_192_ = v___x_187_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v___x_190_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
else
{
lean_object* v_a_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_202_; 
v_a_195_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_202_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_202_ == 0)
{
v___x_197_ = v___x_184_;
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_a_195_);
lean_dec(v___x_184_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v___x_200_; 
if (v_isShared_198_ == 0)
{
v___x_200_ = v___x_197_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_a_195_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
}
}
}
else
{
lean_object* v_val_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_212_; 
lean_dec(v_stx_171_);
v_val_203_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_212_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_212_ == 0)
{
v___x_205_ = v___x_176_;
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_val_203_);
lean_dec(v___x_176_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_210_; 
v___x_207_ = lean_unsigned_to_nat(0u);
v___x_208_ = l_Lean_Syntax_getArg(v_val_203_, v___x_207_);
lean_dec(v_val_203_);
if (v_isShared_206_ == 0)
{
lean_ctor_set_tag(v___x_205_, 0);
lean_ctor_set(v___x_205_, 0, v___x_208_);
v___x_210_ = v___x_205_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v___x_208_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getId___boxed(lean_object* v_stx_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Mathlib_Command_MinImports_getId(v_stx_213_, v_a_214_, v_a_215_);
lean_dec(v_a_215_);
lean_dec_ref(v_a_214_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIds(lean_object* v_x_218_){
_start:
{
switch(lean_obj_tag(v_x_218_))
{
case 1:
{
lean_object* v_args_219_; lean_object* v___x_220_; size_t v_sz_221_; size_t v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; uint8_t v___x_226_; 
v_args_219_ = lean_ctor_get(v_x_218_, 2);
lean_inc_ref(v_args_219_);
lean_dec_ref_known(v_x_218_, 3);
v___x_220_ = l_Lean_NameSet_empty;
v_sz_221_ = lean_array_size(v_args_219_);
v___x_222_ = ((size_t)0ULL);
v___x_223_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0(v_sz_221_, v___x_222_, v_args_219_);
v___x_224_ = lean_unsigned_to_nat(0u);
v___x_225_ = lean_array_get_size(v___x_223_);
v___x_226_ = lean_nat_dec_lt(v___x_224_, v___x_225_);
if (v___x_226_ == 0)
{
lean_dec_ref(v___x_223_);
return v___x_220_;
}
else
{
uint8_t v___x_227_; 
v___x_227_ = lean_nat_dec_le(v___x_225_, v___x_225_);
if (v___x_227_ == 0)
{
if (v___x_226_ == 0)
{
lean_dec_ref(v___x_223_);
return v___x_220_;
}
else
{
size_t v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_usize_of_nat(v___x_225_);
v___x_229_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(v___x_223_, v___x_222_, v___x_228_, v___x_220_);
lean_dec_ref(v___x_223_);
return v___x_229_;
}
}
else
{
size_t v___x_230_; lean_object* v___x_231_; 
v___x_230_ = lean_usize_of_nat(v___x_225_);
v___x_231_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_getSyntaxNodeKinds_spec__1(v___x_223_, v___x_222_, v___x_230_, v___x_220_);
lean_dec_ref(v___x_223_);
return v___x_231_;
}
}
}
case 3:
{
lean_object* v_val_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v_val_232_ = lean_ctor_get(v_x_218_, 2);
lean_inc(v_val_232_);
lean_dec_ref_known(v_x_218_, 4);
v___x_233_ = l_Lean_NameSet_empty;
v___x_234_ = l_Lean_NameSet_insert(v___x_233_, v_val_232_);
return v___x_234_;
}
default: 
{
lean_object* v___x_235_; 
lean_dec(v_x_218_);
v___x_235_ = l_Lean_NameSet_empty;
return v___x_235_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0(size_t v_sz_236_, size_t v_i_237_, lean_object* v_bs_238_){
_start:
{
uint8_t v___x_239_; 
v___x_239_ = lean_usize_dec_lt(v_i_237_, v_sz_236_);
if (v___x_239_ == 0)
{
return v_bs_238_;
}
else
{
lean_object* v_v_240_; lean_object* v___x_241_; lean_object* v_bs_x27_242_; lean_object* v___x_243_; size_t v___x_244_; size_t v___x_245_; lean_object* v___x_246_; 
v_v_240_ = lean_array_uget(v_bs_238_, v_i_237_);
v___x_241_ = lean_unsigned_to_nat(0u);
v_bs_x27_242_ = lean_array_uset(v_bs_238_, v_i_237_, v___x_241_);
v___x_243_ = lp_mathlib_Mathlib_Command_MinImports_getIds(v_v_240_);
v___x_244_ = ((size_t)1ULL);
v___x_245_ = lean_usize_add(v_i_237_, v___x_244_);
v___x_246_ = lean_array_uset(v_bs_x27_242_, v_i_237_, v___x_243_);
v_i_237_ = v___x_245_;
v_bs_238_ = v___x_246_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0___boxed(lean_object* v_sz_248_, lean_object* v_i_249_, lean_object* v_bs_250_){
_start:
{
size_t v_sz_boxed_251_; size_t v_i_boxed_252_; lean_object* v_res_253_; 
v_sz_boxed_251_ = lean_unbox_usize(v_sz_248_);
lean_dec(v_sz_248_);
v_i_boxed_252_ = lean_unbox_usize(v_i_249_);
lean_dec(v_i_249_);
v_res_253_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_getIds_spec__0(v_sz_boxed_251_, v_i_boxed_252_, v_bs_250_);
return v_res_253_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0(lean_object* v_x_261_){
_start:
{
lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___closed__2));
v___x_263_ = l_Lean_Syntax_isOfKind(v_x_261_, v___x_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0___boxed(lean_object* v_x_264_){
_start:
{
uint8_t v_res_265_; lean_object* v_r_266_; 
v_res_265_ = lp_mathlib_Mathlib_Command_MinImports_getAttrNames___lam__0(v_x_264_);
v_r_266_ = lean_box(v_res_265_);
return v_r_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrNames(lean_object* v_stx_268_){
_start:
{
lean_object* v___f_269_; lean_object* v___x_270_; 
v___f_269_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getAttrNames___closed__0));
v___x_270_ = l_Lean_Syntax_find_x3f(v_stx_268_, v___f_269_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v___x_271_; 
v___x_271_ = l_Lean_NameSet_empty;
return v___x_271_;
}
else
{
lean_object* v_val_272_; lean_object* v___x_273_; 
v_val_272_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v_val_272_);
lean_dec_ref_known(v___x_270_, 1);
v___x_273_ = lp_mathlib_Mathlib_Command_MinImports_getIds(v_val_272_);
return v___x_273_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAttrs_spec__0(lean_object* v_env_274_, lean_object* v_init_275_, lean_object* v_x_276_){
_start:
{
if (lean_obj_tag(v_x_276_) == 0)
{
lean_object* v_k_277_; lean_object* v_l_278_; lean_object* v_r_279_; lean_object* v___x_280_; lean_object* v_a_281_; lean_object* v___x_282_; 
v_k_277_ = lean_ctor_get(v_x_276_, 1);
lean_inc(v_k_277_);
v_l_278_ = lean_ctor_get(v_x_276_, 3);
lean_inc(v_l_278_);
v_r_279_ = lean_ctor_get(v_x_276_, 4);
lean_inc(v_r_279_);
lean_dec_ref_known(v_x_276_, 5);
lean_inc_ref_n(v_env_274_, 2);
v___x_280_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAttrs_spec__0(v_env_274_, v_init_275_, v_l_278_);
v_a_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_a_281_);
v___x_282_ = l_Lean_getAttributeImpl(v_env_274_, v_k_277_);
if (lean_obj_tag(v___x_282_) == 0)
{
lean_object* v_a_283_; 
lean_dec_ref_known(v___x_282_, 1);
lean_dec(v_a_281_);
v_a_283_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_a_283_);
lean_dec_ref(v___x_280_);
v_init_275_ = v_a_283_;
v_x_276_ = v_r_279_;
goto _start;
}
else
{
lean_object* v_a_285_; lean_object* v_toAttributeImplCore_286_; lean_object* v_ref_287_; lean_object* v___x_288_; 
lean_dec_ref(v___x_280_);
v_a_285_ = lean_ctor_get(v___x_282_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_282_, 1);
v_toAttributeImplCore_286_ = lean_ctor_get(v_a_285_, 0);
lean_inc_ref(v_toAttributeImplCore_286_);
lean_dec(v_a_285_);
v_ref_287_ = lean_ctor_get(v_toAttributeImplCore_286_, 0);
lean_inc(v_ref_287_);
lean_dec_ref(v_toAttributeImplCore_286_);
v___x_288_ = l_Lean_NameSet_insert(v_a_281_, v_ref_287_);
v_init_275_ = v___x_288_;
v_x_276_ = v_r_279_;
goto _start;
}
}
else
{
lean_object* v___x_290_; 
lean_dec_ref(v_env_274_);
v___x_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_290_, 0, v_init_275_);
return v___x_290_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAttrs(lean_object* v_env_291_, lean_object* v_stx_292_){
_start:
{
lean_object* v_new_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v_a_296_; 
v_new_293_ = l_Lean_NameSet_empty;
v___x_294_ = lp_mathlib_Mathlib_Command_MinImports_getAttrNames(v_stx_292_);
v___x_295_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAttrs_spec__0(v_env_291_, v_new_293_, v___x_294_);
v_a_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc(v_a_296_);
lean_dec_ref(v___x_295_);
return v_a_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0(lean_object* v_s_297_, lean_object* v_pos_298_){
_start:
{
lean_object* v_str_299_; lean_object* v_startInclusive_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; uint8_t v___x_304_; 
v_str_299_ = lean_ctor_get(v_s_297_, 0);
v_startInclusive_300_ = lean_ctor_get(v_s_297_, 1);
v___x_301_ = lean_nat_add(v_startInclusive_300_, v_pos_298_);
v___x_302_ = lean_nat_sub(v___x_301_, v_startInclusive_300_);
v___x_303_ = lean_unsigned_to_nat(0u);
v___x_304_ = lean_nat_dec_eq(v___x_302_, v___x_303_);
if (v___x_304_ == 0)
{
lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; uint32_t v___x_310_; uint32_t v___x_311_; uint8_t v___x_312_; 
lean_inc(v_startInclusive_300_);
lean_inc_ref(v_str_299_);
v___x_305_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_305_, 0, v_str_299_);
lean_ctor_set(v___x_305_, 1, v_startInclusive_300_);
lean_ctor_set(v___x_305_, 2, v___x_301_);
v___x_306_ = lean_unsigned_to_nat(1u);
v___x_307_ = lean_nat_sub(v___x_302_, v___x_306_);
lean_dec(v___x_302_);
v___x_308_ = l_String_Slice_posLE(v___x_305_, v___x_307_);
lean_dec_ref_known(v___x_305_, 3);
v___x_309_ = lean_nat_add(v_startInclusive_300_, v___x_308_);
v___x_310_ = lean_string_utf8_get_fast(v_str_299_, v___x_309_);
lean_dec(v___x_309_);
v___x_311_ = 95;
v___x_312_ = lean_uint32_dec_eq(v___x_310_, v___x_311_);
if (v___x_312_ == 0)
{
uint8_t v___x_313_; 
v___x_313_ = lean_nat_dec_lt(v___x_308_, v_pos_298_);
if (v___x_313_ == 0)
{
lean_dec(v___x_308_);
return v_pos_298_;
}
else
{
lean_dec(v_pos_298_);
v_pos_298_ = v___x_308_;
goto _start;
}
}
else
{
lean_dec(v___x_308_);
return v_pos_298_;
}
}
else
{
lean_dec(v___x_302_);
lean_dec(v___x_301_);
return v_pos_298_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0___boxed(lean_object* v_s_315_, lean_object* v_pos_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0(v_s_315_, v_pos_316_);
lean_dec_ref(v_s_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_previousInstName(lean_object* v_x_320_){
_start:
{
if (lean_obj_tag(v_x_320_) == 1)
{
lean_object* v_pre_321_; lean_object* v_str_322_; lean_object* v___y_324_; lean_object* v_str_325_; lean_object* v_startInclusive_326_; lean_object* v_endExclusive_327_; lean_object* v___y_332_; lean_object* v___y_333_; lean_object* v___y_334_; lean_object* v___y_335_; uint32_t v___y_336_; lean_object* v___y_345_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v_last_363_; lean_object* v___x_364_; 
v_pre_321_ = lean_ctor_get(v_x_320_, 0);
v_str_322_ = lean_ctor_get(v_x_320_, 1);
v___x_359_ = lean_unsigned_to_nat(0u);
v___x_360_ = lean_string_utf8_byte_size(v_str_322_);
lean_inc_ref_n(v_str_322_, 2);
v___x_361_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_361_, 0, v_str_322_);
lean_ctor_set(v___x_361_, 1, v___x_359_);
lean_ctor_set(v___x_361_, 2, v___x_360_);
v___x_362_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0(v___x_361_, v___x_360_);
lean_dec_ref_known(v___x_361_, 3);
v_last_363_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_last_363_, 0, v_str_322_);
lean_ctor_set(v_last_363_, 1, v___x_362_);
lean_ctor_set(v_last_363_, 2, v___x_360_);
v___x_364_ = l_String_Slice_toNat_x3f(v_last_363_);
lean_dec_ref_known(v_last_363_, 3);
if (lean_obj_tag(v___x_364_) == 1)
{
lean_object* v_val_365_; uint8_t v_isZero_366_; 
v_val_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc(v_val_365_);
lean_dec_ref_known(v___x_364_, 1);
v_isZero_366_ = lean_nat_dec_eq(v_val_365_, v___x_359_);
if (v_isZero_366_ == 0)
{
lean_object* v_one_367_; lean_object* v_n_368_; uint8_t v_isZero_369_; 
v_one_367_ = lean_unsigned_to_nat(1u);
v_n_368_ = lean_nat_sub(v_val_365_, v_one_367_);
lean_dec(v_val_365_);
v_isZero_369_ = lean_nat_dec_eq(v_n_368_, v___x_359_);
if (v_isZero_369_ == 0)
{
lean_object* v_n_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
v_n_370_ = lean_nat_sub(v_n_368_, v_one_367_);
lean_dec(v_n_368_);
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__0));
v___x_372_ = lean_nat_add(v_n_370_, v_one_367_);
lean_dec(v_n_370_);
v___x_373_ = l_Nat_reprFast(v___x_372_);
v___x_374_ = lean_string_append(v___x_371_, v___x_373_);
lean_dec_ref(v___x_373_);
v___y_345_ = v___x_374_;
goto v___jp_344_;
}
else
{
lean_object* v___x_375_; 
lean_dec(v_n_368_);
v___x_375_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___y_345_ = v___x_375_;
goto v___jp_344_;
}
}
else
{
lean_object* v___x_376_; 
lean_dec(v_val_365_);
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___y_345_ = v___x_376_;
goto v___jp_344_;
}
}
else
{
lean_object* v___x_377_; 
lean_dec(v___x_364_);
v___x_377_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___y_345_ = v___x_377_;
goto v___jp_344_;
}
v___jp_323_:
{
lean_object* v___x_328_; lean_object* v_newTail_329_; lean_object* v___x_330_; 
v___x_328_ = lean_string_utf8_extract_fast(v_str_325_, v_startInclusive_326_, v_endExclusive_327_);
lean_dec(v_endExclusive_327_);
lean_dec(v_startInclusive_326_);
lean_dec_ref(v_str_325_);
v_newTail_329_ = lean_string_append(v___x_328_, v___y_324_);
lean_dec_ref(v___y_324_);
v___x_330_ = l_Lean_Name_str___override(v_pre_321_, v_newTail_329_);
return v___x_330_;
}
v___jp_331_:
{
uint32_t v___x_337_; uint8_t v___x_338_; 
v___x_337_ = 95;
v___x_338_ = lean_uint32_dec_eq(v___y_336_, v___x_337_);
if (v___x_338_ == 0)
{
lean_object* v_str_339_; lean_object* v_startInclusive_340_; lean_object* v_endExclusive_341_; 
lean_dec(v___y_333_);
lean_dec(v___y_332_);
lean_dec_ref(v_str_322_);
v_str_339_ = lean_ctor_get(v___y_335_, 0);
lean_inc_ref(v_str_339_);
v_startInclusive_340_ = lean_ctor_get(v___y_335_, 1);
lean_inc(v_startInclusive_340_);
v_endExclusive_341_ = lean_ctor_get(v___y_335_, 2);
lean_inc(v_endExclusive_341_);
lean_dec_ref(v___y_335_);
v___y_324_ = v___y_334_;
v_str_325_ = v_str_339_;
v_startInclusive_326_ = v_startInclusive_340_;
v_endExclusive_327_ = v_endExclusive_341_;
goto v___jp_323_;
}
else
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_unsigned_to_nat(1u);
v___x_343_ = l_String_Slice_Pos_prevn(v___y_335_, v___y_333_, v___x_342_);
lean_dec_ref(v___y_335_);
v___y_324_ = v___y_334_;
v_str_325_ = v_str_322_;
v_startInclusive_326_ = v___y_332_;
v_endExclusive_327_ = v___x_343_;
goto v___jp_323_;
}
}
v___jp_344_:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; uint8_t v___x_350_; 
v___x_346_ = lean_unsigned_to_nat(0u);
v___x_347_ = lean_string_utf8_byte_size(v_str_322_);
lean_inc_ref(v_str_322_);
v___x_348_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_348_, 0, v_str_322_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
lean_ctor_set(v___x_348_, 2, v___x_347_);
v___x_349_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Command_MinImports_previousInstName_spec__0(v___x_348_, v___x_347_);
lean_dec_ref_known(v___x_348_, 3);
v___x_350_ = lean_nat_dec_eq(v___x_349_, v___x_346_);
if (v___x_350_ == 0)
{
lean_object* v_newTailPrefix_351_; lean_object* v___x_352_; 
lean_inc_ref_n(v_str_322_, 2);
lean_inc(v_pre_321_);
lean_dec_ref_known(v_x_320_, 2);
lean_inc(v___x_349_);
v_newTailPrefix_351_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_newTailPrefix_351_, 0, v_str_322_);
lean_ctor_set(v_newTailPrefix_351_, 1, v___x_346_);
lean_ctor_set(v_newTailPrefix_351_, 2, v___x_349_);
v___x_352_ = l_String_Slice_Pos_prev_x3f(v_newTailPrefix_351_, v___x_349_);
if (lean_obj_tag(v___x_352_) == 0)
{
uint32_t v___x_353_; 
v___x_353_ = 65;
v___y_332_ = v___x_346_;
v___y_333_ = v___x_349_;
v___y_334_ = v___y_345_;
v___y_335_ = v_newTailPrefix_351_;
v___y_336_ = v___x_353_;
goto v___jp_331_;
}
else
{
lean_object* v_val_354_; lean_object* v___x_355_; 
v_val_354_ = lean_ctor_get(v___x_352_, 0);
lean_inc(v_val_354_);
lean_dec_ref_known(v___x_352_, 1);
v___x_355_ = l_String_Slice_Pos_get_x3f(v_newTailPrefix_351_, v_val_354_);
lean_dec(v_val_354_);
if (lean_obj_tag(v___x_355_) == 0)
{
uint32_t v___x_356_; 
v___x_356_ = 65;
v___y_332_ = v___x_346_;
v___y_333_ = v___x_349_;
v___y_334_ = v___y_345_;
v___y_335_ = v_newTailPrefix_351_;
v___y_336_ = v___x_356_;
goto v___jp_331_;
}
else
{
lean_object* v_val_357_; uint32_t v___x_358_; 
v_val_357_ = lean_ctor_get(v___x_355_, 0);
lean_inc(v_val_357_);
lean_dec_ref_known(v___x_355_, 1);
v___x_358_ = lean_unbox_uint32(v_val_357_);
lean_dec(v_val_357_);
v___y_332_ = v___x_346_;
v___y_333_ = v___x_349_;
v___y_334_ = v___y_345_;
v___y_335_ = v_newTailPrefix_351_;
v___y_336_ = v___x_358_;
goto v___jp_331_;
}
}
}
else
{
lean_dec(v___x_349_);
lean_dec_ref(v___y_345_);
return v_x_320_;
}
}
}
else
{
return v_x_320_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0(lean_object* v_x_384_){
_start:
{
lean_object* v___x_385_; uint8_t v___x_386_; 
v___x_385_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___closed__1));
v___x_386_ = l_Lean_Syntax_isOfKind(v_x_384_, v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0___boxed(lean_object* v_x_387_){
_start:
{
uint8_t v_res_388_; lean_object* v_r_389_; 
v_res_388_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__0(v_x_387_);
v_r_389_ = lean_box(v_res_388_);
return v_r_389_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1(lean_object* v_x_396_){
_start:
{
lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___closed__1));
v___x_398_ = l_Lean_Syntax_isOfKind(v_x_396_, v___x_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1___boxed(lean_object* v_x_399_){
_start:
{
uint8_t v_res_400_; lean_object* v_r_401_; 
v_res_400_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__1(v_x_399_);
v_r_401_ = lean_box(v_res_400_);
return v_r_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2(uint8_t v_visibility_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
if (v_visibility_402_ == 1)
{
lean_object* v___x_407_; lean_object* v_env_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_407_ = lean_st_ref_get(v___y_405_);
v_env_408_ = lean_ctor_get(v___x_407_, 0);
lean_inc_ref(v_env_408_);
lean_dec(v___x_407_);
v___x_409_ = l_Lean_mkPrivateName(v_env_408_, v___y_403_);
lean_dec_ref(v_env_408_);
v___x_410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_410_, 0, v___x_409_);
return v___x_410_;
}
else
{
lean_object* v___x_411_; 
v___x_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_411_, 0, v___y_403_);
return v___x_411_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2___boxed(lean_object* v_visibility_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
uint8_t v_visibility_boxed_417_; lean_object* v_res_418_; 
v_visibility_boxed_417_ = lean_unbox(v_visibility_412_);
v_res_418_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2(v_visibility_boxed_417_, v___y_413_, v___y_414_, v___y_415_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0(lean_object* v___y_419_, uint8_t v_isExporting_420_, lean_object* v_a_x3f_421_){
_start:
{
lean_object* v___x_423_; lean_object* v_env_424_; lean_object* v_messages_425_; lean_object* v_scopes_426_; lean_object* v_usedQuotCtxts_427_; lean_object* v_nextMacroScope_428_; lean_object* v_maxRecDepth_429_; lean_object* v_ngen_430_; lean_object* v_auxDeclNGen_431_; lean_object* v_infoState_432_; lean_object* v_traceState_433_; lean_object* v_snapshotTasks_434_; lean_object* v_prevLinterStates_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_446_; 
v___x_423_ = lean_st_ref_take(v___y_419_);
v_env_424_ = lean_ctor_get(v___x_423_, 0);
v_messages_425_ = lean_ctor_get(v___x_423_, 1);
v_scopes_426_ = lean_ctor_get(v___x_423_, 2);
v_usedQuotCtxts_427_ = lean_ctor_get(v___x_423_, 3);
v_nextMacroScope_428_ = lean_ctor_get(v___x_423_, 4);
v_maxRecDepth_429_ = lean_ctor_get(v___x_423_, 5);
v_ngen_430_ = lean_ctor_get(v___x_423_, 6);
v_auxDeclNGen_431_ = lean_ctor_get(v___x_423_, 7);
v_infoState_432_ = lean_ctor_get(v___x_423_, 8);
v_traceState_433_ = lean_ctor_get(v___x_423_, 9);
v_snapshotTasks_434_ = lean_ctor_get(v___x_423_, 10);
v_prevLinterStates_435_ = lean_ctor_get(v___x_423_, 11);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_446_ == 0)
{
v___x_437_ = v___x_423_;
v_isShared_438_ = v_isSharedCheck_446_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_prevLinterStates_435_);
lean_inc(v_snapshotTasks_434_);
lean_inc(v_traceState_433_);
lean_inc(v_infoState_432_);
lean_inc(v_auxDeclNGen_431_);
lean_inc(v_ngen_430_);
lean_inc(v_maxRecDepth_429_);
lean_inc(v_nextMacroScope_428_);
lean_inc(v_usedQuotCtxts_427_);
lean_inc(v_scopes_426_);
lean_inc(v_messages_425_);
lean_inc(v_env_424_);
lean_dec(v___x_423_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_446_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v___x_439_; lean_object* v___x_441_; 
v___x_439_ = l_Lean_Environment_setExporting(v_env_424_, v_isExporting_420_);
if (v_isShared_438_ == 0)
{
lean_ctor_set(v___x_437_, 0, v___x_439_);
v___x_441_ = v___x_437_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v___x_439_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v_messages_425_);
lean_ctor_set(v_reuseFailAlloc_445_, 2, v_scopes_426_);
lean_ctor_set(v_reuseFailAlloc_445_, 3, v_usedQuotCtxts_427_);
lean_ctor_set(v_reuseFailAlloc_445_, 4, v_nextMacroScope_428_);
lean_ctor_set(v_reuseFailAlloc_445_, 5, v_maxRecDepth_429_);
lean_ctor_set(v_reuseFailAlloc_445_, 6, v_ngen_430_);
lean_ctor_set(v_reuseFailAlloc_445_, 7, v_auxDeclNGen_431_);
lean_ctor_set(v_reuseFailAlloc_445_, 8, v_infoState_432_);
lean_ctor_set(v_reuseFailAlloc_445_, 9, v_traceState_433_);
lean_ctor_set(v_reuseFailAlloc_445_, 10, v_snapshotTasks_434_);
lean_ctor_set(v_reuseFailAlloc_445_, 11, v_prevLinterStates_435_);
v___x_441_ = v_reuseFailAlloc_445_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_442_ = lean_st_ref_set(v___y_419_, v___x_441_);
v___x_443_ = lean_box(0);
v___x_444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
return v___x_444_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0___boxed(lean_object* v___y_447_, lean_object* v_isExporting_448_, lean_object* v_a_x3f_449_, lean_object* v___y_450_){
_start:
{
uint8_t v_isExporting_boxed_451_; lean_object* v_res_452_; 
v_isExporting_boxed_451_ = lean_unbox(v_isExporting_448_);
v_res_452_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0(v___y_447_, v_isExporting_boxed_451_, v_a_x3f_449_);
lean_dec(v_a_x3f_449_);
lean_dec(v___y_447_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg(lean_object* v_x_453_, uint8_t v_isExporting_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v___x_458_; lean_object* v_env_459_; uint8_t v_isExporting_460_; lean_object* v___x_461_; uint8_t v_isModule_462_; 
v___x_458_ = lean_st_ref_get(v___y_456_);
v_env_459_ = lean_ctor_get(v___x_458_, 0);
lean_inc_ref(v_env_459_);
lean_dec(v___x_458_);
v_isExporting_460_ = lean_ctor_get_uint8(v_env_459_, sizeof(void*)*8);
v___x_461_ = l_Lean_Environment_header(v_env_459_);
lean_dec_ref(v_env_459_);
v_isModule_462_ = lean_ctor_get_uint8(v___x_461_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_461_);
if (v_isModule_462_ == 0)
{
lean_object* v___x_463_; 
lean_inc(v___y_456_);
lean_inc_ref(v___y_455_);
v___x_463_ = lean_apply_3(v_x_453_, v___y_455_, v___y_456_, lean_box(0));
return v___x_463_;
}
else
{
if (v_isExporting_460_ == 0)
{
if (v_isExporting_454_ == 0)
{
lean_object* v___x_516_; 
lean_inc(v___y_456_);
lean_inc_ref(v___y_455_);
v___x_516_ = lean_apply_3(v_x_453_, v___y_455_, v___y_456_, lean_box(0));
return v___x_516_;
}
else
{
goto v___jp_464_;
}
}
else
{
if (v_isExporting_454_ == 0)
{
goto v___jp_464_;
}
else
{
lean_object* v___x_517_; 
lean_inc(v___y_456_);
lean_inc_ref(v___y_455_);
v___x_517_ = lean_apply_3(v_x_453_, v___y_455_, v___y_456_, lean_box(0));
return v___x_517_;
}
}
v___jp_464_:
{
lean_object* v___x_465_; lean_object* v_env_466_; lean_object* v_messages_467_; lean_object* v_scopes_468_; lean_object* v_usedQuotCtxts_469_; lean_object* v_nextMacroScope_470_; lean_object* v_maxRecDepth_471_; lean_object* v_ngen_472_; lean_object* v_auxDeclNGen_473_; lean_object* v_infoState_474_; lean_object* v_traceState_475_; lean_object* v_snapshotTasks_476_; lean_object* v_prevLinterStates_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_515_; 
v___x_465_ = lean_st_ref_take(v___y_456_);
v_env_466_ = lean_ctor_get(v___x_465_, 0);
v_messages_467_ = lean_ctor_get(v___x_465_, 1);
v_scopes_468_ = lean_ctor_get(v___x_465_, 2);
v_usedQuotCtxts_469_ = lean_ctor_get(v___x_465_, 3);
v_nextMacroScope_470_ = lean_ctor_get(v___x_465_, 4);
v_maxRecDepth_471_ = lean_ctor_get(v___x_465_, 5);
v_ngen_472_ = lean_ctor_get(v___x_465_, 6);
v_auxDeclNGen_473_ = lean_ctor_get(v___x_465_, 7);
v_infoState_474_ = lean_ctor_get(v___x_465_, 8);
v_traceState_475_ = lean_ctor_get(v___x_465_, 9);
v_snapshotTasks_476_ = lean_ctor_get(v___x_465_, 10);
v_prevLinterStates_477_ = lean_ctor_get(v___x_465_, 11);
v_isSharedCheck_515_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_515_ == 0)
{
v___x_479_ = v___x_465_;
v_isShared_480_ = v_isSharedCheck_515_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_prevLinterStates_477_);
lean_inc(v_snapshotTasks_476_);
lean_inc(v_traceState_475_);
lean_inc(v_infoState_474_);
lean_inc(v_auxDeclNGen_473_);
lean_inc(v_ngen_472_);
lean_inc(v_maxRecDepth_471_);
lean_inc(v_nextMacroScope_470_);
lean_inc(v_usedQuotCtxts_469_);
lean_inc(v_scopes_468_);
lean_inc(v_messages_467_);
lean_inc(v_env_466_);
lean_dec(v___x_465_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_515_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_481_ = l_Lean_Environment_setExporting(v_env_466_, v_isExporting_454_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 0, v___x_481_);
v___x_483_ = v___x_479_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_481_);
lean_ctor_set(v_reuseFailAlloc_514_, 1, v_messages_467_);
lean_ctor_set(v_reuseFailAlloc_514_, 2, v_scopes_468_);
lean_ctor_set(v_reuseFailAlloc_514_, 3, v_usedQuotCtxts_469_);
lean_ctor_set(v_reuseFailAlloc_514_, 4, v_nextMacroScope_470_);
lean_ctor_set(v_reuseFailAlloc_514_, 5, v_maxRecDepth_471_);
lean_ctor_set(v_reuseFailAlloc_514_, 6, v_ngen_472_);
lean_ctor_set(v_reuseFailAlloc_514_, 7, v_auxDeclNGen_473_);
lean_ctor_set(v_reuseFailAlloc_514_, 8, v_infoState_474_);
lean_ctor_set(v_reuseFailAlloc_514_, 9, v_traceState_475_);
lean_ctor_set(v_reuseFailAlloc_514_, 10, v_snapshotTasks_476_);
lean_ctor_set(v_reuseFailAlloc_514_, 11, v_prevLinterStates_477_);
v___x_483_ = v_reuseFailAlloc_514_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_484_; lean_object* v_r_485_; 
v___x_484_ = lean_st_ref_set(v___y_456_, v___x_483_);
lean_inc(v___y_456_);
lean_inc_ref(v___y_455_);
v_r_485_ = lean_apply_3(v_x_453_, v___y_455_, v___y_456_, lean_box(0));
if (lean_obj_tag(v_r_485_) == 0)
{
lean_object* v_a_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_502_; 
v_a_486_ = lean_ctor_get(v_r_485_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v_r_485_);
if (v_isSharedCheck_502_ == 0)
{
v___x_488_ = v_r_485_;
v_isShared_489_ = v_isSharedCheck_502_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_a_486_);
lean_dec(v_r_485_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_502_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v___x_491_; 
lean_inc(v_a_486_);
if (v_isShared_489_ == 0)
{
lean_ctor_set_tag(v___x_488_, 1);
v___x_491_ = v___x_488_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_a_486_);
v___x_491_ = v_reuseFailAlloc_501_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
lean_object* v___x_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
v___x_492_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0(v___y_456_, v_isExporting_460_, v___x_491_);
lean_dec_ref(v___x_491_);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_499_ == 0)
{
lean_object* v_unused_500_; 
v_unused_500_ = lean_ctor_get(v___x_492_, 0);
lean_dec(v_unused_500_);
v___x_494_ = v___x_492_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_dec(v___x_492_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 0, v_a_486_);
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_486_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
}
}
else
{
lean_object* v_a_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_512_; 
v_a_503_ = lean_ctor_get(v_r_485_, 0);
lean_inc(v_a_503_);
lean_dec_ref_known(v_r_485_, 1);
v___x_504_ = lean_box(0);
v___x_505_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___lam__0(v___y_456_, v_isExporting_460_, v___x_504_);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_512_ == 0)
{
lean_object* v_unused_513_; 
v_unused_513_ = lean_ctor_get(v___x_505_, 0);
lean_dec(v_unused_513_);
v___x_507_ = v___x_505_;
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
else
{
lean_dec(v___x_505_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_510_; 
if (v_isShared_508_ == 0)
{
lean_ctor_set_tag(v___x_507_, 1);
lean_ctor_set(v___x_507_, 0, v_a_503_);
v___x_510_ = v___x_507_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v_a_503_);
v___x_510_ = v_reuseFailAlloc_511_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
return v___x_510_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg___boxed(lean_object* v_x_518_, lean_object* v_isExporting_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_){
_start:
{
uint8_t v_isExporting_boxed_523_; lean_object* v_res_524_; 
v_isExporting_boxed_523_ = lean_unbox(v_isExporting_519_);
v_res_524_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg(v_x_518_, v_isExporting_boxed_523_, v___y_520_, v___y_521_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg(lean_object* v_x_525_, uint8_t v_when_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
if (v_when_526_ == 0)
{
lean_object* v___x_530_; 
lean_inc(v___y_528_);
lean_inc_ref(v___y_527_);
v___x_530_ = lean_apply_3(v_x_525_, v___y_527_, v___y_528_, lean_box(0));
return v___x_530_;
}
else
{
uint8_t v___x_531_; lean_object* v___x_532_; 
v___x_531_ = 0;
v___x_532_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg(v_x_525_, v___x_531_, v___y_527_, v___y_528_);
return v___x_532_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg___boxed(lean_object* v_x_533_, lean_object* v_when_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_){
_start:
{
uint8_t v_when_boxed_538_; lean_object* v_res_539_; 
v_when_boxed_538_ = lean_unbox(v_when_534_);
v_res_539_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg(v_x_533_, v_when_boxed_538_, v___y_535_, v___y_536_);
lean_dec(v___y_536_);
lean_dec_ref(v___y_535_);
return v_res_539_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__0);
v___x_542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
return v___x_542_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_543_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1);
v___x_544_ = lean_unsigned_to_nat(0u);
v___x_545_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
lean_ctor_set(v___x_545_, 2, v___x_544_);
lean_ctor_set(v___x_545_, 3, v___x_544_);
lean_ctor_set(v___x_545_, 4, v___x_543_);
lean_ctor_set(v___x_545_, 5, v___x_543_);
lean_ctor_set(v___x_545_, 6, v___x_543_);
lean_ctor_set(v___x_545_, 7, v___x_543_);
lean_ctor_set(v___x_545_, 8, v___x_543_);
lean_ctor_set(v___x_545_, 9, v___x_543_);
return v___x_545_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_546_ = lean_unsigned_to_nat(32u);
v___x_547_ = lean_mk_empty_array_with_capacity(v___x_546_);
v___x_548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4(void){
_start:
{
size_t v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; 
v___x_549_ = ((size_t)5ULL);
v___x_550_ = lean_unsigned_to_nat(0u);
v___x_551_ = lean_unsigned_to_nat(32u);
v___x_552_ = lean_mk_empty_array_with_capacity(v___x_551_);
v___x_553_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__3);
v___x_554_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_554_, 0, v___x_553_);
lean_ctor_set(v___x_554_, 1, v___x_552_);
lean_ctor_set(v___x_554_, 2, v___x_550_);
lean_ctor_set(v___x_554_, 3, v___x_550_);
lean_ctor_set_usize(v___x_554_, 4, v___x_549_);
return v___x_554_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5(void){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_555_ = lean_box(1);
v___x_556_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__4);
v___x_557_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__1);
v___x_558_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
lean_ctor_set(v___x_558_, 1, v___x_556_);
lean_ctor_set(v___x_558_, 2, v___x_555_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_msgData_559_, lean_object* v___y_560_){
_start:
{
lean_object* v___x_562_; lean_object* v_env_563_; lean_object* v___x_564_; lean_object* v_scopes_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v_opts_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_562_ = lean_st_ref_get(v___y_560_);
v_env_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc_ref(v_env_563_);
lean_dec(v___x_562_);
v___x_564_ = lean_st_ref_get(v___y_560_);
v_scopes_565_ = lean_ctor_get(v___x_564_, 2);
lean_inc(v_scopes_565_);
lean_dec(v___x_564_);
v___x_566_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_567_ = l_List_head_x21___redArg(v___x_566_, v_scopes_565_);
lean_dec(v_scopes_565_);
v_opts_568_ = lean_ctor_get(v___x_567_, 1);
lean_inc_ref(v_opts_568_);
lean_dec(v___x_567_);
v___x_569_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__2);
v___x_570_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___closed__5);
v___x_571_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_571_, 0, v_env_563_);
lean_ctor_set(v___x_571_, 1, v___x_569_);
lean_ctor_set(v___x_571_, 2, v___x_570_);
lean_ctor_set(v___x_571_, 3, v_opts_568_);
v___x_572_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_572_, 0, v___x_571_);
lean_ctor_set(v___x_572_, 1, v_msgData_559_);
v___x_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_msgData_574_, lean_object* v___y_575_, lean_object* v___y_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msgData_574_, v___y_575_);
lean_dec(v___y_575_);
return v_res_577_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0(void){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_578_ = lean_box(1);
v___x_579_ = l_Lean_MessageData_ofFormat(v___x_578_);
return v___x_579_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3(void){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__2));
v___x_584_ = l_Lean_MessageData_ofFormat(v___x_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9(lean_object* v_x_585_, lean_object* v_x_586_){
_start:
{
if (lean_obj_tag(v_x_586_) == 0)
{
return v_x_585_;
}
else
{
lean_object* v_head_587_; lean_object* v_tail_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_610_; 
v_head_587_ = lean_ctor_get(v_x_586_, 0);
v_tail_588_ = lean_ctor_get(v_x_586_, 1);
v_isSharedCheck_610_ = !lean_is_exclusive(v_x_586_);
if (v_isSharedCheck_610_ == 0)
{
v___x_590_ = v_x_586_;
v_isShared_591_ = v_isSharedCheck_610_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_tail_588_);
lean_inc(v_head_587_);
lean_dec(v_x_586_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_610_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v_before_592_; lean_object* v___x_594_; uint8_t v_isShared_595_; uint8_t v_isSharedCheck_608_; 
v_before_592_ = lean_ctor_get(v_head_587_, 0);
v_isSharedCheck_608_ = !lean_is_exclusive(v_head_587_);
if (v_isSharedCheck_608_ == 0)
{
lean_object* v_unused_609_; 
v_unused_609_ = lean_ctor_get(v_head_587_, 1);
lean_dec(v_unused_609_);
v___x_594_ = v_head_587_;
v_isShared_595_ = v_isSharedCheck_608_;
goto v_resetjp_593_;
}
else
{
lean_inc(v_before_592_);
lean_dec(v_head_587_);
v___x_594_ = lean_box(0);
v_isShared_595_ = v_isSharedCheck_608_;
goto v_resetjp_593_;
}
v_resetjp_593_:
{
lean_object* v___x_596_; lean_object* v___x_598_; 
v___x_596_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0);
if (v_isShared_595_ == 0)
{
lean_ctor_set_tag(v___x_594_, 7);
lean_ctor_set(v___x_594_, 1, v___x_596_);
lean_ctor_set(v___x_594_, 0, v_x_585_);
v___x_598_ = v___x_594_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_x_585_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v___x_596_);
v___x_598_ = v_reuseFailAlloc_607_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
lean_object* v___x_599_; lean_object* v___x_601_; 
v___x_599_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__3);
if (v_isShared_591_ == 0)
{
lean_ctor_set_tag(v___x_590_, 7);
lean_ctor_set(v___x_590_, 1, v___x_599_);
lean_ctor_set(v___x_590_, 0, v___x_598_);
v___x_601_ = v___x_590_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_598_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v___x_599_);
v___x_601_ = v_reuseFailAlloc_606_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_602_ = l_Lean_MessageData_ofSyntax(v_before_592_);
v___x_603_ = l_Lean_indentD(v___x_602_);
v___x_604_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_604_, 0, v___x_601_);
lean_ctor_set(v___x_604_, 1, v___x_603_);
v_x_585_ = v___x_604_;
v_x_586_ = v_tail_588_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8(lean_object* v_opts_611_, lean_object* v_opt_612_){
_start:
{
lean_object* v_name_613_; lean_object* v_defValue_614_; lean_object* v_map_615_; lean_object* v___x_616_; 
v_name_613_ = lean_ctor_get(v_opt_612_, 0);
v_defValue_614_ = lean_ctor_get(v_opt_612_, 1);
v_map_615_ = lean_ctor_get(v_opts_611_, 0);
v___x_616_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_615_, v_name_613_);
if (lean_obj_tag(v___x_616_) == 0)
{
uint8_t v___x_617_; 
v___x_617_ = lean_unbox(v_defValue_614_);
return v___x_617_;
}
else
{
lean_object* v_val_618_; 
v_val_618_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_val_618_);
lean_dec_ref_known(v___x_616_, 1);
if (lean_obj_tag(v_val_618_) == 1)
{
uint8_t v_v_619_; 
v_v_619_ = lean_ctor_get_uint8(v_val_618_, 0);
lean_dec_ref_known(v_val_618_, 0);
return v_v_619_;
}
else
{
uint8_t v___x_620_; 
lean_dec(v_val_618_);
v___x_620_ = lean_unbox(v_defValue_614_);
return v___x_620_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8___boxed(lean_object* v_opts_621_, lean_object* v_opt_622_){
_start:
{
uint8_t v_res_623_; lean_object* v_r_624_; 
v_res_623_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8(v_opts_621_, v_opt_622_);
lean_dec_ref(v_opt_622_);
lean_dec_ref(v_opts_621_);
v_r_624_ = lean_box(v_res_623_);
return v_r_624_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_628_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__1));
v___x_629_ = l_Lean_MessageData_ofFormat(v___x_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg(lean_object* v_msgData_630_, lean_object* v_macroStack_631_, lean_object* v___y_632_){
_start:
{
lean_object* v___x_634_; lean_object* v_scopes_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v_opts_638_; lean_object* v___x_639_; uint8_t v___x_640_; 
v___x_634_ = lean_st_ref_get(v___y_632_);
v_scopes_635_ = lean_ctor_get(v___x_634_, 2);
lean_inc(v_scopes_635_);
lean_dec(v___x_634_);
v___x_636_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_637_ = l_List_head_x21___redArg(v___x_636_, v_scopes_635_);
lean_dec(v_scopes_635_);
v_opts_638_ = lean_ctor_get(v___x_637_, 1);
lean_inc_ref(v_opts_638_);
lean_dec(v___x_637_);
v___x_639_ = l_Lean_Elab_pp_macroStack;
v___x_640_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8(v_opts_638_, v___x_639_);
lean_dec_ref(v_opts_638_);
if (v___x_640_ == 0)
{
lean_object* v___x_641_; 
lean_dec(v_macroStack_631_);
v___x_641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_641_, 0, v_msgData_630_);
return v___x_641_;
}
else
{
if (lean_obj_tag(v_macroStack_631_) == 0)
{
lean_object* v___x_642_; 
v___x_642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_642_, 0, v_msgData_630_);
return v___x_642_;
}
else
{
lean_object* v_head_643_; lean_object* v_after_644_; lean_object* v___x_646_; uint8_t v_isShared_647_; uint8_t v_isSharedCheck_659_; 
v_head_643_ = lean_ctor_get(v_macroStack_631_, 0);
lean_inc(v_head_643_);
v_after_644_ = lean_ctor_get(v_head_643_, 1);
v_isSharedCheck_659_ = !lean_is_exclusive(v_head_643_);
if (v_isSharedCheck_659_ == 0)
{
lean_object* v_unused_660_; 
v_unused_660_ = lean_ctor_get(v_head_643_, 0);
lean_dec(v_unused_660_);
v___x_646_ = v_head_643_;
v_isShared_647_ = v_isSharedCheck_659_;
goto v_resetjp_645_;
}
else
{
lean_inc(v_after_644_);
lean_dec(v_head_643_);
v___x_646_ = lean_box(0);
v_isShared_647_ = v_isSharedCheck_659_;
goto v_resetjp_645_;
}
v_resetjp_645_:
{
lean_object* v___x_648_; lean_object* v___x_650_; 
v___x_648_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9___closed__0);
if (v_isShared_647_ == 0)
{
lean_ctor_set_tag(v___x_646_, 7);
lean_ctor_set(v___x_646_, 1, v___x_648_);
lean_ctor_set(v___x_646_, 0, v_msgData_630_);
v___x_650_ = v___x_646_;
goto v_reusejp_649_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v_msgData_630_);
lean_ctor_set(v_reuseFailAlloc_658_, 1, v___x_648_);
v___x_650_ = v_reuseFailAlloc_658_;
goto v_reusejp_649_;
}
v_reusejp_649_:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v_msgData_655_; lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_651_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___closed__2);
v___x_652_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_652_, 0, v___x_650_);
lean_ctor_set(v___x_652_, 1, v___x_651_);
v___x_653_ = l_Lean_MessageData_ofSyntax(v_after_644_);
v___x_654_ = l_Lean_indentD(v___x_653_);
v_msgData_655_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_655_, 0, v___x_652_);
lean_ctor_set(v_msgData_655_, 1, v___x_654_);
v___x_656_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__9(v_msgData_655_, v_macroStack_631_);
v___x_657_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_657_, 0, v___x_656_);
return v___x_657_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_msgData_661_, lean_object* v_macroStack_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg(v_msgData_661_, v_macroStack_662_, v___y_663_);
lean_dec(v___y_663_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_msg_666_, lean_object* v___y_667_, lean_object* v___y_668_){
_start:
{
lean_object* v___x_670_; 
v___x_670_ = l_Lean_Elab_Command_getRef___redArg(v___y_667_);
if (lean_obj_tag(v___x_670_) == 0)
{
lean_object* v_a_671_; lean_object* v_macroStack_672_; lean_object* v___x_673_; lean_object* v_a_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v_a_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_685_; 
v_a_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_a_671_);
lean_dec_ref_known(v___x_670_, 1);
v_macroStack_672_ = lean_ctor_get(v___y_667_, 4);
v___x_673_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msg_666_, v___y_668_);
v_a_674_ = lean_ctor_get(v___x_673_, 0);
lean_inc(v_a_674_);
lean_dec_ref(v___x_673_);
v___x_675_ = l_Lean_Elab_getBetterRef(v_a_671_, v_macroStack_672_);
lean_dec(v_a_671_);
lean_inc(v_macroStack_672_);
v___x_676_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg(v_a_674_, v_macroStack_672_, v___y_668_);
v_a_677_ = lean_ctor_get(v___x_676_, 0);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_676_);
if (v_isSharedCheck_685_ == 0)
{
v___x_679_ = v___x_676_;
v_isShared_680_ = v_isSharedCheck_685_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_a_677_);
lean_dec(v___x_676_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_685_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_681_; lean_object* v___x_683_; 
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_675_);
lean_ctor_set(v___x_681_, 1, v_a_677_);
if (v_isShared_680_ == 0)
{
lean_ctor_set_tag(v___x_679_, 1);
lean_ctor_set(v___x_679_, 0, v___x_681_);
v___x_683_ = v___x_679_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v___x_681_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
else
{
lean_object* v_a_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_693_; 
lean_dec_ref(v_msg_666_);
v_a_686_ = lean_ctor_get(v___x_670_, 0);
v_isSharedCheck_693_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_693_ == 0)
{
v___x_688_ = v___x_670_;
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_a_686_);
lean_dec(v___x_670_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_691_; 
if (v_isShared_689_ == 0)
{
v___x_691_ = v___x_688_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v_a_686_);
v___x_691_ = v_reuseFailAlloc_692_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
return v___x_691_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msg_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(v_msg_694_, v___y_695_, v___y_696_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
return v_res_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg(lean_object* v___y_699_){
_start:
{
lean_object* v___x_701_; lean_object* v_env_702_; lean_object* v___x_703_; lean_object* v_mainModule_704_; lean_object* v___x_705_; 
v___x_701_ = lean_st_ref_get(v___y_699_);
v_env_702_ = lean_ctor_get(v___x_701_, 0);
lean_inc_ref(v_env_702_);
lean_dec(v___x_701_);
v___x_703_ = l_Lean_Environment_header(v_env_702_);
lean_dec_ref(v_env_702_);
v_mainModule_704_ = lean_ctor_get(v___x_703_, 0);
lean_inc(v_mainModule_704_);
lean_dec_ref(v___x_703_);
v___x_705_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_705_, 0, v_mainModule_704_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg___boxed(lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg(v___y_706_);
lean_dec(v___y_706_);
return v_res_708_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3(void){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_714_ = l_Lean_maxRecDepthErrorMessage;
v___x_715_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
return v___x_715_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4(void){
_start:
{
lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_716_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__3);
v___x_717_ = l_Lean_MessageData_ofFormat(v___x_716_);
return v___x_717_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5(void){
_start:
{
lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_718_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__4);
v___x_719_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__2));
v___x_720_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_720_, 0, v___x_719_);
lean_ctor_set(v___x_720_, 1, v___x_718_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg(lean_object* v_ref_721_){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_723_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___closed__5);
v___x_724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_724_, 0, v_ref_721_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg___boxed(lean_object* v_ref_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg(v_ref_726_);
return v_res_728_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg(lean_object* v_keys_729_, lean_object* v_i_730_, lean_object* v_k_731_){
_start:
{
lean_object* v___x_732_; uint8_t v___x_733_; 
v___x_732_ = lean_array_get_size(v_keys_729_);
v___x_733_ = lean_nat_dec_lt(v_i_730_, v___x_732_);
if (v___x_733_ == 0)
{
lean_dec(v_i_730_);
return v___x_733_;
}
else
{
lean_object* v_k_x27_734_; uint8_t v___x_735_; 
v_k_x27_734_ = lean_array_fget_borrowed(v_keys_729_, v_i_730_);
v___x_735_ = l_Lean_instBEqExtraModUse_beq(v_k_731_, v_k_x27_734_);
if (v___x_735_ == 0)
{
lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_736_ = lean_unsigned_to_nat(1u);
v___x_737_ = lean_nat_add(v_i_730_, v___x_736_);
lean_dec(v_i_730_);
v_i_730_ = v___x_737_;
goto _start;
}
else
{
lean_dec(v_i_730_);
return v___x_735_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg___boxed(lean_object* v_keys_739_, lean_object* v_i_740_, lean_object* v_k_741_){
_start:
{
uint8_t v_res_742_; lean_object* v_r_743_; 
v_res_742_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg(v_keys_739_, v_i_740_, v_k_741_);
lean_dec_ref(v_k_741_);
lean_dec_ref(v_keys_739_);
v_r_743_ = lean_box(v_res_742_);
return v_r_743_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg(lean_object* v_x_744_, size_t v_x_745_, lean_object* v_x_746_){
_start:
{
if (lean_obj_tag(v_x_744_) == 0)
{
lean_object* v_es_747_; lean_object* v___x_748_; size_t v___x_749_; size_t v___x_750_; lean_object* v_j_751_; lean_object* v___x_752_; 
v_es_747_ = lean_ctor_get(v_x_744_, 0);
v___x_748_ = lean_box(2);
v___x_749_ = ((size_t)31ULL);
v___x_750_ = lean_usize_land(v_x_745_, v___x_749_);
v_j_751_ = lean_usize_to_nat(v___x_750_);
v___x_752_ = lean_array_get_borrowed(v___x_748_, v_es_747_, v_j_751_);
lean_dec(v_j_751_);
switch(lean_obj_tag(v___x_752_))
{
case 0:
{
lean_object* v_key_753_; uint8_t v___x_754_; 
v_key_753_ = lean_ctor_get(v___x_752_, 0);
v___x_754_ = l_Lean_instBEqExtraModUse_beq(v_x_746_, v_key_753_);
return v___x_754_;
}
case 1:
{
lean_object* v_node_755_; size_t v___x_756_; size_t v___x_757_; 
v_node_755_ = lean_ctor_get(v___x_752_, 0);
v___x_756_ = ((size_t)5ULL);
v___x_757_ = lean_usize_shift_right(v_x_745_, v___x_756_);
v_x_744_ = v_node_755_;
v_x_745_ = v___x_757_;
goto _start;
}
default: 
{
uint8_t v___x_759_; 
v___x_759_ = 0;
return v___x_759_;
}
}
}
else
{
lean_object* v_ks_760_; lean_object* v___x_761_; uint8_t v___x_762_; 
v_ks_760_ = lean_ctor_get(v_x_744_, 0);
v___x_761_ = lean_unsigned_to_nat(0u);
v___x_762_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg(v_ks_760_, v___x_761_, v_x_746_);
return v___x_762_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg___boxed(lean_object* v_x_763_, lean_object* v_x_764_, lean_object* v_x_765_){
_start:
{
size_t v_x_21463__boxed_766_; uint8_t v_res_767_; lean_object* v_r_768_; 
v_x_21463__boxed_766_ = lean_unbox_usize(v_x_764_);
lean_dec(v_x_764_);
v_res_767_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg(v_x_763_, v_x_21463__boxed_766_, v_x_765_);
lean_dec_ref(v_x_765_);
lean_dec_ref(v_x_763_);
v_r_768_ = lean_box(v_res_767_);
return v_r_768_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg(lean_object* v_x_769_, lean_object* v_x_770_){
_start:
{
uint64_t v___x_771_; size_t v___x_772_; uint8_t v___x_773_; 
v___x_771_ = l_Lean_instHashableExtraModUse_hash(v_x_770_);
v___x_772_ = lean_uint64_to_usize(v___x_771_);
v___x_773_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg(v_x_769_, v___x_772_, v_x_770_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg___boxed(lean_object* v_x_774_, lean_object* v_x_775_){
_start:
{
uint8_t v_res_776_; lean_object* v_r_777_; 
v_res_776_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg(v_x_774_, v_x_775_);
lean_dec_ref(v_x_775_);
lean_dec_ref(v_x_774_);
v_r_777_ = lean_box(v_res_776_);
return v_r_777_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0(void){
_start:
{
lean_object* v___x_778_; double v___x_779_; 
v___x_778_ = lean_unsigned_to_nat(0u);
v___x_779_ = lean_float_of_nat(v___x_778_);
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21(lean_object* v_cls_782_, lean_object* v_msg_783_, lean_object* v___y_784_, lean_object* v___y_785_){
_start:
{
lean_object* v___x_787_; 
v___x_787_ = l_Lean_Elab_Command_getRef___redArg(v___y_784_);
if (lean_obj_tag(v___x_787_) == 0)
{
lean_object* v_a_788_; lean_object* v___x_789_; lean_object* v_a_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_837_; 
v_a_788_ = lean_ctor_get(v___x_787_, 0);
lean_inc(v_a_788_);
lean_dec_ref_known(v___x_787_, 1);
v___x_789_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msg_783_, v___y_785_);
v_a_790_ = lean_ctor_get(v___x_789_, 0);
v_isSharedCheck_837_ = !lean_is_exclusive(v___x_789_);
if (v_isSharedCheck_837_ == 0)
{
v___x_792_ = v___x_789_;
v_isShared_793_ = v_isSharedCheck_837_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_a_790_);
lean_dec(v___x_789_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_837_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_794_; lean_object* v_traceState_795_; lean_object* v_env_796_; lean_object* v_messages_797_; lean_object* v_scopes_798_; lean_object* v_usedQuotCtxts_799_; lean_object* v_nextMacroScope_800_; lean_object* v_maxRecDepth_801_; lean_object* v_ngen_802_; lean_object* v_auxDeclNGen_803_; lean_object* v_infoState_804_; lean_object* v_snapshotTasks_805_; lean_object* v_prevLinterStates_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_836_; 
v___x_794_ = lean_st_ref_take(v___y_785_);
v_traceState_795_ = lean_ctor_get(v___x_794_, 9);
v_env_796_ = lean_ctor_get(v___x_794_, 0);
v_messages_797_ = lean_ctor_get(v___x_794_, 1);
v_scopes_798_ = lean_ctor_get(v___x_794_, 2);
v_usedQuotCtxts_799_ = lean_ctor_get(v___x_794_, 3);
v_nextMacroScope_800_ = lean_ctor_get(v___x_794_, 4);
v_maxRecDepth_801_ = lean_ctor_get(v___x_794_, 5);
v_ngen_802_ = lean_ctor_get(v___x_794_, 6);
v_auxDeclNGen_803_ = lean_ctor_get(v___x_794_, 7);
v_infoState_804_ = lean_ctor_get(v___x_794_, 8);
v_snapshotTasks_805_ = lean_ctor_get(v___x_794_, 10);
v_prevLinterStates_806_ = lean_ctor_get(v___x_794_, 11);
v_isSharedCheck_836_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_836_ == 0)
{
v___x_808_ = v___x_794_;
v_isShared_809_ = v_isSharedCheck_836_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_prevLinterStates_806_);
lean_inc(v_snapshotTasks_805_);
lean_inc(v_traceState_795_);
lean_inc(v_infoState_804_);
lean_inc(v_auxDeclNGen_803_);
lean_inc(v_ngen_802_);
lean_inc(v_maxRecDepth_801_);
lean_inc(v_nextMacroScope_800_);
lean_inc(v_usedQuotCtxts_799_);
lean_inc(v_scopes_798_);
lean_inc(v_messages_797_);
lean_inc(v_env_796_);
lean_dec(v___x_794_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_836_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
uint64_t v_tid_810_; lean_object* v_traces_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_835_; 
v_tid_810_ = lean_ctor_get_uint64(v_traceState_795_, sizeof(void*)*1);
v_traces_811_ = lean_ctor_get(v_traceState_795_, 0);
v_isSharedCheck_835_ = !lean_is_exclusive(v_traceState_795_);
if (v_isSharedCheck_835_ == 0)
{
v___x_813_ = v_traceState_795_;
v_isShared_814_ = v_isSharedCheck_835_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_traces_811_);
lean_dec(v_traceState_795_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_835_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v___x_815_; double v___x_816_; uint8_t v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_825_; 
v___x_815_ = lean_box(0);
v___x_816_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0, &lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__0);
v___x_817_ = 0;
v___x_818_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___x_819_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_819_, 0, v_cls_782_);
lean_ctor_set(v___x_819_, 1, v___x_815_);
lean_ctor_set(v___x_819_, 2, v___x_818_);
lean_ctor_set_float(v___x_819_, sizeof(void*)*3, v___x_816_);
lean_ctor_set_float(v___x_819_, sizeof(void*)*3 + 8, v___x_816_);
lean_ctor_set_uint8(v___x_819_, sizeof(void*)*3 + 16, v___x_817_);
v___x_820_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___closed__1));
v___x_821_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_821_, 0, v___x_819_);
lean_ctor_set(v___x_821_, 1, v_a_790_);
lean_ctor_set(v___x_821_, 2, v___x_820_);
v___x_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_822_, 0, v_a_788_);
lean_ctor_set(v___x_822_, 1, v___x_821_);
v___x_823_ = l_Lean_PersistentArray_push___redArg(v_traces_811_, v___x_822_);
if (v_isShared_814_ == 0)
{
lean_ctor_set(v___x_813_, 0, v___x_823_);
v___x_825_ = v___x_813_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_823_);
lean_ctor_set_uint64(v_reuseFailAlloc_834_, sizeof(void*)*1, v_tid_810_);
v___x_825_ = v_reuseFailAlloc_834_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
lean_object* v___x_827_; 
if (v_isShared_809_ == 0)
{
lean_ctor_set(v___x_808_, 9, v___x_825_);
v___x_827_ = v___x_808_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_env_796_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v_messages_797_);
lean_ctor_set(v_reuseFailAlloc_833_, 2, v_scopes_798_);
lean_ctor_set(v_reuseFailAlloc_833_, 3, v_usedQuotCtxts_799_);
lean_ctor_set(v_reuseFailAlloc_833_, 4, v_nextMacroScope_800_);
lean_ctor_set(v_reuseFailAlloc_833_, 5, v_maxRecDepth_801_);
lean_ctor_set(v_reuseFailAlloc_833_, 6, v_ngen_802_);
lean_ctor_set(v_reuseFailAlloc_833_, 7, v_auxDeclNGen_803_);
lean_ctor_set(v_reuseFailAlloc_833_, 8, v_infoState_804_);
lean_ctor_set(v_reuseFailAlloc_833_, 9, v___x_825_);
lean_ctor_set(v_reuseFailAlloc_833_, 10, v_snapshotTasks_805_);
lean_ctor_set(v_reuseFailAlloc_833_, 11, v_prevLinterStates_806_);
v___x_827_ = v_reuseFailAlloc_833_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_831_; 
v___x_828_ = lean_st_ref_set(v___y_785_, v___x_827_);
v___x_829_ = lean_box(0);
if (v_isShared_793_ == 0)
{
lean_ctor_set(v___x_792_, 0, v___x_829_);
v___x_831_ = v___x_792_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v___x_829_);
v___x_831_ = v_reuseFailAlloc_832_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
return v___x_831_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_845_; 
lean_dec_ref(v_msg_783_);
lean_dec(v_cls_782_);
v_a_838_ = lean_ctor_get(v___x_787_, 0);
v_isSharedCheck_845_ = !lean_is_exclusive(v___x_787_);
if (v_isSharedCheck_845_ == 0)
{
v___x_840_ = v___x_787_;
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_a_838_);
lean_dec(v___x_787_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v___x_843_; 
if (v_isShared_841_ == 0)
{
v___x_843_ = v___x_840_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_a_838_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21___boxed(lean_object* v_cls_846_, lean_object* v_msg_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21(v_cls_846_, v_msg_847_, v___y_848_, v___y_849_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
return v_res_851_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2(void){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; 
v___x_854_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__1));
v___x_855_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__0));
v___x_856_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_855_, v___x_854_);
return v___x_856_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6(void){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_861_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__5));
v___x_862_ = l_Lean_stringToMessageData(v___x_861_);
return v___x_862_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; 
v___x_864_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__7));
v___x_865_ = l_Lean_stringToMessageData(v___x_864_);
return v___x_865_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9(void){
_start:
{
lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_866_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___x_867_ = l_Lean_stringToMessageData(v___x_866_);
return v___x_867_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12(void){
_start:
{
lean_object* v_cls_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v_cls_871_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__4));
v___x_872_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__11));
v___x_873_ = l_Lean_Name_append(v___x_872_, v_cls_871_);
return v___x_873_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14(void){
_start:
{
lean_object* v___x_875_; lean_object* v___x_876_; 
v___x_875_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__13));
v___x_876_ = l_Lean_stringToMessageData(v___x_875_);
return v___x_876_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16(void){
_start:
{
lean_object* v___x_878_; lean_object* v___x_879_; 
v___x_878_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__15));
v___x_879_ = l_Lean_stringToMessageData(v___x_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29(lean_object* v_mod_884_, uint8_t v_isMeta_885_, lean_object* v_hint_886_, lean_object* v___y_887_, lean_object* v___y_888_){
_start:
{
lean_object* v___x_890_; lean_object* v_env_891_; uint8_t v_isExporting_892_; lean_object* v___x_893_; lean_object* v_env_894_; lean_object* v___x_895_; lean_object* v_entry_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___y_901_; lean_object* v___x_928_; uint8_t v___x_929_; 
v___x_890_ = lean_st_ref_get(v___y_888_);
v_env_891_ = lean_ctor_get(v___x_890_, 0);
lean_inc_ref(v_env_891_);
lean_dec(v___x_890_);
v_isExporting_892_ = lean_ctor_get_uint8(v_env_891_, sizeof(void*)*8);
lean_dec_ref(v_env_891_);
v___x_893_ = lean_st_ref_get(v___y_888_);
v_env_894_ = lean_ctor_get(v___x_893_, 0);
lean_inc_ref(v_env_894_);
lean_dec(v___x_893_);
v___x_895_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__2);
lean_inc(v_mod_884_);
v_entry_896_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_entry_896_, 0, v_mod_884_);
lean_ctor_set_uint8(v_entry_896_, sizeof(void*)*1, v_isExporting_892_);
lean_ctor_set_uint8(v_entry_896_, sizeof(void*)*1 + 1, v_isMeta_885_);
v___x_897_ = l___private_Lean_ExtraModUses_0__Lean_extraModUses;
v___x_898_ = lean_box(1);
v___x_899_ = lean_box(0);
v___x_928_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_895_, v___x_897_, v_env_894_, v___x_898_, v___x_899_);
v___x_929_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg(v___x_928_, v_entry_896_);
lean_dec(v___x_928_);
if (v___x_929_ == 0)
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v_scopes_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v_opts_936_; uint8_t v_hasTrace_937_; 
v___x_930_ = l_Lean_inheritedTraceOptions;
v___x_931_ = lean_st_ref_get(v___x_930_);
v___x_932_ = lean_st_ref_get(v___y_888_);
v_scopes_933_ = lean_ctor_get(v___x_932_, 2);
lean_inc(v_scopes_933_);
lean_dec(v___x_932_);
v___x_934_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_935_ = l_List_head_x21___redArg(v___x_934_, v_scopes_933_);
lean_dec(v_scopes_933_);
v_opts_936_ = lean_ctor_get(v___x_935_, 1);
lean_inc_ref(v_opts_936_);
lean_dec(v___x_935_);
v_hasTrace_937_ = lean_ctor_get_uint8(v_opts_936_, sizeof(void*)*1);
if (v_hasTrace_937_ == 0)
{
lean_dec_ref(v_opts_936_);
lean_dec(v___x_931_);
lean_dec(v_hint_886_);
lean_dec(v_mod_884_);
v___y_901_ = v___y_888_;
goto v___jp_900_;
}
else
{
lean_object* v_cls_938_; lean_object* v___y_940_; lean_object* v___y_941_; lean_object* v___y_945_; lean_object* v___y_946_; lean_object* v___x_958_; uint8_t v___x_959_; 
v_cls_938_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__4));
v___x_958_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__12);
v___x_959_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___x_931_, v_opts_936_, v___x_958_);
lean_dec_ref(v_opts_936_);
lean_dec(v___x_931_);
if (v___x_959_ == 0)
{
lean_dec(v_hint_886_);
lean_dec(v_mod_884_);
v___y_901_ = v___y_888_;
goto v___jp_900_;
}
else
{
lean_object* v___x_960_; lean_object* v___y_962_; 
v___x_960_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__14);
if (v_isExporting_892_ == 0)
{
lean_object* v___x_969_; 
v___x_969_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__19));
v___y_962_ = v___x_969_;
goto v___jp_961_;
}
else
{
lean_object* v___x_970_; 
v___x_970_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__20));
v___y_962_ = v___x_970_;
goto v___jp_961_;
}
v___jp_961_:
{
lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; 
lean_inc_ref(v___y_962_);
v___x_963_ = l_Lean_stringToMessageData(v___y_962_);
v___x_964_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_964_, 0, v___x_960_);
lean_ctor_set(v___x_964_, 1, v___x_963_);
v___x_965_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__16);
v___x_966_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_966_, 0, v___x_964_);
lean_ctor_set(v___x_966_, 1, v___x_965_);
if (v_isMeta_885_ == 0)
{
lean_object* v___x_967_; 
v___x_967_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__17));
v___y_945_ = v___x_966_;
v___y_946_ = v___x_967_;
goto v___jp_944_;
}
else
{
lean_object* v___x_968_; 
v___x_968_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__18));
v___y_945_ = v___x_966_;
v___y_946_ = v___x_968_;
goto v___jp_944_;
}
}
}
v___jp_939_:
{
lean_object* v___x_942_; lean_object* v___x_943_; 
v___x_942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_942_, 0, v___y_940_);
lean_ctor_set(v___x_942_, 1, v___y_941_);
v___x_943_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21(v_cls_938_, v___x_942_, v___y_887_, v___y_888_);
if (lean_obj_tag(v___x_943_) == 0)
{
lean_dec_ref_known(v___x_943_, 1);
v___y_901_ = v___y_888_;
goto v___jp_900_;
}
else
{
lean_dec_ref_known(v_entry_896_, 1);
return v___x_943_;
}
}
v___jp_944_:
{
lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; uint8_t v___x_953_; 
lean_inc_ref(v___y_946_);
v___x_947_ = l_Lean_stringToMessageData(v___y_946_);
v___x_948_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_948_, 0, v___y_945_);
lean_ctor_set(v___x_948_, 1, v___x_947_);
v___x_949_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__6);
v___x_950_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_950_, 0, v___x_948_);
lean_ctor_set(v___x_950_, 1, v___x_949_);
v___x_951_ = l_Lean_MessageData_ofName(v_mod_884_);
v___x_952_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_952_, 0, v___x_950_);
lean_ctor_set(v___x_952_, 1, v___x_951_);
v___x_953_ = l_Lean_Name_isAnonymous(v_hint_886_);
if (v___x_953_ == 0)
{
lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_954_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__8);
v___x_955_ = l_Lean_MessageData_ofName(v_hint_886_);
v___x_956_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_956_, 0, v___x_954_);
lean_ctor_set(v___x_956_, 1, v___x_955_);
v___y_940_ = v___x_952_;
v___y_941_ = v___x_956_;
goto v___jp_939_;
}
else
{
lean_object* v___x_957_; 
lean_dec(v_hint_886_);
v___x_957_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__9);
v___y_940_ = v___x_952_;
v___y_941_ = v___x_957_;
goto v___jp_939_;
}
}
}
}
else
{
lean_object* v___x_971_; lean_object* v___x_972_; 
lean_dec_ref_known(v_entry_896_, 1);
lean_dec(v_hint_886_);
lean_dec(v_mod_884_);
v___x_971_ = lean_box(0);
v___x_972_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_972_, 0, v___x_971_);
return v___x_972_;
}
v___jp_900_:
{
lean_object* v___x_902_; lean_object* v_toEnvExtension_903_; lean_object* v_env_904_; lean_object* v_messages_905_; lean_object* v_scopes_906_; lean_object* v_usedQuotCtxts_907_; lean_object* v_nextMacroScope_908_; lean_object* v_maxRecDepth_909_; lean_object* v_ngen_910_; lean_object* v_auxDeclNGen_911_; lean_object* v_infoState_912_; lean_object* v_traceState_913_; lean_object* v_snapshotTasks_914_; lean_object* v_prevLinterStates_915_; lean_object* v___x_917_; uint8_t v_isShared_918_; uint8_t v_isSharedCheck_927_; 
v___x_902_ = lean_st_ref_take(v___y_901_);
v_toEnvExtension_903_ = lean_ctor_get(v___x_897_, 0);
v_env_904_ = lean_ctor_get(v___x_902_, 0);
v_messages_905_ = lean_ctor_get(v___x_902_, 1);
v_scopes_906_ = lean_ctor_get(v___x_902_, 2);
v_usedQuotCtxts_907_ = lean_ctor_get(v___x_902_, 3);
v_nextMacroScope_908_ = lean_ctor_get(v___x_902_, 4);
v_maxRecDepth_909_ = lean_ctor_get(v___x_902_, 5);
v_ngen_910_ = lean_ctor_get(v___x_902_, 6);
v_auxDeclNGen_911_ = lean_ctor_get(v___x_902_, 7);
v_infoState_912_ = lean_ctor_get(v___x_902_, 8);
v_traceState_913_ = lean_ctor_get(v___x_902_, 9);
v_snapshotTasks_914_ = lean_ctor_get(v___x_902_, 10);
v_prevLinterStates_915_ = lean_ctor_get(v___x_902_, 11);
v_isSharedCheck_927_ = !lean_is_exclusive(v___x_902_);
if (v_isSharedCheck_927_ == 0)
{
v___x_917_ = v___x_902_;
v_isShared_918_ = v_isSharedCheck_927_;
goto v_resetjp_916_;
}
else
{
lean_inc(v_prevLinterStates_915_);
lean_inc(v_snapshotTasks_914_);
lean_inc(v_traceState_913_);
lean_inc(v_infoState_912_);
lean_inc(v_auxDeclNGen_911_);
lean_inc(v_ngen_910_);
lean_inc(v_maxRecDepth_909_);
lean_inc(v_nextMacroScope_908_);
lean_inc(v_usedQuotCtxts_907_);
lean_inc(v_scopes_906_);
lean_inc(v_messages_905_);
lean_inc(v_env_904_);
lean_dec(v___x_902_);
v___x_917_ = lean_box(0);
v_isShared_918_ = v_isSharedCheck_927_;
goto v_resetjp_916_;
}
v_resetjp_916_:
{
lean_object* v_asyncMode_919_; lean_object* v___x_920_; lean_object* v___x_922_; 
v_asyncMode_919_ = lean_ctor_get(v_toEnvExtension_903_, 2);
v___x_920_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_897_, v_env_904_, v_entry_896_, v_asyncMode_919_, v___x_899_);
if (v_isShared_918_ == 0)
{
lean_ctor_set(v___x_917_, 0, v___x_920_);
v___x_922_ = v___x_917_;
goto v_reusejp_921_;
}
else
{
lean_object* v_reuseFailAlloc_926_; 
v_reuseFailAlloc_926_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_926_, 0, v___x_920_);
lean_ctor_set(v_reuseFailAlloc_926_, 1, v_messages_905_);
lean_ctor_set(v_reuseFailAlloc_926_, 2, v_scopes_906_);
lean_ctor_set(v_reuseFailAlloc_926_, 3, v_usedQuotCtxts_907_);
lean_ctor_set(v_reuseFailAlloc_926_, 4, v_nextMacroScope_908_);
lean_ctor_set(v_reuseFailAlloc_926_, 5, v_maxRecDepth_909_);
lean_ctor_set(v_reuseFailAlloc_926_, 6, v_ngen_910_);
lean_ctor_set(v_reuseFailAlloc_926_, 7, v_auxDeclNGen_911_);
lean_ctor_set(v_reuseFailAlloc_926_, 8, v_infoState_912_);
lean_ctor_set(v_reuseFailAlloc_926_, 9, v_traceState_913_);
lean_ctor_set(v_reuseFailAlloc_926_, 10, v_snapshotTasks_914_);
lean_ctor_set(v_reuseFailAlloc_926_, 11, v_prevLinterStates_915_);
v___x_922_ = v_reuseFailAlloc_926_;
goto v_reusejp_921_;
}
v_reusejp_921_:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v___x_923_ = lean_st_ref_set(v___y_901_, v___x_922_);
v___x_924_ = lean_box(0);
v___x_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
return v___x_925_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___boxed(lean_object* v_mod_973_, lean_object* v_isMeta_974_, lean_object* v_hint_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
uint8_t v_isMeta_boxed_979_; lean_object* v_res_980_; 
v_isMeta_boxed_979_ = lean_unbox(v_isMeta_974_);
v_res_980_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29(v_mod_973_, v_isMeta_boxed_979_, v_hint_975_, v___y_976_, v___y_977_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30(lean_object* v___x_981_, lean_object* v_declName_982_, lean_object* v_as_983_, size_t v_sz_984_, size_t v_i_985_, lean_object* v_b_986_, lean_object* v___y_987_, lean_object* v___y_988_){
_start:
{
uint8_t v___x_990_; 
v___x_990_ = lean_usize_dec_lt(v_i_985_, v_sz_984_);
if (v___x_990_ == 0)
{
lean_object* v___x_991_; 
lean_dec(v_declName_982_);
v___x_991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_991_, 0, v_b_986_);
return v___x_991_;
}
else
{
lean_object* v___x_992_; lean_object* v_modules_993_; lean_object* v___x_994_; lean_object* v_a_995_; lean_object* v___x_996_; lean_object* v_toImport_997_; lean_object* v_module_998_; uint8_t v___x_999_; lean_object* v___x_1000_; 
v___x_992_ = l_Lean_Environment_header(v___x_981_);
v_modules_993_ = lean_ctor_get(v___x_992_, 3);
lean_inc_ref(v_modules_993_);
lean_dec_ref(v___x_992_);
v___x_994_ = l_Lean_instInhabitedEffectiveImport_default;
v_a_995_ = lean_array_uget_borrowed(v_as_983_, v_i_985_);
v___x_996_ = lean_array_get(v___x_994_, v_modules_993_, v_a_995_);
lean_dec_ref(v_modules_993_);
v_toImport_997_ = lean_ctor_get(v___x_996_, 0);
lean_inc_ref(v_toImport_997_);
lean_dec(v___x_996_);
v_module_998_ = lean_ctor_get(v_toImport_997_, 0);
lean_inc(v_module_998_);
lean_dec_ref(v_toImport_997_);
v___x_999_ = 0;
lean_inc(v_declName_982_);
v___x_1000_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29(v_module_998_, v___x_999_, v_declName_982_, v___y_987_, v___y_988_);
if (lean_obj_tag(v___x_1000_) == 0)
{
lean_object* v___x_1001_; size_t v___x_1002_; size_t v___x_1003_; 
lean_dec_ref_known(v___x_1000_, 1);
v___x_1001_ = lean_box(0);
v___x_1002_ = ((size_t)1ULL);
v___x_1003_ = lean_usize_add(v_i_985_, v___x_1002_);
v_i_985_ = v___x_1003_;
v_b_986_ = v___x_1001_;
goto _start;
}
else
{
lean_dec(v_declName_982_);
return v___x_1000_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30___boxed(lean_object* v___x_1005_, lean_object* v_declName_1006_, lean_object* v_as_1007_, lean_object* v_sz_1008_, lean_object* v_i_1009_, lean_object* v_b_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_){
_start:
{
size_t v_sz_boxed_1014_; size_t v_i_boxed_1015_; lean_object* v_res_1016_; 
v_sz_boxed_1014_ = lean_unbox_usize(v_sz_1008_);
lean_dec(v_sz_1008_);
v_i_boxed_1015_ = lean_unbox_usize(v_i_1009_);
lean_dec(v_i_1009_);
v_res_1016_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30(v___x_1005_, v_declName_1006_, v_as_1007_, v_sz_boxed_1014_, v_i_boxed_1015_, v_b_1010_, v___y_1011_, v___y_1012_);
lean_dec(v___y_1012_);
lean_dec_ref(v___y_1011_);
lean_dec_ref(v_as_1007_);
lean_dec_ref(v___x_1005_);
return v_res_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg(lean_object* v_a_1017_, lean_object* v_x_1018_){
_start:
{
if (lean_obj_tag(v_x_1018_) == 0)
{
lean_object* v___x_1019_; 
v___x_1019_ = lean_box(0);
return v___x_1019_;
}
else
{
lean_object* v_key_1020_; lean_object* v_value_1021_; lean_object* v_tail_1022_; uint8_t v___x_1023_; 
v_key_1020_ = lean_ctor_get(v_x_1018_, 0);
v_value_1021_ = lean_ctor_get(v_x_1018_, 1);
v_tail_1022_ = lean_ctor_get(v_x_1018_, 2);
v___x_1023_ = lean_name_eq(v_key_1020_, v_a_1017_);
if (v___x_1023_ == 0)
{
v_x_1018_ = v_tail_1022_;
goto _start;
}
else
{
lean_object* v___x_1025_; 
lean_inc(v_value_1021_);
v___x_1025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1025_, 0, v_value_1021_);
return v___x_1025_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg___boxed(lean_object* v_a_1026_, lean_object* v_x_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg(v_a_1026_, v_x_1027_);
lean_dec(v_x_1027_);
lean_dec(v_a_1026_);
return v_res_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg(lean_object* v_m_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v_buckets_1031_; lean_object* v___x_1032_; uint64_t v___y_1034_; 
v_buckets_1031_ = lean_ctor_get(v_m_1029_, 1);
v___x_1032_ = lean_array_get_size(v_buckets_1031_);
if (lean_obj_tag(v_a_1030_) == 0)
{
uint64_t v___x_1048_; 
v___x_1048_ = 1723ULL;
v___y_1034_ = v___x_1048_;
goto v___jp_1033_;
}
else
{
uint64_t v_hash_1049_; 
v_hash_1049_ = lean_ctor_get_uint64(v_a_1030_, sizeof(void*)*2);
v___y_1034_ = v_hash_1049_;
goto v___jp_1033_;
}
v___jp_1033_:
{
uint64_t v___x_1035_; uint64_t v___x_1036_; uint64_t v_fold_1037_; uint64_t v___x_1038_; uint64_t v___x_1039_; uint64_t v___x_1040_; size_t v___x_1041_; size_t v___x_1042_; size_t v___x_1043_; size_t v___x_1044_; size_t v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1035_ = 32ULL;
v___x_1036_ = lean_uint64_shift_right(v___y_1034_, v___x_1035_);
v_fold_1037_ = lean_uint64_xor(v___y_1034_, v___x_1036_);
v___x_1038_ = 16ULL;
v___x_1039_ = lean_uint64_shift_right(v_fold_1037_, v___x_1038_);
v___x_1040_ = lean_uint64_xor(v_fold_1037_, v___x_1039_);
v___x_1041_ = lean_uint64_to_usize(v___x_1040_);
v___x_1042_ = lean_usize_of_nat(v___x_1032_);
v___x_1043_ = ((size_t)1ULL);
v___x_1044_ = lean_usize_sub(v___x_1042_, v___x_1043_);
v___x_1045_ = lean_usize_land(v___x_1041_, v___x_1044_);
v___x_1046_ = lean_array_uget_borrowed(v_buckets_1031_, v___x_1045_);
v___x_1047_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg(v_a_1030_, v___x_1046_);
return v___x_1047_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg___boxed(lean_object* v_m_1050_, lean_object* v_a_1051_){
_start:
{
lean_object* v_res_1052_; 
v_res_1052_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg(v_m_1050_, v_a_1051_);
lean_dec(v_a_1051_);
lean_dec_ref(v_m_1050_);
return v_res_1052_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2(void){
_start:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1055_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__1));
v___x_1056_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__0));
v___x_1057_ = l_Std_HashMap_instInhabited(lean_box(0), lean_box(0), v___x_1056_, v___x_1055_);
return v___x_1057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17(lean_object* v_declName_1060_, uint8_t v_isMeta_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_){
_start:
{
lean_object* v___x_1065_; lean_object* v_env_1069_; lean_object* v___y_1071_; lean_object* v___x_1084_; 
v___x_1065_ = lean_st_ref_get(v___y_1063_);
v_env_1069_ = lean_ctor_get(v___x_1065_, 0);
lean_inc_ref(v_env_1069_);
lean_dec(v___x_1065_);
v___x_1084_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1069_, v_declName_1060_);
if (lean_obj_tag(v___x_1084_) == 0)
{
lean_dec_ref(v_env_1069_);
lean_dec(v_declName_1060_);
goto v___jp_1066_;
}
else
{
lean_object* v_val_1085_; lean_object* v___x_1086_; lean_object* v_modules_1087_; lean_object* v___x_1088_; uint8_t v___x_1089_; 
v_val_1085_ = lean_ctor_get(v___x_1084_, 0);
lean_inc(v_val_1085_);
lean_dec_ref_known(v___x_1084_, 1);
v___x_1086_ = l_Lean_Environment_header(v_env_1069_);
v_modules_1087_ = lean_ctor_get(v___x_1086_, 3);
lean_inc_ref(v_modules_1087_);
lean_dec_ref(v___x_1086_);
v___x_1088_ = lean_array_get_size(v_modules_1087_);
v___x_1089_ = lean_nat_dec_lt(v_val_1085_, v___x_1088_);
if (v___x_1089_ == 0)
{
lean_dec_ref(v_modules_1087_);
lean_dec(v_val_1085_);
lean_dec_ref(v_env_1069_);
lean_dec(v_declName_1060_);
goto v___jp_1066_;
}
else
{
lean_object* v___x_1090_; lean_object* v_env_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; uint8_t v___y_1095_; 
v___x_1090_ = lean_st_ref_get(v___y_1063_);
v_env_1091_ = lean_ctor_get(v___x_1090_, 0);
lean_inc_ref(v_env_1091_);
lean_dec(v___x_1090_);
v___x_1092_ = lean_obj_once(&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2, &lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2_once, _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__2);
v___x_1093_ = lean_array_fget(v_modules_1087_, v_val_1085_);
lean_dec(v_val_1085_);
lean_dec_ref(v_modules_1087_);
if (v_isMeta_1061_ == 0)
{
lean_dec_ref(v_env_1091_);
v___y_1095_ = v_isMeta_1061_;
goto v___jp_1094_;
}
else
{
uint8_t v___x_1106_; 
lean_inc(v_declName_1060_);
v___x_1106_ = l_Lean_isMarkedMeta(v_env_1091_, v_declName_1060_);
if (v___x_1106_ == 0)
{
v___y_1095_ = v_isMeta_1061_;
goto v___jp_1094_;
}
else
{
uint8_t v___x_1107_; 
v___x_1107_ = 0;
v___y_1095_ = v___x_1107_;
goto v___jp_1094_;
}
}
v___jp_1094_:
{
lean_object* v_toImport_1096_; lean_object* v_module_1097_; lean_object* v___x_1098_; 
v_toImport_1096_ = lean_ctor_get(v___x_1093_, 0);
lean_inc_ref(v_toImport_1096_);
lean_dec(v___x_1093_);
v_module_1097_ = lean_ctor_get(v_toImport_1096_, 0);
lean_inc(v_module_1097_);
lean_dec_ref(v_toImport_1096_);
lean_inc(v_declName_1060_);
v___x_1098_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29(v_module_1097_, v___y_1095_, v_declName_1060_, v___y_1062_, v___y_1063_);
if (lean_obj_tag(v___x_1098_) == 0)
{
lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; 
lean_dec_ref_known(v___x_1098_, 1);
v___x_1099_ = l_Lean_indirectModUseExt;
v___x_1100_ = lean_box(1);
v___x_1101_ = lean_box(0);
lean_inc_ref(v_env_1069_);
v___x_1102_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_1092_, v___x_1099_, v_env_1069_, v___x_1100_, v___x_1101_);
v___x_1103_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg(v___x_1102_, v_declName_1060_);
lean_dec(v___x_1102_);
if (lean_obj_tag(v___x_1103_) == 0)
{
lean_object* v___x_1104_; 
v___x_1104_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___closed__3));
v___y_1071_ = v___x_1104_;
goto v___jp_1070_;
}
else
{
lean_object* v_val_1105_; 
v_val_1105_ = lean_ctor_get(v___x_1103_, 0);
lean_inc(v_val_1105_);
lean_dec_ref_known(v___x_1103_, 1);
v___y_1071_ = v_val_1105_;
goto v___jp_1070_;
}
}
else
{
lean_dec_ref(v_env_1069_);
lean_dec(v_declName_1060_);
return v___x_1098_;
}
}
}
}
v___jp_1066_:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1067_ = lean_box(0);
v___x_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1068_, 0, v___x_1067_);
return v___x_1068_;
}
v___jp_1070_:
{
lean_object* v___x_1072_; size_t v_sz_1073_; size_t v___x_1074_; lean_object* v___x_1075_; 
v___x_1072_ = lean_box(0);
v_sz_1073_ = lean_array_size(v___y_1071_);
v___x_1074_ = ((size_t)0ULL);
v___x_1075_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__30(v_env_1069_, v_declName_1060_, v___y_1071_, v_sz_1073_, v___x_1074_, v___x_1072_, v___y_1062_, v___y_1063_);
lean_dec_ref(v___y_1071_);
lean_dec_ref(v_env_1069_);
if (lean_obj_tag(v___x_1075_) == 0)
{
lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1082_; 
v_isSharedCheck_1082_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1082_ == 0)
{
lean_object* v_unused_1083_; 
v_unused_1083_ = lean_ctor_get(v___x_1075_, 0);
lean_dec(v_unused_1083_);
v___x_1077_ = v___x_1075_;
v_isShared_1078_ = v_isSharedCheck_1082_;
goto v_resetjp_1076_;
}
else
{
lean_dec(v___x_1075_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1082_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v___x_1080_; 
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 0, v___x_1072_);
v___x_1080_ = v___x_1077_;
goto v_reusejp_1079_;
}
else
{
lean_object* v_reuseFailAlloc_1081_; 
v_reuseFailAlloc_1081_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1081_, 0, v___x_1072_);
v___x_1080_ = v_reuseFailAlloc_1081_;
goto v_reusejp_1079_;
}
v_reusejp_1079_:
{
return v___x_1080_;
}
}
}
else
{
return v___x_1075_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17___boxed(lean_object* v_declName_1108_, lean_object* v_isMeta_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
uint8_t v_isMeta_boxed_1113_; lean_object* v_res_1114_; 
v_isMeta_boxed_1113_ = lean_unbox(v_isMeta_1109_);
v_res_1114_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17(v_declName_1108_, v_isMeta_boxed_1113_, v___y_1110_, v___y_1111_);
lean_dec(v___y_1111_);
lean_dec_ref(v___y_1110_);
return v_res_1114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg(lean_object* v_as_x27_1115_, lean_object* v_b_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_){
_start:
{
if (lean_obj_tag(v_as_x27_1115_) == 0)
{
lean_object* v___x_1120_; 
v___x_1120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1120_, 0, v_b_1116_);
return v___x_1120_;
}
else
{
lean_object* v_head_1121_; lean_object* v_tail_1122_; uint8_t v___x_1123_; lean_object* v___x_1124_; 
v_head_1121_ = lean_ctor_get(v_as_x27_1115_, 0);
v_tail_1122_ = lean_ctor_get(v_as_x27_1115_, 1);
v___x_1123_ = 1;
lean_inc(v_head_1121_);
v___x_1124_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17(v_head_1121_, v___x_1123_, v___y_1117_, v___y_1118_);
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v___x_1125_; 
lean_dec_ref_known(v___x_1124_, 1);
v___x_1125_ = lean_box(0);
v_as_x27_1115_ = v_tail_1122_;
v_b_1116_ = v___x_1125_;
goto _start;
}
else
{
return v___x_1124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg___boxed(lean_object* v_as_x27_1127_, lean_object* v_b_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_){
_start:
{
lean_object* v_res_1132_; 
v_res_1132_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg(v_as_x27_1127_, v_b_1128_, v___y_1129_, v___y_1130_);
lean_dec(v___y_1130_);
lean_dec_ref(v___y_1129_);
lean_dec(v_as_x27_1127_);
return v_res_1132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4(lean_object* v_env_1133_, lean_object* v_opts_1134_, lean_object* v_currNamespace_1135_, lean_object* v_openDecls_1136_, lean_object* v_n_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_){
_start:
{
lean_object* v___x_1140_; lean_object* v___x_1141_; 
v___x_1140_ = l_Lean_ResolveName_resolveGlobalName(v_env_1133_, v_opts_1134_, v_currNamespace_1135_, v_openDecls_1136_, v_n_1137_);
v___x_1141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1141_, 0, v___x_1140_);
lean_ctor_set(v___x_1141_, 1, v___y_1139_);
return v___x_1141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4___boxed(lean_object* v_env_1142_, lean_object* v_opts_1143_, lean_object* v_currNamespace_1144_, lean_object* v_openDecls_1145_, lean_object* v_n_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_){
_start:
{
lean_object* v_res_1149_; 
v_res_1149_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4(v_env_1142_, v_opts_1143_, v_currNamespace_1144_, v_openDecls_1145_, v_n_1146_, v___y_1147_, v___y_1148_);
lean_dec_ref(v___y_1147_);
lean_dec_ref(v_opts_1143_);
return v_res_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0(lean_object* v_env_1150_, lean_object* v_declName_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_){
_start:
{
uint8_t v___x_1154_; lean_object* v_env_1155_; lean_object* v___x_1156_; uint8_t v___x_1157_; uint8_t v___x_1158_; 
v___x_1154_ = 0;
v_env_1155_ = l_Lean_Environment_setExporting(v_env_1150_, v___x_1154_);
lean_inc(v_declName_1151_);
v___x_1156_ = l_Lean_mkPrivateName(v_env_1155_, v_declName_1151_);
v___x_1157_ = 1;
lean_inc_ref(v_env_1155_);
v___x_1158_ = l_Lean_Environment_contains(v_env_1155_, v___x_1156_, v___x_1157_);
if (v___x_1158_ == 0)
{
lean_object* v___x_1159_; uint8_t v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1159_ = l_Lean_privateToUserName(v_declName_1151_);
v___x_1160_ = l_Lean_Environment_contains(v_env_1155_, v___x_1159_, v___x_1157_);
v___x_1161_ = lean_box(v___x_1160_);
v___x_1162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1162_, 0, v___x_1161_);
lean_ctor_set(v___x_1162_, 1, v___y_1153_);
return v___x_1162_;
}
else
{
lean_object* v___x_1163_; lean_object* v___x_1164_; 
lean_dec_ref(v_env_1155_);
lean_dec(v_declName_1151_);
v___x_1163_ = lean_box(v___x_1158_);
v___x_1164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1164_, 0, v___x_1163_);
lean_ctor_set(v___x_1164_, 1, v___y_1153_);
return v___x_1164_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0___boxed(lean_object* v_env_1165_, lean_object* v_declName_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0(v_env_1165_, v_declName_1166_, v___y_1167_, v___y_1168_);
lean_dec_ref(v___y_1167_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2(lean_object* v_currNamespace_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1173_, 0, v_currNamespace_1170_);
lean_ctor_set(v___x_1173_, 1, v___y_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2___boxed(lean_object* v_currNamespace_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v_res_1177_; 
v_res_1177_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2(v_currNamespace_1174_, v___y_1175_, v___y_1176_);
lean_dec_ref(v___y_1175_);
return v_res_1177_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0(void){
_start:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1178_ = lean_box(0);
v___x_1179_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v___x_1178_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg(){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1182_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___closed__0);
v___x_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1182_);
return v___x_1183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg___boxed(lean_object* v___y_1184_){
_start:
{
lean_object* v_res_1185_; 
v_res_1185_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
return v_res_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3(lean_object* v_env_1186_, lean_object* v_currNamespace_1187_, lean_object* v_openDecls_1188_, lean_object* v_n_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_){
_start:
{
lean_object* v___x_1192_; lean_object* v___x_1193_; 
v___x_1192_ = l_Lean_ResolveName_resolveNamespace(v_env_1186_, v_currNamespace_1187_, v_openDecls_1188_, v_n_1189_);
v___x_1193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1192_);
lean_ctor_set(v___x_1193_, 1, v___y_1191_);
return v___x_1193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3___boxed(lean_object* v_env_1194_, lean_object* v_currNamespace_1195_, lean_object* v_openDecls_1196_, lean_object* v_n_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_){
_start:
{
lean_object* v_res_1200_; 
v_res_1200_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3(v_env_1194_, v_currNamespace_1195_, v_openDecls_1196_, v_n_1197_, v___y_1198_, v___y_1199_);
lean_dec_ref(v___y_1198_);
return v_res_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_1201_, lean_object* v_msg_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_){
_start:
{
lean_object* v___x_1206_; 
v___x_1206_ = l_Lean_Elab_Command_getRef___redArg(v___y_1203_);
if (lean_obj_tag(v___x_1206_) == 0)
{
lean_object* v_a_1207_; lean_object* v_fileName_1208_; lean_object* v_fileMap_1209_; lean_object* v_currRecDepth_1210_; lean_object* v_cmdPos_1211_; lean_object* v_macroStack_1212_; lean_object* v_quotContext_x3f_1213_; lean_object* v_currMacroScope_1214_; lean_object* v_snap_x3f_1215_; lean_object* v_cancelTk_x3f_1216_; uint8_t v_suppressElabErrors_1217_; lean_object* v_ref_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; 
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
lean_inc(v_a_1207_);
lean_dec_ref_known(v___x_1206_, 1);
v_fileName_1208_ = lean_ctor_get(v___y_1203_, 0);
v_fileMap_1209_ = lean_ctor_get(v___y_1203_, 1);
v_currRecDepth_1210_ = lean_ctor_get(v___y_1203_, 2);
v_cmdPos_1211_ = lean_ctor_get(v___y_1203_, 3);
v_macroStack_1212_ = lean_ctor_get(v___y_1203_, 4);
v_quotContext_x3f_1213_ = lean_ctor_get(v___y_1203_, 5);
v_currMacroScope_1214_ = lean_ctor_get(v___y_1203_, 6);
v_snap_x3f_1215_ = lean_ctor_get(v___y_1203_, 8);
v_cancelTk_x3f_1216_ = lean_ctor_get(v___y_1203_, 9);
v_suppressElabErrors_1217_ = lean_ctor_get_uint8(v___y_1203_, sizeof(void*)*10);
v_ref_1218_ = l_Lean_replaceRef(v_ref_1201_, v_a_1207_);
lean_dec(v_a_1207_);
lean_inc(v_cancelTk_x3f_1216_);
lean_inc(v_snap_x3f_1215_);
lean_inc(v_currMacroScope_1214_);
lean_inc(v_quotContext_x3f_1213_);
lean_inc(v_macroStack_1212_);
lean_inc(v_cmdPos_1211_);
lean_inc(v_currRecDepth_1210_);
lean_inc_ref(v_fileMap_1209_);
lean_inc_ref(v_fileName_1208_);
v___x_1219_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_1219_, 0, v_fileName_1208_);
lean_ctor_set(v___x_1219_, 1, v_fileMap_1209_);
lean_ctor_set(v___x_1219_, 2, v_currRecDepth_1210_);
lean_ctor_set(v___x_1219_, 3, v_cmdPos_1211_);
lean_ctor_set(v___x_1219_, 4, v_macroStack_1212_);
lean_ctor_set(v___x_1219_, 5, v_quotContext_x3f_1213_);
lean_ctor_set(v___x_1219_, 6, v_currMacroScope_1214_);
lean_ctor_set(v___x_1219_, 7, v_ref_1218_);
lean_ctor_set(v___x_1219_, 8, v_snap_x3f_1215_);
lean_ctor_set(v___x_1219_, 9, v_cancelTk_x3f_1216_);
lean_ctor_set_uint8(v___x_1219_, sizeof(void*)*10, v_suppressElabErrors_1217_);
v___x_1220_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(v_msg_1202_, v___x_1219_, v___y_1204_);
lean_dec_ref_known(v___x_1219_, 10);
return v___x_1220_;
}
else
{
lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1228_; 
lean_dec_ref(v_msg_1202_);
v_a_1221_ = lean_ctor_get(v___x_1206_, 0);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1223_ = v___x_1206_;
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1206_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v___x_1226_; 
if (v_isShared_1224_ == 0)
{
v___x_1226_ = v___x_1223_;
goto v_reusejp_1225_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_a_1221_);
v___x_1226_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1225_;
}
v_reusejp_1225_:
{
return v___x_1226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_1229_, lean_object* v_msg_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v_res_1234_; 
v_res_1234_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(v_ref_1229_, v_msg_1230_, v___y_1231_, v___y_1232_);
lean_dec(v___y_1232_);
lean_dec_ref(v___y_1231_);
lean_dec(v_ref_1229_);
return v_res_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(lean_object* v_x_1235_, lean_object* v___y_1236_){
_start:
{
if (lean_obj_tag(v_x_1235_) == 0)
{
lean_object* v_a_1237_; lean_object* v___x_1238_; 
v_a_1237_ = lean_ctor_get(v_x_1235_, 0);
lean_inc(v_a_1237_);
v___x_1238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1238_, 0, v_a_1237_);
lean_ctor_set(v___x_1238_, 1, v___y_1236_);
return v___x_1238_;
}
else
{
lean_object* v_a_1239_; lean_object* v___x_1240_; 
v_a_1239_ = lean_ctor_get(v_x_1235_, 0);
lean_inc(v_a_1239_);
v___x_1240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1240_, 0, v_a_1239_);
lean_ctor_set(v___x_1240_, 1, v___y_1236_);
return v___x_1240_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg___boxed(lean_object* v_x_1241_, lean_object* v___y_1242_){
_start:
{
lean_object* v_res_1243_; 
v_res_1243_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(v_x_1241_, v___y_1242_);
lean_dec_ref(v_x_1241_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1(lean_object* v_env_1244_, lean_object* v_stx_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = l_Lean_Elab_expandMacroImpl_x3f(v_env_1244_, v_stx_1245_, v___y_1246_, v___y_1247_);
if (lean_obj_tag(v___x_1248_) == 0)
{
lean_object* v_a_1249_; 
v_a_1249_ = lean_ctor_get(v___x_1248_, 0);
lean_inc(v_a_1249_);
if (lean_obj_tag(v_a_1249_) == 0)
{
lean_object* v_a_1250_; lean_object* v___x_1252_; uint8_t v_isShared_1253_; uint8_t v_isSharedCheck_1258_; 
v_a_1250_ = lean_ctor_get(v___x_1248_, 1);
v_isSharedCheck_1258_ = !lean_is_exclusive(v___x_1248_);
if (v_isSharedCheck_1258_ == 0)
{
lean_object* v_unused_1259_; 
v_unused_1259_ = lean_ctor_get(v___x_1248_, 0);
lean_dec(v_unused_1259_);
v___x_1252_ = v___x_1248_;
v_isShared_1253_ = v_isSharedCheck_1258_;
goto v_resetjp_1251_;
}
else
{
lean_inc(v_a_1250_);
lean_dec(v___x_1248_);
v___x_1252_ = lean_box(0);
v_isShared_1253_ = v_isSharedCheck_1258_;
goto v_resetjp_1251_;
}
v_resetjp_1251_:
{
lean_object* v___x_1254_; lean_object* v___x_1256_; 
v___x_1254_ = lean_box(0);
if (v_isShared_1253_ == 0)
{
lean_ctor_set(v___x_1252_, 0, v___x_1254_);
v___x_1256_ = v___x_1252_;
goto v_reusejp_1255_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v___x_1254_);
lean_ctor_set(v_reuseFailAlloc_1257_, 1, v_a_1250_);
v___x_1256_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1255_;
}
v_reusejp_1255_:
{
return v___x_1256_;
}
}
}
else
{
lean_object* v_val_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1288_; 
v_val_1260_ = lean_ctor_get(v_a_1249_, 0);
v_isSharedCheck_1288_ = !lean_is_exclusive(v_a_1249_);
if (v_isSharedCheck_1288_ == 0)
{
v___x_1262_ = v_a_1249_;
v_isShared_1263_ = v_isSharedCheck_1288_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_val_1260_);
lean_dec(v_a_1249_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1288_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v_snd_1264_; 
v_snd_1264_ = lean_ctor_get(v_val_1260_, 1);
lean_inc(v_snd_1264_);
lean_dec(v_val_1260_);
if (lean_obj_tag(v_snd_1264_) == 0)
{
lean_object* v_a_1265_; lean_object* v_a_1266_; lean_object* v___x_1268_; uint8_t v_isShared_1269_; uint8_t v_isSharedCheck_1274_; 
lean_del_object(v___x_1262_);
v_a_1265_ = lean_ctor_get(v___x_1248_, 1);
lean_inc(v_a_1265_);
lean_dec_ref_known(v___x_1248_, 2);
v_a_1266_ = lean_ctor_get(v_snd_1264_, 0);
v_isSharedCheck_1274_ = !lean_is_exclusive(v_snd_1264_);
if (v_isSharedCheck_1274_ == 0)
{
v___x_1268_ = v_snd_1264_;
v_isShared_1269_ = v_isSharedCheck_1274_;
goto v_resetjp_1267_;
}
else
{
lean_inc(v_a_1266_);
lean_dec(v_snd_1264_);
v___x_1268_ = lean_box(0);
v_isShared_1269_ = v_isSharedCheck_1274_;
goto v_resetjp_1267_;
}
v_resetjp_1267_:
{
lean_object* v___x_1271_; 
if (v_isShared_1269_ == 0)
{
v___x_1271_ = v___x_1268_;
goto v_reusejp_1270_;
}
else
{
lean_object* v_reuseFailAlloc_1273_; 
v_reuseFailAlloc_1273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1273_, 0, v_a_1266_);
v___x_1271_ = v_reuseFailAlloc_1273_;
goto v_reusejp_1270_;
}
v_reusejp_1270_:
{
lean_object* v___x_1272_; 
v___x_1272_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(v___x_1271_, v_a_1265_);
lean_dec_ref(v___x_1271_);
return v___x_1272_;
}
}
}
else
{
lean_object* v_a_1275_; lean_object* v_a_1276_; lean_object* v___x_1278_; uint8_t v_isShared_1279_; uint8_t v_isSharedCheck_1287_; 
v_a_1275_ = lean_ctor_get(v___x_1248_, 1);
lean_inc(v_a_1275_);
lean_dec_ref_known(v___x_1248_, 2);
v_a_1276_ = lean_ctor_get(v_snd_1264_, 0);
v_isSharedCheck_1287_ = !lean_is_exclusive(v_snd_1264_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1278_ = v_snd_1264_;
v_isShared_1279_ = v_isSharedCheck_1287_;
goto v_resetjp_1277_;
}
else
{
lean_inc(v_a_1276_);
lean_dec(v_snd_1264_);
v___x_1278_ = lean_box(0);
v_isShared_1279_ = v_isSharedCheck_1287_;
goto v_resetjp_1277_;
}
v_resetjp_1277_:
{
lean_object* v___x_1281_; 
if (v_isShared_1263_ == 0)
{
lean_ctor_set(v___x_1262_, 0, v_a_1276_);
v___x_1281_ = v___x_1262_;
goto v_reusejp_1280_;
}
else
{
lean_object* v_reuseFailAlloc_1286_; 
v_reuseFailAlloc_1286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1286_, 0, v_a_1276_);
v___x_1281_ = v_reuseFailAlloc_1286_;
goto v_reusejp_1280_;
}
v_reusejp_1280_:
{
lean_object* v___x_1283_; 
if (v_isShared_1279_ == 0)
{
lean_ctor_set(v___x_1278_, 0, v___x_1281_);
v___x_1283_ = v___x_1278_;
goto v_reusejp_1282_;
}
else
{
lean_object* v_reuseFailAlloc_1285_; 
v_reuseFailAlloc_1285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1285_, 0, v___x_1281_);
v___x_1283_ = v_reuseFailAlloc_1285_;
goto v_reusejp_1282_;
}
v_reusejp_1282_:
{
lean_object* v___x_1284_; 
v___x_1284_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(v___x_1283_, v_a_1275_);
lean_dec_ref(v___x_1283_);
return v___x_1284_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1289_; lean_object* v_a_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1297_; 
v_a_1289_ = lean_ctor_get(v___x_1248_, 0);
v_a_1290_ = lean_ctor_get(v___x_1248_, 1);
v_isSharedCheck_1297_ = !lean_is_exclusive(v___x_1248_);
if (v_isSharedCheck_1297_ == 0)
{
v___x_1292_ = v___x_1248_;
v_isShared_1293_ = v_isSharedCheck_1297_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_a_1290_);
lean_inc(v_a_1289_);
lean_dec(v___x_1248_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1297_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v___x_1295_; 
if (v_isShared_1293_ == 0)
{
v___x_1295_ = v___x_1292_;
goto v_reusejp_1294_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v_a_1289_);
lean_ctor_set(v_reuseFailAlloc_1296_, 1, v_a_1290_);
v___x_1295_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1294_;
}
v_reusejp_1294_:
{
return v___x_1295_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1___boxed(lean_object* v_env_1298_, lean_object* v_stx_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_){
_start:
{
lean_object* v_res_1302_; 
v_res_1302_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1(v_env_1298_, v_stx_1299_, v___y_1300_, v___y_1301_);
lean_dec_ref(v___y_1300_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24(lean_object* v_as_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_){
_start:
{
if (lean_obj_tag(v_as_1303_) == 0)
{
lean_object* v___x_1307_; lean_object* v___x_1308_; 
v___x_1307_ = lean_box(0);
v___x_1308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1307_);
return v___x_1308_;
}
else
{
lean_object* v_head_1309_; lean_object* v_tail_1310_; lean_object* v_fst_1311_; lean_object* v_snd_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v_scopes_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v_opts_1319_; uint8_t v_hasTrace_1320_; 
v_head_1309_ = lean_ctor_get(v_as_1303_, 0);
lean_inc(v_head_1309_);
v_tail_1310_ = lean_ctor_get(v_as_1303_, 1);
lean_inc(v_tail_1310_);
lean_dec_ref_known(v_as_1303_, 2);
v_fst_1311_ = lean_ctor_get(v_head_1309_, 0);
lean_inc(v_fst_1311_);
v_snd_1312_ = lean_ctor_get(v_head_1309_, 1);
lean_inc(v_snd_1312_);
lean_dec(v_head_1309_);
v___x_1313_ = l_Lean_inheritedTraceOptions;
v___x_1314_ = lean_st_ref_get(v___x_1313_);
v___x_1315_ = lean_st_ref_get(v___y_1305_);
v_scopes_1316_ = lean_ctor_get(v___x_1315_, 2);
lean_inc(v_scopes_1316_);
lean_dec(v___x_1315_);
v___x_1317_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1318_ = l_List_head_x21___redArg(v___x_1317_, v_scopes_1316_);
lean_dec(v_scopes_1316_);
v_opts_1319_ = lean_ctor_get(v___x_1318_, 1);
lean_inc_ref(v_opts_1319_);
lean_dec(v___x_1318_);
v_hasTrace_1320_ = lean_ctor_get_uint8(v_opts_1319_, sizeof(void*)*1);
if (v_hasTrace_1320_ == 0)
{
lean_dec_ref(v_opts_1319_);
lean_dec(v___x_1314_);
lean_dec(v_snd_1312_);
lean_dec(v_fst_1311_);
v_as_1303_ = v_tail_1310_;
goto _start;
}
else
{
lean_object* v___x_1322_; lean_object* v___x_1323_; uint8_t v___x_1324_; 
v___x_1322_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__11));
lean_inc(v_fst_1311_);
v___x_1323_ = l_Lean_Name_append(v___x_1322_, v_fst_1311_);
v___x_1324_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___x_1314_, v_opts_1319_, v___x_1323_);
lean_dec(v___x_1323_);
lean_dec_ref(v_opts_1319_);
lean_dec(v___x_1314_);
if (v___x_1324_ == 0)
{
lean_dec(v_snd_1312_);
lean_dec(v_fst_1311_);
v_as_1303_ = v_tail_1310_;
goto _start;
}
else
{
lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; 
v___x_1326_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1326_, 0, v_snd_1312_);
v___x_1327_ = l_Lean_MessageData_ofFormat(v___x_1326_);
v___x_1328_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__21(v_fst_1311_, v___x_1327_, v___y_1304_, v___y_1305_);
if (lean_obj_tag(v___x_1328_) == 0)
{
lean_dec_ref_known(v___x_1328_, 1);
v_as_1303_ = v_tail_1310_;
goto _start;
}
else
{
lean_dec(v_tail_1310_);
return v___x_1328_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24___boxed(lean_object* v_as_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
lean_object* v_res_1334_; 
v_res_1334_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24(v_as_1330_, v___y_1331_, v___y_1332_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
return v_res_1334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(lean_object* v_x_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_){
_start:
{
lean_object* v___x_1340_; lean_object* v_env_1341_; lean_object* v___x_1342_; lean_object* v_scopes_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v_opts_1346_; lean_object* v___x_1347_; 
v___x_1340_ = lean_st_ref_get(v___y_1338_);
v_env_1341_ = lean_ctor_get(v___x_1340_, 0);
lean_inc_ref(v_env_1341_);
lean_dec(v___x_1340_);
v___x_1342_ = lean_st_ref_get(v___y_1338_);
v_scopes_1343_ = lean_ctor_get(v___x_1342_, 2);
lean_inc(v_scopes_1343_);
lean_dec(v___x_1342_);
v___x_1344_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1345_ = l_List_head_x21___redArg(v___x_1344_, v_scopes_1343_);
lean_dec(v_scopes_1343_);
v_opts_1346_ = lean_ctor_get(v___x_1345_, 1);
lean_inc_ref(v_opts_1346_);
lean_dec(v___x_1345_);
v___x_1347_ = l_Lean_Elab_Command_getScope___redArg(v___y_1338_);
if (lean_obj_tag(v___x_1347_) == 0)
{
lean_object* v_a_1348_; lean_object* v_currNamespace_1349_; lean_object* v___x_1350_; 
v_a_1348_ = lean_ctor_get(v___x_1347_, 0);
lean_inc(v_a_1348_);
lean_dec_ref_known(v___x_1347_, 1);
v_currNamespace_1349_ = lean_ctor_get(v_a_1348_, 2);
lean_inc(v_currNamespace_1349_);
lean_dec(v_a_1348_);
v___x_1350_ = l_Lean_Elab_Command_getScope___redArg(v___y_1338_);
if (lean_obj_tag(v___x_1350_) == 0)
{
lean_object* v_a_1351_; lean_object* v_openDecls_1352_; lean_object* v___x_1353_; 
v_a_1351_ = lean_ctor_get(v___x_1350_, 0);
lean_inc(v_a_1351_);
lean_dec_ref_known(v___x_1350_, 1);
v_openDecls_1352_ = lean_ctor_get(v_a_1351_, 3);
lean_inc(v_openDecls_1352_);
lean_dec(v_a_1351_);
v___x_1353_ = l_Lean_Elab_Command_getRef___redArg(v___y_1337_);
if (lean_obj_tag(v___x_1353_) == 0)
{
lean_object* v_a_1354_; lean_object* v___x_1355_; 
v_a_1354_ = lean_ctor_get(v___x_1353_, 0);
lean_inc(v_a_1354_);
lean_dec_ref_known(v___x_1353_, 1);
v___x_1355_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_1337_);
if (lean_obj_tag(v___x_1355_) == 0)
{
lean_object* v_a_1356_; lean_object* v_currRecDepth_1357_; lean_object* v_quotContext_x3f_1358_; lean_object* v___f_1359_; lean_object* v___f_1360_; lean_object* v___f_1361_; lean_object* v___f_1362_; lean_object* v___f_1363_; lean_object* v_methods_1364_; lean_object* v_a_1366_; 
v_a_1356_ = lean_ctor_get(v___x_1355_, 0);
lean_inc(v_a_1356_);
lean_dec_ref_known(v___x_1355_, 1);
v_currRecDepth_1357_ = lean_ctor_get(v___y_1337_, 2);
v_quotContext_x3f_1358_ = lean_ctor_get(v___y_1337_, 5);
lean_inc_ref_n(v_env_1341_, 3);
v___f_1359_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1359_, 0, v_env_1341_);
v___f_1360_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_1360_, 0, v_env_1341_);
lean_inc_n(v_currNamespace_1349_, 2);
v___f_1361_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_1361_, 0, v_currNamespace_1349_);
lean_inc(v_openDecls_1352_);
v___f_1362_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__3___boxed), 6, 3);
lean_closure_set(v___f_1362_, 0, v_env_1341_);
lean_closure_set(v___f_1362_, 1, v_currNamespace_1349_);
lean_closure_set(v___f_1362_, 2, v_openDecls_1352_);
v___f_1363_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___lam__4___boxed), 7, 4);
lean_closure_set(v___f_1363_, 0, v_env_1341_);
lean_closure_set(v___f_1363_, 1, v_opts_1346_);
lean_closure_set(v___f_1363_, 2, v_currNamespace_1349_);
lean_closure_set(v___f_1363_, 3, v_openDecls_1352_);
v_methods_1364_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_methods_1364_, 0, v___f_1360_);
lean_ctor_set(v_methods_1364_, 1, v___f_1361_);
lean_ctor_set(v_methods_1364_, 2, v___f_1359_);
lean_ctor_set(v_methods_1364_, 3, v___f_1362_);
lean_ctor_set(v_methods_1364_, 4, v___f_1363_);
if (lean_obj_tag(v_quotContext_x3f_1358_) == 0)
{
lean_object* v___x_1439_; lean_object* v_a_1440_; 
v___x_1439_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg(v___y_1338_);
v_a_1440_ = lean_ctor_get(v___x_1439_, 0);
lean_inc(v_a_1440_);
lean_dec_ref(v___x_1439_);
v_a_1366_ = v_a_1440_;
goto v___jp_1365_;
}
else
{
lean_object* v_val_1441_; 
v_val_1441_ = lean_ctor_get(v_quotContext_x3f_1358_, 0);
lean_inc(v_val_1441_);
v_a_1366_ = v_val_1441_;
goto v___jp_1365_;
}
v___jp_1365_:
{
lean_object* v___x_1367_; lean_object* v_maxRecDepth_1368_; lean_object* v___x_1369_; lean_object* v_nextMacroScope_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1367_ = lean_st_ref_get(v___y_1338_);
v_maxRecDepth_1368_ = lean_ctor_get(v___x_1367_, 5);
lean_inc(v_maxRecDepth_1368_);
lean_dec(v___x_1367_);
v___x_1369_ = lean_st_ref_get(v___y_1338_);
v_nextMacroScope_1370_ = lean_ctor_get(v___x_1369_, 4);
lean_inc(v_nextMacroScope_1370_);
lean_dec(v___x_1369_);
lean_inc(v_currRecDepth_1357_);
v___x_1371_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1371_, 0, v_methods_1364_);
lean_ctor_set(v___x_1371_, 1, v_a_1366_);
lean_ctor_set(v___x_1371_, 2, v_a_1356_);
lean_ctor_set(v___x_1371_, 3, v_currRecDepth_1357_);
lean_ctor_set(v___x_1371_, 4, v_maxRecDepth_1368_);
lean_ctor_set(v___x_1371_, 5, v_a_1354_);
v___x_1372_ = lean_box(0);
v___x_1373_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1373_, 0, v_nextMacroScope_1370_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
lean_ctor_set(v___x_1373_, 2, v___x_1372_);
v___x_1374_ = lean_apply_2(v_x_1336_, v___x_1371_, v___x_1373_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; lean_object* v_a_1376_; lean_object* v_macroScope_1377_; lean_object* v_traceMsgs_1378_; lean_object* v_expandedMacroDecls_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 1);
lean_inc(v_a_1375_);
v_a_1376_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1376_);
lean_dec_ref_known(v___x_1374_, 2);
v_macroScope_1377_ = lean_ctor_get(v_a_1375_, 0);
lean_inc(v_macroScope_1377_);
v_traceMsgs_1378_ = lean_ctor_get(v_a_1375_, 1);
lean_inc(v_traceMsgs_1378_);
v_expandedMacroDecls_1379_ = lean_ctor_get(v_a_1375_, 2);
lean_inc(v_expandedMacroDecls_1379_);
lean_dec(v_a_1375_);
v___x_1380_ = lean_box(0);
v___x_1381_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg(v_expandedMacroDecls_1379_, v___x_1380_, v___y_1337_, v___y_1338_);
lean_dec(v_expandedMacroDecls_1379_);
if (lean_obj_tag(v___x_1381_) == 0)
{
lean_object* v___x_1382_; lean_object* v_env_1383_; lean_object* v_messages_1384_; lean_object* v_scopes_1385_; lean_object* v_usedQuotCtxts_1386_; lean_object* v_maxRecDepth_1387_; lean_object* v_ngen_1388_; lean_object* v_auxDeclNGen_1389_; lean_object* v_infoState_1390_; lean_object* v_traceState_1391_; lean_object* v_snapshotTasks_1392_; lean_object* v_prevLinterStates_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1419_; 
lean_dec_ref_known(v___x_1381_, 1);
v___x_1382_ = lean_st_ref_take(v___y_1338_);
v_env_1383_ = lean_ctor_get(v___x_1382_, 0);
v_messages_1384_ = lean_ctor_get(v___x_1382_, 1);
v_scopes_1385_ = lean_ctor_get(v___x_1382_, 2);
v_usedQuotCtxts_1386_ = lean_ctor_get(v___x_1382_, 3);
v_maxRecDepth_1387_ = lean_ctor_get(v___x_1382_, 5);
v_ngen_1388_ = lean_ctor_get(v___x_1382_, 6);
v_auxDeclNGen_1389_ = lean_ctor_get(v___x_1382_, 7);
v_infoState_1390_ = lean_ctor_get(v___x_1382_, 8);
v_traceState_1391_ = lean_ctor_get(v___x_1382_, 9);
v_snapshotTasks_1392_ = lean_ctor_get(v___x_1382_, 10);
v_prevLinterStates_1393_ = lean_ctor_get(v___x_1382_, 11);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1382_);
if (v_isSharedCheck_1419_ == 0)
{
lean_object* v_unused_1420_; 
v_unused_1420_ = lean_ctor_get(v___x_1382_, 4);
lean_dec(v_unused_1420_);
v___x_1395_ = v___x_1382_;
v_isShared_1396_ = v_isSharedCheck_1419_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_prevLinterStates_1393_);
lean_inc(v_snapshotTasks_1392_);
lean_inc(v_traceState_1391_);
lean_inc(v_infoState_1390_);
lean_inc(v_auxDeclNGen_1389_);
lean_inc(v_ngen_1388_);
lean_inc(v_maxRecDepth_1387_);
lean_inc(v_usedQuotCtxts_1386_);
lean_inc(v_scopes_1385_);
lean_inc(v_messages_1384_);
lean_inc(v_env_1383_);
lean_dec(v___x_1382_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1419_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
lean_object* v___x_1398_; 
if (v_isShared_1396_ == 0)
{
lean_ctor_set(v___x_1395_, 4, v_macroScope_1377_);
v___x_1398_ = v___x_1395_;
goto v_reusejp_1397_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_env_1383_);
lean_ctor_set(v_reuseFailAlloc_1418_, 1, v_messages_1384_);
lean_ctor_set(v_reuseFailAlloc_1418_, 2, v_scopes_1385_);
lean_ctor_set(v_reuseFailAlloc_1418_, 3, v_usedQuotCtxts_1386_);
lean_ctor_set(v_reuseFailAlloc_1418_, 4, v_macroScope_1377_);
lean_ctor_set(v_reuseFailAlloc_1418_, 5, v_maxRecDepth_1387_);
lean_ctor_set(v_reuseFailAlloc_1418_, 6, v_ngen_1388_);
lean_ctor_set(v_reuseFailAlloc_1418_, 7, v_auxDeclNGen_1389_);
lean_ctor_set(v_reuseFailAlloc_1418_, 8, v_infoState_1390_);
lean_ctor_set(v_reuseFailAlloc_1418_, 9, v_traceState_1391_);
lean_ctor_set(v_reuseFailAlloc_1418_, 10, v_snapshotTasks_1392_);
lean_ctor_set(v_reuseFailAlloc_1418_, 11, v_prevLinterStates_1393_);
v___x_1398_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1397_;
}
v_reusejp_1397_:
{
lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; 
v___x_1399_ = lean_st_ref_set(v___y_1338_, v___x_1398_);
v___x_1400_ = l_List_reverse___redArg(v_traceMsgs_1378_);
v___x_1401_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__24(v___x_1400_, v___y_1337_, v___y_1338_);
if (lean_obj_tag(v___x_1401_) == 0)
{
lean_object* v___x_1403_; uint8_t v_isShared_1404_; uint8_t v_isSharedCheck_1408_; 
v_isSharedCheck_1408_ = !lean_is_exclusive(v___x_1401_);
if (v_isSharedCheck_1408_ == 0)
{
lean_object* v_unused_1409_; 
v_unused_1409_ = lean_ctor_get(v___x_1401_, 0);
lean_dec(v_unused_1409_);
v___x_1403_ = v___x_1401_;
v_isShared_1404_ = v_isSharedCheck_1408_;
goto v_resetjp_1402_;
}
else
{
lean_dec(v___x_1401_);
v___x_1403_ = lean_box(0);
v_isShared_1404_ = v_isSharedCheck_1408_;
goto v_resetjp_1402_;
}
v_resetjp_1402_:
{
lean_object* v___x_1406_; 
if (v_isShared_1404_ == 0)
{
lean_ctor_set(v___x_1403_, 0, v_a_1376_);
v___x_1406_ = v___x_1403_;
goto v_reusejp_1405_;
}
else
{
lean_object* v_reuseFailAlloc_1407_; 
v_reuseFailAlloc_1407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1407_, 0, v_a_1376_);
v___x_1406_ = v_reuseFailAlloc_1407_;
goto v_reusejp_1405_;
}
v_reusejp_1405_:
{
return v___x_1406_;
}
}
}
else
{
lean_object* v_a_1410_; lean_object* v___x_1412_; uint8_t v_isShared_1413_; uint8_t v_isSharedCheck_1417_; 
lean_dec(v_a_1376_);
v_a_1410_ = lean_ctor_get(v___x_1401_, 0);
v_isSharedCheck_1417_ = !lean_is_exclusive(v___x_1401_);
if (v_isSharedCheck_1417_ == 0)
{
v___x_1412_ = v___x_1401_;
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
else
{
lean_inc(v_a_1410_);
lean_dec(v___x_1401_);
v___x_1412_ = lean_box(0);
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
v_resetjp_1411_:
{
lean_object* v___x_1415_; 
if (v_isShared_1413_ == 0)
{
v___x_1415_ = v___x_1412_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1416_; 
v_reuseFailAlloc_1416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1416_, 0, v_a_1410_);
v___x_1415_ = v_reuseFailAlloc_1416_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
return v___x_1415_;
}
}
}
}
}
}
else
{
lean_object* v_a_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1428_; 
lean_dec(v_traceMsgs_1378_);
lean_dec(v_macroScope_1377_);
lean_dec(v_a_1376_);
v_a_1421_ = lean_ctor_get(v___x_1381_, 0);
v_isSharedCheck_1428_ = !lean_is_exclusive(v___x_1381_);
if (v_isSharedCheck_1428_ == 0)
{
v___x_1423_ = v___x_1381_;
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_a_1421_);
lean_dec(v___x_1381_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v___x_1426_; 
if (v_isShared_1424_ == 0)
{
v___x_1426_ = v___x_1423_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v_a_1421_);
v___x_1426_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
return v___x_1426_;
}
}
}
}
else
{
lean_object* v_a_1429_; 
v_a_1429_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1429_);
lean_dec_ref_known(v___x_1374_, 2);
if (lean_obj_tag(v_a_1429_) == 0)
{
lean_object* v_a_1430_; lean_object* v_a_1431_; lean_object* v___x_1432_; uint8_t v___x_1433_; 
v_a_1430_ = lean_ctor_get(v_a_1429_, 0);
lean_inc(v_a_1430_);
v_a_1431_ = lean_ctor_get(v_a_1429_, 1);
lean_inc_ref(v_a_1431_);
lean_dec_ref_known(v_a_1429_, 2);
v___x_1432_ = ((lean_object*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___closed__0));
v___x_1433_ = lean_string_dec_eq(v_a_1431_, v___x_1432_);
if (v___x_1433_ == 0)
{
lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; 
v___x_1434_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1434_, 0, v_a_1431_);
v___x_1435_ = l_Lean_MessageData_ofFormat(v___x_1434_);
v___x_1436_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(v_a_1430_, v___x_1435_, v___y_1337_, v___y_1338_);
lean_dec(v_a_1430_);
return v___x_1436_;
}
else
{
lean_object* v___x_1437_; 
lean_dec_ref(v_a_1431_);
v___x_1437_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg(v_a_1430_);
return v___x_1437_;
}
}
else
{
lean_object* v___x_1438_; 
v___x_1438_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
return v___x_1438_;
}
}
}
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec(v_a_1354_);
lean_dec(v_openDecls_1352_);
lean_dec(v_currNamespace_1349_);
lean_dec_ref(v_opts_1346_);
lean_dec_ref(v_env_1341_);
lean_dec_ref(v_x_1336_);
v_a_1442_ = lean_ctor_get(v___x_1355_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1355_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1355_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1355_);
v___x_1444_ = lean_box(0);
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
v_resetjp_1443_:
{
lean_object* v___x_1447_; 
if (v_isShared_1445_ == 0)
{
v___x_1447_ = v___x_1444_;
goto v_reusejp_1446_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v_a_1442_);
v___x_1447_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1446_;
}
v_reusejp_1446_:
{
return v___x_1447_;
}
}
}
}
else
{
lean_object* v_a_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1457_; 
lean_dec(v_openDecls_1352_);
lean_dec(v_currNamespace_1349_);
lean_dec_ref(v_opts_1346_);
lean_dec_ref(v_env_1341_);
lean_dec_ref(v_x_1336_);
v_a_1450_ = lean_ctor_get(v___x_1353_, 0);
v_isSharedCheck_1457_ = !lean_is_exclusive(v___x_1353_);
if (v_isSharedCheck_1457_ == 0)
{
v___x_1452_ = v___x_1353_;
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_a_1450_);
lean_dec(v___x_1353_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1455_; 
if (v_isShared_1453_ == 0)
{
v___x_1455_ = v___x_1452_;
goto v_reusejp_1454_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v_a_1450_);
v___x_1455_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1454_;
}
v_reusejp_1454_:
{
return v___x_1455_;
}
}
}
}
else
{
lean_object* v_a_1458_; lean_object* v___x_1460_; uint8_t v_isShared_1461_; uint8_t v_isSharedCheck_1465_; 
lean_dec(v_currNamespace_1349_);
lean_dec_ref(v_opts_1346_);
lean_dec_ref(v_env_1341_);
lean_dec_ref(v_x_1336_);
v_a_1458_ = lean_ctor_get(v___x_1350_, 0);
v_isSharedCheck_1465_ = !lean_is_exclusive(v___x_1350_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1460_ = v___x_1350_;
v_isShared_1461_ = v_isSharedCheck_1465_;
goto v_resetjp_1459_;
}
else
{
lean_inc(v_a_1458_);
lean_dec(v___x_1350_);
v___x_1460_ = lean_box(0);
v_isShared_1461_ = v_isSharedCheck_1465_;
goto v_resetjp_1459_;
}
v_resetjp_1459_:
{
lean_object* v___x_1463_; 
if (v_isShared_1461_ == 0)
{
v___x_1463_ = v___x_1460_;
goto v_reusejp_1462_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v_a_1458_);
v___x_1463_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1462_;
}
v_reusejp_1462_:
{
return v___x_1463_;
}
}
}
}
else
{
lean_object* v_a_1466_; lean_object* v___x_1468_; uint8_t v_isShared_1469_; uint8_t v_isSharedCheck_1473_; 
lean_dec_ref(v_opts_1346_);
lean_dec_ref(v_env_1341_);
lean_dec_ref(v_x_1336_);
v_a_1466_ = lean_ctor_get(v___x_1347_, 0);
v_isSharedCheck_1473_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1473_ == 0)
{
v___x_1468_ = v___x_1347_;
v_isShared_1469_ = v_isSharedCheck_1473_;
goto v_resetjp_1467_;
}
else
{
lean_inc(v_a_1466_);
lean_dec(v___x_1347_);
v___x_1468_ = lean_box(0);
v_isShared_1469_ = v_isSharedCheck_1473_;
goto v_resetjp_1467_;
}
v_resetjp_1467_:
{
lean_object* v___x_1471_; 
if (v_isShared_1469_ == 0)
{
v___x_1471_ = v___x_1468_;
goto v_reusejp_1470_;
}
else
{
lean_object* v_reuseFailAlloc_1472_; 
v_reuseFailAlloc_1472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1472_, 0, v_a_1466_);
v___x_1471_ = v_reuseFailAlloc_1472_;
goto v_reusejp_1470_;
}
v_reusejp_1470_:
{
return v___x_1471_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg___boxed(lean_object* v_x_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_){
_start:
{
lean_object* v_res_1478_; 
v_res_1478_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(v_x_1474_, v___y_1475_, v___y_1476_);
lean_dec(v___y_1476_);
lean_dec_ref(v___y_1475_);
return v_res_1478_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1480_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__0));
v___x_1481_ = l_Lean_stringToMessageData(v___x_1480_);
return v___x_1481_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1483_; lean_object* v___x_1484_; 
v___x_1483_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__2));
v___x_1484_ = l_Lean_stringToMessageData(v___x_1483_);
return v___x_1484_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5(void){
_start:
{
lean_object* v___x_1486_; lean_object* v___x_1487_; 
v___x_1486_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__4));
v___x_1487_ = l_Lean_stringToMessageData(v___x_1486_);
return v___x_1487_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1489_; lean_object* v___x_1490_; 
v___x_1489_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__6));
v___x_1490_ = l_Lean_stringToMessageData(v___x_1489_);
return v___x_1490_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1492_; lean_object* v___x_1493_; 
v___x_1492_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__8));
v___x_1493_ = l_Lean_stringToMessageData(v___x_1492_);
return v___x_1493_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1495_; lean_object* v___x_1496_; 
v___x_1495_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__10));
v___x_1496_ = l_Lean_stringToMessageData(v___x_1495_);
return v___x_1496_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16(void){
_start:
{
lean_object* v___x_1505_; lean_object* v___x_1506_; 
v___x_1505_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__15));
v___x_1506_ = l_Lean_stringToMessageData(v___x_1505_);
return v___x_1506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1(lean_object* v___x_1507_, lean_object* v_attrInstance_1508_, lean_object* v___f_1509_, lean_object* v___x_1510_, lean_object* v___x_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_){
_start:
{
lean_object* v___x_1515_; 
v___x_1515_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(v___x_1507_, v___y_1512_, v___y_1513_);
if (lean_obj_tag(v___x_1515_) == 0)
{
lean_object* v_a_1516_; lean_object* v___x_1517_; lean_object* v_attr_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; 
v_a_1516_ = lean_ctor_get(v___x_1515_, 0);
lean_inc(v_a_1516_);
lean_dec_ref_known(v___x_1515_, 1);
v___x_1517_ = lean_unsigned_to_nat(1u);
v_attr_1518_ = l_Lean_Syntax_getArg(v_attrInstance_1508_, v___x_1517_);
v___x_1519_ = lean_alloc_closure((void*)(l_Lean_expandMacros), 4, 2);
lean_closure_set(v___x_1519_, 0, v_attr_1518_);
lean_closure_set(v___x_1519_, 1, v___f_1509_);
v___x_1520_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(v___x_1519_, v___y_1512_, v___y_1513_);
if (lean_obj_tag(v___x_1520_) == 0)
{
lean_object* v_a_1521_; lean_object* v___x_1523_; uint8_t v_isShared_1524_; uint8_t v_isSharedCheck_1627_; 
v_a_1521_ = lean_ctor_get(v___x_1520_, 0);
v_isSharedCheck_1627_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1627_ == 0)
{
v___x_1523_ = v___x_1520_;
v_isShared_1524_ = v_isSharedCheck_1627_;
goto v_resetjp_1522_;
}
else
{
lean_inc(v_a_1521_);
lean_dec(v___x_1520_);
v___x_1523_ = lean_box(0);
v_isShared_1524_ = v_isSharedCheck_1627_;
goto v_resetjp_1522_;
}
v_resetjp_1522_:
{
lean_object* v___y_1526_; lean_object* v___y_1533_; uint8_t v___y_1534_; lean_object* v___y_1535_; lean_object* v___y_1536_; lean_object* v___y_1537_; lean_object* v_attrName_1548_; lean_object* v___y_1549_; lean_object* v___y_1550_; lean_object* v___x_1608_; lean_object* v___x_1609_; uint8_t v___x_1610_; 
lean_inc(v_a_1521_);
v___x_1608_ = l_Lean_Syntax_getKind(v_a_1521_);
v___x_1609_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__14));
v___x_1610_ = lean_name_eq(v___x_1608_, v___x_1609_);
if (v___x_1610_ == 0)
{
if (lean_obj_tag(v___x_1608_) == 1)
{
lean_object* v_str_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
v_str_1611_ = lean_ctor_get(v___x_1608_, 1);
lean_inc_ref(v_str_1611_);
lean_dec_ref_known(v___x_1608_, 2);
v___x_1612_ = lean_box(0);
v___x_1613_ = l_Lean_Name_str___override(v___x_1612_, v_str_1611_);
v_attrName_1548_ = v___x_1613_;
v___y_1549_ = v___y_1512_;
v___y_1550_ = v___y_1513_;
goto v___jp_1547_;
}
else
{
lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v_a_1616_; lean_object* v___x_1618_; uint8_t v_isShared_1619_; uint8_t v_isSharedCheck_1623_; 
lean_dec(v___x_1608_);
lean_del_object(v___x_1523_);
lean_dec(v_a_1516_);
lean_dec(v___x_1510_);
v___x_1614_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__16);
v___x_1615_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(v_a_1521_, v___x_1614_, v___y_1512_, v___y_1513_);
lean_dec(v_a_1521_);
v_a_1616_ = lean_ctor_get(v___x_1615_, 0);
v_isSharedCheck_1623_ = !lean_is_exclusive(v___x_1615_);
if (v_isSharedCheck_1623_ == 0)
{
v___x_1618_ = v___x_1615_;
v_isShared_1619_ = v_isSharedCheck_1623_;
goto v_resetjp_1617_;
}
else
{
lean_inc(v_a_1616_);
lean_dec(v___x_1615_);
v___x_1618_ = lean_box(0);
v_isShared_1619_ = v_isSharedCheck_1623_;
goto v_resetjp_1617_;
}
v_resetjp_1617_:
{
lean_object* v___x_1621_; 
if (v_isShared_1619_ == 0)
{
v___x_1621_ = v___x_1618_;
goto v_reusejp_1620_;
}
else
{
lean_object* v_reuseFailAlloc_1622_; 
v_reuseFailAlloc_1622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1622_, 0, v_a_1616_);
v___x_1621_ = v_reuseFailAlloc_1622_;
goto v_reusejp_1620_;
}
v_reusejp_1620_:
{
return v___x_1621_;
}
}
}
}
else
{
lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; 
lean_dec(v___x_1608_);
v___x_1624_ = l_Lean_Syntax_getArg(v_a_1521_, v___x_1511_);
v___x_1625_ = l_Lean_Syntax_getId(v___x_1624_);
lean_dec(v___x_1624_);
v___x_1626_ = l_Lean_Name_eraseMacroScopes(v___x_1625_);
lean_dec(v___x_1625_);
v_attrName_1548_ = v___x_1626_;
v___y_1549_ = v___y_1512_;
v___y_1550_ = v___y_1513_;
goto v___jp_1547_;
}
v___jp_1525_:
{
lean_object* v___x_1527_; uint8_t v___x_1528_; lean_object* v___x_1530_; 
v___x_1527_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1527_, 0, v___y_1526_);
lean_ctor_set(v___x_1527_, 1, v_a_1521_);
v___x_1528_ = lean_unbox(v_a_1516_);
lean_dec(v_a_1516_);
lean_ctor_set_uint8(v___x_1527_, sizeof(void*)*2, v___x_1528_);
if (v_isShared_1524_ == 0)
{
lean_ctor_set(v___x_1523_, 0, v___x_1527_);
v___x_1530_ = v___x_1523_;
goto v_reusejp_1529_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1527_);
v___x_1530_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1529_;
}
v_reusejp_1529_:
{
return v___x_1530_;
}
}
v___jp_1532_:
{
lean_object* v___x_1538_; 
v___x_1538_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17(v___y_1533_, v___y_1534_, v___y_1536_, v___y_1537_);
if (lean_obj_tag(v___x_1538_) == 0)
{
lean_dec_ref_known(v___x_1538_, 1);
v___y_1526_ = v___y_1535_;
goto v___jp_1525_;
}
else
{
lean_object* v_a_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1546_; 
lean_dec(v___y_1535_);
lean_del_object(v___x_1523_);
lean_dec(v_a_1521_);
lean_dec(v_a_1516_);
v_a_1539_ = lean_ctor_get(v___x_1538_, 0);
v_isSharedCheck_1546_ = !lean_is_exclusive(v___x_1538_);
if (v_isSharedCheck_1546_ == 0)
{
v___x_1541_ = v___x_1538_;
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
else
{
lean_inc(v_a_1539_);
lean_dec(v___x_1538_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1544_; 
if (v_isShared_1542_ == 0)
{
v___x_1544_ = v___x_1541_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1545_; 
v_reuseFailAlloc_1545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1545_, 0, v_a_1539_);
v___x_1544_ = v_reuseFailAlloc_1545_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
return v___x_1544_;
}
}
}
}
v___jp_1547_:
{
lean_object* v___x_1551_; lean_object* v_env_1552_; lean_object* v___x_1553_; 
v___x_1551_ = lean_st_ref_get(v___y_1550_);
v_env_1552_ = lean_ctor_get(v___x_1551_, 0);
lean_inc_ref(v_env_1552_);
lean_dec(v___x_1551_);
lean_inc(v_attrName_1548_);
v___x_1553_ = l_Lean_getAttributeImpl(v_env_1552_, v_attrName_1548_);
if (lean_obj_tag(v___x_1553_) == 1)
{
lean_object* v___x_1554_; lean_object* v_env_1555_; lean_object* v___x_1556_; 
lean_dec_ref_known(v___x_1553_, 1);
v___x_1554_ = lean_st_ref_get(v___y_1550_);
v_env_1555_ = lean_ctor_get(v___x_1554_, 0);
lean_inc_ref(v_env_1555_);
lean_dec(v___x_1554_);
lean_inc(v_attrName_1548_);
v___x_1556_ = l_Lean_getAttributeImpl(v_env_1555_, v_attrName_1548_);
if (lean_obj_tag(v___x_1556_) == 1)
{
lean_object* v_a_1557_; lean_object* v___x_1558_; lean_object* v_toAttributeImplCore_1559_; lean_object* v_env_1560_; lean_object* v_ref_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
v_a_1557_ = lean_ctor_get(v___x_1556_, 0);
lean_inc(v_a_1557_);
lean_dec_ref_known(v___x_1556_, 1);
v___x_1558_ = lean_st_ref_get(v___y_1550_);
v_toAttributeImplCore_1559_ = lean_ctor_get(v_a_1557_, 0);
lean_inc_ref(v_toAttributeImplCore_1559_);
lean_dec(v_a_1557_);
v_env_1560_ = lean_ctor_get(v___x_1558_, 0);
lean_inc_ref(v_env_1560_);
lean_dec(v___x_1558_);
v_ref_1561_ = lean_ctor_get(v_toAttributeImplCore_1559_, 0);
lean_inc_n(v_ref_1561_, 2);
lean_dec_ref(v_toAttributeImplCore_1559_);
v___x_1562_ = l_Lean_regularInitAttr;
v___x_1563_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_1510_, v___x_1562_, v_env_1560_, v_ref_1561_);
if (lean_obj_tag(v___x_1563_) == 0)
{
lean_dec(v_ref_1561_);
v___y_1526_ = v_attrName_1548_;
goto v___jp_1525_;
}
else
{
lean_object* v___x_1564_; lean_object* v_env_1565_; uint8_t v___x_1566_; lean_object* v___x_1567_; 
lean_dec_ref_known(v___x_1563_, 1);
v___x_1564_ = lean_st_ref_get(v___y_1550_);
v_env_1565_ = lean_ctor_get(v___x_1564_, 0);
lean_inc_ref(v_env_1565_);
lean_dec(v___x_1564_);
v___x_1566_ = 1;
v___x_1567_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1565_, v_ref_1561_);
lean_dec_ref(v_env_1565_);
if (lean_obj_tag(v___x_1567_) == 1)
{
lean_object* v_val_1568_; lean_object* v___x_1569_; lean_object* v_env_1570_; lean_object* v___x_1571_; lean_object* v_modules_1572_; lean_object* v___x_1573_; uint8_t v___x_1574_; 
v_val_1568_ = lean_ctor_get(v___x_1567_, 0);
lean_inc(v_val_1568_);
lean_dec_ref_known(v___x_1567_, 1);
v___x_1569_ = lean_st_ref_get(v___y_1550_);
v_env_1570_ = lean_ctor_get(v___x_1569_, 0);
lean_inc_ref(v_env_1570_);
lean_dec(v___x_1569_);
v___x_1571_ = l_Lean_Environment_header(v_env_1570_);
lean_dec_ref(v_env_1570_);
v_modules_1572_ = lean_ctor_get(v___x_1571_, 3);
lean_inc_ref(v_modules_1572_);
lean_dec_ref(v___x_1571_);
v___x_1573_ = lean_array_get_size(v_modules_1572_);
v___x_1574_ = lean_nat_dec_lt(v_val_1568_, v___x_1573_);
if (v___x_1574_ == 0)
{
lean_dec_ref(v_modules_1572_);
lean_dec(v_val_1568_);
v___y_1533_ = v_ref_1561_;
v___y_1534_ = v___x_1566_;
v___y_1535_ = v_attrName_1548_;
v___y_1536_ = v___y_1549_;
v___y_1537_ = v___y_1550_;
goto v___jp_1532_;
}
else
{
lean_object* v___x_1575_; uint8_t v_hasData_1576_; 
v___x_1575_ = lean_array_fget_borrowed(v_modules_1572_, v_val_1568_);
v_hasData_1576_ = lean_ctor_get_uint8(v___x_1575_, sizeof(void*)*1 + 1);
if (v_hasData_1576_ == 0)
{
lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v_toImport_1579_; lean_object* v_module_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v_a_1594_; lean_object* v___x_1596_; uint8_t v_isShared_1597_; uint8_t v_isSharedCheck_1601_; 
lean_dec(v_ref_1561_);
lean_del_object(v___x_1523_);
lean_dec(v_a_1521_);
lean_dec(v_a_1516_);
v___x_1577_ = l_Lean_instInhabitedEffectiveImport_default;
v___x_1578_ = lean_array_get(v___x_1577_, v_modules_1572_, v_val_1568_);
lean_dec(v_val_1568_);
lean_dec_ref(v_modules_1572_);
v_toImport_1579_ = lean_ctor_get(v___x_1578_, 0);
lean_inc_ref(v_toImport_1579_);
lean_dec(v___x_1578_);
v_module_1580_ = lean_ctor_get(v_toImport_1579_, 0);
lean_inc(v_module_1580_);
lean_dec_ref(v_toImport_1579_);
v___x_1581_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__1);
v___x_1582_ = l_Lean_MessageData_ofName(v_attrName_1548_);
v___x_1583_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1581_);
lean_ctor_set(v___x_1583_, 1, v___x_1582_);
v___x_1584_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__3);
v___x_1585_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1585_, 0, v___x_1583_);
lean_ctor_set(v___x_1585_, 1, v___x_1584_);
v___x_1586_ = l_Lean_MessageData_ofName(v_module_1580_);
lean_inc_ref(v___x_1586_);
v___x_1587_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1587_, 0, v___x_1585_);
lean_ctor_set(v___x_1587_, 1, v___x_1586_);
v___x_1588_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__5);
v___x_1589_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1589_, 0, v___x_1587_);
lean_ctor_set(v___x_1589_, 1, v___x_1588_);
v___x_1590_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1590_, 0, v___x_1589_);
lean_ctor_set(v___x_1590_, 1, v___x_1586_);
v___x_1591_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__7);
v___x_1592_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1592_, 0, v___x_1590_);
lean_ctor_set(v___x_1592_, 1, v___x_1591_);
v___x_1593_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(v___x_1592_, v___y_1549_, v___y_1550_);
v_a_1594_ = lean_ctor_get(v___x_1593_, 0);
v_isSharedCheck_1601_ = !lean_is_exclusive(v___x_1593_);
if (v_isSharedCheck_1601_ == 0)
{
v___x_1596_ = v___x_1593_;
v_isShared_1597_ = v_isSharedCheck_1601_;
goto v_resetjp_1595_;
}
else
{
lean_inc(v_a_1594_);
lean_dec(v___x_1593_);
v___x_1596_ = lean_box(0);
v_isShared_1597_ = v_isSharedCheck_1601_;
goto v_resetjp_1595_;
}
v_resetjp_1595_:
{
lean_object* v___x_1599_; 
if (v_isShared_1597_ == 0)
{
v___x_1599_ = v___x_1596_;
goto v_reusejp_1598_;
}
else
{
lean_object* v_reuseFailAlloc_1600_; 
v_reuseFailAlloc_1600_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1600_, 0, v_a_1594_);
v___x_1599_ = v_reuseFailAlloc_1600_;
goto v_reusejp_1598_;
}
v_reusejp_1598_:
{
return v___x_1599_;
}
}
}
else
{
lean_dec_ref(v_modules_1572_);
lean_dec(v_val_1568_);
v___y_1533_ = v_ref_1561_;
v___y_1534_ = v___x_1566_;
v___y_1535_ = v_attrName_1548_;
v___y_1536_ = v___y_1549_;
v___y_1537_ = v___y_1550_;
goto v___jp_1532_;
}
}
}
else
{
lean_dec(v___x_1567_);
v___y_1533_ = v_ref_1561_;
v___y_1534_ = v___x_1566_;
v___y_1535_ = v_attrName_1548_;
v___y_1536_ = v___y_1549_;
v___y_1537_ = v___y_1550_;
goto v___jp_1532_;
}
}
}
else
{
lean_dec_ref(v___x_1556_);
lean_dec(v___x_1510_);
v___y_1526_ = v_attrName_1548_;
goto v___jp_1525_;
}
}
else
{
lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; 
lean_dec_ref(v___x_1553_);
lean_del_object(v___x_1523_);
lean_dec(v_a_1521_);
lean_dec(v_a_1516_);
lean_dec(v___x_1510_);
v___x_1602_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__9);
v___x_1603_ = l_Lean_MessageData_ofName(v_attrName_1548_);
v___x_1604_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1604_, 0, v___x_1602_);
lean_ctor_set(v___x_1604_, 1, v___x_1603_);
v___x_1605_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___closed__11);
v___x_1606_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1606_, 0, v___x_1604_);
lean_ctor_set(v___x_1606_, 1, v___x_1605_);
v___x_1607_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(v___x_1606_, v___y_1549_, v___y_1550_);
return v___x_1607_;
}
}
}
}
else
{
lean_object* v_a_1628_; lean_object* v___x_1630_; uint8_t v_isShared_1631_; uint8_t v_isSharedCheck_1635_; 
lean_dec(v_a_1516_);
lean_dec(v___x_1510_);
v_a_1628_ = lean_ctor_get(v___x_1520_, 0);
v_isSharedCheck_1635_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1635_ == 0)
{
v___x_1630_ = v___x_1520_;
v_isShared_1631_ = v_isSharedCheck_1635_;
goto v_resetjp_1629_;
}
else
{
lean_inc(v_a_1628_);
lean_dec(v___x_1520_);
v___x_1630_ = lean_box(0);
v_isShared_1631_ = v_isSharedCheck_1635_;
goto v_resetjp_1629_;
}
v_resetjp_1629_:
{
lean_object* v___x_1633_; 
if (v_isShared_1631_ == 0)
{
v___x_1633_ = v___x_1630_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1634_; 
v_reuseFailAlloc_1634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1634_, 0, v_a_1628_);
v___x_1633_ = v_reuseFailAlloc_1634_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
return v___x_1633_;
}
}
}
}
else
{
lean_object* v_a_1636_; lean_object* v___x_1638_; uint8_t v_isShared_1639_; uint8_t v_isSharedCheck_1643_; 
lean_dec(v___x_1510_);
lean_dec_ref(v___f_1509_);
v_a_1636_ = lean_ctor_get(v___x_1515_, 0);
v_isSharedCheck_1643_ = !lean_is_exclusive(v___x_1515_);
if (v_isSharedCheck_1643_ == 0)
{
v___x_1638_ = v___x_1515_;
v_isShared_1639_ = v_isSharedCheck_1643_;
goto v_resetjp_1637_;
}
else
{
lean_inc(v_a_1636_);
lean_dec(v___x_1515_);
v___x_1638_ = lean_box(0);
v_isShared_1639_ = v_isSharedCheck_1643_;
goto v_resetjp_1637_;
}
v_resetjp_1637_:
{
lean_object* v___x_1641_; 
if (v_isShared_1639_ == 0)
{
v___x_1641_ = v___x_1638_;
goto v_reusejp_1640_;
}
else
{
lean_object* v_reuseFailAlloc_1642_; 
v_reuseFailAlloc_1642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1642_, 0, v_a_1636_);
v___x_1641_ = v_reuseFailAlloc_1642_;
goto v_reusejp_1640_;
}
v_reusejp_1640_:
{
return v___x_1641_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___boxed(lean_object* v___x_1644_, lean_object* v_attrInstance_1645_, lean_object* v___f_1646_, lean_object* v___x_1647_, lean_object* v___x_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_){
_start:
{
lean_object* v_res_1652_; 
v_res_1652_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1(v___x_1644_, v_attrInstance_1645_, v___f_1646_, v___x_1647_, v___x_1648_, v___y_1649_, v___y_1650_);
lean_dec(v___y_1650_);
lean_dec_ref(v___y_1649_);
lean_dec(v___x_1648_);
lean_dec(v_attrInstance_1645_);
return v_res_1652_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0(lean_object* v_k_1659_){
_start:
{
lean_object* v___x_1660_; uint8_t v___x_1661_; 
v___x_1660_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___closed__1));
v___x_1661_ = lean_name_eq(v_k_1659_, v___x_1660_);
if (v___x_1661_ == 0)
{
uint8_t v___x_1662_; 
v___x_1662_ = 1;
return v___x_1662_;
}
else
{
uint8_t v___x_1663_; 
v___x_1663_ = 0;
return v___x_1663_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0___boxed(lean_object* v_k_1664_){
_start:
{
uint8_t v_res_1665_; lean_object* v_r_1666_; 
v_res_1665_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__0(v_k_1664_);
lean_dec(v_k_1664_);
v_r_1666_ = lean_box(v_res_1665_);
return v_r_1666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9(lean_object* v_attrInstance_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_){
_start:
{
lean_object* v___f_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___f_1677_; uint8_t v___x_1678_; lean_object* v___x_1679_; 
v___f_1672_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___closed__0));
v___x_1673_ = lean_box(0);
v___x_1674_ = lean_unsigned_to_nat(0u);
v___x_1675_ = l_Lean_Syntax_getArg(v_attrInstance_1668_, v___x_1674_);
v___x_1676_ = lean_alloc_closure((void*)(l_Lean_Elab_toAttributeKind___boxed), 3, 1);
lean_closure_set(v___x_1676_, 0, v___x_1675_);
v___f_1677_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___lam__1___boxed), 8, 5);
lean_closure_set(v___f_1677_, 0, v___x_1676_);
lean_closure_set(v___f_1677_, 1, v_attrInstance_1668_);
lean_closure_set(v___f_1677_, 2, v___f_1672_);
lean_closure_set(v___f_1677_, 3, v___x_1673_);
lean_closure_set(v___f_1677_, 4, v___x_1674_);
v___x_1678_ = 1;
v___x_1679_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg(v___f_1677_, v___x_1678_, v___y_1669_, v___y_1670_);
return v___x_1679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9___boxed(lean_object* v_attrInstance_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_){
_start:
{
lean_object* v_res_1684_; 
v_res_1684_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9(v_attrInstance_1680_, v___y_1681_, v___y_1682_);
lean_dec(v___y_1682_);
lean_dec_ref(v___y_1681_);
return v_res_1684_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0(uint8_t v___y_1685_, uint8_t v_suppressElabErrors_1686_, lean_object* v_x_1687_){
_start:
{
if (lean_obj_tag(v_x_1687_) == 1)
{
lean_object* v_pre_1688_; 
v_pre_1688_ = lean_ctor_get(v_x_1687_, 0);
if (lean_obj_tag(v_pre_1688_) == 0)
{
lean_object* v_str_1689_; lean_object* v___x_1690_; uint8_t v___x_1691_; 
v_str_1689_ = lean_ctor_get(v_x_1687_, 1);
v___x_1690_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29___closed__10));
v___x_1691_ = lean_string_dec_eq(v_str_1689_, v___x_1690_);
if (v___x_1691_ == 0)
{
return v___y_1685_;
}
else
{
return v_suppressElabErrors_1686_;
}
}
else
{
return v___y_1685_;
}
}
else
{
return v___y_1685_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0___boxed(lean_object* v___y_1692_, lean_object* v_suppressElabErrors_1693_, lean_object* v_x_1694_){
_start:
{
uint8_t v___y_22997__boxed_1695_; uint8_t v_suppressElabErrors_boxed_1696_; uint8_t v_res_1697_; lean_object* v_r_1698_; 
v___y_22997__boxed_1695_ = lean_unbox(v___y_1692_);
v_suppressElabErrors_boxed_1696_ = lean_unbox(v_suppressElabErrors_1693_);
v_res_1697_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0(v___y_22997__boxed_1695_, v_suppressElabErrors_boxed_1696_, v_x_1694_);
lean_dec(v_x_1694_);
v_r_1698_ = lean_box(v_res_1697_);
return v_r_1698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(lean_object* v_ref_1699_, lean_object* v_msgData_1700_, uint8_t v_severity_1701_, uint8_t v_isSilent_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_){
_start:
{
lean_object* v___y_1707_; lean_object* v___y_1708_; uint8_t v___y_1709_; uint8_t v___y_1710_; lean_object* v___y_1711_; lean_object* v___y_1712_; lean_object* v___y_1713_; lean_object* v___y_1714_; uint8_t v___y_1771_; lean_object* v___y_1772_; uint8_t v___y_1773_; uint8_t v___y_1774_; lean_object* v___y_1775_; uint8_t v___y_1799_; uint8_t v___y_1800_; uint8_t v___y_1801_; lean_object* v___y_1802_; lean_object* v___y_1803_; uint8_t v___y_1807_; uint8_t v___y_1808_; uint8_t v___y_1809_; uint8_t v___x_1824_; uint8_t v___y_1826_; uint8_t v___y_1827_; uint8_t v___y_1828_; uint8_t v___y_1830_; uint8_t v___x_1842_; 
v___x_1824_ = 2;
v___x_1842_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1701_, v___x_1824_);
if (v___x_1842_ == 0)
{
v___y_1830_ = v___x_1842_;
goto v___jp_1829_;
}
else
{
uint8_t v___x_1843_; 
lean_inc_ref(v_msgData_1700_);
v___x_1843_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1700_);
v___y_1830_ = v___x_1843_;
goto v___jp_1829_;
}
v___jp_1706_:
{
lean_object* v___x_1715_; 
v___x_1715_ = l_Lean_Elab_Command_getScope___redArg(v___y_1714_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___x_1717_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
lean_inc(v_a_1716_);
lean_dec_ref_known(v___x_1715_, 1);
v___x_1717_ = l_Lean_Elab_Command_getScope___redArg(v___y_1714_);
if (lean_obj_tag(v___x_1717_) == 0)
{
lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1753_; 
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1753_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1720_ = v___x_1717_;
v_isShared_1721_ = v_isSharedCheck_1753_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1717_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1753_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
lean_object* v___x_1722_; lean_object* v_currNamespace_1723_; lean_object* v_openDecls_1724_; lean_object* v_env_1725_; lean_object* v_messages_1726_; lean_object* v_scopes_1727_; lean_object* v_usedQuotCtxts_1728_; lean_object* v_nextMacroScope_1729_; lean_object* v_maxRecDepth_1730_; lean_object* v_ngen_1731_; lean_object* v_auxDeclNGen_1732_; lean_object* v_infoState_1733_; lean_object* v_traceState_1734_; lean_object* v_snapshotTasks_1735_; lean_object* v_prevLinterStates_1736_; lean_object* v___x_1738_; uint8_t v_isShared_1739_; uint8_t v_isSharedCheck_1752_; 
v___x_1722_ = lean_st_ref_take(v___y_1714_);
v_currNamespace_1723_ = lean_ctor_get(v_a_1716_, 2);
lean_inc(v_currNamespace_1723_);
lean_dec(v_a_1716_);
v_openDecls_1724_ = lean_ctor_get(v_a_1718_, 3);
lean_inc(v_openDecls_1724_);
lean_dec(v_a_1718_);
v_env_1725_ = lean_ctor_get(v___x_1722_, 0);
v_messages_1726_ = lean_ctor_get(v___x_1722_, 1);
v_scopes_1727_ = lean_ctor_get(v___x_1722_, 2);
v_usedQuotCtxts_1728_ = lean_ctor_get(v___x_1722_, 3);
v_nextMacroScope_1729_ = lean_ctor_get(v___x_1722_, 4);
v_maxRecDepth_1730_ = lean_ctor_get(v___x_1722_, 5);
v_ngen_1731_ = lean_ctor_get(v___x_1722_, 6);
v_auxDeclNGen_1732_ = lean_ctor_get(v___x_1722_, 7);
v_infoState_1733_ = lean_ctor_get(v___x_1722_, 8);
v_traceState_1734_ = lean_ctor_get(v___x_1722_, 9);
v_snapshotTasks_1735_ = lean_ctor_get(v___x_1722_, 10);
v_prevLinterStates_1736_ = lean_ctor_get(v___x_1722_, 11);
v_isSharedCheck_1752_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1752_ == 0)
{
v___x_1738_ = v___x_1722_;
v_isShared_1739_ = v_isSharedCheck_1752_;
goto v_resetjp_1737_;
}
else
{
lean_inc(v_prevLinterStates_1736_);
lean_inc(v_snapshotTasks_1735_);
lean_inc(v_traceState_1734_);
lean_inc(v_infoState_1733_);
lean_inc(v_auxDeclNGen_1732_);
lean_inc(v_ngen_1731_);
lean_inc(v_maxRecDepth_1730_);
lean_inc(v_nextMacroScope_1729_);
lean_inc(v_usedQuotCtxts_1728_);
lean_inc(v_scopes_1727_);
lean_inc(v_messages_1726_);
lean_inc(v_env_1725_);
lean_dec(v___x_1722_);
v___x_1738_ = lean_box(0);
v_isShared_1739_ = v_isSharedCheck_1752_;
goto v_resetjp_1737_;
}
v_resetjp_1737_:
{
lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1745_; 
v___x_1740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1740_, 0, v_currNamespace_1723_);
lean_ctor_set(v___x_1740_, 1, v_openDecls_1724_);
v___x_1741_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1741_, 0, v___x_1740_);
lean_ctor_set(v___x_1741_, 1, v___y_1711_);
lean_inc_ref(v___y_1712_);
lean_inc_ref(v___y_1707_);
v___x_1742_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1742_, 0, v___y_1707_);
lean_ctor_set(v___x_1742_, 1, v___y_1713_);
lean_ctor_set(v___x_1742_, 2, v___y_1708_);
lean_ctor_set(v___x_1742_, 3, v___y_1712_);
lean_ctor_set(v___x_1742_, 4, v___x_1741_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*5, v___y_1710_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*5 + 1, v___y_1709_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*5 + 2, v_isSilent_1702_);
v___x_1743_ = l_Lean_MessageLog_add(v___x_1742_, v_messages_1726_);
if (v_isShared_1739_ == 0)
{
lean_ctor_set(v___x_1738_, 1, v___x_1743_);
v___x_1745_ = v___x_1738_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1751_; 
v_reuseFailAlloc_1751_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1751_, 0, v_env_1725_);
lean_ctor_set(v_reuseFailAlloc_1751_, 1, v___x_1743_);
lean_ctor_set(v_reuseFailAlloc_1751_, 2, v_scopes_1727_);
lean_ctor_set(v_reuseFailAlloc_1751_, 3, v_usedQuotCtxts_1728_);
lean_ctor_set(v_reuseFailAlloc_1751_, 4, v_nextMacroScope_1729_);
lean_ctor_set(v_reuseFailAlloc_1751_, 5, v_maxRecDepth_1730_);
lean_ctor_set(v_reuseFailAlloc_1751_, 6, v_ngen_1731_);
lean_ctor_set(v_reuseFailAlloc_1751_, 7, v_auxDeclNGen_1732_);
lean_ctor_set(v_reuseFailAlloc_1751_, 8, v_infoState_1733_);
lean_ctor_set(v_reuseFailAlloc_1751_, 9, v_traceState_1734_);
lean_ctor_set(v_reuseFailAlloc_1751_, 10, v_snapshotTasks_1735_);
lean_ctor_set(v_reuseFailAlloc_1751_, 11, v_prevLinterStates_1736_);
v___x_1745_ = v_reuseFailAlloc_1751_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1749_; 
v___x_1746_ = lean_st_ref_set(v___y_1714_, v___x_1745_);
v___x_1747_ = lean_box(0);
if (v_isShared_1721_ == 0)
{
lean_ctor_set(v___x_1720_, 0, v___x_1747_);
v___x_1749_ = v___x_1720_;
goto v_reusejp_1748_;
}
else
{
lean_object* v_reuseFailAlloc_1750_; 
v_reuseFailAlloc_1750_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1750_, 0, v___x_1747_);
v___x_1749_ = v_reuseFailAlloc_1750_;
goto v_reusejp_1748_;
}
v_reusejp_1748_:
{
return v___x_1749_;
}
}
}
}
}
else
{
lean_object* v_a_1754_; lean_object* v___x_1756_; uint8_t v_isShared_1757_; uint8_t v_isSharedCheck_1761_; 
lean_dec(v_a_1716_);
lean_dec_ref(v___y_1713_);
lean_dec_ref(v___y_1711_);
lean_dec(v___y_1708_);
v_a_1754_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1761_ == 0)
{
v___x_1756_ = v___x_1717_;
v_isShared_1757_ = v_isSharedCheck_1761_;
goto v_resetjp_1755_;
}
else
{
lean_inc(v_a_1754_);
lean_dec(v___x_1717_);
v___x_1756_ = lean_box(0);
v_isShared_1757_ = v_isSharedCheck_1761_;
goto v_resetjp_1755_;
}
v_resetjp_1755_:
{
lean_object* v___x_1759_; 
if (v_isShared_1757_ == 0)
{
v___x_1759_ = v___x_1756_;
goto v_reusejp_1758_;
}
else
{
lean_object* v_reuseFailAlloc_1760_; 
v_reuseFailAlloc_1760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1760_, 0, v_a_1754_);
v___x_1759_ = v_reuseFailAlloc_1760_;
goto v_reusejp_1758_;
}
v_reusejp_1758_:
{
return v___x_1759_;
}
}
}
}
else
{
lean_object* v_a_1762_; lean_object* v___x_1764_; uint8_t v_isShared_1765_; uint8_t v_isSharedCheck_1769_; 
lean_dec_ref(v___y_1713_);
lean_dec_ref(v___y_1711_);
lean_dec(v___y_1708_);
v_a_1762_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_1769_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1769_ == 0)
{
v___x_1764_ = v___x_1715_;
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
else
{
lean_inc(v_a_1762_);
lean_dec(v___x_1715_);
v___x_1764_ = lean_box(0);
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
v_resetjp_1763_:
{
lean_object* v___x_1767_; 
if (v_isShared_1765_ == 0)
{
v___x_1767_ = v___x_1764_;
goto v_reusejp_1766_;
}
else
{
lean_object* v_reuseFailAlloc_1768_; 
v_reuseFailAlloc_1768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1768_, 0, v_a_1762_);
v___x_1767_ = v_reuseFailAlloc_1768_;
goto v_reusejp_1766_;
}
v_reusejp_1766_:
{
return v___x_1767_;
}
}
}
}
v___jp_1770_:
{
lean_object* v_fileName_1776_; lean_object* v_fileMap_1777_; uint8_t v_suppressElabErrors_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v_a_1781_; lean_object* v___x_1783_; uint8_t v_isShared_1784_; uint8_t v_isSharedCheck_1797_; 
v_fileName_1776_ = lean_ctor_get(v___y_1703_, 0);
v_fileMap_1777_ = lean_ctor_get(v___y_1703_, 1);
v_suppressElabErrors_1778_ = lean_ctor_get_uint8(v___y_1703_, sizeof(void*)*10);
v___x_1779_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1700_);
v___x_1780_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v___x_1779_, v___y_1704_);
v_a_1781_ = lean_ctor_get(v___x_1780_, 0);
v_isSharedCheck_1797_ = !lean_is_exclusive(v___x_1780_);
if (v_isSharedCheck_1797_ == 0)
{
v___x_1783_ = v___x_1780_;
v_isShared_1784_ = v_isSharedCheck_1797_;
goto v_resetjp_1782_;
}
else
{
lean_inc(v_a_1781_);
lean_dec(v___x_1780_);
v___x_1783_ = lean_box(0);
v_isShared_1784_ = v_isSharedCheck_1797_;
goto v_resetjp_1782_;
}
v_resetjp_1782_:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; 
lean_inc_ref_n(v_fileMap_1777_, 2);
v___x_1785_ = l_Lean_FileMap_toPosition(v_fileMap_1777_, v___y_1772_);
lean_dec(v___y_1772_);
v___x_1786_ = l_Lean_FileMap_toPosition(v_fileMap_1777_, v___y_1775_);
lean_dec(v___y_1775_);
v___x_1787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1787_, 0, v___x_1786_);
v___x_1788_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
if (v_suppressElabErrors_1778_ == 0)
{
lean_del_object(v___x_1783_);
v___y_1707_ = v_fileName_1776_;
v___y_1708_ = v___x_1787_;
v___y_1709_ = v___y_1773_;
v___y_1710_ = v___y_1774_;
v___y_1711_ = v_a_1781_;
v___y_1712_ = v___x_1788_;
v___y_1713_ = v___x_1785_;
v___y_1714_ = v___y_1704_;
goto v___jp_1706_;
}
else
{
lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___f_1791_; uint8_t v___x_1792_; 
v___x_1789_ = lean_box(v___y_1771_);
v___x_1790_ = lean_box(v_suppressElabErrors_1778_);
v___f_1791_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1791_, 0, v___x_1789_);
lean_closure_set(v___f_1791_, 1, v___x_1790_);
lean_inc(v_a_1781_);
v___x_1792_ = l_Lean_MessageData_hasTag(v___f_1791_, v_a_1781_);
if (v___x_1792_ == 0)
{
lean_object* v___x_1793_; lean_object* v___x_1795_; 
lean_dec_ref_known(v___x_1787_, 1);
lean_dec_ref(v___x_1785_);
lean_dec(v_a_1781_);
v___x_1793_ = lean_box(0);
if (v_isShared_1784_ == 0)
{
lean_ctor_set(v___x_1783_, 0, v___x_1793_);
v___x_1795_ = v___x_1783_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1796_; 
v_reuseFailAlloc_1796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1796_, 0, v___x_1793_);
v___x_1795_ = v_reuseFailAlloc_1796_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
return v___x_1795_;
}
}
else
{
lean_del_object(v___x_1783_);
v___y_1707_ = v_fileName_1776_;
v___y_1708_ = v___x_1787_;
v___y_1709_ = v___y_1773_;
v___y_1710_ = v___y_1774_;
v___y_1711_ = v_a_1781_;
v___y_1712_ = v___x_1788_;
v___y_1713_ = v___x_1785_;
v___y_1714_ = v___y_1704_;
goto v___jp_1706_;
}
}
}
}
v___jp_1798_:
{
lean_object* v___x_1804_; 
v___x_1804_ = l_Lean_Syntax_getTailPos_x3f(v___y_1802_, v___y_1801_);
lean_dec(v___y_1802_);
if (lean_obj_tag(v___x_1804_) == 0)
{
lean_inc(v___y_1803_);
v___y_1771_ = v___y_1799_;
v___y_1772_ = v___y_1803_;
v___y_1773_ = v___y_1800_;
v___y_1774_ = v___y_1801_;
v___y_1775_ = v___y_1803_;
goto v___jp_1770_;
}
else
{
lean_object* v_val_1805_; 
v_val_1805_ = lean_ctor_get(v___x_1804_, 0);
lean_inc(v_val_1805_);
lean_dec_ref_known(v___x_1804_, 1);
v___y_1771_ = v___y_1799_;
v___y_1772_ = v___y_1803_;
v___y_1773_ = v___y_1800_;
v___y_1774_ = v___y_1801_;
v___y_1775_ = v_val_1805_;
goto v___jp_1770_;
}
}
v___jp_1806_:
{
lean_object* v___x_1810_; 
v___x_1810_ = l_Lean_Elab_Command_getRef___redArg(v___y_1703_);
if (lean_obj_tag(v___x_1810_) == 0)
{
lean_object* v_a_1811_; lean_object* v_ref_1812_; lean_object* v___x_1813_; 
v_a_1811_ = lean_ctor_get(v___x_1810_, 0);
lean_inc(v_a_1811_);
lean_dec_ref_known(v___x_1810_, 1);
v_ref_1812_ = l_Lean_replaceRef(v_ref_1699_, v_a_1811_);
lean_dec(v_a_1811_);
v___x_1813_ = l_Lean_Syntax_getPos_x3f(v_ref_1812_, v___y_1808_);
if (lean_obj_tag(v___x_1813_) == 0)
{
lean_object* v___x_1814_; 
v___x_1814_ = lean_unsigned_to_nat(0u);
v___y_1799_ = v___y_1807_;
v___y_1800_ = v___y_1809_;
v___y_1801_ = v___y_1808_;
v___y_1802_ = v_ref_1812_;
v___y_1803_ = v___x_1814_;
goto v___jp_1798_;
}
else
{
lean_object* v_val_1815_; 
v_val_1815_ = lean_ctor_get(v___x_1813_, 0);
lean_inc(v_val_1815_);
lean_dec_ref_known(v___x_1813_, 1);
v___y_1799_ = v___y_1807_;
v___y_1800_ = v___y_1809_;
v___y_1801_ = v___y_1808_;
v___y_1802_ = v_ref_1812_;
v___y_1803_ = v_val_1815_;
goto v___jp_1798_;
}
}
else
{
lean_object* v_a_1816_; lean_object* v___x_1818_; uint8_t v_isShared_1819_; uint8_t v_isSharedCheck_1823_; 
lean_dec_ref(v_msgData_1700_);
v_a_1816_ = lean_ctor_get(v___x_1810_, 0);
v_isSharedCheck_1823_ = !lean_is_exclusive(v___x_1810_);
if (v_isSharedCheck_1823_ == 0)
{
v___x_1818_ = v___x_1810_;
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
else
{
lean_inc(v_a_1816_);
lean_dec(v___x_1810_);
v___x_1818_ = lean_box(0);
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
v_resetjp_1817_:
{
lean_object* v___x_1821_; 
if (v_isShared_1819_ == 0)
{
v___x_1821_ = v___x_1818_;
goto v_reusejp_1820_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_a_1816_);
v___x_1821_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1820_;
}
v_reusejp_1820_:
{
return v___x_1821_;
}
}
}
}
v___jp_1825_:
{
if (v___y_1828_ == 0)
{
v___y_1807_ = v___y_1826_;
v___y_1808_ = v___y_1827_;
v___y_1809_ = v_severity_1701_;
goto v___jp_1806_;
}
else
{
v___y_1807_ = v___y_1826_;
v___y_1808_ = v___y_1827_;
v___y_1809_ = v___x_1824_;
goto v___jp_1806_;
}
}
v___jp_1829_:
{
if (v___y_1830_ == 0)
{
lean_object* v___x_1831_; lean_object* v_scopes_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v_opts_1835_; uint8_t v___x_1836_; uint8_t v___x_1837_; 
v___x_1831_ = lean_st_ref_get(v___y_1704_);
v_scopes_1832_ = lean_ctor_get(v___x_1831_, 2);
lean_inc(v_scopes_1832_);
lean_dec(v___x_1831_);
v___x_1833_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1834_ = l_List_head_x21___redArg(v___x_1833_, v_scopes_1832_);
lean_dec(v_scopes_1832_);
v_opts_1835_ = lean_ctor_get(v___x_1834_, 1);
lean_inc_ref(v_opts_1835_);
lean_dec(v___x_1834_);
v___x_1836_ = 1;
v___x_1837_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1701_, v___x_1836_);
if (v___x_1837_ == 0)
{
lean_dec_ref(v_opts_1835_);
v___y_1826_ = v___y_1830_;
v___y_1827_ = v___y_1830_;
v___y_1828_ = v___x_1837_;
goto v___jp_1825_;
}
else
{
lean_object* v___x_1838_; uint8_t v___x_1839_; 
v___x_1838_ = l_Lean_warningAsError;
v___x_1839_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5_spec__8(v_opts_1835_, v___x_1838_);
lean_dec_ref(v_opts_1835_);
v___y_1826_ = v___y_1830_;
v___y_1827_ = v___y_1830_;
v___y_1828_ = v___x_1839_;
goto v___jp_1825_;
}
}
else
{
lean_object* v___x_1840_; lean_object* v___x_1841_; 
lean_dec_ref(v_msgData_1700_);
v___x_1840_ = lean_box(0);
v___x_1841_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1841_, 0, v___x_1840_);
return v___x_1841_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14___boxed(lean_object* v_ref_1844_, lean_object* v_msgData_1845_, lean_object* v_severity_1846_, lean_object* v_isSilent_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_){
_start:
{
uint8_t v_severity_boxed_1851_; uint8_t v_isSilent_boxed_1852_; lean_object* v_res_1853_; 
v_severity_boxed_1851_ = lean_unbox(v_severity_1846_);
v_isSilent_boxed_1852_ = lean_unbox(v_isSilent_1847_);
v_res_1853_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(v_ref_1844_, v_msgData_1845_, v_severity_boxed_1851_, v_isSilent_boxed_1852_, v___y_1848_, v___y_1849_);
lean_dec(v___y_1849_);
lean_dec_ref(v___y_1848_);
lean_dec(v_ref_1844_);
return v_res_1853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13(lean_object* v_ref_1854_, lean_object* v_msgData_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_){
_start:
{
uint8_t v___x_1859_; uint8_t v___x_1860_; lean_object* v___x_1861_; 
v___x_1859_ = 2;
v___x_1860_ = 0;
v___x_1861_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(v_ref_1854_, v_msgData_1855_, v___x_1859_, v___x_1860_, v___y_1856_, v___y_1857_);
return v___x_1861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13___boxed(lean_object* v_ref_1862_, lean_object* v_msgData_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_){
_start:
{
lean_object* v_res_1867_; 
v_res_1867_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13(v_ref_1862_, v_msgData_1863_, v___y_1864_, v___y_1865_);
lean_dec(v___y_1865_);
lean_dec_ref(v___y_1864_);
lean_dec(v_ref_1862_);
return v_res_1867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18(lean_object* v_msgData_1868_, uint8_t v_severity_1869_, uint8_t v_isSilent_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_){
_start:
{
lean_object* v___x_1874_; 
v___x_1874_ = l_Lean_Elab_Command_getRef___redArg(v___y_1871_);
if (lean_obj_tag(v___x_1874_) == 0)
{
lean_object* v_a_1875_; lean_object* v___x_1876_; 
v_a_1875_ = lean_ctor_get(v___x_1874_, 0);
lean_inc(v_a_1875_);
lean_dec_ref_known(v___x_1874_, 1);
v___x_1876_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(v_a_1875_, v_msgData_1868_, v_severity_1869_, v_isSilent_1870_, v___y_1871_, v___y_1872_);
lean_dec(v_a_1875_);
return v___x_1876_;
}
else
{
lean_object* v_a_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1884_; 
lean_dec_ref(v_msgData_1868_);
v_a_1877_ = lean_ctor_get(v___x_1874_, 0);
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1874_);
if (v_isSharedCheck_1884_ == 0)
{
v___x_1879_ = v___x_1874_;
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_a_1877_);
lean_dec(v___x_1874_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v_a_1877_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
return v___x_1882_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18___boxed(lean_object* v_msgData_1885_, lean_object* v_severity_1886_, lean_object* v_isSilent_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_){
_start:
{
uint8_t v_severity_boxed_1891_; uint8_t v_isSilent_boxed_1892_; lean_object* v_res_1893_; 
v_severity_boxed_1891_ = lean_unbox(v_severity_1886_);
v_isSilent_boxed_1892_ = lean_unbox(v_isSilent_1887_);
v_res_1893_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18(v_msgData_1885_, v_severity_boxed_1891_, v_isSilent_boxed_1892_, v___y_1888_, v___y_1889_);
lean_dec(v___y_1889_);
lean_dec_ref(v___y_1888_);
return v_res_1893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14(lean_object* v_msgData_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_){
_start:
{
uint8_t v___x_1898_; uint8_t v___x_1899_; lean_object* v___x_1900_; 
v___x_1898_ = 2;
v___x_1899_ = 0;
v___x_1900_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14_spec__18(v_msgData_1894_, v___x_1898_, v___x_1899_, v___y_1895_, v___y_1896_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14___boxed(lean_object* v_msgData_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14(v_msgData_1901_, v___y_1902_, v___y_1903_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
return v_res_1905_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1(void){
_start:
{
lean_object* v___x_1907_; lean_object* v___x_1908_; 
v___x_1907_ = ((lean_object*)(lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__0));
v___x_1908_ = l_Lean_stringToMessageData(v___x_1907_);
return v___x_1908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8(lean_object* v_ex_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_){
_start:
{
if (lean_obj_tag(v_ex_1909_) == 0)
{
lean_object* v_ref_1913_; lean_object* v_msg_1914_; lean_object* v___x_1915_; 
v_ref_1913_ = lean_ctor_get(v_ex_1909_, 0);
lean_inc(v_ref_1913_);
v_msg_1914_ = lean_ctor_get(v_ex_1909_, 1);
lean_inc_ref(v_msg_1914_);
lean_dec_ref_known(v_ex_1909_, 2);
v___x_1915_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__13(v_ref_1913_, v_msg_1914_, v___y_1910_, v___y_1911_);
lean_dec(v_ref_1913_);
return v___x_1915_;
}
else
{
lean_object* v_id_1916_; uint8_t v___y_1918_; uint8_t v___x_1940_; 
v_id_1916_ = lean_ctor_get(v_ex_1909_, 0);
lean_inc(v_id_1916_);
v___x_1940_ = l_Lean_Elab_isAbortExceptionId(v_id_1916_);
if (v___x_1940_ == 0)
{
uint8_t v___x_1941_; 
v___x_1941_ = l_Lean_Exception_isInterrupt(v_ex_1909_);
lean_dec_ref_known(v_ex_1909_, 2);
v___y_1918_ = v___x_1941_;
goto v___jp_1917_;
}
else
{
lean_dec_ref_known(v_ex_1909_, 2);
v___y_1918_ = v___x_1940_;
goto v___jp_1917_;
}
v___jp_1917_:
{
if (v___y_1918_ == 0)
{
lean_object* v___x_1919_; 
v___x_1919_ = l_Lean_InternalExceptionId_getName(v_id_1916_);
lean_dec(v_id_1916_);
if (lean_obj_tag(v___x_1919_) == 0)
{
lean_object* v_a_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; 
v_a_1920_ = lean_ctor_get(v___x_1919_, 0);
lean_inc(v_a_1920_);
lean_dec_ref_known(v___x_1919_, 1);
v___x_1921_ = lean_obj_once(&lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1, &lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1_once, _init_lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___closed__1);
v___x_1922_ = l_Lean_MessageData_ofName(v_a_1920_);
v___x_1923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1923_, 0, v___x_1921_);
lean_ctor_set(v___x_1923_, 1, v___x_1922_);
v___x_1924_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8_spec__14(v___x_1923_, v___y_1910_, v___y_1911_);
return v___x_1924_;
}
else
{
lean_object* v_a_1925_; lean_object* v___x_1927_; uint8_t v_isShared_1928_; uint8_t v_isSharedCheck_1937_; 
v_a_1925_ = lean_ctor_get(v___x_1919_, 0);
v_isSharedCheck_1937_ = !lean_is_exclusive(v___x_1919_);
if (v_isSharedCheck_1937_ == 0)
{
v___x_1927_ = v___x_1919_;
v_isShared_1928_ = v_isSharedCheck_1937_;
goto v_resetjp_1926_;
}
else
{
lean_inc(v_a_1925_);
lean_dec(v___x_1919_);
v___x_1927_ = lean_box(0);
v_isShared_1928_ = v_isSharedCheck_1937_;
goto v_resetjp_1926_;
}
v_resetjp_1926_:
{
lean_object* v_ref_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1935_; 
v_ref_1929_ = lean_ctor_get(v___y_1910_, 7);
v___x_1930_ = lean_io_error_to_string(v_a_1925_);
v___x_1931_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1931_, 0, v___x_1930_);
v___x_1932_ = l_Lean_MessageData_ofFormat(v___x_1931_);
lean_inc(v_ref_1929_);
v___x_1933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1933_, 0, v_ref_1929_);
lean_ctor_set(v___x_1933_, 1, v___x_1932_);
if (v_isShared_1928_ == 0)
{
lean_ctor_set(v___x_1927_, 0, v___x_1933_);
v___x_1935_ = v___x_1927_;
goto v_reusejp_1934_;
}
else
{
lean_object* v_reuseFailAlloc_1936_; 
v_reuseFailAlloc_1936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1936_, 0, v___x_1933_);
v___x_1935_ = v_reuseFailAlloc_1936_;
goto v_reusejp_1934_;
}
v_reusejp_1934_:
{
return v___x_1935_;
}
}
}
}
else
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
lean_dec(v_id_1916_);
v___x_1938_ = lean_box(0);
v___x_1939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1939_, 0, v___x_1938_);
return v___x_1939_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8___boxed(lean_object* v_ex_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_){
_start:
{
lean_object* v_res_1946_; 
v_res_1946_ = lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8(v_ex_1942_, v___y_1943_, v___y_1944_);
lean_dec(v___y_1944_);
lean_dec_ref(v___y_1943_);
return v_res_1946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10(lean_object* v_as_1947_, size_t v_sz_1948_, size_t v_i_1949_, lean_object* v_b_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_){
_start:
{
lean_object* v_snd_1955_; lean_object* v_a_1960_; uint8_t v___x_1972_; 
v___x_1972_ = lean_usize_dec_lt(v_i_1949_, v_sz_1948_);
if (v___x_1972_ == 0)
{
lean_object* v___x_1973_; 
v___x_1973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1973_, 0, v_b_1950_);
return v___x_1973_;
}
else
{
lean_object* v___x_1974_; 
v___x_1974_ = l_Lean_Elab_Command_getRef___redArg(v___y_1951_);
if (lean_obj_tag(v___x_1974_) == 0)
{
lean_object* v_a_1975_; lean_object* v_fileName_1976_; lean_object* v_fileMap_1977_; lean_object* v_currRecDepth_1978_; lean_object* v_cmdPos_1979_; lean_object* v_macroStack_1980_; lean_object* v_quotContext_x3f_1981_; lean_object* v_currMacroScope_1982_; lean_object* v_snap_x3f_1983_; lean_object* v_cancelTk_x3f_1984_; uint8_t v_suppressElabErrors_1985_; lean_object* v_a_1986_; lean_object* v_ref_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; 
v_a_1975_ = lean_ctor_get(v___x_1974_, 0);
lean_inc(v_a_1975_);
lean_dec_ref_known(v___x_1974_, 1);
v_fileName_1976_ = lean_ctor_get(v___y_1951_, 0);
v_fileMap_1977_ = lean_ctor_get(v___y_1951_, 1);
v_currRecDepth_1978_ = lean_ctor_get(v___y_1951_, 2);
v_cmdPos_1979_ = lean_ctor_get(v___y_1951_, 3);
v_macroStack_1980_ = lean_ctor_get(v___y_1951_, 4);
v_quotContext_x3f_1981_ = lean_ctor_get(v___y_1951_, 5);
v_currMacroScope_1982_ = lean_ctor_get(v___y_1951_, 6);
v_snap_x3f_1983_ = lean_ctor_get(v___y_1951_, 8);
v_cancelTk_x3f_1984_ = lean_ctor_get(v___y_1951_, 9);
v_suppressElabErrors_1985_ = lean_ctor_get_uint8(v___y_1951_, sizeof(void*)*10);
v_a_1986_ = lean_array_uget_borrowed(v_as_1947_, v_i_1949_);
v_ref_1987_ = l_Lean_replaceRef(v_a_1986_, v_a_1975_);
lean_dec(v_a_1975_);
lean_inc(v_cancelTk_x3f_1984_);
lean_inc(v_snap_x3f_1983_);
lean_inc(v_currMacroScope_1982_);
lean_inc(v_quotContext_x3f_1981_);
lean_inc(v_macroStack_1980_);
lean_inc(v_cmdPos_1979_);
lean_inc(v_currRecDepth_1978_);
lean_inc_ref(v_fileMap_1977_);
lean_inc_ref(v_fileName_1976_);
v___x_1988_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_1988_, 0, v_fileName_1976_);
lean_ctor_set(v___x_1988_, 1, v_fileMap_1977_);
lean_ctor_set(v___x_1988_, 2, v_currRecDepth_1978_);
lean_ctor_set(v___x_1988_, 3, v_cmdPos_1979_);
lean_ctor_set(v___x_1988_, 4, v_macroStack_1980_);
lean_ctor_set(v___x_1988_, 5, v_quotContext_x3f_1981_);
lean_ctor_set(v___x_1988_, 6, v_currMacroScope_1982_);
lean_ctor_set(v___x_1988_, 7, v_ref_1987_);
lean_ctor_set(v___x_1988_, 8, v_snap_x3f_1983_);
lean_ctor_set(v___x_1988_, 9, v_cancelTk_x3f_1984_);
lean_ctor_set_uint8(v___x_1988_, sizeof(void*)*10, v_suppressElabErrors_1985_);
lean_inc(v_a_1986_);
v___x_1989_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9(v_a_1986_, v___x_1988_, v___y_1952_);
lean_dec_ref_known(v___x_1988_, 10);
if (lean_obj_tag(v___x_1989_) == 0)
{
lean_object* v_a_1990_; lean_object* v___x_1991_; 
v_a_1990_ = lean_ctor_get(v___x_1989_, 0);
lean_inc(v_a_1990_);
lean_dec_ref_known(v___x_1989_, 1);
v___x_1991_ = lean_array_push(v_b_1950_, v_a_1990_);
v_snd_1955_ = v___x_1991_;
goto v___jp_1954_;
}
else
{
lean_object* v_a_1992_; 
v_a_1992_ = lean_ctor_get(v___x_1989_, 0);
lean_inc(v_a_1992_);
lean_dec_ref_known(v___x_1989_, 1);
v_a_1960_ = v_a_1992_;
goto v___jp_1959_;
}
}
else
{
lean_object* v_a_1993_; 
v_a_1993_ = lean_ctor_get(v___x_1974_, 0);
lean_inc(v_a_1993_);
lean_dec_ref_known(v___x_1974_, 1);
v_a_1960_ = v_a_1993_;
goto v___jp_1959_;
}
}
v___jp_1954_:
{
size_t v___x_1956_; size_t v___x_1957_; 
v___x_1956_ = ((size_t)1ULL);
v___x_1957_ = lean_usize_add(v_i_1949_, v___x_1956_);
v_i_1949_ = v___x_1957_;
v_b_1950_ = v_snd_1955_;
goto _start;
}
v___jp_1959_:
{
uint8_t v___x_1961_; 
v___x_1961_ = l_Lean_Exception_isInterrupt(v_a_1960_);
if (v___x_1961_ == 0)
{
lean_object* v___x_1962_; 
v___x_1962_ = lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__8(v_a_1960_, v___y_1951_, v___y_1952_);
if (lean_obj_tag(v___x_1962_) == 0)
{
lean_dec_ref_known(v___x_1962_, 1);
v_snd_1955_ = v_b_1950_;
goto v___jp_1954_;
}
else
{
lean_object* v_a_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1970_; 
lean_dec_ref(v_b_1950_);
v_a_1963_ = lean_ctor_get(v___x_1962_, 0);
v_isSharedCheck_1970_ = !lean_is_exclusive(v___x_1962_);
if (v_isSharedCheck_1970_ == 0)
{
v___x_1965_ = v___x_1962_;
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_a_1963_);
lean_dec(v___x_1962_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1968_; 
if (v_isShared_1966_ == 0)
{
v___x_1968_ = v___x_1965_;
goto v_reusejp_1967_;
}
else
{
lean_object* v_reuseFailAlloc_1969_; 
v_reuseFailAlloc_1969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1969_, 0, v_a_1963_);
v___x_1968_ = v_reuseFailAlloc_1969_;
goto v_reusejp_1967_;
}
v_reusejp_1967_:
{
return v___x_1968_;
}
}
}
}
else
{
lean_object* v___x_1971_; 
lean_dec_ref(v_b_1950_);
v___x_1971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1971_, 0, v_a_1960_);
return v___x_1971_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10___boxed(lean_object* v_as_1994_, lean_object* v_sz_1995_, lean_object* v_i_1996_, lean_object* v_b_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_){
_start:
{
size_t v_sz_boxed_2001_; size_t v_i_boxed_2002_; lean_object* v_res_2003_; 
v_sz_boxed_2001_ = lean_unbox_usize(v_sz_1995_);
lean_dec(v_sz_1995_);
v_i_boxed_2002_ = lean_unbox_usize(v_i_1996_);
lean_dec(v_i_1996_);
v_res_2003_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10(v_as_1994_, v_sz_boxed_2001_, v_i_boxed_2002_, v_b_1997_, v___y_1998_, v___y_1999_);
lean_dec(v___y_1999_);
lean_dec_ref(v___y_1998_);
lean_dec_ref(v_as_1994_);
return v_res_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4(lean_object* v_attrInstances_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_){
_start:
{
lean_object* v_attrs_2008_; size_t v_sz_2009_; size_t v___x_2010_; lean_object* v___x_2011_; 
v_attrs_2008_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___closed__2));
v_sz_2009_ = lean_array_size(v_attrInstances_2004_);
v___x_2010_ = ((size_t)0ULL);
v___x_2011_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__10(v_attrInstances_2004_, v_sz_2009_, v___x_2010_, v_attrs_2008_, v___y_2005_, v___y_2006_);
return v___x_2011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4___boxed(lean_object* v_attrInstances_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_){
_start:
{
lean_object* v_res_2016_; 
v_res_2016_ = lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4(v_attrInstances_2012_, v___y_2013_, v___y_2014_);
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec_ref(v_attrInstances_2012_);
return v_res_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1(lean_object* v_stx_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_){
_start:
{
lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; 
v___x_2021_ = lean_unsigned_to_nat(1u);
v___x_2022_ = l_Lean_Syntax_getArg(v_stx_2017_, v___x_2021_);
v___x_2023_ = l_Lean_Syntax_getSepArgs(v___x_2022_);
lean_dec(v___x_2022_);
v___x_2024_ = lp_mathlib_Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4(v___x_2023_, v___y_2018_, v___y_2019_);
lean_dec_ref(v___x_2023_);
return v___x_2024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1___boxed(lean_object* v_stx_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_){
_start:
{
lean_object* v_res_2029_; 
v_res_2029_ = lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1(v_stx_2025_, v___y_2026_, v___y_2027_);
lean_dec(v___y_2027_);
lean_dec_ref(v___y_2026_);
lean_dec(v_stx_2025_);
return v_res_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg(lean_object* v_o_2030_, lean_object* v___y_2031_){
_start:
{
lean_object* v___x_2033_; lean_object* v_env_2034_; lean_object* v___x_2035_; lean_object* v_toEnvExtension_2036_; lean_object* v_asyncMode_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v_merged_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2049_; 
v___x_2033_ = lean_st_ref_get(v___y_2031_);
v_env_2034_ = lean_ctor_get(v___x_2033_, 0);
lean_inc_ref(v_env_2034_);
lean_dec(v___x_2033_);
v___x_2035_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_2036_ = lean_ctor_get(v___x_2035_, 0);
v_asyncMode_2037_ = lean_ctor_get(v_toEnvExtension_2036_, 2);
v___x_2038_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_2039_ = lean_box(0);
v___x_2040_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_2038_, v___x_2035_, v_env_2034_, v_asyncMode_2037_, v___x_2039_);
v_merged_2041_ = lean_ctor_get(v___x_2040_, 0);
v_isSharedCheck_2049_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2049_ == 0)
{
lean_object* v_unused_2050_; 
v_unused_2050_ = lean_ctor_get(v___x_2040_, 1);
lean_dec(v_unused_2050_);
v___x_2043_ = v___x_2040_;
v_isShared_2044_ = v_isSharedCheck_2049_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_merged_2041_);
lean_dec(v___x_2040_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2049_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v___x_2046_; 
if (v_isShared_2044_ == 0)
{
lean_ctor_set(v___x_2043_, 1, v_merged_2041_);
lean_ctor_set(v___x_2043_, 0, v_o_2030_);
v___x_2046_ = v___x_2043_;
goto v_reusejp_2045_;
}
else
{
lean_object* v_reuseFailAlloc_2048_; 
v_reuseFailAlloc_2048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2048_, 0, v_o_2030_);
lean_ctor_set(v_reuseFailAlloc_2048_, 1, v_merged_2041_);
v___x_2046_ = v_reuseFailAlloc_2048_;
goto v_reusejp_2045_;
}
v_reusejp_2045_:
{
lean_object* v___x_2047_; 
v___x_2047_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2047_, 0, v___x_2046_);
return v___x_2047_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg___boxed(lean_object* v_o_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_){
_start:
{
lean_object* v_res_2054_; 
v_res_2054_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg(v_o_2051_, v___y_2052_);
lean_dec(v___y_2052_);
return v_res_2054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4(lean_object* v___y_2055_, lean_object* v___y_2056_){
_start:
{
lean_object* v___x_2058_; lean_object* v_scopes_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v_opts_2062_; lean_object* v___x_2063_; 
v___x_2058_ = lean_st_ref_get(v___y_2056_);
v_scopes_2059_ = lean_ctor_get(v___x_2058_, 2);
lean_inc(v_scopes_2059_);
lean_dec(v___x_2058_);
v___x_2060_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2061_ = l_List_head_x21___redArg(v___x_2060_, v_scopes_2059_);
lean_dec(v_scopes_2059_);
v_opts_2062_ = lean_ctor_get(v___x_2061_, 1);
lean_inc_ref(v_opts_2062_);
lean_dec(v___x_2061_);
v___x_2063_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg(v_opts_2062_, v___y_2056_);
return v___x_2063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
lean_object* v_res_2067_; 
v_res_2067_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4(v___y_2064_, v___y_2065_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10(lean_object* v_ref_2068_, lean_object* v_msgData_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_){
_start:
{
uint8_t v___x_2073_; uint8_t v___x_2074_; lean_object* v___x_2075_; 
v___x_2073_ = 1;
v___x_2074_ = 0;
v___x_2075_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(v_ref_2068_, v_msgData_2069_, v___x_2073_, v___x_2074_, v___y_2070_, v___y_2071_);
return v___x_2075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10___boxed(lean_object* v_ref_2076_, lean_object* v_msgData_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_){
_start:
{
lean_object* v_res_2081_; 
v_res_2081_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10(v_ref_2076_, v_msgData_2077_, v___y_2078_, v___y_2079_);
lean_dec(v___y_2079_);
lean_dec_ref(v___y_2078_);
lean_dec(v_ref_2076_);
return v_res_2081_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1(void){
_start:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; 
v___x_2083_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__0));
v___x_2084_ = l_Lean_stringToMessageData(v___x_2083_);
return v___x_2084_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_2086_; lean_object* v___x_2087_; 
v___x_2086_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__2));
v___x_2087_ = l_Lean_stringToMessageData(v___x_2086_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5(lean_object* v_linterOption_2088_, lean_object* v_stx_2089_, lean_object* v_msg_2090_, lean_object* v___y_2091_, lean_object* v___y_2092_){
_start:
{
lean_object* v_name_2094_; lean_object* v___x_2096_; uint8_t v_isShared_2097_; uint8_t v_isSharedCheck_2112_; 
v_name_2094_ = lean_ctor_get(v_linterOption_2088_, 0);
v_isSharedCheck_2112_ = !lean_is_exclusive(v_linterOption_2088_);
if (v_isSharedCheck_2112_ == 0)
{
lean_object* v_unused_2113_; 
v_unused_2113_ = lean_ctor_get(v_linterOption_2088_, 1);
lean_dec(v_unused_2113_);
v___x_2096_ = v_linterOption_2088_;
v_isShared_2097_ = v_isSharedCheck_2112_;
goto v_resetjp_2095_;
}
else
{
lean_inc(v_name_2094_);
lean_dec(v_linterOption_2088_);
v___x_2096_ = lean_box(0);
v_isShared_2097_ = v_isSharedCheck_2112_;
goto v_resetjp_2095_;
}
v_resetjp_2095_:
{
lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2101_; 
v___x_2098_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__1);
lean_inc(v_name_2094_);
v___x_2099_ = l_Lean_MessageData_ofName(v_name_2094_);
if (v_isShared_2097_ == 0)
{
lean_ctor_set_tag(v___x_2096_, 7);
lean_ctor_set(v___x_2096_, 1, v___x_2099_);
lean_ctor_set(v___x_2096_, 0, v___x_2098_);
v___x_2101_ = v___x_2096_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2111_; 
v_reuseFailAlloc_2111_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2111_, 0, v___x_2098_);
lean_ctor_set(v_reuseFailAlloc_2111_, 1, v___x_2099_);
v___x_2101_ = v_reuseFailAlloc_2111_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v_disable_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; 
v___x_2102_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___closed__3);
v___x_2103_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2103_, 0, v___x_2101_);
lean_ctor_set(v___x_2103_, 1, v___x_2102_);
v_disable_2104_ = l_Lean_MessageData_note(v___x_2103_);
v___x_2105_ = l_Lean_Linter_linterMessageTag;
v___x_2106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2106_, 0, v_msg_2090_);
lean_ctor_set(v___x_2106_, 1, v_disable_2104_);
v___x_2107_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2107_, 0, v___x_2105_);
lean_ctor_set(v___x_2107_, 1, v___x_2106_);
v___x_2108_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2108_, 0, v_name_2094_);
lean_ctor_set(v___x_2108_, 1, v___x_2107_);
lean_inc(v_stx_2089_);
v___x_2109_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_2109_, 0, v_stx_2089_);
lean_ctor_set(v___x_2109_, 1, v___x_2108_);
v___x_2110_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10(v_stx_2089_, v___x_2109_, v___y_2091_, v___y_2092_);
lean_dec(v_stx_2089_);
return v___x_2110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5___boxed(lean_object* v_linterOption_2114_, lean_object* v_stx_2115_, lean_object* v_msg_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_){
_start:
{
lean_object* v_res_2120_; 
v_res_2120_ = lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5(v_linterOption_2114_, v_stx_2115_, v_msg_2116_, v___y_2117_, v___y_2118_);
lean_dec(v___y_2118_);
lean_dec_ref(v___y_2117_);
return v_res_2120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2(lean_object* v_linterOption_2121_, lean_object* v_stx_2122_, lean_object* v_msg_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_){
_start:
{
lean_object* v___x_2127_; lean_object* v_a_2128_; lean_object* v___x_2130_; uint8_t v_isShared_2131_; uint8_t v_isSharedCheck_2138_; 
v___x_2127_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4(v___y_2124_, v___y_2125_);
v_a_2128_ = lean_ctor_get(v___x_2127_, 0);
v_isSharedCheck_2138_ = !lean_is_exclusive(v___x_2127_);
if (v_isSharedCheck_2138_ == 0)
{
v___x_2130_ = v___x_2127_;
v_isShared_2131_ = v_isSharedCheck_2138_;
goto v_resetjp_2129_;
}
else
{
lean_inc(v_a_2128_);
lean_dec(v___x_2127_);
v___x_2130_ = lean_box(0);
v_isShared_2131_ = v_isSharedCheck_2138_;
goto v_resetjp_2129_;
}
v_resetjp_2129_:
{
uint8_t v___x_2132_; 
v___x_2132_ = l_Lean_Linter_getLinterValue(v_linterOption_2121_, v_a_2128_);
lean_dec(v_a_2128_);
if (v___x_2132_ == 0)
{
lean_object* v___x_2133_; lean_object* v___x_2135_; 
lean_dec_ref(v_msg_2123_);
lean_dec(v_stx_2122_);
lean_dec_ref(v_linterOption_2121_);
v___x_2133_ = lean_box(0);
if (v_isShared_2131_ == 0)
{
lean_ctor_set(v___x_2130_, 0, v___x_2133_);
v___x_2135_ = v___x_2130_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2136_; 
v_reuseFailAlloc_2136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2136_, 0, v___x_2133_);
v___x_2135_ = v_reuseFailAlloc_2136_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
return v___x_2135_;
}
}
else
{
lean_object* v___x_2137_; 
lean_del_object(v___x_2130_);
v___x_2137_ = lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5(v_linterOption_2121_, v_stx_2122_, v_msg_2123_, v___y_2124_, v___y_2125_);
return v___x_2137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2___boxed(lean_object* v_linterOption_2139_, lean_object* v_stx_2140_, lean_object* v_msg_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_){
_start:
{
lean_object* v_res_2145_; 
v_res_2145_ = lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2(v_linterOption_2139_, v_stx_2140_, v_msg_2141_, v___y_2142_, v___y_2143_);
lean_dec(v___y_2143_);
lean_dec_ref(v___y_2142_);
return v_res_2145_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2147_; lean_object* v___x_2148_; 
v___x_2147_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__0));
v___x_2148_ = l_Lean_stringToMessageData(v___x_2147_);
return v___x_2148_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_2150_; lean_object* v___x_2151_; 
v___x_2150_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__2));
v___x_2151_ = l_Lean_stringToMessageData(v___x_2150_);
return v___x_2151_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8(void){
_start:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; 
v___x_2164_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__7));
v___x_2165_ = l_Lean_stringToMessageData(v___x_2164_);
return v___x_2165_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10(void){
_start:
{
lean_object* v___x_2167_; lean_object* v___x_2168_; 
v___x_2167_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__9));
v___x_2168_ = l_Lean_stringToMessageData(v___x_2167_);
return v___x_2168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0(lean_object* v_vis_x3f_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_){
_start:
{
lean_object* v___x_2181_; 
v___x_2181_ = lean_st_ref_get(v___y_2171_);
if (lean_obj_tag(v_vis_x3f_2169_) == 0)
{
uint8_t v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; 
lean_dec(v___x_2181_);
v___x_2182_ = 0;
v___x_2183_ = lean_box(v___x_2182_);
v___x_2184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2184_, 0, v___x_2183_);
return v___x_2184_;
}
else
{
lean_object* v_env_2185_; lean_object* v_val_2186_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___y_2190_; lean_object* v___x_2215_; uint8_t v___x_2216_; 
v_env_2185_ = lean_ctor_get(v___x_2181_, 0);
lean_inc_ref(v_env_2185_);
lean_dec(v___x_2181_);
v_val_2186_ = lean_ctor_get(v_vis_x3f_2169_, 0);
lean_inc_n(v_val_2186_, 2);
lean_dec_ref_known(v_vis_x3f_2169_, 1);
v___x_2215_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__5));
v___x_2216_ = l_Lean_Syntax_isOfKind(v_val_2186_, v___x_2215_);
if (v___x_2216_ == 0)
{
lean_object* v___x_2217_; uint8_t v___x_2218_; 
v___x_2217_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__6));
lean_inc(v_val_2186_);
v___x_2218_ = l_Lean_Syntax_isOfKind(v_val_2186_, v___x_2217_);
if (v___x_2218_ == 0)
{
lean_object* v___x_2219_; lean_object* v___x_2220_; 
lean_dec_ref(v_env_2185_);
v___x_2219_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8, &lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__8);
v___x_2220_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(v_val_2186_, v___x_2219_, v___y_2170_, v___y_2171_);
lean_dec(v_val_2186_);
return v___x_2220_;
}
else
{
lean_object* v___x_2221_; 
v___x_2221_ = l_Lean_Syntax_getHeadInfo(v_val_2186_);
if (lean_obj_tag(v___x_2221_) == 0)
{
lean_dec_ref_known(v___x_2221_, 4);
goto v___jp_2211_;
}
else
{
lean_dec(v___x_2221_);
if (v___x_2216_ == 0)
{
lean_dec(v_val_2186_);
lean_dec_ref(v_env_2185_);
goto v___jp_2177_;
}
else
{
goto v___jp_2211_;
}
}
}
}
else
{
lean_object* v___x_2222_; 
v___x_2222_ = l_Lean_Syntax_getHeadInfo(v_val_2186_);
if (lean_obj_tag(v___x_2222_) == 0)
{
lean_object* v___x_2223_; uint8_t v_isModule_2224_; 
lean_dec_ref_known(v___x_2222_, 4);
v___x_2223_ = l_Lean_Environment_header(v_env_2185_);
v_isModule_2224_ = lean_ctor_get_uint8(v___x_2223_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_2223_);
if (v_isModule_2224_ == 0)
{
lean_dec(v_val_2186_);
lean_dec_ref(v_env_2185_);
goto v___jp_2173_;
}
else
{
uint8_t v_isExporting_2225_; 
v_isExporting_2225_ = lean_ctor_get_uint8(v_env_2185_, sizeof(void*)*8);
lean_dec_ref(v_env_2185_);
if (v_isExporting_2225_ == 0)
{
lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; 
v___x_2226_ = l_Lean_linter_redundantVisibility;
v___x_2227_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10, &lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10_once, _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__10);
v___x_2228_ = lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2(v___x_2226_, v_val_2186_, v___x_2227_, v___y_2170_, v___y_2171_);
if (lean_obj_tag(v___x_2228_) == 0)
{
lean_dec_ref_known(v___x_2228_, 1);
goto v___jp_2173_;
}
else
{
lean_object* v_a_2229_; lean_object* v___x_2231_; uint8_t v_isShared_2232_; uint8_t v_isSharedCheck_2236_; 
v_a_2229_ = lean_ctor_get(v___x_2228_, 0);
v_isSharedCheck_2236_ = !lean_is_exclusive(v___x_2228_);
if (v_isSharedCheck_2236_ == 0)
{
v___x_2231_ = v___x_2228_;
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
else
{
lean_inc(v_a_2229_);
lean_dec(v___x_2228_);
v___x_2231_ = lean_box(0);
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
v_resetjp_2230_:
{
lean_object* v___x_2234_; 
if (v_isShared_2232_ == 0)
{
v___x_2234_ = v___x_2231_;
goto v_reusejp_2233_;
}
else
{
lean_object* v_reuseFailAlloc_2235_; 
v_reuseFailAlloc_2235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2235_, 0, v_a_2229_);
v___x_2234_ = v_reuseFailAlloc_2235_;
goto v_reusejp_2233_;
}
v_reusejp_2233_:
{
return v___x_2234_;
}
}
}
}
else
{
lean_dec(v_val_2186_);
goto v___jp_2173_;
}
}
}
else
{
lean_dec(v___x_2222_);
lean_dec(v_val_2186_);
lean_dec_ref(v_env_2185_);
goto v___jp_2173_;
}
}
v___jp_2187_:
{
lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; 
lean_inc_ref(v___y_2190_);
v___x_2191_ = l_Lean_stringToMessageData(v___y_2190_);
lean_inc_ref(v___y_2189_);
v___x_2192_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2192_, 0, v___y_2189_);
lean_ctor_set(v___x_2192_, 1, v___x_2191_);
v___x_2193_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1, &lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__1);
v___x_2194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2192_);
lean_ctor_set(v___x_2194_, 1, v___x_2193_);
lean_inc_ref(v___y_2188_);
v___x_2195_ = lp_mathlib_Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2(v___y_2188_, v_val_2186_, v___x_2194_, v___y_2170_, v___y_2171_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_dec_ref_known(v___x_2195_, 1);
goto v___jp_2177_;
}
else
{
lean_object* v_a_2196_; lean_object* v___x_2198_; uint8_t v_isShared_2199_; uint8_t v_isSharedCheck_2203_; 
v_a_2196_ = lean_ctor_get(v___x_2195_, 0);
v_isSharedCheck_2203_ = !lean_is_exclusive(v___x_2195_);
if (v_isSharedCheck_2203_ == 0)
{
v___x_2198_ = v___x_2195_;
v_isShared_2199_ = v_isSharedCheck_2203_;
goto v_resetjp_2197_;
}
else
{
lean_inc(v_a_2196_);
lean_dec(v___x_2195_);
v___x_2198_ = lean_box(0);
v_isShared_2199_ = v_isSharedCheck_2203_;
goto v_resetjp_2197_;
}
v_resetjp_2197_:
{
lean_object* v___x_2201_; 
if (v_isShared_2199_ == 0)
{
v___x_2201_ = v___x_2198_;
goto v_reusejp_2200_;
}
else
{
lean_object* v_reuseFailAlloc_2202_; 
v_reuseFailAlloc_2202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2202_, 0, v_a_2196_);
v___x_2201_ = v_reuseFailAlloc_2202_;
goto v_reusejp_2200_;
}
v_reusejp_2200_:
{
return v___x_2201_;
}
}
}
}
v___jp_2204_:
{
lean_object* v___x_2205_; uint8_t v_isModule_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; 
v___x_2205_ = l_Lean_Environment_header(v_env_2185_);
lean_dec_ref(v_env_2185_);
v_isModule_2206_ = lean_ctor_get_uint8(v___x_2205_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_2205_);
v___x_2207_ = l_Lean_linter_redundantVisibility;
v___x_2208_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3, &lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__3);
if (v_isModule_2206_ == 0)
{
lean_object* v___x_2209_; 
v___x_2209_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_previousInstName___closed__1));
v___y_2188_ = v___x_2207_;
v___y_2189_ = v___x_2208_;
v___y_2190_ = v___x_2209_;
goto v___jp_2187_;
}
else
{
lean_object* v___x_2210_; 
v___x_2210_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___closed__4));
v___y_2188_ = v___x_2207_;
v___y_2189_ = v___x_2208_;
v___y_2190_ = v___x_2210_;
goto v___jp_2187_;
}
}
v___jp_2211_:
{
uint8_t v_isExporting_2212_; 
v_isExporting_2212_ = lean_ctor_get_uint8(v_env_2185_, sizeof(void*)*8);
if (v_isExporting_2212_ == 0)
{
lean_object* v___x_2213_; uint8_t v_isModule_2214_; 
v___x_2213_ = l_Lean_Environment_header(v_env_2185_);
v_isModule_2214_ = lean_ctor_get_uint8(v___x_2213_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_2213_);
if (v_isModule_2214_ == 0)
{
goto v___jp_2204_;
}
else
{
lean_dec(v_val_2186_);
lean_dec_ref(v_env_2185_);
goto v___jp_2177_;
}
}
else
{
goto v___jp_2204_;
}
}
}
v___jp_2173_:
{
uint8_t v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; 
v___x_2174_ = 1;
v___x_2175_ = lean_box(v___x_2174_);
v___x_2176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2176_, 0, v___x_2175_);
return v___x_2176_;
}
v___jp_2177_:
{
uint8_t v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; 
v___x_2178_ = 2;
v___x_2179_ = lean_box(v___x_2178_);
v___x_2180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2180_, 0, v___x_2179_);
return v___x_2180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0___boxed(lean_object* v_vis_x3f_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_){
_start:
{
lean_object* v_res_2241_; 
v_res_2241_ = lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0(v_vis_x3f_2237_, v___y_2238_, v___y_2239_);
lean_dec(v___y_2239_);
lean_dec_ref(v___y_2238_);
return v_res_2241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0(lean_object* v_stx_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_){
_start:
{
lean_object* v___y_2258_; uint8_t v___y_2259_; uint8_t v___y_2260_; uint8_t v___y_2261_; lean_object* v___y_2262_; uint8_t v___y_2263_; uint8_t v___y_2264_; lean_object* v___y_2268_; lean_object* v___y_2269_; uint8_t v___y_2270_; uint8_t v___y_2271_; uint8_t v___y_2272_; uint8_t v___y_2273_; lean_object* v_attrs_2274_; lean_object* v___x_2278_; lean_object* v_docCommentStx_2279_; lean_object* v___x_2280_; lean_object* v_attrsStx_2281_; lean_object* v___y_2283_; uint8_t v___y_2284_; lean_object* v___y_2285_; uint8_t v___y_2286_; uint8_t v___y_2287_; uint8_t v___y_2288_; lean_object* v___x_2302_; lean_object* v_visibilityStx_2303_; lean_object* v___x_2304_; lean_object* v_protectedStx_2305_; lean_object* v___y_2307_; lean_object* v___y_2308_; uint8_t v___y_2309_; uint8_t v___y_2310_; lean_object* v___y_2311_; uint8_t v___y_2328_; lean_object* v___y_2329_; uint8_t v___y_2330_; lean_object* v___y_2331_; lean_object* v___y_2343_; uint8_t v___y_2344_; uint8_t v___y_2345_; uint8_t v___y_2357_; lean_object* v___x_2370_; lean_object* v___x_2371_; uint8_t v___x_2372_; 
v___x_2278_ = lean_unsigned_to_nat(0u);
v_docCommentStx_2279_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2278_);
v___x_2280_ = lean_unsigned_to_nat(1u);
v_attrsStx_2281_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2280_);
v___x_2302_ = lean_unsigned_to_nat(2u);
v_visibilityStx_2303_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2302_);
v___x_2304_ = lean_unsigned_to_nat(3u);
v_protectedStx_2305_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2304_);
v___x_2370_ = lean_unsigned_to_nat(4u);
v___x_2371_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2370_);
v___x_2372_ = l_Lean_Syntax_isNone(v___x_2371_);
if (v___x_2372_ == 0)
{
lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; uint8_t v___x_2376_; 
v___x_2373_ = l_Lean_Syntax_getArg(v___x_2371_, v___x_2278_);
lean_dec(v___x_2371_);
v___x_2374_ = l_Lean_Syntax_getKind(v___x_2373_);
v___x_2375_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__2));
v___x_2376_ = lean_name_eq(v___x_2374_, v___x_2375_);
lean_dec(v___x_2374_);
if (v___x_2376_ == 0)
{
uint8_t v___x_2377_; 
v___x_2377_ = 2;
v___y_2357_ = v___x_2377_;
goto v___jp_2356_;
}
else
{
uint8_t v___x_2378_; 
v___x_2378_ = 1;
v___y_2357_ = v___x_2378_;
goto v___jp_2356_;
}
}
else
{
uint8_t v___x_2379_; 
lean_dec(v___x_2371_);
v___x_2379_ = 0;
v___y_2357_ = v___x_2379_;
goto v___jp_2356_;
}
v___jp_2257_:
{
lean_object* v___x_2265_; lean_object* v___x_2266_; 
v___x_2265_ = lean_alloc_ctor(0, 3, 5);
lean_ctor_set(v___x_2265_, 0, v_stx_2253_);
lean_ctor_set(v___x_2265_, 1, v___y_2258_);
lean_ctor_set(v___x_2265_, 2, v___y_2262_);
lean_ctor_set_uint8(v___x_2265_, sizeof(void*)*3, v___y_2263_);
lean_ctor_set_uint8(v___x_2265_, sizeof(void*)*3 + 1, v___y_2261_);
lean_ctor_set_uint8(v___x_2265_, sizeof(void*)*3 + 2, v___y_2259_);
lean_ctor_set_uint8(v___x_2265_, sizeof(void*)*3 + 3, v___y_2260_);
lean_ctor_set_uint8(v___x_2265_, sizeof(void*)*3 + 4, v___y_2264_);
v___x_2266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2266_, 0, v___x_2265_);
return v___x_2266_;
}
v___jp_2267_:
{
uint8_t v___x_2275_; 
v___x_2275_ = l_Lean_Syntax_isNone(v___y_2269_);
lean_dec(v___y_2269_);
if (v___x_2275_ == 0)
{
uint8_t v___x_2276_; 
v___x_2276_ = 1;
v___y_2258_ = v___y_2268_;
v___y_2259_ = v___y_2270_;
v___y_2260_ = v___y_2272_;
v___y_2261_ = v___y_2271_;
v___y_2262_ = v_attrs_2274_;
v___y_2263_ = v___y_2273_;
v___y_2264_ = v___x_2276_;
goto v___jp_2257_;
}
else
{
uint8_t v___x_2277_; 
v___x_2277_ = 0;
v___y_2258_ = v___y_2268_;
v___y_2259_ = v___y_2270_;
v___y_2260_ = v___y_2272_;
v___y_2261_ = v___y_2271_;
v___y_2262_ = v_attrs_2274_;
v___y_2263_ = v___y_2273_;
v___y_2264_ = v___x_2277_;
goto v___jp_2257_;
}
}
v___jp_2282_:
{
lean_object* v___x_2289_; 
v___x_2289_ = l_Lean_Syntax_getOptional_x3f(v_attrsStx_2281_);
lean_dec(v_attrsStx_2281_);
if (lean_obj_tag(v___x_2289_) == 0)
{
lean_object* v___x_2290_; 
v___x_2290_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getId___closed__2));
v___y_2268_ = v___y_2283_;
v___y_2269_ = v___y_2285_;
v___y_2270_ = v___y_2284_;
v___y_2271_ = v___y_2288_;
v___y_2272_ = v___y_2286_;
v___y_2273_ = v___y_2287_;
v_attrs_2274_ = v___x_2290_;
goto v___jp_2267_;
}
else
{
lean_object* v_val_2291_; lean_object* v___x_2292_; 
v_val_2291_ = lean_ctor_get(v___x_2289_, 0);
lean_inc(v_val_2291_);
lean_dec_ref_known(v___x_2289_, 1);
v___x_2292_ = lp_mathlib_Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1(v_val_2291_, v___y_2254_, v___y_2255_);
lean_dec(v_val_2291_);
if (lean_obj_tag(v___x_2292_) == 0)
{
lean_object* v_a_2293_; 
v_a_2293_ = lean_ctor_get(v___x_2292_, 0);
lean_inc(v_a_2293_);
lean_dec_ref_known(v___x_2292_, 1);
v___y_2268_ = v___y_2283_;
v___y_2269_ = v___y_2285_;
v___y_2270_ = v___y_2284_;
v___y_2271_ = v___y_2288_;
v___y_2272_ = v___y_2286_;
v___y_2273_ = v___y_2287_;
v_attrs_2274_ = v_a_2293_;
goto v___jp_2267_;
}
else
{
lean_object* v_a_2294_; lean_object* v___x_2296_; uint8_t v_isShared_2297_; uint8_t v_isSharedCheck_2301_; 
lean_dec(v___y_2285_);
lean_dec(v___y_2283_);
lean_dec(v_stx_2253_);
v_a_2294_ = lean_ctor_get(v___x_2292_, 0);
v_isSharedCheck_2301_ = !lean_is_exclusive(v___x_2292_);
if (v_isSharedCheck_2301_ == 0)
{
v___x_2296_ = v___x_2292_;
v_isShared_2297_ = v_isSharedCheck_2301_;
goto v_resetjp_2295_;
}
else
{
lean_inc(v_a_2294_);
lean_dec(v___x_2292_);
v___x_2296_ = lean_box(0);
v_isShared_2297_ = v_isSharedCheck_2301_;
goto v_resetjp_2295_;
}
v_resetjp_2295_:
{
lean_object* v___x_2299_; 
if (v_isShared_2297_ == 0)
{
v___x_2299_ = v___x_2296_;
goto v_reusejp_2298_;
}
else
{
lean_object* v_reuseFailAlloc_2300_; 
v_reuseFailAlloc_2300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2300_, 0, v_a_2294_);
v___x_2299_ = v_reuseFailAlloc_2300_;
goto v_reusejp_2298_;
}
v_reusejp_2298_:
{
return v___x_2299_;
}
}
}
}
}
v___jp_2306_:
{
lean_object* v___x_2312_; 
v___x_2312_ = lp_mathlib_Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0(v___y_2311_, v___y_2254_, v___y_2255_);
if (lean_obj_tag(v___x_2312_) == 0)
{
lean_object* v_a_2313_; uint8_t v___x_2314_; 
v_a_2313_ = lean_ctor_get(v___x_2312_, 0);
lean_inc(v_a_2313_);
lean_dec_ref_known(v___x_2312_, 1);
v___x_2314_ = l_Lean_Syntax_isNone(v_protectedStx_2305_);
lean_dec(v_protectedStx_2305_);
if (v___x_2314_ == 0)
{
uint8_t v___x_2315_; uint8_t v___x_2316_; 
v___x_2315_ = 1;
v___x_2316_ = lean_unbox(v_a_2313_);
lean_dec(v_a_2313_);
v___y_2283_ = v___y_2307_;
v___y_2284_ = v___y_2309_;
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___y_2310_;
v___y_2287_ = v___x_2316_;
v___y_2288_ = v___x_2315_;
goto v___jp_2282_;
}
else
{
uint8_t v___x_2317_; uint8_t v___x_2318_; 
v___x_2317_ = 0;
v___x_2318_ = lean_unbox(v_a_2313_);
lean_dec(v_a_2313_);
v___y_2283_ = v___y_2307_;
v___y_2284_ = v___y_2309_;
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___y_2310_;
v___y_2287_ = v___x_2318_;
v___y_2288_ = v___x_2317_;
goto v___jp_2282_;
}
}
else
{
lean_object* v_a_2319_; lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2326_; 
lean_dec(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec(v_protectedStx_2305_);
lean_dec(v_attrsStx_2281_);
lean_dec(v_stx_2253_);
v_a_2319_ = lean_ctor_get(v___x_2312_, 0);
v_isSharedCheck_2326_ = !lean_is_exclusive(v___x_2312_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2321_ = v___x_2312_;
v_isShared_2322_ = v_isSharedCheck_2326_;
goto v_resetjp_2320_;
}
else
{
lean_inc(v_a_2319_);
lean_dec(v___x_2312_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2326_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
lean_object* v___x_2324_; 
if (v_isShared_2322_ == 0)
{
v___x_2324_ = v___x_2321_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v_a_2319_);
v___x_2324_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
return v___x_2324_;
}
}
}
}
v___jp_2327_:
{
lean_object* v___x_2332_; 
v___x_2332_ = l_Lean_Syntax_getOptional_x3f(v_visibilityStx_2303_);
lean_dec(v_visibilityStx_2303_);
if (lean_obj_tag(v___x_2332_) == 0)
{
lean_object* v___x_2333_; 
v___x_2333_ = lean_box(0);
v___y_2307_ = v___y_2331_;
v___y_2308_ = v___y_2329_;
v___y_2309_ = v___y_2328_;
v___y_2310_ = v___y_2330_;
v___y_2311_ = v___x_2333_;
goto v___jp_2306_;
}
else
{
lean_object* v_val_2334_; lean_object* v___x_2336_; uint8_t v_isShared_2337_; uint8_t v_isSharedCheck_2341_; 
v_val_2334_ = lean_ctor_get(v___x_2332_, 0);
v_isSharedCheck_2341_ = !lean_is_exclusive(v___x_2332_);
if (v_isSharedCheck_2341_ == 0)
{
v___x_2336_ = v___x_2332_;
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
else
{
lean_inc(v_val_2334_);
lean_dec(v___x_2332_);
v___x_2336_ = lean_box(0);
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
v_resetjp_2335_:
{
lean_object* v___x_2339_; 
if (v_isShared_2337_ == 0)
{
v___x_2339_ = v___x_2336_;
goto v_reusejp_2338_;
}
else
{
lean_object* v_reuseFailAlloc_2340_; 
v_reuseFailAlloc_2340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2340_, 0, v_val_2334_);
v___x_2339_ = v_reuseFailAlloc_2340_;
goto v_reusejp_2338_;
}
v_reusejp_2338_:
{
v___y_2307_ = v___y_2331_;
v___y_2308_ = v___y_2329_;
v___y_2309_ = v___y_2328_;
v___y_2310_ = v___y_2330_;
v___y_2311_ = v___x_2339_;
goto v___jp_2306_;
}
}
}
}
v___jp_2342_:
{
lean_object* v___x_2346_; 
v___x_2346_ = l_Lean_Syntax_getOptional_x3f(v_docCommentStx_2279_);
lean_dec(v_docCommentStx_2279_);
if (lean_obj_tag(v___x_2346_) == 0)
{
lean_object* v___x_2347_; 
v___x_2347_ = lean_box(0);
v___y_2328_ = v___y_2344_;
v___y_2329_ = v___y_2343_;
v___y_2330_ = v___y_2345_;
v___y_2331_ = v___x_2347_;
goto v___jp_2327_;
}
else
{
lean_object* v_val_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2355_; 
v_val_2348_ = lean_ctor_get(v___x_2346_, 0);
v_isSharedCheck_2355_ = !lean_is_exclusive(v___x_2346_);
if (v_isSharedCheck_2355_ == 0)
{
v___x_2350_ = v___x_2346_;
v_isShared_2351_ = v_isSharedCheck_2355_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_val_2348_);
lean_dec(v___x_2346_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2355_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v___x_2353_; 
if (v_isShared_2351_ == 0)
{
v___x_2353_ = v___x_2350_;
goto v_reusejp_2352_;
}
else
{
lean_object* v_reuseFailAlloc_2354_; 
v_reuseFailAlloc_2354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2354_, 0, v_val_2348_);
v___x_2353_ = v_reuseFailAlloc_2354_;
goto v_reusejp_2352_;
}
v_reusejp_2352_:
{
v___y_2328_ = v___y_2344_;
v___y_2329_ = v___y_2343_;
v___y_2330_ = v___y_2345_;
v___y_2331_ = v___x_2353_;
goto v___jp_2327_;
}
}
}
}
v___jp_2356_:
{
lean_object* v___x_2358_; lean_object* v_unsafeStx_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; uint8_t v___x_2362_; 
v___x_2358_ = lean_unsigned_to_nat(5u);
v_unsafeStx_2359_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2358_);
v___x_2360_ = lean_unsigned_to_nat(6u);
v___x_2361_ = l_Lean_Syntax_getArg(v_stx_2253_, v___x_2360_);
v___x_2362_ = l_Lean_Syntax_isNone(v___x_2361_);
if (v___x_2362_ == 0)
{
lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; uint8_t v___x_2366_; 
v___x_2363_ = l_Lean_Syntax_getArg(v___x_2361_, v___x_2278_);
lean_dec(v___x_2361_);
v___x_2364_ = l_Lean_Syntax_getKind(v___x_2363_);
v___x_2365_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___closed__1));
v___x_2366_ = lean_name_eq(v___x_2364_, v___x_2365_);
lean_dec(v___x_2364_);
if (v___x_2366_ == 0)
{
uint8_t v___x_2367_; 
v___x_2367_ = 1;
v___y_2343_ = v_unsafeStx_2359_;
v___y_2344_ = v___y_2357_;
v___y_2345_ = v___x_2367_;
goto v___jp_2342_;
}
else
{
uint8_t v___x_2368_; 
v___x_2368_ = 0;
v___y_2343_ = v_unsafeStx_2359_;
v___y_2344_ = v___y_2357_;
v___y_2345_ = v___x_2368_;
goto v___jp_2342_;
}
}
else
{
uint8_t v___x_2369_; 
lean_dec(v___x_2361_);
v___x_2369_ = 2;
v___y_2343_ = v_unsafeStx_2359_;
v___y_2344_ = v___y_2357_;
v___y_2345_ = v___x_2369_;
goto v___jp_2342_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0___boxed(lean_object* v_stx_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_){
_start:
{
lean_object* v_res_2384_; 
v_res_2384_ = lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0(v_stx_2380_, v___y_2381_, v___y_2382_);
lean_dec(v___y_2382_);
lean_dec_ref(v___y_2381_);
return v_res_2384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName(lean_object* v_cmd_2390_, lean_object* v_a_2391_, lean_object* v_a_2392_){
_start:
{
lean_object* v___x_2394_; 
v___x_2394_ = l_Lean_Elab_Command_getScope___redArg(v_a_2392_);
if (lean_obj_tag(v___x_2394_) == 0)
{
lean_object* v_a_2395_; lean_object* v___x_2396_; 
v_a_2395_ = lean_ctor_get(v___x_2394_, 0);
lean_inc(v_a_2395_);
lean_dec_ref_known(v___x_2394_, 1);
lean_inc(v_cmd_2390_);
v___x_2396_ = lp_mathlib_Mathlib_Command_MinImports_getId(v_cmd_2390_, v_a_2391_, v_a_2392_);
if (lean_obj_tag(v___x_2396_) == 0)
{
lean_object* v_a_2397_; lean_object* v___x_2399_; uint8_t v_isShared_2400_; uint8_t v_isSharedCheck_2453_; 
v_a_2397_ = lean_ctor_get(v___x_2396_, 0);
v_isSharedCheck_2453_ = !lean_is_exclusive(v___x_2396_);
if (v_isSharedCheck_2453_ == 0)
{
v___x_2399_ = v___x_2396_;
v_isShared_2400_ = v_isSharedCheck_2453_;
goto v_resetjp_2398_;
}
else
{
lean_inc(v_a_2397_);
lean_dec(v___x_2396_);
v___x_2399_ = lean_box(0);
v_isShared_2400_ = v_isSharedCheck_2453_;
goto v_resetjp_2398_;
}
v_resetjp_2398_:
{
lean_object* v___f_2401_; lean_object* v___x_2402_; 
v___f_2401_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__0));
v___x_2402_ = l_Lean_Syntax_find_x3f(v_cmd_2390_, v___f_2401_);
if (lean_obj_tag(v___x_2402_) == 1)
{
lean_object* v_val_2403_; lean_object* v___f_2404_; lean_object* v___x_2405_; 
v_val_2403_ = lean_ctor_get(v___x_2402_, 0);
lean_inc(v_val_2403_);
lean_dec_ref_known(v___x_2402_, 1);
v___f_2404_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__1));
v___x_2405_ = l_Lean_Syntax_find_x3f(v_val_2403_, v___f_2404_);
if (lean_obj_tag(v___x_2405_) == 1)
{
lean_object* v_val_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
lean_del_object(v___x_2399_);
v_val_2406_ = lean_ctor_get(v___x_2405_, 0);
lean_inc(v_val_2406_);
lean_dec_ref_known(v___x_2405_, 1);
v___x_2407_ = lean_st_ref_get(v_a_2392_);
v___x_2408_ = lp_mathlib_Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0(v_val_2406_, v_a_2391_, v_a_2392_);
if (lean_obj_tag(v___x_2408_) == 0)
{
lean_object* v_a_2409_; lean_object* v___x_2410_; lean_object* v___y_2412_; lean_object* v_currNamespace_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v_name_2420_; lean_object* v_imported_2421_; lean_object* v_ctx_2422_; lean_object* v_scopes_2423_; lean_object* v___x_2425_; uint8_t v_isShared_2426_; uint8_t v_isSharedCheck_2436_; 
v_a_2409_ = lean_ctor_get(v___x_2408_, 0);
lean_inc(v_a_2409_);
lean_dec_ref_known(v___x_2408_, 1);
v___x_2410_ = lean_st_ref_set(v_a_2392_, v___x_2407_);
v_currNamespace_2417_ = lean_ctor_get(v_a_2395_, 2);
lean_inc(v_currNamespace_2417_);
lean_dec(v_a_2395_);
v___x_2418_ = l_Lean_Syntax_getId(v_a_2397_);
lean_dec(v_a_2397_);
lean_inc(v___x_2418_);
v___x_2419_ = l_Lean_extractMacroScopes(v___x_2418_);
v_name_2420_ = lean_ctor_get(v___x_2419_, 0);
v_imported_2421_ = lean_ctor_get(v___x_2419_, 1);
v_ctx_2422_ = lean_ctor_get(v___x_2419_, 2);
v_scopes_2423_ = lean_ctor_get(v___x_2419_, 3);
v_isSharedCheck_2436_ = !lean_is_exclusive(v___x_2419_);
if (v_isSharedCheck_2436_ == 0)
{
v___x_2425_ = v___x_2419_;
v_isShared_2426_ = v_isSharedCheck_2436_;
goto v_resetjp_2424_;
}
else
{
lean_inc(v_scopes_2423_);
lean_inc(v_ctx_2422_);
lean_inc(v_imported_2421_);
lean_inc(v_name_2420_);
lean_dec(v___x_2419_);
v___x_2425_ = lean_box(0);
v_isShared_2426_ = v_isSharedCheck_2436_;
goto v_resetjp_2424_;
}
v___jp_2411_:
{
uint8_t v_visibility_2413_; lean_object* v___x_2414_; lean_object* v___f_2415_; lean_object* v___x_2416_; 
v_visibility_2413_ = lean_ctor_get_uint8(v_a_2409_, sizeof(void*)*3);
lean_dec(v_a_2409_);
v___x_2414_ = lean_box(v_visibility_2413_);
v___f_2415_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___lam__2___boxed), 5, 2);
lean_closure_set(v___f_2415_, 0, v___x_2414_);
lean_closure_set(v___f_2415_, 1, v___y_2412_);
v___x_2416_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_2415_, v_a_2391_, v_a_2392_);
return v___x_2416_;
}
v_resetjp_2424_:
{
lean_object* v___x_2427_; uint8_t v___x_2428_; 
v___x_2427_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getDeclName___closed__3));
v___x_2428_ = l_Lean_Name_isPrefixOf(v___x_2427_, v_name_2420_);
if (v___x_2428_ == 0)
{
lean_object* v___x_2429_; 
lean_del_object(v___x_2425_);
lean_dec(v_scopes_2423_);
lean_dec(v_ctx_2422_);
lean_dec(v_imported_2421_);
lean_dec(v_name_2420_);
v___x_2429_ = l_Lean_Name_append(v_currNamespace_2417_, v___x_2418_);
v___y_2412_ = v___x_2429_;
goto v___jp_2411_;
}
else
{
lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2433_; 
lean_dec(v___x_2418_);
lean_dec(v_currNamespace_2417_);
v___x_2430_ = lean_box(0);
v___x_2431_ = l_Lean_Name_replacePrefix(v_name_2420_, v___x_2427_, v___x_2430_);
if (v_isShared_2426_ == 0)
{
lean_ctor_set(v___x_2425_, 0, v___x_2431_);
v___x_2433_ = v___x_2425_;
goto v_reusejp_2432_;
}
else
{
lean_object* v_reuseFailAlloc_2435_; 
v_reuseFailAlloc_2435_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2435_, 0, v___x_2431_);
lean_ctor_set(v_reuseFailAlloc_2435_, 1, v_imported_2421_);
lean_ctor_set(v_reuseFailAlloc_2435_, 2, v_ctx_2422_);
lean_ctor_set(v_reuseFailAlloc_2435_, 3, v_scopes_2423_);
v___x_2433_ = v_reuseFailAlloc_2435_;
goto v_reusejp_2432_;
}
v_reusejp_2432_:
{
lean_object* v___x_2434_; 
v___x_2434_ = l_Lean_MacroScopesView_review(v___x_2433_);
v___y_2412_ = v___x_2434_;
goto v___jp_2411_;
}
}
}
}
else
{
lean_object* v_a_2437_; lean_object* v___x_2439_; uint8_t v_isShared_2440_; uint8_t v_isSharedCheck_2444_; 
lean_dec(v___x_2407_);
lean_dec(v_a_2397_);
lean_dec(v_a_2395_);
v_a_2437_ = lean_ctor_get(v___x_2408_, 0);
v_isSharedCheck_2444_ = !lean_is_exclusive(v___x_2408_);
if (v_isSharedCheck_2444_ == 0)
{
v___x_2439_ = v___x_2408_;
v_isShared_2440_ = v_isSharedCheck_2444_;
goto v_resetjp_2438_;
}
else
{
lean_inc(v_a_2437_);
lean_dec(v___x_2408_);
v___x_2439_ = lean_box(0);
v_isShared_2440_ = v_isSharedCheck_2444_;
goto v_resetjp_2438_;
}
v_resetjp_2438_:
{
lean_object* v___x_2442_; 
if (v_isShared_2440_ == 0)
{
v___x_2442_ = v___x_2439_;
goto v_reusejp_2441_;
}
else
{
lean_object* v_reuseFailAlloc_2443_; 
v_reuseFailAlloc_2443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2443_, 0, v_a_2437_);
v___x_2442_ = v_reuseFailAlloc_2443_;
goto v_reusejp_2441_;
}
v_reusejp_2441_:
{
return v___x_2442_;
}
}
}
}
else
{
lean_object* v___x_2445_; lean_object* v___x_2447_; 
lean_dec(v___x_2405_);
lean_dec(v_a_2397_);
lean_dec(v_a_2395_);
v___x_2445_ = lean_box(0);
if (v_isShared_2400_ == 0)
{
lean_ctor_set(v___x_2399_, 0, v___x_2445_);
v___x_2447_ = v___x_2399_;
goto v_reusejp_2446_;
}
else
{
lean_object* v_reuseFailAlloc_2448_; 
v_reuseFailAlloc_2448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2448_, 0, v___x_2445_);
v___x_2447_ = v_reuseFailAlloc_2448_;
goto v_reusejp_2446_;
}
v_reusejp_2446_:
{
return v___x_2447_;
}
}
}
else
{
lean_object* v___x_2449_; lean_object* v___x_2451_; 
lean_dec(v___x_2402_);
lean_dec(v_a_2397_);
lean_dec(v_a_2395_);
v___x_2449_ = lean_box(0);
if (v_isShared_2400_ == 0)
{
lean_ctor_set(v___x_2399_, 0, v___x_2449_);
v___x_2451_ = v___x_2399_;
goto v_reusejp_2450_;
}
else
{
lean_object* v_reuseFailAlloc_2452_; 
v_reuseFailAlloc_2452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2452_, 0, v___x_2449_);
v___x_2451_ = v_reuseFailAlloc_2452_;
goto v_reusejp_2450_;
}
v_reusejp_2450_:
{
return v___x_2451_;
}
}
}
}
else
{
lean_object* v_a_2454_; lean_object* v___x_2456_; uint8_t v_isShared_2457_; uint8_t v_isSharedCheck_2461_; 
lean_dec(v_a_2395_);
lean_dec(v_cmd_2390_);
v_a_2454_ = lean_ctor_get(v___x_2396_, 0);
v_isSharedCheck_2461_ = !lean_is_exclusive(v___x_2396_);
if (v_isSharedCheck_2461_ == 0)
{
v___x_2456_ = v___x_2396_;
v_isShared_2457_ = v_isSharedCheck_2461_;
goto v_resetjp_2455_;
}
else
{
lean_inc(v_a_2454_);
lean_dec(v___x_2396_);
v___x_2456_ = lean_box(0);
v_isShared_2457_ = v_isSharedCheck_2461_;
goto v_resetjp_2455_;
}
v_resetjp_2455_:
{
lean_object* v___x_2459_; 
if (v_isShared_2457_ == 0)
{
v___x_2459_ = v___x_2456_;
goto v_reusejp_2458_;
}
else
{
lean_object* v_reuseFailAlloc_2460_; 
v_reuseFailAlloc_2460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2460_, 0, v_a_2454_);
v___x_2459_ = v_reuseFailAlloc_2460_;
goto v_reusejp_2458_;
}
v_reusejp_2458_:
{
return v___x_2459_;
}
}
}
}
else
{
lean_object* v_a_2462_; lean_object* v___x_2464_; uint8_t v_isShared_2465_; uint8_t v_isSharedCheck_2469_; 
lean_dec(v_cmd_2390_);
v_a_2462_ = lean_ctor_get(v___x_2394_, 0);
v_isSharedCheck_2469_ = !lean_is_exclusive(v___x_2394_);
if (v_isSharedCheck_2469_ == 0)
{
v___x_2464_ = v___x_2394_;
v_isShared_2465_ = v_isSharedCheck_2469_;
goto v_resetjp_2463_;
}
else
{
lean_inc(v_a_2462_);
lean_dec(v___x_2394_);
v___x_2464_ = lean_box(0);
v_isShared_2465_ = v_isSharedCheck_2469_;
goto v_resetjp_2463_;
}
v_resetjp_2463_:
{
lean_object* v___x_2467_; 
if (v_isShared_2465_ == 0)
{
v___x_2467_ = v___x_2464_;
goto v_reusejp_2466_;
}
else
{
lean_object* v_reuseFailAlloc_2468_; 
v_reuseFailAlloc_2468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2468_, 0, v_a_2462_);
v___x_2467_ = v_reuseFailAlloc_2468_;
goto v_reusejp_2466_;
}
v_reusejp_2466_:
{
return v___x_2467_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getDeclName___boxed(lean_object* v_cmd_2470_, lean_object* v_a_2471_, lean_object* v_a_2472_, lean_object* v_a_2473_){
_start:
{
lean_object* v_res_2474_; 
v_res_2474_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName(v_cmd_2470_, v_a_2471_, v_a_2472_);
lean_dec(v_a_2472_);
lean_dec_ref(v_a_2471_);
return v_res_2474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_2475_, lean_object* v_ref_2476_, lean_object* v_msg_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_){
_start:
{
lean_object* v___x_2481_; 
v___x_2481_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___redArg(v_ref_2476_, v_msg_2477_, v___y_2478_, v___y_2479_);
return v___x_2481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_2482_, lean_object* v_ref_2483_, lean_object* v_msg_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_){
_start:
{
lean_object* v_res_2488_; 
v_res_2488_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1(v_00_u03b1_2482_, v_ref_2483_, v_msg_2484_, v___y_2485_, v___y_2486_);
lean_dec(v___y_2486_);
lean_dec_ref(v___y_2485_);
lean_dec(v_ref_2483_);
return v_res_2488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_msgData_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_){
_start:
{
lean_object* v___x_2493_; 
v___x_2493_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_msgData_2489_, v___y_2491_);
return v___x_2493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_msgData_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_){
_start:
{
lean_object* v_res_2498_; 
v_res_2498_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__4(v_msgData_2494_, v___y_2495_, v___y_2496_);
lean_dec(v___y_2496_);
lean_dec_ref(v___y_2495_);
return v_res_2498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_2499_, lean_object* v_msg_2500_, lean_object* v___y_2501_, lean_object* v___y_2502_){
_start:
{
lean_object* v___x_2504_; 
v___x_2504_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___redArg(v_msg_2500_, v___y_2501_, v___y_2502_);
return v___x_2504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_2505_, lean_object* v_msg_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_){
_start:
{
lean_object* v_res_2510_; 
v_res_2510_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_2505_, v_msg_2506_, v___y_2507_, v___y_2508_);
lean_dec(v___y_2508_);
lean_dec_ref(v___y_2507_);
return v_res_2510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8(lean_object* v_o_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_){
_start:
{
lean_object* v___x_2515_; 
v___x_2515_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___redArg(v_o_2511_, v___y_2513_);
return v___x_2515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8___boxed(lean_object* v_o_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_, lean_object* v___y_2519_){
_start:
{
lean_object* v_res_2520_; 
v_res_2520_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__4_spec__8(v_o_2516_, v___y_2517_, v___y_2518_);
lean_dec(v___y_2518_);
lean_dec_ref(v___y_2517_);
return v_res_2520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5(lean_object* v_msgData_2521_, lean_object* v_macroStack_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_){
_start:
{
lean_object* v___x_2526_; 
v___x_2526_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___redArg(v_msgData_2521_, v_macroStack_2522_, v___y_2524_);
return v___x_2526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5___boxed(lean_object* v_msgData_2527_, lean_object* v_macroStack_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_){
_start:
{
lean_object* v_res_2532_; 
v_res_2532_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__1_spec__2_spec__5(v_msgData_2527_, v_macroStack_2528_, v___y_2529_, v___y_2530_);
lean_dec(v___y_2530_);
lean_dec_ref(v___y_2529_);
return v_res_2532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22(lean_object* v_00_u03b1_2533_, lean_object* v_x_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_){
_start:
{
lean_object* v___x_2537_; 
v___x_2537_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___redArg(v_x_2534_, v___y_2536_);
return v___x_2537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22___boxed(lean_object* v_00_u03b1_2538_, lean_object* v_x_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_){
_start:
{
lean_object* v_res_2542_; 
v_res_2542_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__22(v_00_u03b1_2538_, v_x_2539_, v___y_2540_, v___y_2541_);
lean_dec_ref(v___y_2540_);
lean_dec_ref(v_x_2539_);
return v_res_2542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25(lean_object* v_00_u03b1_2543_, lean_object* v_ref_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_){
_start:
{
lean_object* v___x_2548_; 
v___x_2548_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___redArg(v_ref_2544_);
return v___x_2548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25___boxed(lean_object* v_00_u03b1_2549_, lean_object* v_ref_2550_, lean_object* v___y_2551_, lean_object* v___y_2552_, lean_object* v___y_2553_){
_start:
{
lean_object* v_res_2554_; 
v_res_2554_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__25(v_00_u03b1_2549_, v_ref_2550_, v___y_2551_, v___y_2552_);
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
return v_res_2554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26(lean_object* v_00_u03b1_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_){
_start:
{
lean_object* v___x_2559_; 
v___x_2559_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
return v___x_2559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___boxed(lean_object* v_00_u03b1_2560_, lean_object* v___y_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_){
_start:
{
lean_object* v_res_2564_; 
v_res_2564_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26(v_00_u03b1_2560_, v___y_2561_, v___y_2562_);
lean_dec(v___y_2562_);
lean_dec_ref(v___y_2561_);
return v_res_2564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27(lean_object* v___y_2565_, lean_object* v___y_2566_){
_start:
{
lean_object* v___x_2568_; 
v___x_2568_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___redArg(v___y_2566_);
return v___x_2568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27___boxed(lean_object* v___y_2569_, lean_object* v___y_2570_, lean_object* v___y_2571_){
_start:
{
lean_object* v_res_2572_; 
v_res_2572_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__27(v___y_2569_, v___y_2570_);
lean_dec(v___y_2570_);
lean_dec_ref(v___y_2569_);
return v_res_2572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16(lean_object* v_00_u03b1_2573_, lean_object* v_x_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_){
_start:
{
lean_object* v___x_2578_; 
v___x_2578_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___redArg(v_x_2574_, v___y_2575_, v___y_2576_);
return v___x_2578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16___boxed(lean_object* v_00_u03b1_2579_, lean_object* v_x_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_){
_start:
{
lean_object* v_res_2584_; 
v_res_2584_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16(v_00_u03b1_2579_, v_x_2580_, v___y_2581_, v___y_2582_);
lean_dec(v___y_2582_);
lean_dec_ref(v___y_2581_);
return v_res_2584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33(lean_object* v_00_u03b1_2585_, lean_object* v_x_2586_, uint8_t v_isExporting_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_){
_start:
{
lean_object* v___x_2591_; 
v___x_2591_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___redArg(v_x_2586_, v_isExporting_2587_, v___y_2588_, v___y_2589_);
return v___x_2591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33___boxed(lean_object* v_00_u03b1_2592_, lean_object* v_x_2593_, lean_object* v_isExporting_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_){
_start:
{
uint8_t v_isExporting_boxed_2598_; lean_object* v_res_2599_; 
v_isExporting_boxed_2598_ = lean_unbox(v_isExporting_2594_);
v_res_2599_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18_spec__33(v_00_u03b1_2592_, v_x_2593_, v_isExporting_boxed_2598_, v___y_2595_, v___y_2596_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2595_);
return v_res_2599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18(lean_object* v_00_u03b1_2600_, lean_object* v_x_2601_, uint8_t v_when_2602_, lean_object* v___y_2603_, lean_object* v___y_2604_){
_start:
{
lean_object* v___x_2606_; 
v___x_2606_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___redArg(v_x_2601_, v_when_2602_, v___y_2603_, v___y_2604_);
return v___x_2606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18___boxed(lean_object* v_00_u03b1_2607_, lean_object* v_x_2608_, lean_object* v_when_2609_, lean_object* v___y_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_){
_start:
{
uint8_t v_when_boxed_2613_; lean_object* v_res_2614_; 
v_when_boxed_2613_ = lean_unbox(v_when_2609_);
v_res_2614_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__18(v_00_u03b1_2607_, v_x_2608_, v_when_boxed_2613_, v___y_2610_, v___y_2611_);
lean_dec(v___y_2611_);
lean_dec_ref(v___y_2610_);
return v_res_2614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23(lean_object* v_as_2615_, lean_object* v_as_x27_2616_, lean_object* v_b_2617_, lean_object* v_a_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_){
_start:
{
lean_object* v___x_2622_; 
v___x_2622_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___redArg(v_as_x27_2616_, v_b_2617_, v___y_2619_, v___y_2620_);
return v___x_2622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23___boxed(lean_object* v_as_2623_, lean_object* v_as_x27_2624_, lean_object* v_b_2625_, lean_object* v_a_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_){
_start:
{
lean_object* v_res_2630_; 
v_res_2630_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__23(v_as_2623_, v_as_x27_2624_, v_b_2625_, v_a_2626_, v___y_2627_, v___y_2628_);
lean_dec(v___y_2628_);
lean_dec_ref(v___y_2627_);
lean_dec(v_as_x27_2624_);
lean_dec(v_as_2623_);
return v_res_2630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31(lean_object* v_00_u03b2_2631_, lean_object* v_m_2632_, lean_object* v_a_2633_){
_start:
{
lean_object* v___x_2634_; 
v___x_2634_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___redArg(v_m_2632_, v_a_2633_);
return v___x_2634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31___boxed(lean_object* v_00_u03b2_2635_, lean_object* v_m_2636_, lean_object* v_a_2637_){
_start:
{
lean_object* v_res_2638_; 
v_res_2638_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31(v_00_u03b2_2635_, v_m_2636_, v_a_2637_);
lean_dec(v_a_2637_);
lean_dec_ref(v_m_2636_);
return v_res_2638_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31(lean_object* v_00_u03b2_2639_, lean_object* v_x_2640_, lean_object* v_x_2641_){
_start:
{
uint8_t v___x_2642_; 
v___x_2642_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___redArg(v_x_2640_, v_x_2641_);
return v___x_2642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31___boxed(lean_object* v_00_u03b2_2643_, lean_object* v_x_2644_, lean_object* v_x_2645_){
_start:
{
uint8_t v_res_2646_; lean_object* v_r_2647_; 
v_res_2646_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31(v_00_u03b2_2643_, v_x_2644_, v_x_2645_);
lean_dec_ref(v_x_2645_);
lean_dec_ref(v_x_2644_);
v_r_2647_ = lean_box(v_res_2646_);
return v_r_2647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34(lean_object* v_00_u03b2_2648_, lean_object* v_a_2649_, lean_object* v_x_2650_){
_start:
{
lean_object* v___x_2651_; 
v___x_2651_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___redArg(v_a_2649_, v_x_2650_);
return v___x_2651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34___boxed(lean_object* v_00_u03b2_2652_, lean_object* v_a_2653_, lean_object* v_x_2654_){
_start:
{
lean_object* v_res_2655_; 
v_res_2655_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__31_spec__34(v_00_u03b2_2652_, v_a_2653_, v_x_2654_);
lean_dec(v_x_2654_);
lean_dec(v_a_2653_);
return v_res_2655_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34(lean_object* v_00_u03b2_2656_, lean_object* v_x_2657_, size_t v_x_2658_, lean_object* v_x_2659_){
_start:
{
uint8_t v___x_2660_; 
v___x_2660_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___redArg(v_x_2657_, v_x_2658_, v_x_2659_);
return v___x_2660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34___boxed(lean_object* v_00_u03b2_2661_, lean_object* v_x_2662_, lean_object* v_x_2663_, lean_object* v_x_2664_){
_start:
{
size_t v_x_24472__boxed_2665_; uint8_t v_res_2666_; lean_object* v_r_2667_; 
v_x_24472__boxed_2665_ = lean_unbox_usize(v_x_2663_);
lean_dec(v_x_2663_);
v_res_2666_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34(v_00_u03b2_2661_, v_x_2662_, v_x_24472__boxed_2665_, v_x_2664_);
lean_dec_ref(v_x_2664_);
lean_dec_ref(v_x_2662_);
v_r_2667_ = lean_box(v_res_2666_);
return v_r_2667_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37(lean_object* v_00_u03b2_2668_, lean_object* v_keys_2669_, lean_object* v_vals_2670_, lean_object* v_heq_2671_, lean_object* v_i_2672_, lean_object* v_k_2673_){
_start:
{
uint8_t v___x_2674_; 
v___x_2674_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___redArg(v_keys_2669_, v_i_2672_, v_k_2673_);
return v___x_2674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37___boxed(lean_object* v_00_u03b2_2675_, lean_object* v_keys_2676_, lean_object* v_vals_2677_, lean_object* v_heq_2678_, lean_object* v_i_2679_, lean_object* v_k_2680_){
_start:
{
uint8_t v_res_2681_; lean_object* v_r_2682_; 
v_res_2681_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__17_spec__29_spec__31_spec__34_spec__37(v_00_u03b2_2675_, v_keys_2676_, v_vals_2677_, v_heq_2678_, v_i_2679_, v_k_2680_);
lean_dec_ref(v_k_2680_);
lean_dec_ref(v_vals_2677_);
lean_dec_ref(v_keys_2676_);
v_r_2682_ = lean_box(v_res_2681_);
return v_r_2682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(lean_object* v_cmd_2683_, lean_object* v_id_2684_, lean_object* v_a_2685_, lean_object* v_a_2686_){
_start:
{
lean_object* v___x_2688_; lean_object* v___x_2689_; 
v___x_2688_ = lean_st_ref_get(v_a_2686_);
lean_inc(v_cmd_2683_);
v___x_2689_ = lp_mathlib_Mathlib_Command_MinImports_getDeclName(v_cmd_2683_, v_a_2685_, v_a_2686_);
if (lean_obj_tag(v___x_2689_) == 0)
{
lean_object* v_a_2690_; lean_object* v___x_2691_; 
v_a_2690_ = lean_ctor_get(v___x_2689_, 0);
lean_inc(v_a_2690_);
lean_dec_ref_known(v___x_2689_, 1);
v___x_2691_ = lp_mathlib_Mathlib_Command_MinImports_getVisited(v_a_2690_, v_a_2685_, v_a_2686_);
if (lean_obj_tag(v___x_2691_) == 0)
{
lean_object* v_a_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; 
v_a_2692_ = lean_ctor_get(v___x_2691_, 0);
lean_inc(v_a_2692_);
lean_dec_ref_known(v___x_2691_, 1);
v___x_2693_ = l_Lean_Syntax_getId(v_id_2684_);
v___x_2694_ = lp_mathlib_Mathlib_Command_MinImports_getVisited(v___x_2693_, v_a_2685_, v_a_2686_);
if (lean_obj_tag(v___x_2694_) == 0)
{
lean_object* v_a_2695_; lean_object* v___x_2697_; uint8_t v_isShared_2698_; uint8_t v_isSharedCheck_2708_; 
v_a_2695_ = lean_ctor_get(v___x_2694_, 0);
v_isSharedCheck_2708_ = !lean_is_exclusive(v___x_2694_);
if (v_isSharedCheck_2708_ == 0)
{
v___x_2697_ = v___x_2694_;
v_isShared_2698_ = v_isSharedCheck_2708_;
goto v_resetjp_2696_;
}
else
{
lean_inc(v_a_2695_);
lean_dec(v___x_2694_);
v___x_2697_ = lean_box(0);
v_isShared_2698_ = v_isSharedCheck_2708_;
goto v_resetjp_2696_;
}
v_resetjp_2696_:
{
lean_object* v_env_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2706_; 
v_env_2699_ = lean_ctor_get(v___x_2688_, 0);
lean_inc_ref(v_env_2699_);
lean_dec(v___x_2688_);
v___x_2700_ = l_Lean_NameSet_append(v_a_2692_, v_a_2695_);
lean_inc(v_cmd_2683_);
v___x_2701_ = lp_mathlib_Mathlib_Command_MinImports_getSyntaxNodeKinds(v_cmd_2683_);
v___x_2702_ = l_Lean_NameSet_append(v___x_2700_, v___x_2701_);
v___x_2703_ = lp_mathlib_Mathlib_Command_MinImports_getAttrs(v_env_2699_, v_cmd_2683_);
v___x_2704_ = l_Lean_NameSet_append(v___x_2702_, v___x_2703_);
if (v_isShared_2698_ == 0)
{
lean_ctor_set(v___x_2697_, 0, v___x_2704_);
v___x_2706_ = v___x_2697_;
goto v_reusejp_2705_;
}
else
{
lean_object* v_reuseFailAlloc_2707_; 
v_reuseFailAlloc_2707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2707_, 0, v___x_2704_);
v___x_2706_ = v_reuseFailAlloc_2707_;
goto v_reusejp_2705_;
}
v_reusejp_2705_:
{
return v___x_2706_;
}
}
}
else
{
lean_dec(v_a_2692_);
lean_dec(v___x_2688_);
lean_dec(v_cmd_2683_);
return v___x_2694_;
}
}
else
{
lean_dec(v___x_2688_);
lean_dec(v_cmd_2683_);
return v___x_2691_;
}
}
else
{
lean_object* v_a_2709_; lean_object* v___x_2711_; uint8_t v_isShared_2712_; uint8_t v_isSharedCheck_2716_; 
lean_dec(v___x_2688_);
lean_dec(v_cmd_2683_);
v_a_2709_ = lean_ctor_get(v___x_2689_, 0);
v_isSharedCheck_2716_ = !lean_is_exclusive(v___x_2689_);
if (v_isSharedCheck_2716_ == 0)
{
v___x_2711_ = v___x_2689_;
v_isShared_2712_ = v_isSharedCheck_2716_;
goto v_resetjp_2710_;
}
else
{
lean_inc(v_a_2709_);
lean_dec(v___x_2689_);
v___x_2711_ = lean_box(0);
v_isShared_2712_ = v_isSharedCheck_2716_;
goto v_resetjp_2710_;
}
v_resetjp_2710_:
{
lean_object* v___x_2714_; 
if (v_isShared_2712_ == 0)
{
v___x_2714_ = v___x_2711_;
goto v_reusejp_2713_;
}
else
{
lean_object* v_reuseFailAlloc_2715_; 
v_reuseFailAlloc_2715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2715_, 0, v_a_2709_);
v___x_2714_ = v_reuseFailAlloc_2715_;
goto v_reusejp_2713_;
}
v_reusejp_2713_:
{
return v___x_2714_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllDependencies___boxed(lean_object* v_cmd_2717_, lean_object* v_id_2718_, lean_object* v_a_2719_, lean_object* v_a_2720_, lean_object* v_a_2721_){
_start:
{
lean_object* v_res_2722_; 
v_res_2722_ = lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(v_cmd_2717_, v_id_2718_, v_a_2719_, v_a_2720_);
lean_dec(v_a_2720_);
lean_dec_ref(v_a_2719_);
lean_dec(v_id_2718_);
return v_res_2722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Command_MinImports_getAllImports_spec__4(lean_object* v_msg_2723_){
_start:
{
lean_object* v___x_2724_; lean_object* v___x_2725_; 
v___x_2724_ = lean_box(0);
v___x_2725_ = lean_panic_fn_borrowed(v___x_2724_, v_msg_2723_);
return v___x_2725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(lean_object* v_k_2726_, lean_object* v_t_2727_){
_start:
{
if (lean_obj_tag(v_t_2727_) == 0)
{
lean_object* v_k_2728_; lean_object* v_v_2729_; lean_object* v_l_2730_; lean_object* v_r_2731_; lean_object* v___x_2733_; uint8_t v_isShared_2734_; uint8_t v_isSharedCheck_3385_; 
v_k_2728_ = lean_ctor_get(v_t_2727_, 1);
v_v_2729_ = lean_ctor_get(v_t_2727_, 2);
v_l_2730_ = lean_ctor_get(v_t_2727_, 3);
v_r_2731_ = lean_ctor_get(v_t_2727_, 4);
v_isSharedCheck_3385_ = !lean_is_exclusive(v_t_2727_);
if (v_isSharedCheck_3385_ == 0)
{
lean_object* v_unused_3386_; 
v_unused_3386_ = lean_ctor_get(v_t_2727_, 0);
lean_dec(v_unused_3386_);
v___x_2733_ = v_t_2727_;
v_isShared_2734_ = v_isSharedCheck_3385_;
goto v_resetjp_2732_;
}
else
{
lean_inc(v_r_2731_);
lean_inc(v_l_2730_);
lean_inc(v_v_2729_);
lean_inc(v_k_2728_);
lean_dec(v_t_2727_);
v___x_2733_ = lean_box(0);
v_isShared_2734_ = v_isSharedCheck_3385_;
goto v_resetjp_2732_;
}
v_resetjp_2732_:
{
uint8_t v___x_2735_; 
v___x_2735_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_2726_, v_k_2728_);
switch(v___x_2735_)
{
case 0:
{
lean_object* v_impl_2736_; lean_object* v___x_2737_; 
v_impl_2736_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v_k_2726_, v_l_2730_);
v___x_2737_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_2736_) == 0)
{
if (lean_obj_tag(v_r_2731_) == 0)
{
lean_object* v_size_2738_; lean_object* v_size_2739_; lean_object* v_k_2740_; lean_object* v_v_2741_; lean_object* v_l_2742_; lean_object* v_r_2743_; lean_object* v___x_2744_; lean_object* v___x_2745_; uint8_t v___x_2746_; 
v_size_2738_ = lean_ctor_get(v_impl_2736_, 0);
lean_inc(v_size_2738_);
v_size_2739_ = lean_ctor_get(v_r_2731_, 0);
v_k_2740_ = lean_ctor_get(v_r_2731_, 1);
v_v_2741_ = lean_ctor_get(v_r_2731_, 2);
v_l_2742_ = lean_ctor_get(v_r_2731_, 3);
lean_inc(v_l_2742_);
v_r_2743_ = lean_ctor_get(v_r_2731_, 4);
v___x_2744_ = lean_unsigned_to_nat(3u);
v___x_2745_ = lean_nat_mul(v___x_2744_, v_size_2738_);
v___x_2746_ = lean_nat_dec_lt(v___x_2745_, v_size_2739_);
lean_dec(v___x_2745_);
if (v___x_2746_ == 0)
{
lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2750_; 
lean_dec(v_l_2742_);
v___x_2747_ = lean_nat_add(v___x_2737_, v_size_2738_);
lean_dec(v_size_2738_);
v___x_2748_ = lean_nat_add(v___x_2747_, v_size_2739_);
lean_dec(v___x_2747_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 3, v_impl_2736_);
lean_ctor_set(v___x_2733_, 0, v___x_2748_);
v___x_2750_ = v___x_2733_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2751_; 
v_reuseFailAlloc_2751_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2751_, 0, v___x_2748_);
lean_ctor_set(v_reuseFailAlloc_2751_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2751_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2751_, 3, v_impl_2736_);
lean_ctor_set(v_reuseFailAlloc_2751_, 4, v_r_2731_);
v___x_2750_ = v_reuseFailAlloc_2751_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
return v___x_2750_;
}
}
else
{
lean_object* v___x_2753_; uint8_t v_isShared_2754_; uint8_t v_isSharedCheck_2815_; 
lean_inc(v_r_2743_);
lean_inc(v_v_2741_);
lean_inc(v_k_2740_);
lean_inc(v_size_2739_);
v_isSharedCheck_2815_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2815_ == 0)
{
lean_object* v_unused_2816_; lean_object* v_unused_2817_; lean_object* v_unused_2818_; lean_object* v_unused_2819_; lean_object* v_unused_2820_; 
v_unused_2816_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2816_);
v_unused_2817_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2817_);
v_unused_2818_ = lean_ctor_get(v_r_2731_, 2);
lean_dec(v_unused_2818_);
v_unused_2819_ = lean_ctor_get(v_r_2731_, 1);
lean_dec(v_unused_2819_);
v_unused_2820_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_2820_);
v___x_2753_ = v_r_2731_;
v_isShared_2754_ = v_isSharedCheck_2815_;
goto v_resetjp_2752_;
}
else
{
lean_dec(v_r_2731_);
v___x_2753_ = lean_box(0);
v_isShared_2754_ = v_isSharedCheck_2815_;
goto v_resetjp_2752_;
}
v_resetjp_2752_:
{
lean_object* v_size_2755_; lean_object* v_k_2756_; lean_object* v_v_2757_; lean_object* v_l_2758_; lean_object* v_r_2759_; lean_object* v_size_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; uint8_t v___x_2763_; 
v_size_2755_ = lean_ctor_get(v_l_2742_, 0);
v_k_2756_ = lean_ctor_get(v_l_2742_, 1);
v_v_2757_ = lean_ctor_get(v_l_2742_, 2);
v_l_2758_ = lean_ctor_get(v_l_2742_, 3);
v_r_2759_ = lean_ctor_get(v_l_2742_, 4);
v_size_2760_ = lean_ctor_get(v_r_2743_, 0);
v___x_2761_ = lean_unsigned_to_nat(2u);
v___x_2762_ = lean_nat_mul(v___x_2761_, v_size_2760_);
v___x_2763_ = lean_nat_dec_lt(v_size_2755_, v___x_2762_);
lean_dec(v___x_2762_);
if (v___x_2763_ == 0)
{
lean_object* v___x_2765_; uint8_t v_isShared_2766_; uint8_t v_isSharedCheck_2791_; 
lean_inc(v_r_2759_);
lean_inc(v_l_2758_);
lean_inc(v_v_2757_);
lean_inc(v_k_2756_);
v_isSharedCheck_2791_ = !lean_is_exclusive(v_l_2742_);
if (v_isSharedCheck_2791_ == 0)
{
lean_object* v_unused_2792_; lean_object* v_unused_2793_; lean_object* v_unused_2794_; lean_object* v_unused_2795_; lean_object* v_unused_2796_; 
v_unused_2792_ = lean_ctor_get(v_l_2742_, 4);
lean_dec(v_unused_2792_);
v_unused_2793_ = lean_ctor_get(v_l_2742_, 3);
lean_dec(v_unused_2793_);
v_unused_2794_ = lean_ctor_get(v_l_2742_, 2);
lean_dec(v_unused_2794_);
v_unused_2795_ = lean_ctor_get(v_l_2742_, 1);
lean_dec(v_unused_2795_);
v_unused_2796_ = lean_ctor_get(v_l_2742_, 0);
lean_dec(v_unused_2796_);
v___x_2765_ = v_l_2742_;
v_isShared_2766_ = v_isSharedCheck_2791_;
goto v_resetjp_2764_;
}
else
{
lean_dec(v_l_2742_);
v___x_2765_ = lean_box(0);
v_isShared_2766_ = v_isSharedCheck_2791_;
goto v_resetjp_2764_;
}
v_resetjp_2764_:
{
lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v___y_2770_; lean_object* v___y_2771_; lean_object* v___y_2772_; lean_object* v___y_2781_; 
v___x_2767_ = lean_nat_add(v___x_2737_, v_size_2738_);
lean_dec(v_size_2738_);
v___x_2768_ = lean_nat_add(v___x_2767_, v_size_2739_);
lean_dec(v_size_2739_);
if (lean_obj_tag(v_l_2758_) == 0)
{
lean_object* v_size_2789_; 
v_size_2789_ = lean_ctor_get(v_l_2758_, 0);
lean_inc(v_size_2789_);
v___y_2781_ = v_size_2789_;
goto v___jp_2780_;
}
else
{
lean_object* v___x_2790_; 
v___x_2790_ = lean_unsigned_to_nat(0u);
v___y_2781_ = v___x_2790_;
goto v___jp_2780_;
}
v___jp_2769_:
{
lean_object* v___x_2773_; lean_object* v___x_2775_; 
v___x_2773_ = lean_nat_add(v___y_2770_, v___y_2772_);
lean_dec(v___y_2772_);
lean_dec(v___y_2770_);
if (v_isShared_2766_ == 0)
{
lean_ctor_set(v___x_2765_, 4, v_r_2743_);
lean_ctor_set(v___x_2765_, 3, v_r_2759_);
lean_ctor_set(v___x_2765_, 2, v_v_2741_);
lean_ctor_set(v___x_2765_, 1, v_k_2740_);
lean_ctor_set(v___x_2765_, 0, v___x_2773_);
v___x_2775_ = v___x_2765_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2779_; 
v_reuseFailAlloc_2779_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2779_, 0, v___x_2773_);
lean_ctor_set(v_reuseFailAlloc_2779_, 1, v_k_2740_);
lean_ctor_set(v_reuseFailAlloc_2779_, 2, v_v_2741_);
lean_ctor_set(v_reuseFailAlloc_2779_, 3, v_r_2759_);
lean_ctor_set(v_reuseFailAlloc_2779_, 4, v_r_2743_);
v___x_2775_ = v_reuseFailAlloc_2779_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
lean_object* v___x_2777_; 
if (v_isShared_2754_ == 0)
{
lean_ctor_set(v___x_2753_, 4, v___x_2775_);
lean_ctor_set(v___x_2753_, 3, v___y_2771_);
lean_ctor_set(v___x_2753_, 2, v_v_2757_);
lean_ctor_set(v___x_2753_, 1, v_k_2756_);
lean_ctor_set(v___x_2753_, 0, v___x_2768_);
v___x_2777_ = v___x_2753_;
goto v_reusejp_2776_;
}
else
{
lean_object* v_reuseFailAlloc_2778_; 
v_reuseFailAlloc_2778_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2778_, 0, v___x_2768_);
lean_ctor_set(v_reuseFailAlloc_2778_, 1, v_k_2756_);
lean_ctor_set(v_reuseFailAlloc_2778_, 2, v_v_2757_);
lean_ctor_set(v_reuseFailAlloc_2778_, 3, v___y_2771_);
lean_ctor_set(v_reuseFailAlloc_2778_, 4, v___x_2775_);
v___x_2777_ = v_reuseFailAlloc_2778_;
goto v_reusejp_2776_;
}
v_reusejp_2776_:
{
return v___x_2777_;
}
}
}
v___jp_2780_:
{
lean_object* v___x_2782_; lean_object* v___x_2784_; 
v___x_2782_ = lean_nat_add(v___x_2767_, v___y_2781_);
lean_dec(v___y_2781_);
lean_dec(v___x_2767_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_l_2758_);
lean_ctor_set(v___x_2733_, 3, v_impl_2736_);
lean_ctor_set(v___x_2733_, 0, v___x_2782_);
v___x_2784_ = v___x_2733_;
goto v_reusejp_2783_;
}
else
{
lean_object* v_reuseFailAlloc_2788_; 
v_reuseFailAlloc_2788_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2788_, 0, v___x_2782_);
lean_ctor_set(v_reuseFailAlloc_2788_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2788_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2788_, 3, v_impl_2736_);
lean_ctor_set(v_reuseFailAlloc_2788_, 4, v_l_2758_);
v___x_2784_ = v_reuseFailAlloc_2788_;
goto v_reusejp_2783_;
}
v_reusejp_2783_:
{
lean_object* v___x_2785_; 
v___x_2785_ = lean_nat_add(v___x_2737_, v_size_2760_);
if (lean_obj_tag(v_r_2759_) == 0)
{
lean_object* v_size_2786_; 
v_size_2786_ = lean_ctor_get(v_r_2759_, 0);
lean_inc(v_size_2786_);
v___y_2770_ = v___x_2785_;
v___y_2771_ = v___x_2784_;
v___y_2772_ = v_size_2786_;
goto v___jp_2769_;
}
else
{
lean_object* v___x_2787_; 
v___x_2787_ = lean_unsigned_to_nat(0u);
v___y_2770_ = v___x_2785_;
v___y_2771_ = v___x_2784_;
v___y_2772_ = v___x_2787_;
goto v___jp_2769_;
}
}
}
}
}
else
{
lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2801_; 
lean_del_object(v___x_2733_);
v___x_2797_ = lean_nat_add(v___x_2737_, v_size_2738_);
lean_dec(v_size_2738_);
v___x_2798_ = lean_nat_add(v___x_2797_, v_size_2739_);
lean_dec(v_size_2739_);
v___x_2799_ = lean_nat_add(v___x_2797_, v_size_2755_);
lean_dec(v___x_2797_);
lean_inc_ref(v_impl_2736_);
if (v_isShared_2754_ == 0)
{
lean_ctor_set(v___x_2753_, 4, v_l_2742_);
lean_ctor_set(v___x_2753_, 3, v_impl_2736_);
lean_ctor_set(v___x_2753_, 2, v_v_2729_);
lean_ctor_set(v___x_2753_, 1, v_k_2728_);
lean_ctor_set(v___x_2753_, 0, v___x_2799_);
v___x_2801_ = v___x_2753_;
goto v_reusejp_2800_;
}
else
{
lean_object* v_reuseFailAlloc_2814_; 
v_reuseFailAlloc_2814_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2814_, 0, v___x_2799_);
lean_ctor_set(v_reuseFailAlloc_2814_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2814_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2814_, 3, v_impl_2736_);
lean_ctor_set(v_reuseFailAlloc_2814_, 4, v_l_2742_);
v___x_2801_ = v_reuseFailAlloc_2814_;
goto v_reusejp_2800_;
}
v_reusejp_2800_:
{
lean_object* v___x_2803_; uint8_t v_isShared_2804_; uint8_t v_isSharedCheck_2808_; 
v_isSharedCheck_2808_ = !lean_is_exclusive(v_impl_2736_);
if (v_isSharedCheck_2808_ == 0)
{
lean_object* v_unused_2809_; lean_object* v_unused_2810_; lean_object* v_unused_2811_; lean_object* v_unused_2812_; lean_object* v_unused_2813_; 
v_unused_2809_ = lean_ctor_get(v_impl_2736_, 4);
lean_dec(v_unused_2809_);
v_unused_2810_ = lean_ctor_get(v_impl_2736_, 3);
lean_dec(v_unused_2810_);
v_unused_2811_ = lean_ctor_get(v_impl_2736_, 2);
lean_dec(v_unused_2811_);
v_unused_2812_ = lean_ctor_get(v_impl_2736_, 1);
lean_dec(v_unused_2812_);
v_unused_2813_ = lean_ctor_get(v_impl_2736_, 0);
lean_dec(v_unused_2813_);
v___x_2803_ = v_impl_2736_;
v_isShared_2804_ = v_isSharedCheck_2808_;
goto v_resetjp_2802_;
}
else
{
lean_dec(v_impl_2736_);
v___x_2803_ = lean_box(0);
v_isShared_2804_ = v_isSharedCheck_2808_;
goto v_resetjp_2802_;
}
v_resetjp_2802_:
{
lean_object* v___x_2806_; 
if (v_isShared_2804_ == 0)
{
lean_ctor_set(v___x_2803_, 4, v_r_2743_);
lean_ctor_set(v___x_2803_, 3, v___x_2801_);
lean_ctor_set(v___x_2803_, 2, v_v_2741_);
lean_ctor_set(v___x_2803_, 1, v_k_2740_);
lean_ctor_set(v___x_2803_, 0, v___x_2798_);
v___x_2806_ = v___x_2803_;
goto v_reusejp_2805_;
}
else
{
lean_object* v_reuseFailAlloc_2807_; 
v_reuseFailAlloc_2807_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2807_, 0, v___x_2798_);
lean_ctor_set(v_reuseFailAlloc_2807_, 1, v_k_2740_);
lean_ctor_set(v_reuseFailAlloc_2807_, 2, v_v_2741_);
lean_ctor_set(v_reuseFailAlloc_2807_, 3, v___x_2801_);
lean_ctor_set(v_reuseFailAlloc_2807_, 4, v_r_2743_);
v___x_2806_ = v_reuseFailAlloc_2807_;
goto v_reusejp_2805_;
}
v_reusejp_2805_:
{
return v___x_2806_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_2821_; lean_object* v___x_2822_; lean_object* v___x_2824_; 
v_size_2821_ = lean_ctor_get(v_impl_2736_, 0);
lean_inc(v_size_2821_);
v___x_2822_ = lean_nat_add(v___x_2737_, v_size_2821_);
lean_dec(v_size_2821_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 3, v_impl_2736_);
lean_ctor_set(v___x_2733_, 0, v___x_2822_);
v___x_2824_ = v___x_2733_;
goto v_reusejp_2823_;
}
else
{
lean_object* v_reuseFailAlloc_2825_; 
v_reuseFailAlloc_2825_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2825_, 0, v___x_2822_);
lean_ctor_set(v_reuseFailAlloc_2825_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2825_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2825_, 3, v_impl_2736_);
lean_ctor_set(v_reuseFailAlloc_2825_, 4, v_r_2731_);
v___x_2824_ = v_reuseFailAlloc_2825_;
goto v_reusejp_2823_;
}
v_reusejp_2823_:
{
return v___x_2824_;
}
}
}
else
{
if (lean_obj_tag(v_r_2731_) == 0)
{
lean_object* v_l_2826_; 
v_l_2826_ = lean_ctor_get(v_r_2731_, 3);
lean_inc(v_l_2826_);
if (lean_obj_tag(v_l_2826_) == 0)
{
lean_object* v_r_2827_; 
v_r_2827_ = lean_ctor_get(v_r_2731_, 4);
lean_inc(v_r_2827_);
if (lean_obj_tag(v_r_2827_) == 0)
{
lean_object* v_size_2828_; lean_object* v_k_2829_; lean_object* v_v_2830_; lean_object* v___x_2832_; uint8_t v_isShared_2833_; uint8_t v_isSharedCheck_2843_; 
v_size_2828_ = lean_ctor_get(v_r_2731_, 0);
v_k_2829_ = lean_ctor_get(v_r_2731_, 1);
v_v_2830_ = lean_ctor_get(v_r_2731_, 2);
v_isSharedCheck_2843_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2843_ == 0)
{
lean_object* v_unused_2844_; lean_object* v_unused_2845_; 
v_unused_2844_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2844_);
v_unused_2845_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2845_);
v___x_2832_ = v_r_2731_;
v_isShared_2833_ = v_isSharedCheck_2843_;
goto v_resetjp_2831_;
}
else
{
lean_inc(v_v_2830_);
lean_inc(v_k_2829_);
lean_inc(v_size_2828_);
lean_dec(v_r_2731_);
v___x_2832_ = lean_box(0);
v_isShared_2833_ = v_isSharedCheck_2843_;
goto v_resetjp_2831_;
}
v_resetjp_2831_:
{
lean_object* v_size_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2838_; 
v_size_2834_ = lean_ctor_get(v_l_2826_, 0);
v___x_2835_ = lean_nat_add(v___x_2737_, v_size_2828_);
lean_dec(v_size_2828_);
v___x_2836_ = lean_nat_add(v___x_2737_, v_size_2834_);
if (v_isShared_2833_ == 0)
{
lean_ctor_set(v___x_2832_, 4, v_l_2826_);
lean_ctor_set(v___x_2832_, 3, v_impl_2736_);
lean_ctor_set(v___x_2832_, 2, v_v_2729_);
lean_ctor_set(v___x_2832_, 1, v_k_2728_);
lean_ctor_set(v___x_2832_, 0, v___x_2836_);
v___x_2838_ = v___x_2832_;
goto v_reusejp_2837_;
}
else
{
lean_object* v_reuseFailAlloc_2842_; 
v_reuseFailAlloc_2842_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2842_, 0, v___x_2836_);
lean_ctor_set(v_reuseFailAlloc_2842_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2842_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2842_, 3, v_impl_2736_);
lean_ctor_set(v_reuseFailAlloc_2842_, 4, v_l_2826_);
v___x_2838_ = v_reuseFailAlloc_2842_;
goto v_reusejp_2837_;
}
v_reusejp_2837_:
{
lean_object* v___x_2840_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_r_2827_);
lean_ctor_set(v___x_2733_, 3, v___x_2838_);
lean_ctor_set(v___x_2733_, 2, v_v_2830_);
lean_ctor_set(v___x_2733_, 1, v_k_2829_);
lean_ctor_set(v___x_2733_, 0, v___x_2835_);
v___x_2840_ = v___x_2733_;
goto v_reusejp_2839_;
}
else
{
lean_object* v_reuseFailAlloc_2841_; 
v_reuseFailAlloc_2841_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2841_, 0, v___x_2835_);
lean_ctor_set(v_reuseFailAlloc_2841_, 1, v_k_2829_);
lean_ctor_set(v_reuseFailAlloc_2841_, 2, v_v_2830_);
lean_ctor_set(v_reuseFailAlloc_2841_, 3, v___x_2838_);
lean_ctor_set(v_reuseFailAlloc_2841_, 4, v_r_2827_);
v___x_2840_ = v_reuseFailAlloc_2841_;
goto v_reusejp_2839_;
}
v_reusejp_2839_:
{
return v___x_2840_;
}
}
}
}
else
{
lean_object* v_k_2846_; lean_object* v_v_2847_; lean_object* v___x_2849_; uint8_t v_isShared_2850_; uint8_t v_isSharedCheck_2870_; 
v_k_2846_ = lean_ctor_get(v_r_2731_, 1);
v_v_2847_ = lean_ctor_get(v_r_2731_, 2);
v_isSharedCheck_2870_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2870_ == 0)
{
lean_object* v_unused_2871_; lean_object* v_unused_2872_; lean_object* v_unused_2873_; 
v_unused_2871_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2871_);
v_unused_2872_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2872_);
v_unused_2873_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_2873_);
v___x_2849_ = v_r_2731_;
v_isShared_2850_ = v_isSharedCheck_2870_;
goto v_resetjp_2848_;
}
else
{
lean_inc(v_v_2847_);
lean_inc(v_k_2846_);
lean_dec(v_r_2731_);
v___x_2849_ = lean_box(0);
v_isShared_2850_ = v_isSharedCheck_2870_;
goto v_resetjp_2848_;
}
v_resetjp_2848_:
{
lean_object* v_k_2851_; lean_object* v_v_2852_; lean_object* v___x_2854_; uint8_t v_isShared_2855_; uint8_t v_isSharedCheck_2866_; 
v_k_2851_ = lean_ctor_get(v_l_2826_, 1);
v_v_2852_ = lean_ctor_get(v_l_2826_, 2);
v_isSharedCheck_2866_ = !lean_is_exclusive(v_l_2826_);
if (v_isSharedCheck_2866_ == 0)
{
lean_object* v_unused_2867_; lean_object* v_unused_2868_; lean_object* v_unused_2869_; 
v_unused_2867_ = lean_ctor_get(v_l_2826_, 4);
lean_dec(v_unused_2867_);
v_unused_2868_ = lean_ctor_get(v_l_2826_, 3);
lean_dec(v_unused_2868_);
v_unused_2869_ = lean_ctor_get(v_l_2826_, 0);
lean_dec(v_unused_2869_);
v___x_2854_ = v_l_2826_;
v_isShared_2855_ = v_isSharedCheck_2866_;
goto v_resetjp_2853_;
}
else
{
lean_inc(v_v_2852_);
lean_inc(v_k_2851_);
lean_dec(v_l_2826_);
v___x_2854_ = lean_box(0);
v_isShared_2855_ = v_isSharedCheck_2866_;
goto v_resetjp_2853_;
}
v_resetjp_2853_:
{
lean_object* v___x_2856_; lean_object* v___x_2858_; 
v___x_2856_ = lean_unsigned_to_nat(3u);
if (v_isShared_2855_ == 0)
{
lean_ctor_set(v___x_2854_, 4, v_r_2827_);
lean_ctor_set(v___x_2854_, 3, v_r_2827_);
lean_ctor_set(v___x_2854_, 2, v_v_2729_);
lean_ctor_set(v___x_2854_, 1, v_k_2728_);
lean_ctor_set(v___x_2854_, 0, v___x_2737_);
v___x_2858_ = v___x_2854_;
goto v_reusejp_2857_;
}
else
{
lean_object* v_reuseFailAlloc_2865_; 
v_reuseFailAlloc_2865_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2865_, 0, v___x_2737_);
lean_ctor_set(v_reuseFailAlloc_2865_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2865_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2865_, 3, v_r_2827_);
lean_ctor_set(v_reuseFailAlloc_2865_, 4, v_r_2827_);
v___x_2858_ = v_reuseFailAlloc_2865_;
goto v_reusejp_2857_;
}
v_reusejp_2857_:
{
lean_object* v___x_2860_; 
if (v_isShared_2850_ == 0)
{
lean_ctor_set(v___x_2849_, 3, v_r_2827_);
lean_ctor_set(v___x_2849_, 0, v___x_2737_);
v___x_2860_ = v___x_2849_;
goto v_reusejp_2859_;
}
else
{
lean_object* v_reuseFailAlloc_2864_; 
v_reuseFailAlloc_2864_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2864_, 0, v___x_2737_);
lean_ctor_set(v_reuseFailAlloc_2864_, 1, v_k_2846_);
lean_ctor_set(v_reuseFailAlloc_2864_, 2, v_v_2847_);
lean_ctor_set(v_reuseFailAlloc_2864_, 3, v_r_2827_);
lean_ctor_set(v_reuseFailAlloc_2864_, 4, v_r_2827_);
v___x_2860_ = v_reuseFailAlloc_2864_;
goto v_reusejp_2859_;
}
v_reusejp_2859_:
{
lean_object* v___x_2862_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v___x_2860_);
lean_ctor_set(v___x_2733_, 3, v___x_2858_);
lean_ctor_set(v___x_2733_, 2, v_v_2852_);
lean_ctor_set(v___x_2733_, 1, v_k_2851_);
lean_ctor_set(v___x_2733_, 0, v___x_2856_);
v___x_2862_ = v___x_2733_;
goto v_reusejp_2861_;
}
else
{
lean_object* v_reuseFailAlloc_2863_; 
v_reuseFailAlloc_2863_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2863_, 0, v___x_2856_);
lean_ctor_set(v_reuseFailAlloc_2863_, 1, v_k_2851_);
lean_ctor_set(v_reuseFailAlloc_2863_, 2, v_v_2852_);
lean_ctor_set(v_reuseFailAlloc_2863_, 3, v___x_2858_);
lean_ctor_set(v_reuseFailAlloc_2863_, 4, v___x_2860_);
v___x_2862_ = v_reuseFailAlloc_2863_;
goto v_reusejp_2861_;
}
v_reusejp_2861_:
{
return v___x_2862_;
}
}
}
}
}
}
}
else
{
lean_object* v_r_2874_; 
v_r_2874_ = lean_ctor_get(v_r_2731_, 4);
lean_inc(v_r_2874_);
if (lean_obj_tag(v_r_2874_) == 0)
{
lean_object* v_k_2875_; lean_object* v_v_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_2887_; 
v_k_2875_ = lean_ctor_get(v_r_2731_, 1);
v_v_2876_ = lean_ctor_get(v_r_2731_, 2);
v_isSharedCheck_2887_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2887_ == 0)
{
lean_object* v_unused_2888_; lean_object* v_unused_2889_; lean_object* v_unused_2890_; 
v_unused_2888_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2888_);
v_unused_2889_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2889_);
v_unused_2890_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_2890_);
v___x_2878_ = v_r_2731_;
v_isShared_2879_ = v_isSharedCheck_2887_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_v_2876_);
lean_inc(v_k_2875_);
lean_dec(v_r_2731_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_2887_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
lean_object* v___x_2880_; lean_object* v___x_2882_; 
v___x_2880_ = lean_unsigned_to_nat(3u);
if (v_isShared_2879_ == 0)
{
lean_ctor_set(v___x_2878_, 4, v_l_2826_);
lean_ctor_set(v___x_2878_, 2, v_v_2729_);
lean_ctor_set(v___x_2878_, 1, v_k_2728_);
lean_ctor_set(v___x_2878_, 0, v___x_2737_);
v___x_2882_ = v___x_2878_;
goto v_reusejp_2881_;
}
else
{
lean_object* v_reuseFailAlloc_2886_; 
v_reuseFailAlloc_2886_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2886_, 0, v___x_2737_);
lean_ctor_set(v_reuseFailAlloc_2886_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2886_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2886_, 3, v_l_2826_);
lean_ctor_set(v_reuseFailAlloc_2886_, 4, v_l_2826_);
v___x_2882_ = v_reuseFailAlloc_2886_;
goto v_reusejp_2881_;
}
v_reusejp_2881_:
{
lean_object* v___x_2884_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_r_2874_);
lean_ctor_set(v___x_2733_, 3, v___x_2882_);
lean_ctor_set(v___x_2733_, 2, v_v_2876_);
lean_ctor_set(v___x_2733_, 1, v_k_2875_);
lean_ctor_set(v___x_2733_, 0, v___x_2880_);
v___x_2884_ = v___x_2733_;
goto v_reusejp_2883_;
}
else
{
lean_object* v_reuseFailAlloc_2885_; 
v_reuseFailAlloc_2885_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2885_, 0, v___x_2880_);
lean_ctor_set(v_reuseFailAlloc_2885_, 1, v_k_2875_);
lean_ctor_set(v_reuseFailAlloc_2885_, 2, v_v_2876_);
lean_ctor_set(v_reuseFailAlloc_2885_, 3, v___x_2882_);
lean_ctor_set(v_reuseFailAlloc_2885_, 4, v_r_2874_);
v___x_2884_ = v_reuseFailAlloc_2885_;
goto v_reusejp_2883_;
}
v_reusejp_2883_:
{
return v___x_2884_;
}
}
}
}
else
{
lean_object* v_size_2891_; lean_object* v_k_2892_; lean_object* v_v_2893_; lean_object* v___x_2895_; uint8_t v_isShared_2896_; uint8_t v_isSharedCheck_2904_; 
v_size_2891_ = lean_ctor_get(v_r_2731_, 0);
v_k_2892_ = lean_ctor_get(v_r_2731_, 1);
v_v_2893_ = lean_ctor_get(v_r_2731_, 2);
v_isSharedCheck_2904_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2904_ == 0)
{
lean_object* v_unused_2905_; lean_object* v_unused_2906_; 
v_unused_2905_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2905_);
v_unused_2906_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2906_);
v___x_2895_ = v_r_2731_;
v_isShared_2896_ = v_isSharedCheck_2904_;
goto v_resetjp_2894_;
}
else
{
lean_inc(v_v_2893_);
lean_inc(v_k_2892_);
lean_inc(v_size_2891_);
lean_dec(v_r_2731_);
v___x_2895_ = lean_box(0);
v_isShared_2896_ = v_isSharedCheck_2904_;
goto v_resetjp_2894_;
}
v_resetjp_2894_:
{
lean_object* v___x_2898_; 
if (v_isShared_2896_ == 0)
{
lean_ctor_set(v___x_2895_, 3, v_r_2874_);
v___x_2898_ = v___x_2895_;
goto v_reusejp_2897_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v_size_2891_);
lean_ctor_set(v_reuseFailAlloc_2903_, 1, v_k_2892_);
lean_ctor_set(v_reuseFailAlloc_2903_, 2, v_v_2893_);
lean_ctor_set(v_reuseFailAlloc_2903_, 3, v_r_2874_);
lean_ctor_set(v_reuseFailAlloc_2903_, 4, v_r_2874_);
v___x_2898_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2897_;
}
v_reusejp_2897_:
{
lean_object* v___x_2899_; lean_object* v___x_2901_; 
v___x_2899_ = lean_unsigned_to_nat(2u);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v___x_2898_);
lean_ctor_set(v___x_2733_, 3, v_r_2874_);
lean_ctor_set(v___x_2733_, 0, v___x_2899_);
v___x_2901_ = v___x_2733_;
goto v_reusejp_2900_;
}
else
{
lean_object* v_reuseFailAlloc_2902_; 
v_reuseFailAlloc_2902_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2902_, 0, v___x_2899_);
lean_ctor_set(v_reuseFailAlloc_2902_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2902_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2902_, 3, v_r_2874_);
lean_ctor_set(v_reuseFailAlloc_2902_, 4, v___x_2898_);
v___x_2901_ = v_reuseFailAlloc_2902_;
goto v_reusejp_2900_;
}
v_reusejp_2900_:
{
return v___x_2901_;
}
}
}
}
}
}
else
{
lean_object* v___x_2908_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 3, v_r_2731_);
lean_ctor_set(v___x_2733_, 0, v___x_2737_);
v___x_2908_ = v___x_2733_;
goto v_reusejp_2907_;
}
else
{
lean_object* v_reuseFailAlloc_2909_; 
v_reuseFailAlloc_2909_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2909_, 0, v___x_2737_);
lean_ctor_set(v_reuseFailAlloc_2909_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_2909_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_2909_, 3, v_r_2731_);
lean_ctor_set(v_reuseFailAlloc_2909_, 4, v_r_2731_);
v___x_2908_ = v_reuseFailAlloc_2909_;
goto v_reusejp_2907_;
}
v_reusejp_2907_:
{
return v___x_2908_;
}
}
}
}
case 1:
{
lean_del_object(v___x_2733_);
lean_dec(v_v_2729_);
lean_dec(v_k_2728_);
if (lean_obj_tag(v_l_2730_) == 0)
{
if (lean_obj_tag(v_r_2731_) == 0)
{
lean_object* v_size_2910_; lean_object* v_k_2911_; lean_object* v_v_2912_; lean_object* v_l_2913_; lean_object* v_r_2914_; lean_object* v_size_2915_; lean_object* v_k_2916_; lean_object* v_v_2917_; lean_object* v_l_2918_; lean_object* v_r_2919_; lean_object* v___x_2920_; uint8_t v___x_2921_; 
v_size_2910_ = lean_ctor_get(v_l_2730_, 0);
v_k_2911_ = lean_ctor_get(v_l_2730_, 1);
v_v_2912_ = lean_ctor_get(v_l_2730_, 2);
v_l_2913_ = lean_ctor_get(v_l_2730_, 3);
v_r_2914_ = lean_ctor_get(v_l_2730_, 4);
lean_inc(v_r_2914_);
v_size_2915_ = lean_ctor_get(v_r_2731_, 0);
v_k_2916_ = lean_ctor_get(v_r_2731_, 1);
v_v_2917_ = lean_ctor_get(v_r_2731_, 2);
v_l_2918_ = lean_ctor_get(v_r_2731_, 3);
lean_inc(v_l_2918_);
v_r_2919_ = lean_ctor_get(v_r_2731_, 4);
v___x_2920_ = lean_unsigned_to_nat(1u);
v___x_2921_ = lean_nat_dec_lt(v_size_2910_, v_size_2915_);
if (v___x_2921_ == 0)
{
lean_object* v___x_2923_; uint8_t v_isShared_2924_; uint8_t v_isSharedCheck_3057_; 
lean_inc(v_l_2913_);
lean_inc(v_v_2912_);
lean_inc(v_k_2911_);
v_isSharedCheck_3057_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3057_ == 0)
{
lean_object* v_unused_3058_; lean_object* v_unused_3059_; lean_object* v_unused_3060_; lean_object* v_unused_3061_; lean_object* v_unused_3062_; 
v_unused_3058_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3058_);
v_unused_3059_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3059_);
v_unused_3060_ = lean_ctor_get(v_l_2730_, 2);
lean_dec(v_unused_3060_);
v_unused_3061_ = lean_ctor_get(v_l_2730_, 1);
lean_dec(v_unused_3061_);
v_unused_3062_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3062_);
v___x_2923_ = v_l_2730_;
v_isShared_2924_ = v_isSharedCheck_3057_;
goto v_resetjp_2922_;
}
else
{
lean_dec(v_l_2730_);
v___x_2923_ = lean_box(0);
v_isShared_2924_ = v_isSharedCheck_3057_;
goto v_resetjp_2922_;
}
v_resetjp_2922_:
{
lean_object* v___x_2925_; lean_object* v_tree_2926_; 
v___x_2925_ = l_Std_DTreeMap_Internal_Impl_maxView___redArg(v_k_2911_, v_v_2912_, v_l_2913_, v_r_2914_);
v_tree_2926_ = lean_ctor_get(v___x_2925_, 2);
lean_inc(v_tree_2926_);
if (lean_obj_tag(v_tree_2926_) == 0)
{
lean_object* v_k_2927_; lean_object* v_v_2928_; lean_object* v_size_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; uint8_t v___x_2932_; 
v_k_2927_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_k_2927_);
v_v_2928_ = lean_ctor_get(v___x_2925_, 1);
lean_inc(v_v_2928_);
lean_dec_ref(v___x_2925_);
v_size_2929_ = lean_ctor_get(v_tree_2926_, 0);
v___x_2930_ = lean_unsigned_to_nat(3u);
v___x_2931_ = lean_nat_mul(v___x_2930_, v_size_2929_);
v___x_2932_ = lean_nat_dec_lt(v___x_2931_, v_size_2915_);
lean_dec(v___x_2931_);
if (v___x_2932_ == 0)
{
lean_object* v___x_2933_; lean_object* v___x_2934_; lean_object* v___x_2936_; 
lean_dec(v_l_2918_);
v___x_2933_ = lean_nat_add(v___x_2920_, v_size_2929_);
v___x_2934_ = lean_nat_add(v___x_2933_, v_size_2915_);
lean_dec(v___x_2933_);
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v_r_2731_);
lean_ctor_set(v___x_2923_, 3, v_tree_2926_);
lean_ctor_set(v___x_2923_, 2, v_v_2928_);
lean_ctor_set(v___x_2923_, 1, v_k_2927_);
lean_ctor_set(v___x_2923_, 0, v___x_2934_);
v___x_2936_ = v___x_2923_;
goto v_reusejp_2935_;
}
else
{
lean_object* v_reuseFailAlloc_2937_; 
v_reuseFailAlloc_2937_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2937_, 0, v___x_2934_);
lean_ctor_set(v_reuseFailAlloc_2937_, 1, v_k_2927_);
lean_ctor_set(v_reuseFailAlloc_2937_, 2, v_v_2928_);
lean_ctor_set(v_reuseFailAlloc_2937_, 3, v_tree_2926_);
lean_ctor_set(v_reuseFailAlloc_2937_, 4, v_r_2731_);
v___x_2936_ = v_reuseFailAlloc_2937_;
goto v_reusejp_2935_;
}
v_reusejp_2935_:
{
return v___x_2936_;
}
}
else
{
lean_object* v___x_2939_; uint8_t v_isShared_2940_; uint8_t v_isSharedCheck_2992_; 
lean_inc(v_r_2919_);
lean_inc(v_v_2917_);
lean_inc(v_k_2916_);
lean_inc(v_size_2915_);
v_isSharedCheck_2992_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_2992_ == 0)
{
lean_object* v_unused_2993_; lean_object* v_unused_2994_; lean_object* v_unused_2995_; lean_object* v_unused_2996_; lean_object* v_unused_2997_; 
v_unused_2993_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_2993_);
v_unused_2994_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_2994_);
v_unused_2995_ = lean_ctor_get(v_r_2731_, 2);
lean_dec(v_unused_2995_);
v_unused_2996_ = lean_ctor_get(v_r_2731_, 1);
lean_dec(v_unused_2996_);
v_unused_2997_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_2997_);
v___x_2939_ = v_r_2731_;
v_isShared_2940_ = v_isSharedCheck_2992_;
goto v_resetjp_2938_;
}
else
{
lean_dec(v_r_2731_);
v___x_2939_ = lean_box(0);
v_isShared_2940_ = v_isSharedCheck_2992_;
goto v_resetjp_2938_;
}
v_resetjp_2938_:
{
lean_object* v_size_2941_; lean_object* v_k_2942_; lean_object* v_v_2943_; lean_object* v_l_2944_; lean_object* v_r_2945_; lean_object* v_size_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; uint8_t v___x_2949_; 
v_size_2941_ = lean_ctor_get(v_l_2918_, 0);
v_k_2942_ = lean_ctor_get(v_l_2918_, 1);
v_v_2943_ = lean_ctor_get(v_l_2918_, 2);
v_l_2944_ = lean_ctor_get(v_l_2918_, 3);
v_r_2945_ = lean_ctor_get(v_l_2918_, 4);
v_size_2946_ = lean_ctor_get(v_r_2919_, 0);
v___x_2947_ = lean_unsigned_to_nat(2u);
v___x_2948_ = lean_nat_mul(v___x_2947_, v_size_2946_);
v___x_2949_ = lean_nat_dec_lt(v_size_2941_, v___x_2948_);
lean_dec(v___x_2948_);
if (v___x_2949_ == 0)
{
lean_object* v___x_2951_; uint8_t v_isShared_2952_; uint8_t v_isSharedCheck_2977_; 
lean_inc(v_r_2945_);
lean_inc(v_l_2944_);
lean_inc(v_v_2943_);
lean_inc(v_k_2942_);
v_isSharedCheck_2977_ = !lean_is_exclusive(v_l_2918_);
if (v_isSharedCheck_2977_ == 0)
{
lean_object* v_unused_2978_; lean_object* v_unused_2979_; lean_object* v_unused_2980_; lean_object* v_unused_2981_; lean_object* v_unused_2982_; 
v_unused_2978_ = lean_ctor_get(v_l_2918_, 4);
lean_dec(v_unused_2978_);
v_unused_2979_ = lean_ctor_get(v_l_2918_, 3);
lean_dec(v_unused_2979_);
v_unused_2980_ = lean_ctor_get(v_l_2918_, 2);
lean_dec(v_unused_2980_);
v_unused_2981_ = lean_ctor_get(v_l_2918_, 1);
lean_dec(v_unused_2981_);
v_unused_2982_ = lean_ctor_get(v_l_2918_, 0);
lean_dec(v_unused_2982_);
v___x_2951_ = v_l_2918_;
v_isShared_2952_ = v_isSharedCheck_2977_;
goto v_resetjp_2950_;
}
else
{
lean_dec(v_l_2918_);
v___x_2951_ = lean_box(0);
v_isShared_2952_ = v_isSharedCheck_2977_;
goto v_resetjp_2950_;
}
v_resetjp_2950_:
{
lean_object* v___x_2953_; lean_object* v___x_2954_; lean_object* v___y_2956_; lean_object* v___y_2957_; lean_object* v___y_2958_; lean_object* v___y_2967_; 
v___x_2953_ = lean_nat_add(v___x_2920_, v_size_2929_);
v___x_2954_ = lean_nat_add(v___x_2953_, v_size_2915_);
lean_dec(v_size_2915_);
if (lean_obj_tag(v_l_2944_) == 0)
{
lean_object* v_size_2975_; 
v_size_2975_ = lean_ctor_get(v_l_2944_, 0);
lean_inc(v_size_2975_);
v___y_2967_ = v_size_2975_;
goto v___jp_2966_;
}
else
{
lean_object* v___x_2976_; 
v___x_2976_ = lean_unsigned_to_nat(0u);
v___y_2967_ = v___x_2976_;
goto v___jp_2966_;
}
v___jp_2955_:
{
lean_object* v___x_2959_; lean_object* v___x_2961_; 
v___x_2959_ = lean_nat_add(v___y_2956_, v___y_2958_);
lean_dec(v___y_2958_);
lean_dec(v___y_2956_);
if (v_isShared_2952_ == 0)
{
lean_ctor_set(v___x_2951_, 4, v_r_2919_);
lean_ctor_set(v___x_2951_, 3, v_r_2945_);
lean_ctor_set(v___x_2951_, 2, v_v_2917_);
lean_ctor_set(v___x_2951_, 1, v_k_2916_);
lean_ctor_set(v___x_2951_, 0, v___x_2959_);
v___x_2961_ = v___x_2951_;
goto v_reusejp_2960_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v___x_2959_);
lean_ctor_set(v_reuseFailAlloc_2965_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_2965_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_2965_, 3, v_r_2945_);
lean_ctor_set(v_reuseFailAlloc_2965_, 4, v_r_2919_);
v___x_2961_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2960_;
}
v_reusejp_2960_:
{
lean_object* v___x_2963_; 
if (v_isShared_2940_ == 0)
{
lean_ctor_set(v___x_2939_, 4, v___x_2961_);
lean_ctor_set(v___x_2939_, 3, v___y_2957_);
lean_ctor_set(v___x_2939_, 2, v_v_2943_);
lean_ctor_set(v___x_2939_, 1, v_k_2942_);
lean_ctor_set(v___x_2939_, 0, v___x_2954_);
v___x_2963_ = v___x_2939_;
goto v_reusejp_2962_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v___x_2954_);
lean_ctor_set(v_reuseFailAlloc_2964_, 1, v_k_2942_);
lean_ctor_set(v_reuseFailAlloc_2964_, 2, v_v_2943_);
lean_ctor_set(v_reuseFailAlloc_2964_, 3, v___y_2957_);
lean_ctor_set(v_reuseFailAlloc_2964_, 4, v___x_2961_);
v___x_2963_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2962_;
}
v_reusejp_2962_:
{
return v___x_2963_;
}
}
}
v___jp_2966_:
{
lean_object* v___x_2968_; lean_object* v___x_2970_; 
v___x_2968_ = lean_nat_add(v___x_2953_, v___y_2967_);
lean_dec(v___y_2967_);
lean_dec(v___x_2953_);
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v_l_2944_);
lean_ctor_set(v___x_2923_, 3, v_tree_2926_);
lean_ctor_set(v___x_2923_, 2, v_v_2928_);
lean_ctor_set(v___x_2923_, 1, v_k_2927_);
lean_ctor_set(v___x_2923_, 0, v___x_2968_);
v___x_2970_ = v___x_2923_;
goto v_reusejp_2969_;
}
else
{
lean_object* v_reuseFailAlloc_2974_; 
v_reuseFailAlloc_2974_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2974_, 0, v___x_2968_);
lean_ctor_set(v_reuseFailAlloc_2974_, 1, v_k_2927_);
lean_ctor_set(v_reuseFailAlloc_2974_, 2, v_v_2928_);
lean_ctor_set(v_reuseFailAlloc_2974_, 3, v_tree_2926_);
lean_ctor_set(v_reuseFailAlloc_2974_, 4, v_l_2944_);
v___x_2970_ = v_reuseFailAlloc_2974_;
goto v_reusejp_2969_;
}
v_reusejp_2969_:
{
lean_object* v___x_2971_; 
v___x_2971_ = lean_nat_add(v___x_2920_, v_size_2946_);
if (lean_obj_tag(v_r_2945_) == 0)
{
lean_object* v_size_2972_; 
v_size_2972_ = lean_ctor_get(v_r_2945_, 0);
lean_inc(v_size_2972_);
v___y_2956_ = v___x_2971_;
v___y_2957_ = v___x_2970_;
v___y_2958_ = v_size_2972_;
goto v___jp_2955_;
}
else
{
lean_object* v___x_2973_; 
v___x_2973_ = lean_unsigned_to_nat(0u);
v___y_2956_ = v___x_2971_;
v___y_2957_ = v___x_2970_;
v___y_2958_ = v___x_2973_;
goto v___jp_2955_;
}
}
}
}
}
else
{
lean_object* v___x_2983_; lean_object* v___x_2984_; lean_object* v___x_2985_; lean_object* v___x_2987_; 
v___x_2983_ = lean_nat_add(v___x_2920_, v_size_2929_);
v___x_2984_ = lean_nat_add(v___x_2983_, v_size_2915_);
lean_dec(v_size_2915_);
v___x_2985_ = lean_nat_add(v___x_2983_, v_size_2941_);
lean_dec(v___x_2983_);
if (v_isShared_2940_ == 0)
{
lean_ctor_set(v___x_2939_, 4, v_l_2918_);
lean_ctor_set(v___x_2939_, 3, v_tree_2926_);
lean_ctor_set(v___x_2939_, 2, v_v_2928_);
lean_ctor_set(v___x_2939_, 1, v_k_2927_);
lean_ctor_set(v___x_2939_, 0, v___x_2985_);
v___x_2987_ = v___x_2939_;
goto v_reusejp_2986_;
}
else
{
lean_object* v_reuseFailAlloc_2991_; 
v_reuseFailAlloc_2991_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2991_, 0, v___x_2985_);
lean_ctor_set(v_reuseFailAlloc_2991_, 1, v_k_2927_);
lean_ctor_set(v_reuseFailAlloc_2991_, 2, v_v_2928_);
lean_ctor_set(v_reuseFailAlloc_2991_, 3, v_tree_2926_);
lean_ctor_set(v_reuseFailAlloc_2991_, 4, v_l_2918_);
v___x_2987_ = v_reuseFailAlloc_2991_;
goto v_reusejp_2986_;
}
v_reusejp_2986_:
{
lean_object* v___x_2989_; 
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v_r_2919_);
lean_ctor_set(v___x_2923_, 3, v___x_2987_);
lean_ctor_set(v___x_2923_, 2, v_v_2917_);
lean_ctor_set(v___x_2923_, 1, v_k_2916_);
lean_ctor_set(v___x_2923_, 0, v___x_2984_);
v___x_2989_ = v___x_2923_;
goto v_reusejp_2988_;
}
else
{
lean_object* v_reuseFailAlloc_2990_; 
v_reuseFailAlloc_2990_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2990_, 0, v___x_2984_);
lean_ctor_set(v_reuseFailAlloc_2990_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_2990_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_2990_, 3, v___x_2987_);
lean_ctor_set(v_reuseFailAlloc_2990_, 4, v_r_2919_);
v___x_2989_ = v_reuseFailAlloc_2990_;
goto v_reusejp_2988_;
}
v_reusejp_2988_:
{
return v___x_2989_;
}
}
}
}
}
}
else
{
lean_object* v___x_2999_; uint8_t v_isShared_3000_; uint8_t v_isSharedCheck_3051_; 
lean_inc(v_r_2919_);
lean_inc(v_v_2917_);
lean_inc(v_k_2916_);
lean_inc(v_size_2915_);
v_isSharedCheck_3051_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_3051_ == 0)
{
lean_object* v_unused_3052_; lean_object* v_unused_3053_; lean_object* v_unused_3054_; lean_object* v_unused_3055_; lean_object* v_unused_3056_; 
v_unused_3052_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_3052_);
v_unused_3053_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_3053_);
v_unused_3054_ = lean_ctor_get(v_r_2731_, 2);
lean_dec(v_unused_3054_);
v_unused_3055_ = lean_ctor_get(v_r_2731_, 1);
lean_dec(v_unused_3055_);
v_unused_3056_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_3056_);
v___x_2999_ = v_r_2731_;
v_isShared_3000_ = v_isSharedCheck_3051_;
goto v_resetjp_2998_;
}
else
{
lean_dec(v_r_2731_);
v___x_2999_ = lean_box(0);
v_isShared_3000_ = v_isSharedCheck_3051_;
goto v_resetjp_2998_;
}
v_resetjp_2998_:
{
if (lean_obj_tag(v_l_2918_) == 0)
{
if (lean_obj_tag(v_r_2919_) == 0)
{
lean_object* v_k_3001_; lean_object* v_v_3002_; lean_object* v_size_3003_; lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3007_; 
v_k_3001_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_k_3001_);
v_v_3002_ = lean_ctor_get(v___x_2925_, 1);
lean_inc(v_v_3002_);
lean_dec_ref(v___x_2925_);
v_size_3003_ = lean_ctor_get(v_l_2918_, 0);
v___x_3004_ = lean_nat_add(v___x_2920_, v_size_2915_);
lean_dec(v_size_2915_);
v___x_3005_ = lean_nat_add(v___x_2920_, v_size_3003_);
if (v_isShared_3000_ == 0)
{
lean_ctor_set(v___x_2999_, 4, v_l_2918_);
lean_ctor_set(v___x_2999_, 3, v_tree_2926_);
lean_ctor_set(v___x_2999_, 2, v_v_3002_);
lean_ctor_set(v___x_2999_, 1, v_k_3001_);
lean_ctor_set(v___x_2999_, 0, v___x_3005_);
v___x_3007_ = v___x_2999_;
goto v_reusejp_3006_;
}
else
{
lean_object* v_reuseFailAlloc_3011_; 
v_reuseFailAlloc_3011_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3011_, 0, v___x_3005_);
lean_ctor_set(v_reuseFailAlloc_3011_, 1, v_k_3001_);
lean_ctor_set(v_reuseFailAlloc_3011_, 2, v_v_3002_);
lean_ctor_set(v_reuseFailAlloc_3011_, 3, v_tree_2926_);
lean_ctor_set(v_reuseFailAlloc_3011_, 4, v_l_2918_);
v___x_3007_ = v_reuseFailAlloc_3011_;
goto v_reusejp_3006_;
}
v_reusejp_3006_:
{
lean_object* v___x_3009_; 
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v_r_2919_);
lean_ctor_set(v___x_2923_, 3, v___x_3007_);
lean_ctor_set(v___x_2923_, 2, v_v_2917_);
lean_ctor_set(v___x_2923_, 1, v_k_2916_);
lean_ctor_set(v___x_2923_, 0, v___x_3004_);
v___x_3009_ = v___x_2923_;
goto v_reusejp_3008_;
}
else
{
lean_object* v_reuseFailAlloc_3010_; 
v_reuseFailAlloc_3010_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3010_, 0, v___x_3004_);
lean_ctor_set(v_reuseFailAlloc_3010_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_3010_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_3010_, 3, v___x_3007_);
lean_ctor_set(v_reuseFailAlloc_3010_, 4, v_r_2919_);
v___x_3009_ = v_reuseFailAlloc_3010_;
goto v_reusejp_3008_;
}
v_reusejp_3008_:
{
return v___x_3009_;
}
}
}
else
{
lean_object* v_k_3012_; lean_object* v_v_3013_; lean_object* v_k_3014_; lean_object* v_v_3015_; lean_object* v___x_3017_; uint8_t v_isShared_3018_; uint8_t v_isSharedCheck_3029_; 
lean_dec(v_size_2915_);
v_k_3012_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_k_3012_);
v_v_3013_ = lean_ctor_get(v___x_2925_, 1);
lean_inc(v_v_3013_);
lean_dec_ref(v___x_2925_);
v_k_3014_ = lean_ctor_get(v_l_2918_, 1);
v_v_3015_ = lean_ctor_get(v_l_2918_, 2);
v_isSharedCheck_3029_ = !lean_is_exclusive(v_l_2918_);
if (v_isSharedCheck_3029_ == 0)
{
lean_object* v_unused_3030_; lean_object* v_unused_3031_; lean_object* v_unused_3032_; 
v_unused_3030_ = lean_ctor_get(v_l_2918_, 4);
lean_dec(v_unused_3030_);
v_unused_3031_ = lean_ctor_get(v_l_2918_, 3);
lean_dec(v_unused_3031_);
v_unused_3032_ = lean_ctor_get(v_l_2918_, 0);
lean_dec(v_unused_3032_);
v___x_3017_ = v_l_2918_;
v_isShared_3018_ = v_isSharedCheck_3029_;
goto v_resetjp_3016_;
}
else
{
lean_inc(v_v_3015_);
lean_inc(v_k_3014_);
lean_dec(v_l_2918_);
v___x_3017_ = lean_box(0);
v_isShared_3018_ = v_isSharedCheck_3029_;
goto v_resetjp_3016_;
}
v_resetjp_3016_:
{
lean_object* v___x_3019_; lean_object* v___x_3021_; 
v___x_3019_ = lean_unsigned_to_nat(3u);
if (v_isShared_3018_ == 0)
{
lean_ctor_set(v___x_3017_, 4, v_r_2919_);
lean_ctor_set(v___x_3017_, 3, v_r_2919_);
lean_ctor_set(v___x_3017_, 2, v_v_3013_);
lean_ctor_set(v___x_3017_, 1, v_k_3012_);
lean_ctor_set(v___x_3017_, 0, v___x_2920_);
v___x_3021_ = v___x_3017_;
goto v_reusejp_3020_;
}
else
{
lean_object* v_reuseFailAlloc_3028_; 
v_reuseFailAlloc_3028_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3028_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3028_, 1, v_k_3012_);
lean_ctor_set(v_reuseFailAlloc_3028_, 2, v_v_3013_);
lean_ctor_set(v_reuseFailAlloc_3028_, 3, v_r_2919_);
lean_ctor_set(v_reuseFailAlloc_3028_, 4, v_r_2919_);
v___x_3021_ = v_reuseFailAlloc_3028_;
goto v_reusejp_3020_;
}
v_reusejp_3020_:
{
lean_object* v___x_3023_; 
if (v_isShared_3000_ == 0)
{
lean_ctor_set(v___x_2999_, 3, v_r_2919_);
lean_ctor_set(v___x_2999_, 0, v___x_2920_);
v___x_3023_ = v___x_2999_;
goto v_reusejp_3022_;
}
else
{
lean_object* v_reuseFailAlloc_3027_; 
v_reuseFailAlloc_3027_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3027_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3027_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_3027_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_3027_, 3, v_r_2919_);
lean_ctor_set(v_reuseFailAlloc_3027_, 4, v_r_2919_);
v___x_3023_ = v_reuseFailAlloc_3027_;
goto v_reusejp_3022_;
}
v_reusejp_3022_:
{
lean_object* v___x_3025_; 
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v___x_3023_);
lean_ctor_set(v___x_2923_, 3, v___x_3021_);
lean_ctor_set(v___x_2923_, 2, v_v_3015_);
lean_ctor_set(v___x_2923_, 1, v_k_3014_);
lean_ctor_set(v___x_2923_, 0, v___x_3019_);
v___x_3025_ = v___x_2923_;
goto v_reusejp_3024_;
}
else
{
lean_object* v_reuseFailAlloc_3026_; 
v_reuseFailAlloc_3026_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3026_, 0, v___x_3019_);
lean_ctor_set(v_reuseFailAlloc_3026_, 1, v_k_3014_);
lean_ctor_set(v_reuseFailAlloc_3026_, 2, v_v_3015_);
lean_ctor_set(v_reuseFailAlloc_3026_, 3, v___x_3021_);
lean_ctor_set(v_reuseFailAlloc_3026_, 4, v___x_3023_);
v___x_3025_ = v_reuseFailAlloc_3026_;
goto v_reusejp_3024_;
}
v_reusejp_3024_:
{
return v___x_3025_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_2919_) == 0)
{
lean_object* v_k_3033_; lean_object* v_v_3034_; lean_object* v___x_3035_; lean_object* v___x_3037_; 
lean_dec(v_size_2915_);
v_k_3033_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_k_3033_);
v_v_3034_ = lean_ctor_get(v___x_2925_, 1);
lean_inc(v_v_3034_);
lean_dec_ref(v___x_2925_);
v___x_3035_ = lean_unsigned_to_nat(3u);
if (v_isShared_3000_ == 0)
{
lean_ctor_set(v___x_2999_, 4, v_l_2918_);
lean_ctor_set(v___x_2999_, 2, v_v_3034_);
lean_ctor_set(v___x_2999_, 1, v_k_3033_);
lean_ctor_set(v___x_2999_, 0, v___x_2920_);
v___x_3037_ = v___x_2999_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3041_; 
v_reuseFailAlloc_3041_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3041_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3041_, 1, v_k_3033_);
lean_ctor_set(v_reuseFailAlloc_3041_, 2, v_v_3034_);
lean_ctor_set(v_reuseFailAlloc_3041_, 3, v_l_2918_);
lean_ctor_set(v_reuseFailAlloc_3041_, 4, v_l_2918_);
v___x_3037_ = v_reuseFailAlloc_3041_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
lean_object* v___x_3039_; 
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v_r_2919_);
lean_ctor_set(v___x_2923_, 3, v___x_3037_);
lean_ctor_set(v___x_2923_, 2, v_v_2917_);
lean_ctor_set(v___x_2923_, 1, v_k_2916_);
lean_ctor_set(v___x_2923_, 0, v___x_3035_);
v___x_3039_ = v___x_2923_;
goto v_reusejp_3038_;
}
else
{
lean_object* v_reuseFailAlloc_3040_; 
v_reuseFailAlloc_3040_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3040_, 0, v___x_3035_);
lean_ctor_set(v_reuseFailAlloc_3040_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_3040_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_3040_, 3, v___x_3037_);
lean_ctor_set(v_reuseFailAlloc_3040_, 4, v_r_2919_);
v___x_3039_ = v_reuseFailAlloc_3040_;
goto v_reusejp_3038_;
}
v_reusejp_3038_:
{
return v___x_3039_;
}
}
}
else
{
lean_object* v_k_3042_; lean_object* v_v_3043_; lean_object* v___x_3045_; 
v_k_3042_ = lean_ctor_get(v___x_2925_, 0);
lean_inc(v_k_3042_);
v_v_3043_ = lean_ctor_get(v___x_2925_, 1);
lean_inc(v_v_3043_);
lean_dec_ref(v___x_2925_);
if (v_isShared_3000_ == 0)
{
lean_ctor_set(v___x_2999_, 3, v_r_2919_);
v___x_3045_ = v___x_2999_;
goto v_reusejp_3044_;
}
else
{
lean_object* v_reuseFailAlloc_3050_; 
v_reuseFailAlloc_3050_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3050_, 0, v_size_2915_);
lean_ctor_set(v_reuseFailAlloc_3050_, 1, v_k_2916_);
lean_ctor_set(v_reuseFailAlloc_3050_, 2, v_v_2917_);
lean_ctor_set(v_reuseFailAlloc_3050_, 3, v_r_2919_);
lean_ctor_set(v_reuseFailAlloc_3050_, 4, v_r_2919_);
v___x_3045_ = v_reuseFailAlloc_3050_;
goto v_reusejp_3044_;
}
v_reusejp_3044_:
{
lean_object* v___x_3046_; lean_object* v___x_3048_; 
v___x_3046_ = lean_unsigned_to_nat(2u);
if (v_isShared_2924_ == 0)
{
lean_ctor_set(v___x_2923_, 4, v___x_3045_);
lean_ctor_set(v___x_2923_, 3, v_r_2919_);
lean_ctor_set(v___x_2923_, 2, v_v_3043_);
lean_ctor_set(v___x_2923_, 1, v_k_3042_);
lean_ctor_set(v___x_2923_, 0, v___x_3046_);
v___x_3048_ = v___x_2923_;
goto v_reusejp_3047_;
}
else
{
lean_object* v_reuseFailAlloc_3049_; 
v_reuseFailAlloc_3049_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3049_, 0, v___x_3046_);
lean_ctor_set(v_reuseFailAlloc_3049_, 1, v_k_3042_);
lean_ctor_set(v_reuseFailAlloc_3049_, 2, v_v_3043_);
lean_ctor_set(v_reuseFailAlloc_3049_, 3, v_r_2919_);
lean_ctor_set(v_reuseFailAlloc_3049_, 4, v___x_3045_);
v___x_3048_ = v_reuseFailAlloc_3049_;
goto v_reusejp_3047_;
}
v_reusejp_3047_:
{
return v___x_3048_;
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
lean_object* v___x_3064_; uint8_t v_isShared_3065_; uint8_t v_isSharedCheck_3215_; 
lean_inc(v_r_2919_);
lean_inc(v_v_2917_);
lean_inc(v_k_2916_);
v_isSharedCheck_3215_ = !lean_is_exclusive(v_r_2731_);
if (v_isSharedCheck_3215_ == 0)
{
lean_object* v_unused_3216_; lean_object* v_unused_3217_; lean_object* v_unused_3218_; lean_object* v_unused_3219_; lean_object* v_unused_3220_; 
v_unused_3216_ = lean_ctor_get(v_r_2731_, 4);
lean_dec(v_unused_3216_);
v_unused_3217_ = lean_ctor_get(v_r_2731_, 3);
lean_dec(v_unused_3217_);
v_unused_3218_ = lean_ctor_get(v_r_2731_, 2);
lean_dec(v_unused_3218_);
v_unused_3219_ = lean_ctor_get(v_r_2731_, 1);
lean_dec(v_unused_3219_);
v_unused_3220_ = lean_ctor_get(v_r_2731_, 0);
lean_dec(v_unused_3220_);
v___x_3064_ = v_r_2731_;
v_isShared_3065_ = v_isSharedCheck_3215_;
goto v_resetjp_3063_;
}
else
{
lean_dec(v_r_2731_);
v___x_3064_ = lean_box(0);
v_isShared_3065_ = v_isSharedCheck_3215_;
goto v_resetjp_3063_;
}
v_resetjp_3063_:
{
lean_object* v___x_3066_; lean_object* v_tree_3067_; 
v___x_3066_ = l_Std_DTreeMap_Internal_Impl_minView___redArg(v_k_2916_, v_v_2917_, v_l_2918_, v_r_2919_);
v_tree_3067_ = lean_ctor_get(v___x_3066_, 2);
lean_inc(v_tree_3067_);
if (lean_obj_tag(v_tree_3067_) == 0)
{
lean_object* v_k_3068_; lean_object* v_v_3069_; lean_object* v_size_3070_; lean_object* v___x_3071_; lean_object* v___x_3072_; uint8_t v___x_3073_; 
v_k_3068_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_k_3068_);
v_v_3069_ = lean_ctor_get(v___x_3066_, 1);
lean_inc(v_v_3069_);
lean_dec_ref(v___x_3066_);
v_size_3070_ = lean_ctor_get(v_tree_3067_, 0);
v___x_3071_ = lean_unsigned_to_nat(3u);
v___x_3072_ = lean_nat_mul(v___x_3071_, v_size_3070_);
v___x_3073_ = lean_nat_dec_lt(v___x_3072_, v_size_2910_);
lean_dec(v___x_3072_);
if (v___x_3073_ == 0)
{
lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3077_; 
lean_dec(v_r_2914_);
v___x_3074_ = lean_nat_add(v___x_2920_, v_size_2910_);
v___x_3075_ = lean_nat_add(v___x_3074_, v_size_3070_);
lean_dec(v___x_3074_);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_tree_3067_);
lean_ctor_set(v___x_3064_, 3, v_l_2730_);
lean_ctor_set(v___x_3064_, 2, v_v_3069_);
lean_ctor_set(v___x_3064_, 1, v_k_3068_);
lean_ctor_set(v___x_3064_, 0, v___x_3075_);
v___x_3077_ = v___x_3064_;
goto v_reusejp_3076_;
}
else
{
lean_object* v_reuseFailAlloc_3078_; 
v_reuseFailAlloc_3078_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3078_, 0, v___x_3075_);
lean_ctor_set(v_reuseFailAlloc_3078_, 1, v_k_3068_);
lean_ctor_set(v_reuseFailAlloc_3078_, 2, v_v_3069_);
lean_ctor_set(v_reuseFailAlloc_3078_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3078_, 4, v_tree_3067_);
v___x_3077_ = v_reuseFailAlloc_3078_;
goto v_reusejp_3076_;
}
v_reusejp_3076_:
{
return v___x_3077_;
}
}
else
{
lean_object* v___x_3080_; uint8_t v_isShared_3081_; uint8_t v_isSharedCheck_3144_; 
lean_inc(v_l_2913_);
lean_inc(v_v_2912_);
lean_inc(v_k_2911_);
lean_inc(v_size_2910_);
v_isSharedCheck_3144_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3144_ == 0)
{
lean_object* v_unused_3145_; lean_object* v_unused_3146_; lean_object* v_unused_3147_; lean_object* v_unused_3148_; lean_object* v_unused_3149_; 
v_unused_3145_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3145_);
v_unused_3146_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3146_);
v_unused_3147_ = lean_ctor_get(v_l_2730_, 2);
lean_dec(v_unused_3147_);
v_unused_3148_ = lean_ctor_get(v_l_2730_, 1);
lean_dec(v_unused_3148_);
v_unused_3149_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3149_);
v___x_3080_ = v_l_2730_;
v_isShared_3081_ = v_isSharedCheck_3144_;
goto v_resetjp_3079_;
}
else
{
lean_dec(v_l_2730_);
v___x_3080_ = lean_box(0);
v_isShared_3081_ = v_isSharedCheck_3144_;
goto v_resetjp_3079_;
}
v_resetjp_3079_:
{
lean_object* v_size_3082_; lean_object* v_size_3083_; lean_object* v_k_3084_; lean_object* v_v_3085_; lean_object* v_l_3086_; lean_object* v_r_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; uint8_t v___x_3090_; 
v_size_3082_ = lean_ctor_get(v_l_2913_, 0);
v_size_3083_ = lean_ctor_get(v_r_2914_, 0);
v_k_3084_ = lean_ctor_get(v_r_2914_, 1);
v_v_3085_ = lean_ctor_get(v_r_2914_, 2);
v_l_3086_ = lean_ctor_get(v_r_2914_, 3);
v_r_3087_ = lean_ctor_get(v_r_2914_, 4);
v___x_3088_ = lean_unsigned_to_nat(2u);
v___x_3089_ = lean_nat_mul(v___x_3088_, v_size_3082_);
v___x_3090_ = lean_nat_dec_lt(v_size_3083_, v___x_3089_);
lean_dec(v___x_3089_);
if (v___x_3090_ == 0)
{
lean_object* v___x_3092_; uint8_t v_isShared_3093_; uint8_t v_isSharedCheck_3128_; 
lean_inc(v_r_3087_);
lean_inc(v_l_3086_);
lean_inc(v_v_3085_);
lean_inc(v_k_3084_);
lean_del_object(v___x_3080_);
v_isSharedCheck_3128_ = !lean_is_exclusive(v_r_2914_);
if (v_isSharedCheck_3128_ == 0)
{
lean_object* v_unused_3129_; lean_object* v_unused_3130_; lean_object* v_unused_3131_; lean_object* v_unused_3132_; lean_object* v_unused_3133_; 
v_unused_3129_ = lean_ctor_get(v_r_2914_, 4);
lean_dec(v_unused_3129_);
v_unused_3130_ = lean_ctor_get(v_r_2914_, 3);
lean_dec(v_unused_3130_);
v_unused_3131_ = lean_ctor_get(v_r_2914_, 2);
lean_dec(v_unused_3131_);
v_unused_3132_ = lean_ctor_get(v_r_2914_, 1);
lean_dec(v_unused_3132_);
v_unused_3133_ = lean_ctor_get(v_r_2914_, 0);
lean_dec(v_unused_3133_);
v___x_3092_ = v_r_2914_;
v_isShared_3093_ = v_isSharedCheck_3128_;
goto v_resetjp_3091_;
}
else
{
lean_dec(v_r_2914_);
v___x_3092_ = lean_box(0);
v_isShared_3093_ = v_isSharedCheck_3128_;
goto v_resetjp_3091_;
}
v_resetjp_3091_:
{
lean_object* v___x_3094_; lean_object* v___x_3095_; lean_object* v___y_3097_; lean_object* v___y_3098_; lean_object* v___y_3099_; lean_object* v___x_3116_; lean_object* v___y_3118_; 
v___x_3094_ = lean_nat_add(v___x_2920_, v_size_2910_);
lean_dec(v_size_2910_);
v___x_3095_ = lean_nat_add(v___x_3094_, v_size_3070_);
lean_dec(v___x_3094_);
v___x_3116_ = lean_nat_add(v___x_2920_, v_size_3082_);
if (lean_obj_tag(v_l_3086_) == 0)
{
lean_object* v_size_3126_; 
v_size_3126_ = lean_ctor_get(v_l_3086_, 0);
lean_inc(v_size_3126_);
v___y_3118_ = v_size_3126_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3127_; 
v___x_3127_ = lean_unsigned_to_nat(0u);
v___y_3118_ = v___x_3127_;
goto v___jp_3117_;
}
v___jp_3096_:
{
lean_object* v___x_3100_; lean_object* v___x_3102_; 
v___x_3100_ = lean_nat_add(v___y_3097_, v___y_3099_);
lean_dec(v___y_3099_);
lean_dec(v___y_3097_);
lean_inc_ref(v_tree_3067_);
if (v_isShared_3093_ == 0)
{
lean_ctor_set(v___x_3092_, 4, v_tree_3067_);
lean_ctor_set(v___x_3092_, 3, v_r_3087_);
lean_ctor_set(v___x_3092_, 2, v_v_3069_);
lean_ctor_set(v___x_3092_, 1, v_k_3068_);
lean_ctor_set(v___x_3092_, 0, v___x_3100_);
v___x_3102_ = v___x_3092_;
goto v_reusejp_3101_;
}
else
{
lean_object* v_reuseFailAlloc_3115_; 
v_reuseFailAlloc_3115_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3115_, 0, v___x_3100_);
lean_ctor_set(v_reuseFailAlloc_3115_, 1, v_k_3068_);
lean_ctor_set(v_reuseFailAlloc_3115_, 2, v_v_3069_);
lean_ctor_set(v_reuseFailAlloc_3115_, 3, v_r_3087_);
lean_ctor_set(v_reuseFailAlloc_3115_, 4, v_tree_3067_);
v___x_3102_ = v_reuseFailAlloc_3115_;
goto v_reusejp_3101_;
}
v_reusejp_3101_:
{
lean_object* v___x_3104_; uint8_t v_isShared_3105_; uint8_t v_isSharedCheck_3109_; 
v_isSharedCheck_3109_ = !lean_is_exclusive(v_tree_3067_);
if (v_isSharedCheck_3109_ == 0)
{
lean_object* v_unused_3110_; lean_object* v_unused_3111_; lean_object* v_unused_3112_; lean_object* v_unused_3113_; lean_object* v_unused_3114_; 
v_unused_3110_ = lean_ctor_get(v_tree_3067_, 4);
lean_dec(v_unused_3110_);
v_unused_3111_ = lean_ctor_get(v_tree_3067_, 3);
lean_dec(v_unused_3111_);
v_unused_3112_ = lean_ctor_get(v_tree_3067_, 2);
lean_dec(v_unused_3112_);
v_unused_3113_ = lean_ctor_get(v_tree_3067_, 1);
lean_dec(v_unused_3113_);
v_unused_3114_ = lean_ctor_get(v_tree_3067_, 0);
lean_dec(v_unused_3114_);
v___x_3104_ = v_tree_3067_;
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
else
{
lean_dec(v_tree_3067_);
v___x_3104_ = lean_box(0);
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
v_resetjp_3103_:
{
lean_object* v___x_3107_; 
if (v_isShared_3105_ == 0)
{
lean_ctor_set(v___x_3104_, 4, v___x_3102_);
lean_ctor_set(v___x_3104_, 3, v___y_3098_);
lean_ctor_set(v___x_3104_, 2, v_v_3085_);
lean_ctor_set(v___x_3104_, 1, v_k_3084_);
lean_ctor_set(v___x_3104_, 0, v___x_3095_);
v___x_3107_ = v___x_3104_;
goto v_reusejp_3106_;
}
else
{
lean_object* v_reuseFailAlloc_3108_; 
v_reuseFailAlloc_3108_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3108_, 0, v___x_3095_);
lean_ctor_set(v_reuseFailAlloc_3108_, 1, v_k_3084_);
lean_ctor_set(v_reuseFailAlloc_3108_, 2, v_v_3085_);
lean_ctor_set(v_reuseFailAlloc_3108_, 3, v___y_3098_);
lean_ctor_set(v_reuseFailAlloc_3108_, 4, v___x_3102_);
v___x_3107_ = v_reuseFailAlloc_3108_;
goto v_reusejp_3106_;
}
v_reusejp_3106_:
{
return v___x_3107_;
}
}
}
}
v___jp_3117_:
{
lean_object* v___x_3119_; lean_object* v___x_3121_; 
v___x_3119_ = lean_nat_add(v___x_3116_, v___y_3118_);
lean_dec(v___y_3118_);
lean_dec(v___x_3116_);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_l_3086_);
lean_ctor_set(v___x_3064_, 3, v_l_2913_);
lean_ctor_set(v___x_3064_, 2, v_v_2912_);
lean_ctor_set(v___x_3064_, 1, v_k_2911_);
lean_ctor_set(v___x_3064_, 0, v___x_3119_);
v___x_3121_ = v___x_3064_;
goto v_reusejp_3120_;
}
else
{
lean_object* v_reuseFailAlloc_3125_; 
v_reuseFailAlloc_3125_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3125_, 0, v___x_3119_);
lean_ctor_set(v_reuseFailAlloc_3125_, 1, v_k_2911_);
lean_ctor_set(v_reuseFailAlloc_3125_, 2, v_v_2912_);
lean_ctor_set(v_reuseFailAlloc_3125_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3125_, 4, v_l_3086_);
v___x_3121_ = v_reuseFailAlloc_3125_;
goto v_reusejp_3120_;
}
v_reusejp_3120_:
{
lean_object* v___x_3122_; 
v___x_3122_ = lean_nat_add(v___x_2920_, v_size_3070_);
if (lean_obj_tag(v_r_3087_) == 0)
{
lean_object* v_size_3123_; 
v_size_3123_ = lean_ctor_get(v_r_3087_, 0);
lean_inc(v_size_3123_);
v___y_3097_ = v___x_3122_;
v___y_3098_ = v___x_3121_;
v___y_3099_ = v_size_3123_;
goto v___jp_3096_;
}
else
{
lean_object* v___x_3124_; 
v___x_3124_ = lean_unsigned_to_nat(0u);
v___y_3097_ = v___x_3122_;
v___y_3098_ = v___x_3121_;
v___y_3099_ = v___x_3124_;
goto v___jp_3096_;
}
}
}
}
}
else
{
lean_object* v___x_3134_; lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; lean_object* v___x_3139_; 
v___x_3134_ = lean_nat_add(v___x_2920_, v_size_2910_);
lean_dec(v_size_2910_);
v___x_3135_ = lean_nat_add(v___x_3134_, v_size_3070_);
lean_dec(v___x_3134_);
v___x_3136_ = lean_nat_add(v___x_2920_, v_size_3070_);
v___x_3137_ = lean_nat_add(v___x_3136_, v_size_3083_);
lean_dec(v___x_3136_);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_tree_3067_);
lean_ctor_set(v___x_3064_, 3, v_r_2914_);
lean_ctor_set(v___x_3064_, 2, v_v_3069_);
lean_ctor_set(v___x_3064_, 1, v_k_3068_);
lean_ctor_set(v___x_3064_, 0, v___x_3137_);
v___x_3139_ = v___x_3064_;
goto v_reusejp_3138_;
}
else
{
lean_object* v_reuseFailAlloc_3143_; 
v_reuseFailAlloc_3143_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3143_, 0, v___x_3137_);
lean_ctor_set(v_reuseFailAlloc_3143_, 1, v_k_3068_);
lean_ctor_set(v_reuseFailAlloc_3143_, 2, v_v_3069_);
lean_ctor_set(v_reuseFailAlloc_3143_, 3, v_r_2914_);
lean_ctor_set(v_reuseFailAlloc_3143_, 4, v_tree_3067_);
v___x_3139_ = v_reuseFailAlloc_3143_;
goto v_reusejp_3138_;
}
v_reusejp_3138_:
{
lean_object* v___x_3141_; 
if (v_isShared_3081_ == 0)
{
lean_ctor_set(v___x_3080_, 4, v___x_3139_);
lean_ctor_set(v___x_3080_, 0, v___x_3135_);
v___x_3141_ = v___x_3080_;
goto v_reusejp_3140_;
}
else
{
lean_object* v_reuseFailAlloc_3142_; 
v_reuseFailAlloc_3142_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3142_, 0, v___x_3135_);
lean_ctor_set(v_reuseFailAlloc_3142_, 1, v_k_2911_);
lean_ctor_set(v_reuseFailAlloc_3142_, 2, v_v_2912_);
lean_ctor_set(v_reuseFailAlloc_3142_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3142_, 4, v___x_3139_);
v___x_3141_ = v_reuseFailAlloc_3142_;
goto v_reusejp_3140_;
}
v_reusejp_3140_:
{
return v___x_3141_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_l_2913_) == 0)
{
lean_object* v___x_3151_; uint8_t v_isShared_3152_; uint8_t v_isSharedCheck_3173_; 
lean_inc_ref(v_l_2913_);
lean_inc(v_v_2912_);
lean_inc(v_k_2911_);
lean_inc(v_size_2910_);
v_isSharedCheck_3173_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3173_ == 0)
{
lean_object* v_unused_3174_; lean_object* v_unused_3175_; lean_object* v_unused_3176_; lean_object* v_unused_3177_; lean_object* v_unused_3178_; 
v_unused_3174_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3174_);
v_unused_3175_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3175_);
v_unused_3176_ = lean_ctor_get(v_l_2730_, 2);
lean_dec(v_unused_3176_);
v_unused_3177_ = lean_ctor_get(v_l_2730_, 1);
lean_dec(v_unused_3177_);
v_unused_3178_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3178_);
v___x_3151_ = v_l_2730_;
v_isShared_3152_ = v_isSharedCheck_3173_;
goto v_resetjp_3150_;
}
else
{
lean_dec(v_l_2730_);
v___x_3151_ = lean_box(0);
v_isShared_3152_ = v_isSharedCheck_3173_;
goto v_resetjp_3150_;
}
v_resetjp_3150_:
{
if (lean_obj_tag(v_r_2914_) == 0)
{
lean_object* v_k_3153_; lean_object* v_v_3154_; lean_object* v_size_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3159_; 
v_k_3153_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_k_3153_);
v_v_3154_ = lean_ctor_get(v___x_3066_, 1);
lean_inc(v_v_3154_);
lean_dec_ref(v___x_3066_);
v_size_3155_ = lean_ctor_get(v_r_2914_, 0);
v___x_3156_ = lean_nat_add(v___x_2920_, v_size_2910_);
lean_dec(v_size_2910_);
v___x_3157_ = lean_nat_add(v___x_2920_, v_size_3155_);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_tree_3067_);
lean_ctor_set(v___x_3064_, 3, v_r_2914_);
lean_ctor_set(v___x_3064_, 2, v_v_3154_);
lean_ctor_set(v___x_3064_, 1, v_k_3153_);
lean_ctor_set(v___x_3064_, 0, v___x_3157_);
v___x_3159_ = v___x_3064_;
goto v_reusejp_3158_;
}
else
{
lean_object* v_reuseFailAlloc_3163_; 
v_reuseFailAlloc_3163_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3163_, 0, v___x_3157_);
lean_ctor_set(v_reuseFailAlloc_3163_, 1, v_k_3153_);
lean_ctor_set(v_reuseFailAlloc_3163_, 2, v_v_3154_);
lean_ctor_set(v_reuseFailAlloc_3163_, 3, v_r_2914_);
lean_ctor_set(v_reuseFailAlloc_3163_, 4, v_tree_3067_);
v___x_3159_ = v_reuseFailAlloc_3163_;
goto v_reusejp_3158_;
}
v_reusejp_3158_:
{
lean_object* v___x_3161_; 
if (v_isShared_3152_ == 0)
{
lean_ctor_set(v___x_3151_, 4, v___x_3159_);
lean_ctor_set(v___x_3151_, 0, v___x_3156_);
v___x_3161_ = v___x_3151_;
goto v_reusejp_3160_;
}
else
{
lean_object* v_reuseFailAlloc_3162_; 
v_reuseFailAlloc_3162_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3162_, 0, v___x_3156_);
lean_ctor_set(v_reuseFailAlloc_3162_, 1, v_k_2911_);
lean_ctor_set(v_reuseFailAlloc_3162_, 2, v_v_2912_);
lean_ctor_set(v_reuseFailAlloc_3162_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3162_, 4, v___x_3159_);
v___x_3161_ = v_reuseFailAlloc_3162_;
goto v_reusejp_3160_;
}
v_reusejp_3160_:
{
return v___x_3161_;
}
}
}
else
{
lean_object* v_k_3164_; lean_object* v_v_3165_; lean_object* v___x_3166_; lean_object* v___x_3168_; 
lean_dec(v_size_2910_);
v_k_3164_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_k_3164_);
v_v_3165_ = lean_ctor_get(v___x_3066_, 1);
lean_inc(v_v_3165_);
lean_dec_ref(v___x_3066_);
v___x_3166_ = lean_unsigned_to_nat(3u);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_r_2914_);
lean_ctor_set(v___x_3064_, 3, v_r_2914_);
lean_ctor_set(v___x_3064_, 2, v_v_3165_);
lean_ctor_set(v___x_3064_, 1, v_k_3164_);
lean_ctor_set(v___x_3064_, 0, v___x_2920_);
v___x_3168_ = v___x_3064_;
goto v_reusejp_3167_;
}
else
{
lean_object* v_reuseFailAlloc_3172_; 
v_reuseFailAlloc_3172_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3172_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3172_, 1, v_k_3164_);
lean_ctor_set(v_reuseFailAlloc_3172_, 2, v_v_3165_);
lean_ctor_set(v_reuseFailAlloc_3172_, 3, v_r_2914_);
lean_ctor_set(v_reuseFailAlloc_3172_, 4, v_r_2914_);
v___x_3168_ = v_reuseFailAlloc_3172_;
goto v_reusejp_3167_;
}
v_reusejp_3167_:
{
lean_object* v___x_3170_; 
if (v_isShared_3152_ == 0)
{
lean_ctor_set(v___x_3151_, 4, v___x_3168_);
lean_ctor_set(v___x_3151_, 0, v___x_3166_);
v___x_3170_ = v___x_3151_;
goto v_reusejp_3169_;
}
else
{
lean_object* v_reuseFailAlloc_3171_; 
v_reuseFailAlloc_3171_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3171_, 0, v___x_3166_);
lean_ctor_set(v_reuseFailAlloc_3171_, 1, v_k_2911_);
lean_ctor_set(v_reuseFailAlloc_3171_, 2, v_v_2912_);
lean_ctor_set(v_reuseFailAlloc_3171_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3171_, 4, v___x_3168_);
v___x_3170_ = v_reuseFailAlloc_3171_;
goto v_reusejp_3169_;
}
v_reusejp_3169_:
{
return v___x_3170_;
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_2914_) == 0)
{
lean_object* v___x_3180_; uint8_t v_isShared_3181_; uint8_t v_isSharedCheck_3203_; 
lean_inc(v_l_2913_);
lean_inc(v_v_2912_);
lean_inc(v_k_2911_);
v_isSharedCheck_3203_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3203_ == 0)
{
lean_object* v_unused_3204_; lean_object* v_unused_3205_; lean_object* v_unused_3206_; lean_object* v_unused_3207_; lean_object* v_unused_3208_; 
v_unused_3204_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3204_);
v_unused_3205_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3205_);
v_unused_3206_ = lean_ctor_get(v_l_2730_, 2);
lean_dec(v_unused_3206_);
v_unused_3207_ = lean_ctor_get(v_l_2730_, 1);
lean_dec(v_unused_3207_);
v_unused_3208_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3208_);
v___x_3180_ = v_l_2730_;
v_isShared_3181_ = v_isSharedCheck_3203_;
goto v_resetjp_3179_;
}
else
{
lean_dec(v_l_2730_);
v___x_3180_ = lean_box(0);
v_isShared_3181_ = v_isSharedCheck_3203_;
goto v_resetjp_3179_;
}
v_resetjp_3179_:
{
lean_object* v_k_3182_; lean_object* v_v_3183_; lean_object* v_k_3184_; lean_object* v_v_3185_; lean_object* v___x_3187_; uint8_t v_isShared_3188_; uint8_t v_isSharedCheck_3199_; 
v_k_3182_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_k_3182_);
v_v_3183_ = lean_ctor_get(v___x_3066_, 1);
lean_inc(v_v_3183_);
lean_dec_ref(v___x_3066_);
v_k_3184_ = lean_ctor_get(v_r_2914_, 1);
v_v_3185_ = lean_ctor_get(v_r_2914_, 2);
v_isSharedCheck_3199_ = !lean_is_exclusive(v_r_2914_);
if (v_isSharedCheck_3199_ == 0)
{
lean_object* v_unused_3200_; lean_object* v_unused_3201_; lean_object* v_unused_3202_; 
v_unused_3200_ = lean_ctor_get(v_r_2914_, 4);
lean_dec(v_unused_3200_);
v_unused_3201_ = lean_ctor_get(v_r_2914_, 3);
lean_dec(v_unused_3201_);
v_unused_3202_ = lean_ctor_get(v_r_2914_, 0);
lean_dec(v_unused_3202_);
v___x_3187_ = v_r_2914_;
v_isShared_3188_ = v_isSharedCheck_3199_;
goto v_resetjp_3186_;
}
else
{
lean_inc(v_v_3185_);
lean_inc(v_k_3184_);
lean_dec(v_r_2914_);
v___x_3187_ = lean_box(0);
v_isShared_3188_ = v_isSharedCheck_3199_;
goto v_resetjp_3186_;
}
v_resetjp_3186_:
{
lean_object* v___x_3189_; lean_object* v___x_3191_; 
v___x_3189_ = lean_unsigned_to_nat(3u);
if (v_isShared_3188_ == 0)
{
lean_ctor_set(v___x_3187_, 4, v_l_2913_);
lean_ctor_set(v___x_3187_, 3, v_l_2913_);
lean_ctor_set(v___x_3187_, 2, v_v_2912_);
lean_ctor_set(v___x_3187_, 1, v_k_2911_);
lean_ctor_set(v___x_3187_, 0, v___x_2920_);
v___x_3191_ = v___x_3187_;
goto v_reusejp_3190_;
}
else
{
lean_object* v_reuseFailAlloc_3198_; 
v_reuseFailAlloc_3198_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3198_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3198_, 1, v_k_2911_);
lean_ctor_set(v_reuseFailAlloc_3198_, 2, v_v_2912_);
lean_ctor_set(v_reuseFailAlloc_3198_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3198_, 4, v_l_2913_);
v___x_3191_ = v_reuseFailAlloc_3198_;
goto v_reusejp_3190_;
}
v_reusejp_3190_:
{
lean_object* v___x_3193_; 
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_l_2913_);
lean_ctor_set(v___x_3064_, 3, v_l_2913_);
lean_ctor_set(v___x_3064_, 2, v_v_3183_);
lean_ctor_set(v___x_3064_, 1, v_k_3182_);
lean_ctor_set(v___x_3064_, 0, v___x_2920_);
v___x_3193_ = v___x_3064_;
goto v_reusejp_3192_;
}
else
{
lean_object* v_reuseFailAlloc_3197_; 
v_reuseFailAlloc_3197_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3197_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_3197_, 1, v_k_3182_);
lean_ctor_set(v_reuseFailAlloc_3197_, 2, v_v_3183_);
lean_ctor_set(v_reuseFailAlloc_3197_, 3, v_l_2913_);
lean_ctor_set(v_reuseFailAlloc_3197_, 4, v_l_2913_);
v___x_3193_ = v_reuseFailAlloc_3197_;
goto v_reusejp_3192_;
}
v_reusejp_3192_:
{
lean_object* v___x_3195_; 
if (v_isShared_3181_ == 0)
{
lean_ctor_set(v___x_3180_, 4, v___x_3193_);
lean_ctor_set(v___x_3180_, 3, v___x_3191_);
lean_ctor_set(v___x_3180_, 2, v_v_3185_);
lean_ctor_set(v___x_3180_, 1, v_k_3184_);
lean_ctor_set(v___x_3180_, 0, v___x_3189_);
v___x_3195_ = v___x_3180_;
goto v_reusejp_3194_;
}
else
{
lean_object* v_reuseFailAlloc_3196_; 
v_reuseFailAlloc_3196_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3196_, 0, v___x_3189_);
lean_ctor_set(v_reuseFailAlloc_3196_, 1, v_k_3184_);
lean_ctor_set(v_reuseFailAlloc_3196_, 2, v_v_3185_);
lean_ctor_set(v_reuseFailAlloc_3196_, 3, v___x_3191_);
lean_ctor_set(v_reuseFailAlloc_3196_, 4, v___x_3193_);
v___x_3195_ = v_reuseFailAlloc_3196_;
goto v_reusejp_3194_;
}
v_reusejp_3194_:
{
return v___x_3195_;
}
}
}
}
}
}
else
{
lean_object* v_k_3209_; lean_object* v_v_3210_; lean_object* v___x_3211_; lean_object* v___x_3213_; 
v_k_3209_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_k_3209_);
v_v_3210_ = lean_ctor_get(v___x_3066_, 1);
lean_inc(v_v_3210_);
lean_dec_ref(v___x_3066_);
v___x_3211_ = lean_unsigned_to_nat(2u);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 4, v_r_2914_);
lean_ctor_set(v___x_3064_, 3, v_l_2730_);
lean_ctor_set(v___x_3064_, 2, v_v_3210_);
lean_ctor_set(v___x_3064_, 1, v_k_3209_);
lean_ctor_set(v___x_3064_, 0, v___x_3211_);
v___x_3213_ = v___x_3064_;
goto v_reusejp_3212_;
}
else
{
lean_object* v_reuseFailAlloc_3214_; 
v_reuseFailAlloc_3214_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3214_, 0, v___x_3211_);
lean_ctor_set(v_reuseFailAlloc_3214_, 1, v_k_3209_);
lean_ctor_set(v_reuseFailAlloc_3214_, 2, v_v_3210_);
lean_ctor_set(v_reuseFailAlloc_3214_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3214_, 4, v_r_2914_);
v___x_3213_ = v_reuseFailAlloc_3214_;
goto v_reusejp_3212_;
}
v_reusejp_3212_:
{
return v___x_3213_;
}
}
}
}
}
}
}
else
{
return v_l_2730_;
}
}
else
{
return v_r_2731_;
}
}
default: 
{
lean_object* v_impl_3221_; lean_object* v___x_3222_; 
v_impl_3221_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v_k_2726_, v_r_2731_);
v___x_3222_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_3221_) == 0)
{
if (lean_obj_tag(v_l_2730_) == 0)
{
lean_object* v_size_3223_; lean_object* v_size_3224_; lean_object* v_k_3225_; lean_object* v_v_3226_; lean_object* v_l_3227_; lean_object* v_r_3228_; lean_object* v___x_3229_; lean_object* v___x_3230_; uint8_t v___x_3231_; 
v_size_3223_ = lean_ctor_get(v_impl_3221_, 0);
lean_inc(v_size_3223_);
v_size_3224_ = lean_ctor_get(v_l_2730_, 0);
v_k_3225_ = lean_ctor_get(v_l_2730_, 1);
v_v_3226_ = lean_ctor_get(v_l_2730_, 2);
v_l_3227_ = lean_ctor_get(v_l_2730_, 3);
v_r_3228_ = lean_ctor_get(v_l_2730_, 4);
lean_inc(v_r_3228_);
v___x_3229_ = lean_unsigned_to_nat(3u);
v___x_3230_ = lean_nat_mul(v___x_3229_, v_size_3223_);
v___x_3231_ = lean_nat_dec_lt(v___x_3230_, v_size_3224_);
lean_dec(v___x_3230_);
if (v___x_3231_ == 0)
{
lean_object* v___x_3232_; lean_object* v___x_3233_; lean_object* v___x_3235_; 
lean_dec(v_r_3228_);
v___x_3232_ = lean_nat_add(v___x_3222_, v_size_3224_);
v___x_3233_ = lean_nat_add(v___x_3232_, v_size_3223_);
lean_dec(v_size_3223_);
lean_dec(v___x_3232_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_impl_3221_);
lean_ctor_set(v___x_2733_, 0, v___x_3233_);
v___x_3235_ = v___x_2733_;
goto v_reusejp_3234_;
}
else
{
lean_object* v_reuseFailAlloc_3236_; 
v_reuseFailAlloc_3236_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3236_, 0, v___x_3233_);
lean_ctor_set(v_reuseFailAlloc_3236_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3236_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3236_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3236_, 4, v_impl_3221_);
v___x_3235_ = v_reuseFailAlloc_3236_;
goto v_reusejp_3234_;
}
v_reusejp_3234_:
{
return v___x_3235_;
}
}
else
{
lean_object* v___x_3238_; uint8_t v_isShared_3239_; uint8_t v_isSharedCheck_3302_; 
lean_inc(v_l_3227_);
lean_inc(v_v_3226_);
lean_inc(v_k_3225_);
lean_inc(v_size_3224_);
v_isSharedCheck_3302_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3302_ == 0)
{
lean_object* v_unused_3303_; lean_object* v_unused_3304_; lean_object* v_unused_3305_; lean_object* v_unused_3306_; lean_object* v_unused_3307_; 
v_unused_3303_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3303_);
v_unused_3304_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3304_);
v_unused_3305_ = lean_ctor_get(v_l_2730_, 2);
lean_dec(v_unused_3305_);
v_unused_3306_ = lean_ctor_get(v_l_2730_, 1);
lean_dec(v_unused_3306_);
v_unused_3307_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3307_);
v___x_3238_ = v_l_2730_;
v_isShared_3239_ = v_isSharedCheck_3302_;
goto v_resetjp_3237_;
}
else
{
lean_dec(v_l_2730_);
v___x_3238_ = lean_box(0);
v_isShared_3239_ = v_isSharedCheck_3302_;
goto v_resetjp_3237_;
}
v_resetjp_3237_:
{
lean_object* v_size_3240_; lean_object* v_size_3241_; lean_object* v_k_3242_; lean_object* v_v_3243_; lean_object* v_l_3244_; lean_object* v_r_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; uint8_t v___x_3248_; 
v_size_3240_ = lean_ctor_get(v_l_3227_, 0);
v_size_3241_ = lean_ctor_get(v_r_3228_, 0);
v_k_3242_ = lean_ctor_get(v_r_3228_, 1);
v_v_3243_ = lean_ctor_get(v_r_3228_, 2);
v_l_3244_ = lean_ctor_get(v_r_3228_, 3);
v_r_3245_ = lean_ctor_get(v_r_3228_, 4);
v___x_3246_ = lean_unsigned_to_nat(2u);
v___x_3247_ = lean_nat_mul(v___x_3246_, v_size_3240_);
v___x_3248_ = lean_nat_dec_lt(v_size_3241_, v___x_3247_);
lean_dec(v___x_3247_);
if (v___x_3248_ == 0)
{
lean_object* v___x_3250_; uint8_t v_isShared_3251_; uint8_t v_isSharedCheck_3277_; 
lean_inc(v_r_3245_);
lean_inc(v_l_3244_);
lean_inc(v_v_3243_);
lean_inc(v_k_3242_);
v_isSharedCheck_3277_ = !lean_is_exclusive(v_r_3228_);
if (v_isSharedCheck_3277_ == 0)
{
lean_object* v_unused_3278_; lean_object* v_unused_3279_; lean_object* v_unused_3280_; lean_object* v_unused_3281_; lean_object* v_unused_3282_; 
v_unused_3278_ = lean_ctor_get(v_r_3228_, 4);
lean_dec(v_unused_3278_);
v_unused_3279_ = lean_ctor_get(v_r_3228_, 3);
lean_dec(v_unused_3279_);
v_unused_3280_ = lean_ctor_get(v_r_3228_, 2);
lean_dec(v_unused_3280_);
v_unused_3281_ = lean_ctor_get(v_r_3228_, 1);
lean_dec(v_unused_3281_);
v_unused_3282_ = lean_ctor_get(v_r_3228_, 0);
lean_dec(v_unused_3282_);
v___x_3250_ = v_r_3228_;
v_isShared_3251_ = v_isSharedCheck_3277_;
goto v_resetjp_3249_;
}
else
{
lean_dec(v_r_3228_);
v___x_3250_ = lean_box(0);
v_isShared_3251_ = v_isSharedCheck_3277_;
goto v_resetjp_3249_;
}
v_resetjp_3249_:
{
lean_object* v___x_3252_; lean_object* v___x_3253_; lean_object* v___y_3255_; lean_object* v___y_3256_; lean_object* v___y_3257_; lean_object* v___x_3265_; lean_object* v___y_3267_; 
v___x_3252_ = lean_nat_add(v___x_3222_, v_size_3224_);
lean_dec(v_size_3224_);
v___x_3253_ = lean_nat_add(v___x_3252_, v_size_3223_);
lean_dec(v___x_3252_);
v___x_3265_ = lean_nat_add(v___x_3222_, v_size_3240_);
if (lean_obj_tag(v_l_3244_) == 0)
{
lean_object* v_size_3275_; 
v_size_3275_ = lean_ctor_get(v_l_3244_, 0);
lean_inc(v_size_3275_);
v___y_3267_ = v_size_3275_;
goto v___jp_3266_;
}
else
{
lean_object* v___x_3276_; 
v___x_3276_ = lean_unsigned_to_nat(0u);
v___y_3267_ = v___x_3276_;
goto v___jp_3266_;
}
v___jp_3254_:
{
lean_object* v___x_3258_; lean_object* v___x_3260_; 
v___x_3258_ = lean_nat_add(v___y_3256_, v___y_3257_);
lean_dec(v___y_3257_);
lean_dec(v___y_3256_);
if (v_isShared_3251_ == 0)
{
lean_ctor_set(v___x_3250_, 4, v_impl_3221_);
lean_ctor_set(v___x_3250_, 3, v_r_3245_);
lean_ctor_set(v___x_3250_, 2, v_v_2729_);
lean_ctor_set(v___x_3250_, 1, v_k_2728_);
lean_ctor_set(v___x_3250_, 0, v___x_3258_);
v___x_3260_ = v___x_3250_;
goto v_reusejp_3259_;
}
else
{
lean_object* v_reuseFailAlloc_3264_; 
v_reuseFailAlloc_3264_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3264_, 0, v___x_3258_);
lean_ctor_set(v_reuseFailAlloc_3264_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3264_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3264_, 3, v_r_3245_);
lean_ctor_set(v_reuseFailAlloc_3264_, 4, v_impl_3221_);
v___x_3260_ = v_reuseFailAlloc_3264_;
goto v_reusejp_3259_;
}
v_reusejp_3259_:
{
lean_object* v___x_3262_; 
if (v_isShared_3239_ == 0)
{
lean_ctor_set(v___x_3238_, 4, v___x_3260_);
lean_ctor_set(v___x_3238_, 3, v___y_3255_);
lean_ctor_set(v___x_3238_, 2, v_v_3243_);
lean_ctor_set(v___x_3238_, 1, v_k_3242_);
lean_ctor_set(v___x_3238_, 0, v___x_3253_);
v___x_3262_ = v___x_3238_;
goto v_reusejp_3261_;
}
else
{
lean_object* v_reuseFailAlloc_3263_; 
v_reuseFailAlloc_3263_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3263_, 0, v___x_3253_);
lean_ctor_set(v_reuseFailAlloc_3263_, 1, v_k_3242_);
lean_ctor_set(v_reuseFailAlloc_3263_, 2, v_v_3243_);
lean_ctor_set(v_reuseFailAlloc_3263_, 3, v___y_3255_);
lean_ctor_set(v_reuseFailAlloc_3263_, 4, v___x_3260_);
v___x_3262_ = v_reuseFailAlloc_3263_;
goto v_reusejp_3261_;
}
v_reusejp_3261_:
{
return v___x_3262_;
}
}
}
v___jp_3266_:
{
lean_object* v___x_3268_; lean_object* v___x_3270_; 
v___x_3268_ = lean_nat_add(v___x_3265_, v___y_3267_);
lean_dec(v___y_3267_);
lean_dec(v___x_3265_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_l_3244_);
lean_ctor_set(v___x_2733_, 3, v_l_3227_);
lean_ctor_set(v___x_2733_, 2, v_v_3226_);
lean_ctor_set(v___x_2733_, 1, v_k_3225_);
lean_ctor_set(v___x_2733_, 0, v___x_3268_);
v___x_3270_ = v___x_2733_;
goto v_reusejp_3269_;
}
else
{
lean_object* v_reuseFailAlloc_3274_; 
v_reuseFailAlloc_3274_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3274_, 0, v___x_3268_);
lean_ctor_set(v_reuseFailAlloc_3274_, 1, v_k_3225_);
lean_ctor_set(v_reuseFailAlloc_3274_, 2, v_v_3226_);
lean_ctor_set(v_reuseFailAlloc_3274_, 3, v_l_3227_);
lean_ctor_set(v_reuseFailAlloc_3274_, 4, v_l_3244_);
v___x_3270_ = v_reuseFailAlloc_3274_;
goto v_reusejp_3269_;
}
v_reusejp_3269_:
{
lean_object* v___x_3271_; 
v___x_3271_ = lean_nat_add(v___x_3222_, v_size_3223_);
lean_dec(v_size_3223_);
if (lean_obj_tag(v_r_3245_) == 0)
{
lean_object* v_size_3272_; 
v_size_3272_ = lean_ctor_get(v_r_3245_, 0);
lean_inc(v_size_3272_);
v___y_3255_ = v___x_3270_;
v___y_3256_ = v___x_3271_;
v___y_3257_ = v_size_3272_;
goto v___jp_3254_;
}
else
{
lean_object* v___x_3273_; 
v___x_3273_ = lean_unsigned_to_nat(0u);
v___y_3255_ = v___x_3270_;
v___y_3256_ = v___x_3271_;
v___y_3257_ = v___x_3273_;
goto v___jp_3254_;
}
}
}
}
}
else
{
lean_object* v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3285_; lean_object* v___x_3286_; lean_object* v___x_3288_; 
lean_del_object(v___x_2733_);
v___x_3283_ = lean_nat_add(v___x_3222_, v_size_3224_);
lean_dec(v_size_3224_);
v___x_3284_ = lean_nat_add(v___x_3283_, v_size_3223_);
lean_dec(v___x_3283_);
v___x_3285_ = lean_nat_add(v___x_3222_, v_size_3223_);
lean_dec(v_size_3223_);
v___x_3286_ = lean_nat_add(v___x_3285_, v_size_3241_);
lean_dec(v___x_3285_);
lean_inc_ref(v_impl_3221_);
if (v_isShared_3239_ == 0)
{
lean_ctor_set(v___x_3238_, 4, v_impl_3221_);
lean_ctor_set(v___x_3238_, 3, v_r_3228_);
lean_ctor_set(v___x_3238_, 2, v_v_2729_);
lean_ctor_set(v___x_3238_, 1, v_k_2728_);
lean_ctor_set(v___x_3238_, 0, v___x_3286_);
v___x_3288_ = v___x_3238_;
goto v_reusejp_3287_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v___x_3286_);
lean_ctor_set(v_reuseFailAlloc_3301_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3301_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3301_, 3, v_r_3228_);
lean_ctor_set(v_reuseFailAlloc_3301_, 4, v_impl_3221_);
v___x_3288_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3287_;
}
v_reusejp_3287_:
{
lean_object* v___x_3290_; uint8_t v_isShared_3291_; uint8_t v_isSharedCheck_3295_; 
v_isSharedCheck_3295_ = !lean_is_exclusive(v_impl_3221_);
if (v_isSharedCheck_3295_ == 0)
{
lean_object* v_unused_3296_; lean_object* v_unused_3297_; lean_object* v_unused_3298_; lean_object* v_unused_3299_; lean_object* v_unused_3300_; 
v_unused_3296_ = lean_ctor_get(v_impl_3221_, 4);
lean_dec(v_unused_3296_);
v_unused_3297_ = lean_ctor_get(v_impl_3221_, 3);
lean_dec(v_unused_3297_);
v_unused_3298_ = lean_ctor_get(v_impl_3221_, 2);
lean_dec(v_unused_3298_);
v_unused_3299_ = lean_ctor_get(v_impl_3221_, 1);
lean_dec(v_unused_3299_);
v_unused_3300_ = lean_ctor_get(v_impl_3221_, 0);
lean_dec(v_unused_3300_);
v___x_3290_ = v_impl_3221_;
v_isShared_3291_ = v_isSharedCheck_3295_;
goto v_resetjp_3289_;
}
else
{
lean_dec(v_impl_3221_);
v___x_3290_ = lean_box(0);
v_isShared_3291_ = v_isSharedCheck_3295_;
goto v_resetjp_3289_;
}
v_resetjp_3289_:
{
lean_object* v___x_3293_; 
if (v_isShared_3291_ == 0)
{
lean_ctor_set(v___x_3290_, 4, v___x_3288_);
lean_ctor_set(v___x_3290_, 3, v_l_3227_);
lean_ctor_set(v___x_3290_, 2, v_v_3226_);
lean_ctor_set(v___x_3290_, 1, v_k_3225_);
lean_ctor_set(v___x_3290_, 0, v___x_3284_);
v___x_3293_ = v___x_3290_;
goto v_reusejp_3292_;
}
else
{
lean_object* v_reuseFailAlloc_3294_; 
v_reuseFailAlloc_3294_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3294_, 0, v___x_3284_);
lean_ctor_set(v_reuseFailAlloc_3294_, 1, v_k_3225_);
lean_ctor_set(v_reuseFailAlloc_3294_, 2, v_v_3226_);
lean_ctor_set(v_reuseFailAlloc_3294_, 3, v_l_3227_);
lean_ctor_set(v_reuseFailAlloc_3294_, 4, v___x_3288_);
v___x_3293_ = v_reuseFailAlloc_3294_;
goto v_reusejp_3292_;
}
v_reusejp_3292_:
{
return v___x_3293_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_3308_; lean_object* v___x_3309_; lean_object* v___x_3311_; 
v_size_3308_ = lean_ctor_get(v_impl_3221_, 0);
lean_inc(v_size_3308_);
v___x_3309_ = lean_nat_add(v___x_3222_, v_size_3308_);
lean_dec(v_size_3308_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_impl_3221_);
lean_ctor_set(v___x_2733_, 0, v___x_3309_);
v___x_3311_ = v___x_2733_;
goto v_reusejp_3310_;
}
else
{
lean_object* v_reuseFailAlloc_3312_; 
v_reuseFailAlloc_3312_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3312_, 0, v___x_3309_);
lean_ctor_set(v_reuseFailAlloc_3312_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3312_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3312_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3312_, 4, v_impl_3221_);
v___x_3311_ = v_reuseFailAlloc_3312_;
goto v_reusejp_3310_;
}
v_reusejp_3310_:
{
return v___x_3311_;
}
}
}
else
{
if (lean_obj_tag(v_l_2730_) == 0)
{
lean_object* v_l_3313_; 
v_l_3313_ = lean_ctor_get(v_l_2730_, 3);
if (lean_obj_tag(v_l_3313_) == 0)
{
lean_object* v_r_3314_; 
lean_inc_ref(v_l_3313_);
v_r_3314_ = lean_ctor_get(v_l_2730_, 4);
lean_inc(v_r_3314_);
if (lean_obj_tag(v_r_3314_) == 0)
{
lean_object* v_size_3315_; lean_object* v_k_3316_; lean_object* v_v_3317_; lean_object* v___x_3319_; uint8_t v_isShared_3320_; uint8_t v_isSharedCheck_3330_; 
v_size_3315_ = lean_ctor_get(v_l_2730_, 0);
v_k_3316_ = lean_ctor_get(v_l_2730_, 1);
v_v_3317_ = lean_ctor_get(v_l_2730_, 2);
v_isSharedCheck_3330_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3330_ == 0)
{
lean_object* v_unused_3331_; lean_object* v_unused_3332_; 
v_unused_3331_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3331_);
v_unused_3332_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3332_);
v___x_3319_ = v_l_2730_;
v_isShared_3320_ = v_isSharedCheck_3330_;
goto v_resetjp_3318_;
}
else
{
lean_inc(v_v_3317_);
lean_inc(v_k_3316_);
lean_inc(v_size_3315_);
lean_dec(v_l_2730_);
v___x_3319_ = lean_box(0);
v_isShared_3320_ = v_isSharedCheck_3330_;
goto v_resetjp_3318_;
}
v_resetjp_3318_:
{
lean_object* v_size_3321_; lean_object* v___x_3322_; lean_object* v___x_3323_; lean_object* v___x_3325_; 
v_size_3321_ = lean_ctor_get(v_r_3314_, 0);
v___x_3322_ = lean_nat_add(v___x_3222_, v_size_3315_);
lean_dec(v_size_3315_);
v___x_3323_ = lean_nat_add(v___x_3222_, v_size_3321_);
if (v_isShared_3320_ == 0)
{
lean_ctor_set(v___x_3319_, 4, v_impl_3221_);
lean_ctor_set(v___x_3319_, 3, v_r_3314_);
lean_ctor_set(v___x_3319_, 2, v_v_2729_);
lean_ctor_set(v___x_3319_, 1, v_k_2728_);
lean_ctor_set(v___x_3319_, 0, v___x_3323_);
v___x_3325_ = v___x_3319_;
goto v_reusejp_3324_;
}
else
{
lean_object* v_reuseFailAlloc_3329_; 
v_reuseFailAlloc_3329_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3329_, 0, v___x_3323_);
lean_ctor_set(v_reuseFailAlloc_3329_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3329_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3329_, 3, v_r_3314_);
lean_ctor_set(v_reuseFailAlloc_3329_, 4, v_impl_3221_);
v___x_3325_ = v_reuseFailAlloc_3329_;
goto v_reusejp_3324_;
}
v_reusejp_3324_:
{
lean_object* v___x_3327_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v___x_3325_);
lean_ctor_set(v___x_2733_, 3, v_l_3313_);
lean_ctor_set(v___x_2733_, 2, v_v_3317_);
lean_ctor_set(v___x_2733_, 1, v_k_3316_);
lean_ctor_set(v___x_2733_, 0, v___x_3322_);
v___x_3327_ = v___x_2733_;
goto v_reusejp_3326_;
}
else
{
lean_object* v_reuseFailAlloc_3328_; 
v_reuseFailAlloc_3328_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3328_, 0, v___x_3322_);
lean_ctor_set(v_reuseFailAlloc_3328_, 1, v_k_3316_);
lean_ctor_set(v_reuseFailAlloc_3328_, 2, v_v_3317_);
lean_ctor_set(v_reuseFailAlloc_3328_, 3, v_l_3313_);
lean_ctor_set(v_reuseFailAlloc_3328_, 4, v___x_3325_);
v___x_3327_ = v_reuseFailAlloc_3328_;
goto v_reusejp_3326_;
}
v_reusejp_3326_:
{
return v___x_3327_;
}
}
}
}
else
{
lean_object* v_k_3333_; lean_object* v_v_3334_; lean_object* v___x_3336_; uint8_t v_isShared_3337_; uint8_t v_isSharedCheck_3345_; 
v_k_3333_ = lean_ctor_get(v_l_2730_, 1);
v_v_3334_ = lean_ctor_get(v_l_2730_, 2);
v_isSharedCheck_3345_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3345_ == 0)
{
lean_object* v_unused_3346_; lean_object* v_unused_3347_; lean_object* v_unused_3348_; 
v_unused_3346_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3346_);
v_unused_3347_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3347_);
v_unused_3348_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3348_);
v___x_3336_ = v_l_2730_;
v_isShared_3337_ = v_isSharedCheck_3345_;
goto v_resetjp_3335_;
}
else
{
lean_inc(v_v_3334_);
lean_inc(v_k_3333_);
lean_dec(v_l_2730_);
v___x_3336_ = lean_box(0);
v_isShared_3337_ = v_isSharedCheck_3345_;
goto v_resetjp_3335_;
}
v_resetjp_3335_:
{
lean_object* v___x_3338_; lean_object* v___x_3340_; 
v___x_3338_ = lean_unsigned_to_nat(3u);
if (v_isShared_3337_ == 0)
{
lean_ctor_set(v___x_3336_, 3, v_r_3314_);
lean_ctor_set(v___x_3336_, 2, v_v_2729_);
lean_ctor_set(v___x_3336_, 1, v_k_2728_);
lean_ctor_set(v___x_3336_, 0, v___x_3222_);
v___x_3340_ = v___x_3336_;
goto v_reusejp_3339_;
}
else
{
lean_object* v_reuseFailAlloc_3344_; 
v_reuseFailAlloc_3344_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3344_, 0, v___x_3222_);
lean_ctor_set(v_reuseFailAlloc_3344_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3344_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3344_, 3, v_r_3314_);
lean_ctor_set(v_reuseFailAlloc_3344_, 4, v_r_3314_);
v___x_3340_ = v_reuseFailAlloc_3344_;
goto v_reusejp_3339_;
}
v_reusejp_3339_:
{
lean_object* v___x_3342_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v___x_3340_);
lean_ctor_set(v___x_2733_, 3, v_l_3313_);
lean_ctor_set(v___x_2733_, 2, v_v_3334_);
lean_ctor_set(v___x_2733_, 1, v_k_3333_);
lean_ctor_set(v___x_2733_, 0, v___x_3338_);
v___x_3342_ = v___x_2733_;
goto v_reusejp_3341_;
}
else
{
lean_object* v_reuseFailAlloc_3343_; 
v_reuseFailAlloc_3343_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3343_, 0, v___x_3338_);
lean_ctor_set(v_reuseFailAlloc_3343_, 1, v_k_3333_);
lean_ctor_set(v_reuseFailAlloc_3343_, 2, v_v_3334_);
lean_ctor_set(v_reuseFailAlloc_3343_, 3, v_l_3313_);
lean_ctor_set(v_reuseFailAlloc_3343_, 4, v___x_3340_);
v___x_3342_ = v_reuseFailAlloc_3343_;
goto v_reusejp_3341_;
}
v_reusejp_3341_:
{
return v___x_3342_;
}
}
}
}
}
else
{
lean_object* v_r_3349_; 
v_r_3349_ = lean_ctor_get(v_l_2730_, 4);
lean_inc(v_r_3349_);
if (lean_obj_tag(v_r_3349_) == 0)
{
lean_object* v_k_3350_; lean_object* v_v_3351_; lean_object* v___x_3353_; uint8_t v_isShared_3354_; uint8_t v_isSharedCheck_3374_; 
lean_inc(v_l_3313_);
v_k_3350_ = lean_ctor_get(v_l_2730_, 1);
v_v_3351_ = lean_ctor_get(v_l_2730_, 2);
v_isSharedCheck_3374_ = !lean_is_exclusive(v_l_2730_);
if (v_isSharedCheck_3374_ == 0)
{
lean_object* v_unused_3375_; lean_object* v_unused_3376_; lean_object* v_unused_3377_; 
v_unused_3375_ = lean_ctor_get(v_l_2730_, 4);
lean_dec(v_unused_3375_);
v_unused_3376_ = lean_ctor_get(v_l_2730_, 3);
lean_dec(v_unused_3376_);
v_unused_3377_ = lean_ctor_get(v_l_2730_, 0);
lean_dec(v_unused_3377_);
v___x_3353_ = v_l_2730_;
v_isShared_3354_ = v_isSharedCheck_3374_;
goto v_resetjp_3352_;
}
else
{
lean_inc(v_v_3351_);
lean_inc(v_k_3350_);
lean_dec(v_l_2730_);
v___x_3353_ = lean_box(0);
v_isShared_3354_ = v_isSharedCheck_3374_;
goto v_resetjp_3352_;
}
v_resetjp_3352_:
{
lean_object* v_k_3355_; lean_object* v_v_3356_; lean_object* v___x_3358_; uint8_t v_isShared_3359_; uint8_t v_isSharedCheck_3370_; 
v_k_3355_ = lean_ctor_get(v_r_3349_, 1);
v_v_3356_ = lean_ctor_get(v_r_3349_, 2);
v_isSharedCheck_3370_ = !lean_is_exclusive(v_r_3349_);
if (v_isSharedCheck_3370_ == 0)
{
lean_object* v_unused_3371_; lean_object* v_unused_3372_; lean_object* v_unused_3373_; 
v_unused_3371_ = lean_ctor_get(v_r_3349_, 4);
lean_dec(v_unused_3371_);
v_unused_3372_ = lean_ctor_get(v_r_3349_, 3);
lean_dec(v_unused_3372_);
v_unused_3373_ = lean_ctor_get(v_r_3349_, 0);
lean_dec(v_unused_3373_);
v___x_3358_ = v_r_3349_;
v_isShared_3359_ = v_isSharedCheck_3370_;
goto v_resetjp_3357_;
}
else
{
lean_inc(v_v_3356_);
lean_inc(v_k_3355_);
lean_dec(v_r_3349_);
v___x_3358_ = lean_box(0);
v_isShared_3359_ = v_isSharedCheck_3370_;
goto v_resetjp_3357_;
}
v_resetjp_3357_:
{
lean_object* v___x_3360_; lean_object* v___x_3362_; 
v___x_3360_ = lean_unsigned_to_nat(3u);
if (v_isShared_3359_ == 0)
{
lean_ctor_set(v___x_3358_, 4, v_l_3313_);
lean_ctor_set(v___x_3358_, 3, v_l_3313_);
lean_ctor_set(v___x_3358_, 2, v_v_3351_);
lean_ctor_set(v___x_3358_, 1, v_k_3350_);
lean_ctor_set(v___x_3358_, 0, v___x_3222_);
v___x_3362_ = v___x_3358_;
goto v_reusejp_3361_;
}
else
{
lean_object* v_reuseFailAlloc_3369_; 
v_reuseFailAlloc_3369_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3369_, 0, v___x_3222_);
lean_ctor_set(v_reuseFailAlloc_3369_, 1, v_k_3350_);
lean_ctor_set(v_reuseFailAlloc_3369_, 2, v_v_3351_);
lean_ctor_set(v_reuseFailAlloc_3369_, 3, v_l_3313_);
lean_ctor_set(v_reuseFailAlloc_3369_, 4, v_l_3313_);
v___x_3362_ = v_reuseFailAlloc_3369_;
goto v_reusejp_3361_;
}
v_reusejp_3361_:
{
lean_object* v___x_3364_; 
if (v_isShared_3354_ == 0)
{
lean_ctor_set(v___x_3353_, 4, v_l_3313_);
lean_ctor_set(v___x_3353_, 2, v_v_2729_);
lean_ctor_set(v___x_3353_, 1, v_k_2728_);
lean_ctor_set(v___x_3353_, 0, v___x_3222_);
v___x_3364_ = v___x_3353_;
goto v_reusejp_3363_;
}
else
{
lean_object* v_reuseFailAlloc_3368_; 
v_reuseFailAlloc_3368_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3368_, 0, v___x_3222_);
lean_ctor_set(v_reuseFailAlloc_3368_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3368_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3368_, 3, v_l_3313_);
lean_ctor_set(v_reuseFailAlloc_3368_, 4, v_l_3313_);
v___x_3364_ = v_reuseFailAlloc_3368_;
goto v_reusejp_3363_;
}
v_reusejp_3363_:
{
lean_object* v___x_3366_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v___x_3364_);
lean_ctor_set(v___x_2733_, 3, v___x_3362_);
lean_ctor_set(v___x_2733_, 2, v_v_3356_);
lean_ctor_set(v___x_2733_, 1, v_k_3355_);
lean_ctor_set(v___x_2733_, 0, v___x_3360_);
v___x_3366_ = v___x_2733_;
goto v_reusejp_3365_;
}
else
{
lean_object* v_reuseFailAlloc_3367_; 
v_reuseFailAlloc_3367_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3367_, 0, v___x_3360_);
lean_ctor_set(v_reuseFailAlloc_3367_, 1, v_k_3355_);
lean_ctor_set(v_reuseFailAlloc_3367_, 2, v_v_3356_);
lean_ctor_set(v_reuseFailAlloc_3367_, 3, v___x_3362_);
lean_ctor_set(v_reuseFailAlloc_3367_, 4, v___x_3364_);
v___x_3366_ = v_reuseFailAlloc_3367_;
goto v_reusejp_3365_;
}
v_reusejp_3365_:
{
return v___x_3366_;
}
}
}
}
}
}
else
{
lean_object* v___x_3378_; lean_object* v___x_3380_; 
v___x_3378_ = lean_unsigned_to_nat(2u);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_r_3349_);
lean_ctor_set(v___x_2733_, 0, v___x_3378_);
v___x_3380_ = v___x_2733_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3381_; 
v_reuseFailAlloc_3381_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3381_, 0, v___x_3378_);
lean_ctor_set(v_reuseFailAlloc_3381_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3381_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3381_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3381_, 4, v_r_3349_);
v___x_3380_ = v_reuseFailAlloc_3381_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
return v___x_3380_;
}
}
}
}
else
{
lean_object* v___x_3383_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 4, v_l_2730_);
lean_ctor_set(v___x_2733_, 0, v___x_3222_);
v___x_3383_ = v___x_2733_;
goto v_reusejp_3382_;
}
else
{
lean_object* v_reuseFailAlloc_3384_; 
v_reuseFailAlloc_3384_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3384_, 0, v___x_3222_);
lean_ctor_set(v_reuseFailAlloc_3384_, 1, v_k_2728_);
lean_ctor_set(v_reuseFailAlloc_3384_, 2, v_v_2729_);
lean_ctor_set(v_reuseFailAlloc_3384_, 3, v_l_2730_);
lean_ctor_set(v_reuseFailAlloc_3384_, 4, v_l_2730_);
v___x_3383_ = v_reuseFailAlloc_3384_;
goto v_reusejp_3382_;
}
v_reusejp_3382_:
{
return v___x_3383_;
}
}
}
}
}
}
}
else
{
return v_t_2727_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg___boxed(lean_object* v_k_3387_, lean_object* v_t_3388_){
_start:
{
lean_object* v_res_3389_; 
v_res_3389_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v_k_3387_, v_t_3388_);
lean_dec(v_k_3387_);
return v_res_3389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg(lean_object* v_a_3390_, lean_object* v_x_3391_){
_start:
{
if (lean_obj_tag(v_x_3391_) == 0)
{
lean_object* v___x_3392_; 
v___x_3392_ = lean_box(0);
return v___x_3392_;
}
else
{
lean_object* v_key_3393_; lean_object* v_value_3394_; lean_object* v_tail_3395_; uint8_t v___x_3396_; 
v_key_3393_ = lean_ctor_get(v_x_3391_, 0);
v_value_3394_ = lean_ctor_get(v_x_3391_, 1);
v_tail_3395_ = lean_ctor_get(v_x_3391_, 2);
v___x_3396_ = lean_nat_dec_eq(v_key_3393_, v_a_3390_);
if (v___x_3396_ == 0)
{
v_x_3391_ = v_tail_3395_;
goto _start;
}
else
{
lean_object* v___x_3398_; 
lean_inc(v_value_3394_);
v___x_3398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3398_, 0, v_value_3394_);
return v___x_3398_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg___boxed(lean_object* v_a_3399_, lean_object* v_x_3400_){
_start:
{
lean_object* v_res_3401_; 
v_res_3401_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg(v_a_3399_, v_x_3400_);
lean_dec(v_x_3400_);
lean_dec(v_a_3399_);
return v_res_3401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg(lean_object* v_m_3402_, lean_object* v_a_3403_){
_start:
{
lean_object* v_buckets_3404_; lean_object* v___x_3405_; uint64_t v___x_3406_; uint64_t v___x_3407_; uint64_t v___x_3408_; uint64_t v_fold_3409_; uint64_t v___x_3410_; uint64_t v___x_3411_; uint64_t v___x_3412_; size_t v___x_3413_; size_t v___x_3414_; size_t v___x_3415_; size_t v___x_3416_; size_t v___x_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; 
v_buckets_3404_ = lean_ctor_get(v_m_3402_, 1);
v___x_3405_ = lean_array_get_size(v_buckets_3404_);
v___x_3406_ = lean_uint64_of_nat(v_a_3403_);
v___x_3407_ = 32ULL;
v___x_3408_ = lean_uint64_shift_right(v___x_3406_, v___x_3407_);
v_fold_3409_ = lean_uint64_xor(v___x_3406_, v___x_3408_);
v___x_3410_ = 16ULL;
v___x_3411_ = lean_uint64_shift_right(v_fold_3409_, v___x_3410_);
v___x_3412_ = lean_uint64_xor(v_fold_3409_, v___x_3411_);
v___x_3413_ = lean_uint64_to_usize(v___x_3412_);
v___x_3414_ = lean_usize_of_nat(v___x_3405_);
v___x_3415_ = ((size_t)1ULL);
v___x_3416_ = lean_usize_sub(v___x_3414_, v___x_3415_);
v___x_3417_ = lean_usize_land(v___x_3413_, v___x_3416_);
v___x_3418_ = lean_array_uget_borrowed(v_buckets_3404_, v___x_3417_);
v___x_3419_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg(v_a_3403_, v___x_3418_);
return v___x_3419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg___boxed(lean_object* v_m_3420_, lean_object* v_a_3421_){
_start:
{
lean_object* v_res_3422_; 
v_res_3422_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg(v_m_3420_, v_a_3421_);
lean_dec(v_a_3421_);
lean_dec_ref(v_m_3420_);
return v_res_3422_;
}
}
static lean_object* _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; 
v___x_3426_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__2));
v___x_3427_ = lean_unsigned_to_nat(14u);
v___x_3428_ = lean_unsigned_to_nat(22u);
v___x_3429_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__1));
v___x_3430_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__0));
v___x_3431_ = l_mkPanicMessageWithDecl(v___x_3430_, v___x_3429_, v___x_3428_, v___x_3427_, v___x_3426_);
return v___x_3431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(lean_object* v___x_3432_, lean_object* v_a_3433_, lean_object* v_init_3434_, lean_object* v_x_3435_){
_start:
{
lean_object* v_d_3438_; 
if (lean_obj_tag(v_x_3435_) == 0)
{
lean_object* v_k_3441_; lean_object* v_l_3442_; lean_object* v_r_3443_; lean_object* v___x_3444_; lean_object* v_a_3445_; 
v_k_3441_ = lean_ctor_get(v_x_3435_, 1);
v_l_3442_ = lean_ctor_get(v_x_3435_, 3);
v_r_3443_ = lean_ctor_get(v_x_3435_, 4);
v___x_3444_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(v___x_3432_, v_a_3433_, v_init_3434_, v_l_3442_);
v_a_3445_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3445_);
if (lean_obj_tag(v_a_3445_) == 0)
{
lean_object* v_a_3446_; 
lean_dec_ref(v___x_3444_);
v_a_3446_ = lean_ctor_get(v_a_3445_, 0);
lean_inc(v_a_3446_);
lean_dec_ref_known(v_a_3445_, 1);
v_d_3438_ = v_a_3446_;
goto v___jp_3437_;
}
else
{
lean_object* v_a_3447_; lean_object* v___y_3449_; lean_object* v___x_3457_; 
v_a_3447_ = lean_ctor_get(v_a_3445_, 0);
lean_inc(v_a_3447_);
lean_dec_ref_known(v_a_3445_, 1);
v___x_3457_ = l_Lean_Environment_getModuleIdxFor_x3f(v___x_3432_, v_k_3441_);
if (lean_obj_tag(v___x_3457_) == 0)
{
lean_object* v___x_3458_; 
v___x_3458_ = lean_box(0);
v___y_3449_ = v___x_3458_;
goto v___jp_3448_;
}
else
{
lean_object* v_val_3459_; lean_object* v___x_3460_; 
v_val_3459_ = lean_ctor_get(v___x_3457_, 0);
lean_inc(v_val_3459_);
lean_dec_ref_known(v___x_3457_, 1);
v___x_3460_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg(v_a_3433_, v_val_3459_);
lean_dec(v_val_3459_);
if (lean_obj_tag(v___x_3460_) == 0)
{
lean_object* v___x_3461_; lean_object* v___x_3462_; 
v___x_3461_ = lean_obj_once(&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3, &lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3_once, _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___closed__3);
v___x_3462_ = lp_mathlib_panic___at___00Mathlib_Command_MinImports_getAllImports_spec__4(v___x_3461_);
v___y_3449_ = v___x_3462_;
goto v___jp_3448_;
}
else
{
lean_object* v_val_3463_; 
v_val_3463_ = lean_ctor_get(v___x_3460_, 0);
lean_inc(v_val_3463_);
lean_dec_ref_known(v___x_3460_, 1);
v___y_3449_ = v_val_3463_;
goto v___jp_3448_;
}
}
v___jp_3448_:
{
uint8_t v___x_3450_; 
v___x_3450_ = l_Lean_NameSet_contains(v_a_3447_, v___y_3449_);
if (v___x_3450_ == 0)
{
lean_object* v___x_3451_; 
lean_dec_ref(v___x_3444_);
v___x_3451_ = l_Lean_NameSet_insert(v_a_3447_, v___y_3449_);
v_init_3434_ = v___x_3451_;
v_x_3435_ = v_r_3443_;
goto _start;
}
else
{
lean_object* v_a_3453_; 
lean_dec(v___y_3449_);
lean_dec(v_a_3447_);
v_a_3453_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3453_);
lean_dec_ref(v___x_3444_);
if (lean_obj_tag(v_a_3453_) == 0)
{
lean_object* v_a_3454_; 
v_a_3454_ = lean_ctor_get(v_a_3453_, 0);
lean_inc(v_a_3454_);
lean_dec_ref_known(v_a_3453_, 1);
v_d_3438_ = v_a_3454_;
goto v___jp_3437_;
}
else
{
lean_object* v_a_3455_; 
v_a_3455_ = lean_ctor_get(v_a_3453_, 0);
lean_inc(v_a_3455_);
lean_dec_ref_known(v_a_3453_, 1);
v_init_3434_ = v_a_3455_;
v_x_3435_ = v_r_3443_;
goto _start;
}
}
}
}
}
else
{
lean_object* v___x_3464_; lean_object* v___x_3465_; 
v___x_3464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3464_, 0, v_init_3434_);
v___x_3465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3465_, 0, v___x_3464_);
return v___x_3465_;
}
v___jp_3437_:
{
lean_object* v___x_3439_; lean_object* v___x_3440_; 
v___x_3439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3439_, 0, v_d_3438_);
v___x_3440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3440_, 0, v___x_3439_);
return v___x_3440_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg___boxed(lean_object* v___x_3466_, lean_object* v_a_3467_, lean_object* v_init_3468_, lean_object* v_x_3469_, lean_object* v___y_3470_){
_start:
{
lean_object* v_res_3471_; 
v_res_3471_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(v___x_3466_, v_a_3467_, v_init_3468_, v_x_3469_);
lean_dec(v_x_3469_);
lean_dec_ref(v_a_3467_);
lean_dec_ref(v___x_3466_);
return v_res_3471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3___redArg(lean_object* v_a_3472_, lean_object* v_b_3473_, lean_object* v_x_3474_){
_start:
{
if (lean_obj_tag(v_x_3474_) == 0)
{
lean_dec(v_b_3473_);
lean_dec(v_a_3472_);
return v_x_3474_;
}
else
{
lean_object* v_key_3475_; lean_object* v_value_3476_; lean_object* v_tail_3477_; lean_object* v___x_3479_; uint8_t v_isShared_3480_; uint8_t v_isSharedCheck_3489_; 
v_key_3475_ = lean_ctor_get(v_x_3474_, 0);
v_value_3476_ = lean_ctor_get(v_x_3474_, 1);
v_tail_3477_ = lean_ctor_get(v_x_3474_, 2);
v_isSharedCheck_3489_ = !lean_is_exclusive(v_x_3474_);
if (v_isSharedCheck_3489_ == 0)
{
v___x_3479_ = v_x_3474_;
v_isShared_3480_ = v_isSharedCheck_3489_;
goto v_resetjp_3478_;
}
else
{
lean_inc(v_tail_3477_);
lean_inc(v_value_3476_);
lean_inc(v_key_3475_);
lean_dec(v_x_3474_);
v___x_3479_ = lean_box(0);
v_isShared_3480_ = v_isSharedCheck_3489_;
goto v_resetjp_3478_;
}
v_resetjp_3478_:
{
uint8_t v___x_3481_; 
v___x_3481_ = lean_nat_dec_eq(v_key_3475_, v_a_3472_);
if (v___x_3481_ == 0)
{
lean_object* v___x_3482_; lean_object* v___x_3484_; 
v___x_3482_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3___redArg(v_a_3472_, v_b_3473_, v_tail_3477_);
if (v_isShared_3480_ == 0)
{
lean_ctor_set(v___x_3479_, 2, v___x_3482_);
v___x_3484_ = v___x_3479_;
goto v_reusejp_3483_;
}
else
{
lean_object* v_reuseFailAlloc_3485_; 
v_reuseFailAlloc_3485_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3485_, 0, v_key_3475_);
lean_ctor_set(v_reuseFailAlloc_3485_, 1, v_value_3476_);
lean_ctor_set(v_reuseFailAlloc_3485_, 2, v___x_3482_);
v___x_3484_ = v_reuseFailAlloc_3485_;
goto v_reusejp_3483_;
}
v_reusejp_3483_:
{
return v___x_3484_;
}
}
else
{
lean_object* v___x_3487_; 
lean_dec(v_value_3476_);
lean_dec(v_key_3475_);
if (v_isShared_3480_ == 0)
{
lean_ctor_set(v___x_3479_, 1, v_b_3473_);
lean_ctor_set(v___x_3479_, 0, v_a_3472_);
v___x_3487_ = v___x_3479_;
goto v_reusejp_3486_;
}
else
{
lean_object* v_reuseFailAlloc_3488_; 
v_reuseFailAlloc_3488_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3488_, 0, v_a_3472_);
lean_ctor_set(v_reuseFailAlloc_3488_, 1, v_b_3473_);
lean_ctor_set(v_reuseFailAlloc_3488_, 2, v_tail_3477_);
v___x_3487_ = v_reuseFailAlloc_3488_;
goto v_reusejp_3486_;
}
v_reusejp_3486_:
{
return v___x_3487_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg(lean_object* v_a_3490_, lean_object* v_x_3491_){
_start:
{
if (lean_obj_tag(v_x_3491_) == 0)
{
uint8_t v___x_3492_; 
v___x_3492_ = 0;
return v___x_3492_;
}
else
{
lean_object* v_key_3493_; lean_object* v_tail_3494_; uint8_t v___x_3495_; 
v_key_3493_ = lean_ctor_get(v_x_3491_, 0);
v_tail_3494_ = lean_ctor_get(v_x_3491_, 2);
v___x_3495_ = lean_nat_dec_eq(v_key_3493_, v_a_3490_);
if (v___x_3495_ == 0)
{
v_x_3491_ = v_tail_3494_;
goto _start;
}
else
{
return v___x_3495_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg___boxed(lean_object* v_a_3497_, lean_object* v_x_3498_){
_start:
{
uint8_t v_res_3499_; lean_object* v_r_3500_; 
v_res_3499_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg(v_a_3497_, v_x_3498_);
lean_dec(v_x_3498_);
lean_dec(v_a_3497_);
v_r_3500_ = lean_box(v_res_3499_);
return v_r_3500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11___redArg(lean_object* v_x_3501_, lean_object* v_x_3502_){
_start:
{
if (lean_obj_tag(v_x_3502_) == 0)
{
return v_x_3501_;
}
else
{
lean_object* v_key_3503_; lean_object* v_value_3504_; lean_object* v_tail_3505_; lean_object* v___x_3507_; uint8_t v_isShared_3508_; uint8_t v_isSharedCheck_3528_; 
v_key_3503_ = lean_ctor_get(v_x_3502_, 0);
v_value_3504_ = lean_ctor_get(v_x_3502_, 1);
v_tail_3505_ = lean_ctor_get(v_x_3502_, 2);
v_isSharedCheck_3528_ = !lean_is_exclusive(v_x_3502_);
if (v_isSharedCheck_3528_ == 0)
{
v___x_3507_ = v_x_3502_;
v_isShared_3508_ = v_isSharedCheck_3528_;
goto v_resetjp_3506_;
}
else
{
lean_inc(v_tail_3505_);
lean_inc(v_value_3504_);
lean_inc(v_key_3503_);
lean_dec(v_x_3502_);
v___x_3507_ = lean_box(0);
v_isShared_3508_ = v_isSharedCheck_3528_;
goto v_resetjp_3506_;
}
v_resetjp_3506_:
{
lean_object* v___x_3509_; uint64_t v___x_3510_; uint64_t v___x_3511_; uint64_t v___x_3512_; uint64_t v_fold_3513_; uint64_t v___x_3514_; uint64_t v___x_3515_; uint64_t v___x_3516_; size_t v___x_3517_; size_t v___x_3518_; size_t v___x_3519_; size_t v___x_3520_; size_t v___x_3521_; lean_object* v___x_3522_; lean_object* v___x_3524_; 
v___x_3509_ = lean_array_get_size(v_x_3501_);
v___x_3510_ = lean_uint64_of_nat(v_key_3503_);
v___x_3511_ = 32ULL;
v___x_3512_ = lean_uint64_shift_right(v___x_3510_, v___x_3511_);
v_fold_3513_ = lean_uint64_xor(v___x_3510_, v___x_3512_);
v___x_3514_ = 16ULL;
v___x_3515_ = lean_uint64_shift_right(v_fold_3513_, v___x_3514_);
v___x_3516_ = lean_uint64_xor(v_fold_3513_, v___x_3515_);
v___x_3517_ = lean_uint64_to_usize(v___x_3516_);
v___x_3518_ = lean_usize_of_nat(v___x_3509_);
v___x_3519_ = ((size_t)1ULL);
v___x_3520_ = lean_usize_sub(v___x_3518_, v___x_3519_);
v___x_3521_ = lean_usize_land(v___x_3517_, v___x_3520_);
v___x_3522_ = lean_array_uget_borrowed(v_x_3501_, v___x_3521_);
lean_inc(v___x_3522_);
if (v_isShared_3508_ == 0)
{
lean_ctor_set(v___x_3507_, 2, v___x_3522_);
v___x_3524_ = v___x_3507_;
goto v_reusejp_3523_;
}
else
{
lean_object* v_reuseFailAlloc_3527_; 
v_reuseFailAlloc_3527_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3527_, 0, v_key_3503_);
lean_ctor_set(v_reuseFailAlloc_3527_, 1, v_value_3504_);
lean_ctor_set(v_reuseFailAlloc_3527_, 2, v___x_3522_);
v___x_3524_ = v_reuseFailAlloc_3527_;
goto v_reusejp_3523_;
}
v_reusejp_3523_:
{
lean_object* v___x_3525_; 
v___x_3525_ = lean_array_uset(v_x_3501_, v___x_3521_, v___x_3524_);
v_x_3501_ = v___x_3525_;
v_x_3502_ = v_tail_3505_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4___redArg(lean_object* v_i_3529_, lean_object* v_source_3530_, lean_object* v_target_3531_){
_start:
{
lean_object* v___x_3532_; uint8_t v___x_3533_; 
v___x_3532_ = lean_array_get_size(v_source_3530_);
v___x_3533_ = lean_nat_dec_lt(v_i_3529_, v___x_3532_);
if (v___x_3533_ == 0)
{
lean_dec_ref(v_source_3530_);
lean_dec(v_i_3529_);
return v_target_3531_;
}
else
{
lean_object* v_es_3534_; lean_object* v___x_3535_; lean_object* v_source_3536_; lean_object* v_target_3537_; lean_object* v___x_3538_; lean_object* v___x_3539_; 
v_es_3534_ = lean_array_fget(v_source_3530_, v_i_3529_);
v___x_3535_ = lean_box(0);
v_source_3536_ = lean_array_fset(v_source_3530_, v_i_3529_, v___x_3535_);
v_target_3537_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11___redArg(v_target_3531_, v_es_3534_);
v___x_3538_ = lean_unsigned_to_nat(1u);
v___x_3539_ = lean_nat_add(v_i_3529_, v___x_3538_);
lean_dec(v_i_3529_);
v_i_3529_ = v___x_3539_;
v_source_3530_ = v_source_3536_;
v_target_3531_ = v_target_3537_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2___redArg(lean_object* v_data_3541_){
_start:
{
lean_object* v___x_3542_; lean_object* v___x_3543_; lean_object* v_nbuckets_3544_; lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; 
v___x_3542_ = lean_array_get_size(v_data_3541_);
v___x_3543_ = lean_unsigned_to_nat(2u);
v_nbuckets_3544_ = lean_nat_mul(v___x_3542_, v___x_3543_);
v___x_3545_ = lean_unsigned_to_nat(0u);
v___x_3546_ = lean_box(0);
v___x_3547_ = lean_mk_array(v_nbuckets_3544_, v___x_3546_);
v___x_3548_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4___redArg(v___x_3545_, v_data_3541_, v___x_3547_);
return v___x_3548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1___redArg(lean_object* v_m_3549_, lean_object* v_a_3550_, lean_object* v_b_3551_){
_start:
{
lean_object* v_size_3552_; lean_object* v_buckets_3553_; lean_object* v___x_3555_; uint8_t v_isShared_3556_; uint8_t v_isSharedCheck_3596_; 
v_size_3552_ = lean_ctor_get(v_m_3549_, 0);
v_buckets_3553_ = lean_ctor_get(v_m_3549_, 1);
v_isSharedCheck_3596_ = !lean_is_exclusive(v_m_3549_);
if (v_isSharedCheck_3596_ == 0)
{
v___x_3555_ = v_m_3549_;
v_isShared_3556_ = v_isSharedCheck_3596_;
goto v_resetjp_3554_;
}
else
{
lean_inc(v_buckets_3553_);
lean_inc(v_size_3552_);
lean_dec(v_m_3549_);
v___x_3555_ = lean_box(0);
v_isShared_3556_ = v_isSharedCheck_3596_;
goto v_resetjp_3554_;
}
v_resetjp_3554_:
{
lean_object* v___x_3557_; uint64_t v___x_3558_; uint64_t v___x_3559_; uint64_t v___x_3560_; uint64_t v_fold_3561_; uint64_t v___x_3562_; uint64_t v___x_3563_; uint64_t v___x_3564_; size_t v___x_3565_; size_t v___x_3566_; size_t v___x_3567_; size_t v___x_3568_; size_t v___x_3569_; lean_object* v_bkt_3570_; uint8_t v___x_3571_; 
v___x_3557_ = lean_array_get_size(v_buckets_3553_);
v___x_3558_ = lean_uint64_of_nat(v_a_3550_);
v___x_3559_ = 32ULL;
v___x_3560_ = lean_uint64_shift_right(v___x_3558_, v___x_3559_);
v_fold_3561_ = lean_uint64_xor(v___x_3558_, v___x_3560_);
v___x_3562_ = 16ULL;
v___x_3563_ = lean_uint64_shift_right(v_fold_3561_, v___x_3562_);
v___x_3564_ = lean_uint64_xor(v_fold_3561_, v___x_3563_);
v___x_3565_ = lean_uint64_to_usize(v___x_3564_);
v___x_3566_ = lean_usize_of_nat(v___x_3557_);
v___x_3567_ = ((size_t)1ULL);
v___x_3568_ = lean_usize_sub(v___x_3566_, v___x_3567_);
v___x_3569_ = lean_usize_land(v___x_3565_, v___x_3568_);
v_bkt_3570_ = lean_array_uget_borrowed(v_buckets_3553_, v___x_3569_);
v___x_3571_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg(v_a_3550_, v_bkt_3570_);
if (v___x_3571_ == 0)
{
lean_object* v___x_3572_; lean_object* v_size_x27_3573_; lean_object* v___x_3574_; lean_object* v_buckets_x27_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; uint8_t v___x_3581_; 
v___x_3572_ = lean_unsigned_to_nat(1u);
v_size_x27_3573_ = lean_nat_add(v_size_3552_, v___x_3572_);
lean_dec(v_size_3552_);
lean_inc(v_bkt_3570_);
v___x_3574_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3574_, 0, v_a_3550_);
lean_ctor_set(v___x_3574_, 1, v_b_3551_);
lean_ctor_set(v___x_3574_, 2, v_bkt_3570_);
v_buckets_x27_3575_ = lean_array_uset(v_buckets_3553_, v___x_3569_, v___x_3574_);
v___x_3576_ = lean_unsigned_to_nat(4u);
v___x_3577_ = lean_nat_mul(v_size_x27_3573_, v___x_3576_);
v___x_3578_ = lean_unsigned_to_nat(3u);
v___x_3579_ = lean_nat_div(v___x_3577_, v___x_3578_);
lean_dec(v___x_3577_);
v___x_3580_ = lean_array_get_size(v_buckets_x27_3575_);
v___x_3581_ = lean_nat_dec_le(v___x_3579_, v___x_3580_);
lean_dec(v___x_3579_);
if (v___x_3581_ == 0)
{
lean_object* v_val_3582_; lean_object* v___x_3584_; 
v_val_3582_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2___redArg(v_buckets_x27_3575_);
if (v_isShared_3556_ == 0)
{
lean_ctor_set(v___x_3555_, 1, v_val_3582_);
lean_ctor_set(v___x_3555_, 0, v_size_x27_3573_);
v___x_3584_ = v___x_3555_;
goto v_reusejp_3583_;
}
else
{
lean_object* v_reuseFailAlloc_3585_; 
v_reuseFailAlloc_3585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3585_, 0, v_size_x27_3573_);
lean_ctor_set(v_reuseFailAlloc_3585_, 1, v_val_3582_);
v___x_3584_ = v_reuseFailAlloc_3585_;
goto v_reusejp_3583_;
}
v_reusejp_3583_:
{
return v___x_3584_;
}
}
else
{
lean_object* v___x_3587_; 
if (v_isShared_3556_ == 0)
{
lean_ctor_set(v___x_3555_, 1, v_buckets_x27_3575_);
lean_ctor_set(v___x_3555_, 0, v_size_x27_3573_);
v___x_3587_ = v___x_3555_;
goto v_reusejp_3586_;
}
else
{
lean_object* v_reuseFailAlloc_3588_; 
v_reuseFailAlloc_3588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3588_, 0, v_size_x27_3573_);
lean_ctor_set(v_reuseFailAlloc_3588_, 1, v_buckets_x27_3575_);
v___x_3587_ = v_reuseFailAlloc_3588_;
goto v_reusejp_3586_;
}
v_reusejp_3586_:
{
return v___x_3587_;
}
}
}
else
{
lean_object* v___x_3589_; lean_object* v_buckets_x27_3590_; lean_object* v___x_3591_; lean_object* v___x_3592_; lean_object* v___x_3594_; 
lean_inc(v_bkt_3570_);
v___x_3589_ = lean_box(0);
v_buckets_x27_3590_ = lean_array_uset(v_buckets_3553_, v___x_3569_, v___x_3589_);
v___x_3591_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3___redArg(v_a_3550_, v_b_3551_, v_bkt_3570_);
v___x_3592_ = lean_array_uset(v_buckets_x27_3590_, v___x_3569_, v___x_3591_);
if (v_isShared_3556_ == 0)
{
lean_ctor_set(v___x_3555_, 1, v___x_3592_);
v___x_3594_ = v___x_3555_;
goto v_reusejp_3593_;
}
else
{
lean_object* v_reuseFailAlloc_3595_; 
v_reuseFailAlloc_3595_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3595_, 0, v_size_3552_);
lean_ctor_set(v_reuseFailAlloc_3595_, 1, v___x_3592_);
v___x_3594_ = v_reuseFailAlloc_3595_;
goto v_reusejp_3593_;
}
v_reusejp_3593_:
{
return v___x_3594_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg(lean_object* v___x_3597_, lean_object* v_as_3598_, size_t v_sz_3599_, size_t v_i_3600_, lean_object* v_b_3601_){
_start:
{
uint8_t v___x_3603_; 
v___x_3603_ = lean_usize_dec_lt(v_i_3600_, v_sz_3599_);
if (v___x_3603_ == 0)
{
lean_object* v___x_3604_; 
v___x_3604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3604_, 0, v_b_3601_);
return v___x_3604_;
}
else
{
lean_object* v_a_3605_; lean_object* v___y_3607_; lean_object* v___x_3612_; 
v_a_3605_ = lean_array_uget_borrowed(v_as_3598_, v_i_3600_);
v___x_3612_ = l_Lean_Environment_getModuleIdx_x3f(v___x_3597_, v_a_3605_);
if (lean_obj_tag(v___x_3612_) == 0)
{
lean_object* v___x_3613_; 
v___x_3613_ = lean_unsigned_to_nat(0u);
v___y_3607_ = v___x_3613_;
goto v___jp_3606_;
}
else
{
lean_object* v_val_3614_; 
v_val_3614_ = lean_ctor_get(v___x_3612_, 0);
lean_inc(v_val_3614_);
lean_dec_ref_known(v___x_3612_, 1);
v___y_3607_ = v_val_3614_;
goto v___jp_3606_;
}
v___jp_3606_:
{
lean_object* v___x_3608_; size_t v___x_3609_; size_t v___x_3610_; 
lean_inc(v_a_3605_);
v___x_3608_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1___redArg(v_b_3601_, v___y_3607_, v_a_3605_);
v___x_3609_ = ((size_t)1ULL);
v___x_3610_ = lean_usize_add(v_i_3600_, v___x_3609_);
v_i_3600_ = v___x_3610_;
v_b_3601_ = v___x_3608_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg___boxed(lean_object* v___x_3615_, lean_object* v_as_3616_, lean_object* v_sz_3617_, lean_object* v_i_3618_, lean_object* v_b_3619_, lean_object* v___y_3620_){
_start:
{
size_t v_sz_boxed_3621_; size_t v_i_boxed_3622_; lean_object* v_res_3623_; 
v_sz_boxed_3621_ = lean_unbox_usize(v_sz_3617_);
lean_dec(v_sz_3617_);
v_i_boxed_3622_ = lean_unbox_usize(v_i_3618_);
lean_dec(v_i_3618_);
v_res_3623_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg(v___x_3615_, v_as_3616_, v_sz_boxed_3621_, v_i_boxed_3622_, v_b_3619_);
lean_dec_ref(v_as_3616_);
lean_dec_ref(v___x_3615_);
return v_res_3623_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0(void){
_start:
{
lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3626_; 
v___x_3624_ = lean_box(0);
v___x_3625_ = lean_unsigned_to_nat(16u);
v___x_3626_ = lean_mk_array(v___x_3625_, v___x_3624_);
return v___x_3626_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3627_; lean_object* v___x_3628_; lean_object* v___x_3629_; 
v___x_3627_ = lean_obj_once(&lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0, &lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__0);
v___x_3628_ = lean_unsigned_to_nat(0u);
v___x_3629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3629_, 0, v___x_3628_);
lean_ctor_set(v___x_3629_, 1, v___x_3627_);
return v___x_3629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0(lean_object* v_env_3630_, lean_object* v_a_3631_, lean_object* v_____r_3632_, lean_object* v___y_3633_, lean_object* v___y_3634_){
_start:
{
lean_object* v___x_3636_; lean_object* v___x_3637_; lean_object* v___x_3638_; size_t v_sz_3639_; size_t v___x_3640_; lean_object* v___x_3641_; 
v___x_3636_ = lean_obj_once(&lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1, &lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___closed__1);
v___x_3637_ = l_Lean_Environment_header(v_env_3630_);
v___x_3638_ = l_Lean_EnvironmentHeader_moduleNames(v___x_3637_);
v_sz_3639_ = lean_array_size(v___x_3638_);
v___x_3640_ = ((size_t)0ULL);
v___x_3641_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg(v_env_3630_, v___x_3638_, v_sz_3639_, v___x_3640_, v___x_3636_);
lean_dec_ref(v___x_3638_);
if (lean_obj_tag(v___x_3641_) == 0)
{
lean_object* v_a_3642_; lean_object* v___x_3643_; lean_object* v___x_3644_; lean_object* v_a_3645_; lean_object* v___x_3647_; uint8_t v_isShared_3648_; uint8_t v_isSharedCheck_3657_; 
v_a_3642_ = lean_ctor_get(v___x_3641_, 0);
lean_inc(v_a_3642_);
lean_dec_ref_known(v___x_3641_, 1);
v___x_3643_ = l_Lean_NameSet_empty;
v___x_3644_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(v_env_3630_, v_a_3642_, v___x_3643_, v_a_3631_);
lean_dec(v_a_3642_);
v_a_3645_ = lean_ctor_get(v___x_3644_, 0);
v_isSharedCheck_3657_ = !lean_is_exclusive(v___x_3644_);
if (v_isSharedCheck_3657_ == 0)
{
v___x_3647_ = v___x_3644_;
v_isShared_3648_ = v_isSharedCheck_3657_;
goto v_resetjp_3646_;
}
else
{
lean_inc(v_a_3645_);
lean_dec(v___x_3644_);
v___x_3647_ = lean_box(0);
v_isShared_3648_ = v_isSharedCheck_3657_;
goto v_resetjp_3646_;
}
v_resetjp_3646_:
{
lean_object* v_a_3650_; lean_object* v_a_3656_; 
v_a_3656_ = lean_ctor_get(v_a_3645_, 0);
lean_inc(v_a_3656_);
lean_dec(v_a_3645_);
v_a_3650_ = v_a_3656_;
goto v___jp_3649_;
v___jp_3649_:
{
lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3654_; 
v___x_3651_ = lean_box(0);
v___x_3652_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v___x_3651_, v_a_3650_);
if (v_isShared_3648_ == 0)
{
lean_ctor_set(v___x_3647_, 0, v___x_3652_);
v___x_3654_ = v___x_3647_;
goto v_reusejp_3653_;
}
else
{
lean_object* v_reuseFailAlloc_3655_; 
v_reuseFailAlloc_3655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3655_, 0, v___x_3652_);
v___x_3654_ = v_reuseFailAlloc_3655_;
goto v_reusejp_3653_;
}
v_reusejp_3653_:
{
return v___x_3654_;
}
}
}
}
else
{
lean_object* v_a_3658_; lean_object* v___x_3660_; uint8_t v_isShared_3661_; uint8_t v_isSharedCheck_3665_; 
v_a_3658_ = lean_ctor_get(v___x_3641_, 0);
v_isSharedCheck_3665_ = !lean_is_exclusive(v___x_3641_);
if (v_isSharedCheck_3665_ == 0)
{
v___x_3660_ = v___x_3641_;
v_isShared_3661_ = v_isSharedCheck_3665_;
goto v_resetjp_3659_;
}
else
{
lean_inc(v_a_3658_);
lean_dec(v___x_3641_);
v___x_3660_ = lean_box(0);
v_isShared_3661_ = v_isSharedCheck_3665_;
goto v_resetjp_3659_;
}
v_resetjp_3659_:
{
lean_object* v___x_3663_; 
if (v_isShared_3661_ == 0)
{
v___x_3663_ = v___x_3660_;
goto v_reusejp_3662_;
}
else
{
lean_object* v_reuseFailAlloc_3664_; 
v_reuseFailAlloc_3664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3664_, 0, v_a_3658_);
v___x_3663_ = v_reuseFailAlloc_3664_;
goto v_reusejp_3662_;
}
v_reusejp_3662_:
{
return v___x_3663_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___boxed(lean_object* v_env_3666_, lean_object* v_a_3667_, lean_object* v_____r_3668_, lean_object* v___y_3669_, lean_object* v___y_3670_, lean_object* v___y_3671_){
_start:
{
lean_object* v_res_3672_; 
v_res_3672_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0(v_env_3666_, v_a_3667_, v_____r_3668_, v___y_3669_, v___y_3670_);
lean_dec(v___y_3670_);
lean_dec_ref(v___y_3669_);
lean_dec(v_a_3667_);
lean_dec_ref(v_env_3666_);
return v_res_3672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1(lean_object* v___f_3673_, lean_object* v_x_3674_, lean_object* v___y_3675_, lean_object* v___y_3676_){
_start:
{
lean_object* v___x_3678_; lean_object* v___x_3679_; 
v___x_3678_ = lean_box(0);
lean_inc(v___y_3676_);
lean_inc_ref(v___y_3675_);
v___x_3679_ = lean_apply_4(v___f_3673_, v___x_3678_, v___y_3675_, v___y_3676_, lean_box(0));
return v___x_3679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1___boxed(lean_object* v___f_3680_, lean_object* v_x_3681_, lean_object* v___y_3682_, lean_object* v___y_3683_, lean_object* v___y_3684_){
_start:
{
lean_object* v_res_3685_; 
v_res_3685_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1(v___f_3680_, v_x_3681_, v___y_3682_, v___y_3683_);
lean_dec(v___y_3683_);
lean_dec_ref(v___y_3682_);
return v_res_3685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(lean_object* v_init_3686_, lean_object* v_x_3687_){
_start:
{
if (lean_obj_tag(v_x_3687_) == 0)
{
lean_object* v_k_3688_; lean_object* v_l_3689_; lean_object* v_r_3690_; lean_object* v___x_3691_; lean_object* v___x_3692_; 
v_k_3688_ = lean_ctor_get(v_x_3687_, 1);
lean_inc(v_k_3688_);
v_l_3689_ = lean_ctor_get(v_x_3687_, 3);
lean_inc(v_l_3689_);
v_r_3690_ = lean_ctor_get(v_x_3687_, 4);
lean_inc(v_r_3690_);
lean_dec_ref_known(v_x_3687_, 5);
v___x_3691_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(v_init_3686_, v_l_3689_);
v___x_3692_ = lean_array_push(v___x_3691_, v_k_3688_);
v_init_3686_ = v___x_3692_;
v_x_3687_ = v_r_3690_;
goto _start;
}
else
{
return v_init_3686_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg(lean_object* v_hi_3694_, lean_object* v_pivot_3695_, lean_object* v_as_3696_, lean_object* v_i_3697_, lean_object* v_k_3698_){
_start:
{
uint8_t v___x_3699_; 
v___x_3699_ = lean_nat_dec_lt(v_k_3698_, v_hi_3694_);
if (v___x_3699_ == 0)
{
lean_object* v___x_3700_; lean_object* v___x_3701_; 
lean_dec(v_k_3698_);
v___x_3700_ = lean_array_fswap(v_as_3696_, v_i_3697_, v_hi_3694_);
v___x_3701_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3701_, 0, v_i_3697_);
lean_ctor_set(v___x_3701_, 1, v___x_3700_);
return v___x_3701_;
}
else
{
lean_object* v___x_3702_; uint8_t v___x_3703_; 
v___x_3702_ = lean_array_fget_borrowed(v_as_3696_, v_k_3698_);
v___x_3703_ = l_Lean_Name_lt(v___x_3702_, v_pivot_3695_);
if (v___x_3703_ == 0)
{
lean_object* v___x_3704_; lean_object* v___x_3705_; 
v___x_3704_ = lean_unsigned_to_nat(1u);
v___x_3705_ = lean_nat_add(v_k_3698_, v___x_3704_);
lean_dec(v_k_3698_);
v_k_3698_ = v___x_3705_;
goto _start;
}
else
{
lean_object* v___x_3707_; lean_object* v___x_3708_; lean_object* v___x_3709_; lean_object* v___x_3710_; 
v___x_3707_ = lean_array_fswap(v_as_3696_, v_i_3697_, v_k_3698_);
v___x_3708_ = lean_unsigned_to_nat(1u);
v___x_3709_ = lean_nat_add(v_i_3697_, v___x_3708_);
lean_dec(v_i_3697_);
v___x_3710_ = lean_nat_add(v_k_3698_, v___x_3708_);
lean_dec(v_k_3698_);
v_as_3696_ = v___x_3707_;
v_i_3697_ = v___x_3709_;
v_k_3698_ = v___x_3710_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg___boxed(lean_object* v_hi_3712_, lean_object* v_pivot_3713_, lean_object* v_as_3714_, lean_object* v_i_3715_, lean_object* v_k_3716_){
_start:
{
lean_object* v_res_3717_; 
v_res_3717_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg(v_hi_3712_, v_pivot_3713_, v_as_3714_, v_i_3715_, v_k_3716_);
lean_dec(v_pivot_3713_);
lean_dec(v_hi_3712_);
return v_res_3717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(lean_object* v_n_3718_, lean_object* v_as_3719_, lean_object* v_lo_3720_, lean_object* v_hi_3721_){
_start:
{
lean_object* v___y_3723_; uint8_t v___x_3733_; 
v___x_3733_ = lean_nat_dec_lt(v_lo_3720_, v_hi_3721_);
if (v___x_3733_ == 0)
{
lean_dec(v_lo_3720_);
return v_as_3719_;
}
else
{
lean_object* v___x_3734_; lean_object* v___x_3735_; lean_object* v_mid_3736_; lean_object* v___y_3738_; lean_object* v___y_3744_; lean_object* v___x_3749_; lean_object* v___x_3750_; uint8_t v___x_3751_; 
v___x_3734_ = lean_nat_add(v_lo_3720_, v_hi_3721_);
v___x_3735_ = lean_unsigned_to_nat(1u);
v_mid_3736_ = lean_nat_shiftr(v___x_3734_, v___x_3735_);
lean_dec(v___x_3734_);
v___x_3749_ = lean_array_fget_borrowed(v_as_3719_, v_mid_3736_);
v___x_3750_ = lean_array_fget_borrowed(v_as_3719_, v_lo_3720_);
v___x_3751_ = l_Lean_Name_lt(v___x_3749_, v___x_3750_);
if (v___x_3751_ == 0)
{
v___y_3744_ = v_as_3719_;
goto v___jp_3743_;
}
else
{
lean_object* v___x_3752_; 
v___x_3752_ = lean_array_fswap(v_as_3719_, v_lo_3720_, v_mid_3736_);
v___y_3744_ = v___x_3752_;
goto v___jp_3743_;
}
v___jp_3737_:
{
lean_object* v___x_3739_; lean_object* v___x_3740_; uint8_t v___x_3741_; 
v___x_3739_ = lean_array_fget_borrowed(v___y_3738_, v_mid_3736_);
v___x_3740_ = lean_array_fget_borrowed(v___y_3738_, v_hi_3721_);
v___x_3741_ = l_Lean_Name_lt(v___x_3739_, v___x_3740_);
if (v___x_3741_ == 0)
{
lean_dec(v_mid_3736_);
v___y_3723_ = v___y_3738_;
goto v___jp_3722_;
}
else
{
lean_object* v___x_3742_; 
v___x_3742_ = lean_array_fswap(v___y_3738_, v_mid_3736_, v_hi_3721_);
lean_dec(v_mid_3736_);
v___y_3723_ = v___x_3742_;
goto v___jp_3722_;
}
}
v___jp_3743_:
{
lean_object* v___x_3745_; lean_object* v___x_3746_; uint8_t v___x_3747_; 
v___x_3745_ = lean_array_fget_borrowed(v___y_3744_, v_hi_3721_);
v___x_3746_ = lean_array_fget_borrowed(v___y_3744_, v_lo_3720_);
v___x_3747_ = l_Lean_Name_lt(v___x_3745_, v___x_3746_);
if (v___x_3747_ == 0)
{
v___y_3738_ = v___y_3744_;
goto v___jp_3737_;
}
else
{
lean_object* v___x_3748_; 
v___x_3748_ = lean_array_fswap(v___y_3744_, v_lo_3720_, v_hi_3721_);
v___y_3738_ = v___x_3748_;
goto v___jp_3737_;
}
}
}
v___jp_3722_:
{
lean_object* v_pivot_3724_; lean_object* v___x_3725_; lean_object* v_fst_3726_; lean_object* v_snd_3727_; uint8_t v___x_3728_; 
v_pivot_3724_ = lean_array_fget(v___y_3723_, v_hi_3721_);
lean_inc_n(v_lo_3720_, 2);
v___x_3725_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg(v_hi_3721_, v_pivot_3724_, v___y_3723_, v_lo_3720_, v_lo_3720_);
lean_dec(v_pivot_3724_);
v_fst_3726_ = lean_ctor_get(v___x_3725_, 0);
lean_inc(v_fst_3726_);
v_snd_3727_ = lean_ctor_get(v___x_3725_, 1);
lean_inc(v_snd_3727_);
lean_dec_ref(v___x_3725_);
v___x_3728_ = lean_nat_dec_le(v_hi_3721_, v_fst_3726_);
if (v___x_3728_ == 0)
{
lean_object* v___x_3729_; lean_object* v___x_3730_; lean_object* v___x_3731_; 
v___x_3729_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(v_n_3718_, v_snd_3727_, v_lo_3720_, v_fst_3726_);
v___x_3730_ = lean_unsigned_to_nat(1u);
v___x_3731_ = lean_nat_add(v_fst_3726_, v___x_3730_);
lean_dec(v_fst_3726_);
v_as_3719_ = v___x_3729_;
v_lo_3720_ = v___x_3731_;
goto _start;
}
else
{
lean_dec(v_fst_3726_);
lean_dec(v_lo_3720_);
return v_snd_3727_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg___boxed(lean_object* v_n_3753_, lean_object* v_as_3754_, lean_object* v_lo_3755_, lean_object* v_hi_3756_){
_start:
{
lean_object* v_res_3757_; 
v_res_3757_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(v_n_3753_, v_as_3754_, v_lo_3755_, v_hi_3756_);
lean_dec(v_hi_3756_);
lean_dec(v_n_3753_);
return v_res_3757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10(lean_object* v_x_3759_, lean_object* v_x_3760_){
_start:
{
if (lean_obj_tag(v_x_3760_) == 0)
{
return v_x_3759_;
}
else
{
lean_object* v_head_3761_; lean_object* v_tail_3762_; lean_object* v___x_3763_; lean_object* v___x_3764_; uint8_t v___x_3765_; lean_object* v___x_3766_; lean_object* v___x_3767_; 
v_head_3761_ = lean_ctor_get(v_x_3760_, 0);
lean_inc(v_head_3761_);
v_tail_3762_ = lean_ctor_get(v_x_3760_, 1);
lean_inc(v_tail_3762_);
lean_dec_ref_known(v_x_3760_, 2);
v___x_3763_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10___closed__0));
v___x_3764_ = lean_string_append(v_x_3759_, v___x_3763_);
v___x_3765_ = 1;
v___x_3766_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_3761_, v___x_3765_);
v___x_3767_ = lean_string_append(v___x_3764_, v___x_3766_);
lean_dec_ref(v___x_3766_);
v_x_3759_ = v___x_3767_;
v_x_3760_ = v_tail_3762_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6(lean_object* v_x_3772_){
_start:
{
if (lean_obj_tag(v_x_3772_) == 0)
{
lean_object* v___x_3773_; 
v___x_3773_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__0));
return v___x_3773_;
}
else
{
lean_object* v_tail_3774_; 
v_tail_3774_ = lean_ctor_get(v_x_3772_, 1);
if (lean_obj_tag(v_tail_3774_) == 0)
{
lean_object* v_head_3775_; lean_object* v___x_3776_; uint8_t v___x_3777_; lean_object* v___x_3778_; lean_object* v___x_3779_; lean_object* v___x_3780_; lean_object* v___x_3781_; 
v_head_3775_ = lean_ctor_get(v_x_3772_, 0);
lean_inc(v_head_3775_);
lean_dec_ref_known(v_x_3772_, 2);
v___x_3776_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__1));
v___x_3777_ = 1;
v___x_3778_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_3775_, v___x_3777_);
v___x_3779_ = lean_string_append(v___x_3776_, v___x_3778_);
lean_dec_ref(v___x_3778_);
v___x_3780_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__2));
v___x_3781_ = lean_string_append(v___x_3779_, v___x_3780_);
return v___x_3781_;
}
else
{
lean_object* v_head_3782_; lean_object* v___x_3783_; uint8_t v___x_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; uint32_t v___x_3788_; lean_object* v___x_3789_; 
lean_inc(v_tail_3774_);
v_head_3782_ = lean_ctor_get(v_x_3772_, 0);
lean_inc(v_head_3782_);
lean_dec_ref_known(v_x_3772_, 2);
v___x_3783_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6___closed__1));
v___x_3784_ = 1;
v___x_3785_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_3782_, v___x_3784_);
v___x_3786_ = lean_string_append(v___x_3783_, v___x_3785_);
lean_dec_ref(v___x_3785_);
v___x_3787_ = lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6_spec__10(v___x_3786_, v_tail_3774_);
v___x_3788_ = 93;
v___x_3789_ = lean_string_push(v___x_3787_, v___x_3788_);
return v___x_3789_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports(lean_object* v_cmd_3791_, lean_object* v_id_3792_, uint8_t v_dbg_x3f_3793_, lean_object* v_a_3794_, lean_object* v_a_3795_){
_start:
{
lean_object* v___x_3797_; lean_object* v___x_3798_; 
v___x_3797_ = lean_st_ref_get(v_a_3795_);
v___x_3798_ = lp_mathlib_Mathlib_Command_MinImports_getAllDependencies(v_cmd_3791_, v_id_3792_, v_a_3794_, v_a_3795_);
if (lean_obj_tag(v___x_3798_) == 0)
{
lean_object* v_a_3799_; lean_object* v_env_3800_; lean_object* v___f_3801_; 
v_a_3799_ = lean_ctor_get(v___x_3798_, 0);
lean_inc_n(v_a_3799_, 2);
lean_dec_ref_known(v___x_3798_, 1);
v_env_3800_ = lean_ctor_get(v___x_3797_, 0);
lean_inc_ref_n(v_env_3800_, 2);
lean_dec(v___x_3797_);
v___f_3801_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0___boxed), 6, 2);
lean_closure_set(v___f_3801_, 0, v_env_3800_);
lean_closure_set(v___f_3801_, 1, v_a_3799_);
if (v_dbg_x3f_3793_ == 0)
{
lean_object* v___x_3802_; lean_object* v___x_3803_; 
lean_dec_ref(v___f_3801_);
v___x_3802_ = lean_box(0);
v___x_3803_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__0(v_env_3800_, v_a_3799_, v___x_3802_, v_a_3794_, v_a_3795_);
lean_dec(v_a_3799_);
lean_dec_ref(v_env_3800_);
return v___x_3803_;
}
else
{
lean_object* v___f_3804_; lean_object* v___y_3806_; lean_object* v___y_3814_; lean_object* v___y_3815_; lean_object* v___y_3816_; lean_object* v___y_3817_; lean_object* v___y_3820_; lean_object* v___y_3821_; lean_object* v___y_3822_; lean_object* v___y_3823_; lean_object* v___y_3826_; 
lean_dec_ref(v_env_3800_);
v___f_3804_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_MinImports_getAllImports___lam__1___boxed), 5, 1);
lean_closure_set(v___f_3804_, 0, v___f_3801_);
if (lean_obj_tag(v_a_3799_) == 0)
{
lean_object* v_size_3835_; 
v_size_3835_ = lean_ctor_get(v_a_3799_, 0);
lean_inc(v_size_3835_);
v___y_3826_ = v_size_3835_;
goto v___jp_3825_;
}
else
{
lean_object* v___x_3836_; 
v___x_3836_ = lean_unsigned_to_nat(0u);
v___y_3826_ = v___x_3836_;
goto v___jp_3825_;
}
v___jp_3805_:
{
lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_1994__overap_3811_; lean_object* v___x_3812_; 
v___x_3807_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_getAllImports___closed__0));
v___x_3808_ = lean_array_to_list(v___y_3806_);
v___x_3809_ = lp_mathlib_List_toString___at___00Mathlib_Command_MinImports_getAllImports_spec__6(v___x_3808_);
v___x_3810_ = lean_string_append(v___x_3807_, v___x_3809_);
lean_dec_ref(v___x_3809_);
v___x_1994__overap_3811_ = lean_dbg_trace(v___x_3810_, v___f_3804_);
lean_inc(v_a_3795_);
lean_inc_ref(v_a_3794_);
v___x_3812_ = lean_apply_3(v___x_1994__overap_3811_, v_a_3794_, v_a_3795_, lean_box(0));
return v___x_3812_;
}
v___jp_3813_:
{
lean_object* v___x_3818_; 
v___x_3818_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(v___y_3814_, v___y_3815_, v___y_3816_, v___y_3817_);
lean_dec(v___y_3817_);
lean_dec(v___y_3814_);
v___y_3806_ = v___x_3818_;
goto v___jp_3805_;
}
v___jp_3819_:
{
uint8_t v___x_3824_; 
v___x_3824_ = lean_nat_dec_le(v___y_3823_, v___y_3820_);
if (v___x_3824_ == 0)
{
lean_dec(v___y_3820_);
lean_inc(v___y_3823_);
v___y_3814_ = v___y_3821_;
v___y_3815_ = v___y_3822_;
v___y_3816_ = v___y_3823_;
v___y_3817_ = v___y_3823_;
goto v___jp_3813_;
}
else
{
v___y_3814_ = v___y_3821_;
v___y_3815_ = v___y_3822_;
v___y_3816_ = v___y_3823_;
v___y_3817_ = v___y_3820_;
goto v___jp_3813_;
}
}
v___jp_3825_:
{
lean_object* v___x_3827_; lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3830_; uint8_t v___x_3831_; 
v___x_3827_ = lean_mk_empty_array_with_capacity(v___y_3826_);
lean_dec(v___y_3826_);
v___x_3828_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(v___x_3827_, v_a_3799_);
v___x_3829_ = lean_array_get_size(v___x_3828_);
v___x_3830_ = lean_unsigned_to_nat(0u);
v___x_3831_ = lean_nat_dec_eq(v___x_3829_, v___x_3830_);
if (v___x_3831_ == 0)
{
lean_object* v___x_3832_; lean_object* v___x_3833_; uint8_t v___x_3834_; 
v___x_3832_ = lean_unsigned_to_nat(1u);
v___x_3833_ = lean_nat_sub(v___x_3829_, v___x_3832_);
v___x_3834_ = lean_nat_dec_le(v___x_3830_, v___x_3833_);
if (v___x_3834_ == 0)
{
lean_inc(v___x_3833_);
v___y_3820_ = v___x_3833_;
v___y_3821_ = v___x_3829_;
v___y_3822_ = v___x_3828_;
v___y_3823_ = v___x_3833_;
goto v___jp_3819_;
}
else
{
v___y_3820_ = v___x_3833_;
v___y_3821_ = v___x_3829_;
v___y_3822_ = v___x_3828_;
v___y_3823_ = v___x_3830_;
goto v___jp_3819_;
}
}
else
{
v___y_3806_ = v___x_3828_;
goto v___jp_3805_;
}
}
}
}
else
{
lean_dec(v___x_3797_);
return v___x_3798_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports___boxed(lean_object* v_cmd_3837_, lean_object* v_id_3838_, lean_object* v_dbg_x3f_3839_, lean_object* v_a_3840_, lean_object* v_a_3841_, lean_object* v_a_3842_){
_start:
{
uint8_t v_dbg_x3f_boxed_3843_; lean_object* v_res_3844_; 
v_dbg_x3f_boxed_3843_ = lean_unbox(v_dbg_x3f_3839_);
v_res_3844_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports(v_cmd_3837_, v_id_3838_, v_dbg_x3f_boxed_3843_, v_a_3840_, v_a_3841_);
lean_dec(v_a_3841_);
lean_dec_ref(v_a_3840_);
lean_dec(v_id_3838_);
return v_res_3844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0(lean_object* v_00_u03b2_3845_, lean_object* v_k_3846_, lean_object* v_t_3847_, lean_object* v_h_3848_){
_start:
{
lean_object* v___x_3849_; 
v___x_3849_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v_k_3846_, v_t_3847_);
return v___x_3849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___boxed(lean_object* v_00_u03b2_3850_, lean_object* v_k_3851_, lean_object* v_t_3852_, lean_object* v_h_3853_){
_start:
{
lean_object* v_res_3854_; 
v_res_3854_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0(v_00_u03b2_3850_, v_k_3851_, v_t_3852_, v_h_3853_);
lean_dec(v_k_3851_);
return v_res_3854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1(lean_object* v_00_u03b2_3855_, lean_object* v_m_3856_, lean_object* v_a_3857_, lean_object* v_b_3858_){
_start:
{
lean_object* v___x_3859_; 
v___x_3859_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1___redArg(v_m_3856_, v_a_3857_, v_b_3858_);
return v___x_3859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2(lean_object* v___x_3860_, lean_object* v_as_3861_, size_t v_sz_3862_, size_t v_i_3863_, lean_object* v_b_3864_, lean_object* v___y_3865_, lean_object* v___y_3866_){
_start:
{
lean_object* v___x_3868_; 
v___x_3868_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___redArg(v___x_3860_, v_as_3861_, v_sz_3862_, v_i_3863_, v_b_3864_);
return v___x_3868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2___boxed(lean_object* v___x_3869_, lean_object* v_as_3870_, lean_object* v_sz_3871_, lean_object* v_i_3872_, lean_object* v_b_3873_, lean_object* v___y_3874_, lean_object* v___y_3875_, lean_object* v___y_3876_){
_start:
{
size_t v_sz_boxed_3877_; size_t v_i_boxed_3878_; lean_object* v_res_3879_; 
v_sz_boxed_3877_ = lean_unbox_usize(v_sz_3871_);
lean_dec(v_sz_3871_);
v_i_boxed_3878_ = lean_unbox_usize(v_i_3872_);
lean_dec(v_i_3872_);
v_res_3879_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_MinImports_getAllImports_spec__2(v___x_3869_, v_as_3870_, v_sz_boxed_3877_, v_i_boxed_3878_, v_b_3873_, v___y_3874_, v___y_3875_);
lean_dec(v___y_3875_);
lean_dec_ref(v___y_3874_);
lean_dec_ref(v_as_3870_);
lean_dec_ref(v___x_3869_);
return v_res_3879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3(lean_object* v_00_u03b2_3880_, lean_object* v_m_3881_, lean_object* v_a_3882_){
_start:
{
lean_object* v___x_3883_; 
v___x_3883_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___redArg(v_m_3881_, v_a_3882_);
return v___x_3883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3___boxed(lean_object* v_00_u03b2_3884_, lean_object* v_m_3885_, lean_object* v_a_3886_){
_start:
{
lean_object* v_res_3887_; 
v_res_3887_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3(v_00_u03b2_3884_, v_m_3885_, v_a_3886_);
lean_dec(v_a_3886_);
lean_dec_ref(v_m_3885_);
return v_res_3887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5(lean_object* v___x_3888_, lean_object* v_a_3889_, lean_object* v_init_3890_, lean_object* v_x_3891_, lean_object* v___y_3892_, lean_object* v___y_3893_){
_start:
{
lean_object* v___x_3895_; 
v___x_3895_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___redArg(v___x_3888_, v_a_3889_, v_init_3890_, v_x_3891_);
return v___x_3895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5___boxed(lean_object* v___x_3896_, lean_object* v_a_3897_, lean_object* v_init_3898_, lean_object* v_x_3899_, lean_object* v___y_3900_, lean_object* v___y_3901_, lean_object* v___y_3902_){
_start:
{
lean_object* v_res_3903_; 
v_res_3903_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Command_MinImports_getAllImports_spec__5(v___x_3896_, v_a_3897_, v_init_3898_, v_x_3899_, v___y_3900_, v___y_3901_);
lean_dec(v___y_3901_);
lean_dec_ref(v___y_3900_);
lean_dec(v_x_3899_);
lean_dec_ref(v_a_3897_);
lean_dec_ref(v___x_3896_);
return v_res_3903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7(lean_object* v_init_3904_, lean_object* v_t_3905_){
_start:
{
lean_object* v___x_3906_; 
v___x_3906_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(v_init_3904_, v_t_3905_);
return v___x_3906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8(lean_object* v_n_3907_, lean_object* v_as_3908_, lean_object* v_lo_3909_, lean_object* v_hi_3910_, lean_object* v_w_3911_, lean_object* v_hlo_3912_, lean_object* v_hhi_3913_){
_start:
{
lean_object* v___x_3914_; 
v___x_3914_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(v_n_3907_, v_as_3908_, v_lo_3909_, v_hi_3910_);
return v___x_3914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___boxed(lean_object* v_n_3915_, lean_object* v_as_3916_, lean_object* v_lo_3917_, lean_object* v_hi_3918_, lean_object* v_w_3919_, lean_object* v_hlo_3920_, lean_object* v_hhi_3921_){
_start:
{
lean_object* v_res_3922_; 
v_res_3922_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8(v_n_3915_, v_as_3916_, v_lo_3917_, v_hi_3918_, v_w_3919_, v_hlo_3920_, v_hhi_3921_);
lean_dec(v_hi_3918_);
lean_dec(v_n_3915_);
return v_res_3922_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1(lean_object* v_00_u03b2_3923_, lean_object* v_a_3924_, lean_object* v_x_3925_){
_start:
{
uint8_t v___x_3926_; 
v___x_3926_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___redArg(v_a_3924_, v_x_3925_);
return v___x_3926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1___boxed(lean_object* v_00_u03b2_3927_, lean_object* v_a_3928_, lean_object* v_x_3929_){
_start:
{
uint8_t v_res_3930_; lean_object* v_r_3931_; 
v_res_3930_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__1(v_00_u03b2_3927_, v_a_3928_, v_x_3929_);
lean_dec(v_x_3929_);
lean_dec(v_a_3928_);
v_r_3931_ = lean_box(v_res_3930_);
return v_r_3931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2(lean_object* v_00_u03b2_3932_, lean_object* v_data_3933_){
_start:
{
lean_object* v___x_3934_; 
v___x_3934_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2___redArg(v_data_3933_);
return v___x_3934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3(lean_object* v_00_u03b2_3935_, lean_object* v_a_3936_, lean_object* v_b_3937_, lean_object* v_x_3938_){
_start:
{
lean_object* v___x_3939_; 
v___x_3939_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__3___redArg(v_a_3936_, v_b_3937_, v_x_3938_);
return v___x_3939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6(lean_object* v_00_u03b2_3940_, lean_object* v_a_3941_, lean_object* v_x_3942_){
_start:
{
lean_object* v___x_3943_; 
v___x_3943_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___redArg(v_a_3941_, v_x_3942_);
return v___x_3943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6___boxed(lean_object* v_00_u03b2_3944_, lean_object* v_a_3945_, lean_object* v_x_3946_){
_start:
{
lean_object* v_res_3947_; 
v_res_3947_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Command_MinImports_getAllImports_spec__3_spec__6(v_00_u03b2_3944_, v_a_3945_, v_x_3946_);
lean_dec(v_x_3946_);
lean_dec(v_a_3945_);
return v_res_3947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14(lean_object* v_n_3948_, lean_object* v_lo_3949_, lean_object* v_hi_3950_, lean_object* v_hhi_3951_, lean_object* v_pivot_3952_, lean_object* v_as_3953_, lean_object* v_i_3954_, lean_object* v_k_3955_, lean_object* v_ilo_3956_, lean_object* v_ik_3957_, lean_object* v_w_3958_){
_start:
{
lean_object* v___x_3959_; 
v___x_3959_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___redArg(v_hi_3950_, v_pivot_3952_, v_as_3953_, v_i_3954_, v_k_3955_);
return v___x_3959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14___boxed(lean_object* v_n_3960_, lean_object* v_lo_3961_, lean_object* v_hi_3962_, lean_object* v_hhi_3963_, lean_object* v_pivot_3964_, lean_object* v_as_3965_, lean_object* v_i_3966_, lean_object* v_k_3967_, lean_object* v_ilo_3968_, lean_object* v_ik_3969_, lean_object* v_w_3970_){
_start:
{
lean_object* v_res_3971_; 
v_res_3971_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8_spec__14(v_n_3960_, v_lo_3961_, v_hi_3962_, v_hhi_3963_, v_pivot_3964_, v_as_3965_, v_i_3966_, v_k_3967_, v_ilo_3968_, v_ik_3969_, v_w_3970_);
lean_dec(v_pivot_3964_);
lean_dec(v_hi_3962_);
lean_dec(v_lo_3961_);
lean_dec(v_n_3960_);
return v_res_3971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_3972_, lean_object* v_i_3973_, lean_object* v_source_3974_, lean_object* v_target_3975_){
_start:
{
lean_object* v___x_3976_; 
v___x_3976_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4___redArg(v_i_3973_, v_source_3974_, v_target_3975_);
return v___x_3976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11(lean_object* v_00_u03b2_3977_, lean_object* v_x_3978_, lean_object* v_x_3979_){
_start:
{
lean_object* v___x_3980_; 
v___x_3980_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Command_MinImports_getAllImports_spec__1_spec__2_spec__4_spec__11___redArg(v_x_3978_, v_x_3979_);
return v___x_3980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(lean_object* v_init_3981_, lean_object* v_x_3982_){
_start:
{
if (lean_obj_tag(v_x_3982_) == 0)
{
lean_object* v_k_3983_; lean_object* v_l_3984_; lean_object* v_r_3985_; lean_object* v___x_3986_; lean_object* v___x_3987_; 
v_k_3983_ = lean_ctor_get(v_x_3982_, 1);
v_l_3984_ = lean_ctor_get(v_x_3982_, 3);
v_r_3985_ = lean_ctor_get(v_x_3982_, 4);
v___x_3986_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(v_init_3981_, v_l_3984_);
v___x_3987_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Command_MinImports_getAllImports_spec__0___redArg(v_k_3983_, v___x_3986_);
v_init_3981_ = v___x_3987_;
v_x_3982_ = v_r_3985_;
goto _start;
}
else
{
return v_init_3981_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0___boxed(lean_object* v_init_3989_, lean_object* v_x_3990_){
_start:
{
lean_object* v_res_3991_; 
v_res_3991_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(v_init_3989_, v_x_3990_);
lean_dec(v_x_3990_);
return v_res_3991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(lean_object* v_env_3992_, lean_object* v_importNames_3993_){
_start:
{
lean_object* v___y_3995_; 
if (lean_obj_tag(v_importNames_3993_) == 0)
{
lean_object* v_size_4000_; 
v_size_4000_ = lean_ctor_get(v_importNames_3993_, 0);
lean_inc(v_size_4000_);
v___y_3995_ = v_size_4000_;
goto v___jp_3994_;
}
else
{
lean_object* v___x_4001_; 
v___x_4001_ = lean_unsigned_to_nat(0u);
v___y_3995_ = v___x_4001_;
goto v___jp_3994_;
}
v___jp_3994_:
{
lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; 
v___x_3996_ = lean_mk_empty_array_with_capacity(v___y_3995_);
lean_dec(v___y_3995_);
lean_inc(v_importNames_3993_);
v___x_3997_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(v___x_3996_, v_importNames_3993_);
v___x_3998_ = lp_importGraph_Lean_Environment_findRedundantImports(v_env_3992_, v___x_3997_);
lean_dec_ref(v___x_3997_);
v___x_3999_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(v_importNames_3993_, v___x_3998_);
lean_dec(v___x_3998_);
return v___x_3999_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports___boxed(lean_object* v_env_4002_, lean_object* v_importNames_4003_){
_start:
{
lean_object* v_res_4004_; 
v_res_4004_ = lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(v_env_4002_, v_importNames_4003_);
lean_dec_ref(v_env_4002_);
return v_res_4004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0(lean_object* v_init_4005_, lean_object* v_t_4006_){
_start:
{
lean_object* v___x_4007_; 
v___x_4007_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0_spec__0(v_init_4005_, v_t_4006_);
return v___x_4007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0___boxed(lean_object* v_init_4008_, lean_object* v_t_4009_){
_start:
{
lean_object* v_res_4010_; 
v_res_4010_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getIrredundantImports_spec__0(v_init_4008_, v_t_4009_);
lean_dec(v_t_4009_);
return v_res_4010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1(lean_object* v_ref_4011_, lean_object* v_msgData_4012_, lean_object* v___y_4013_, lean_object* v___y_4014_){
_start:
{
uint8_t v___x_4016_; uint8_t v___x_4017_; lean_object* v___x_4018_; 
v___x_4016_ = 0;
v___x_4017_ = 0;
v___x_4018_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00Lean_Elab_elabVisibility___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__0_spec__2_spec__5_spec__10_spec__14(v_ref_4011_, v_msgData_4012_, v___x_4016_, v___x_4017_, v___y_4013_, v___y_4014_);
return v___x_4018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1___boxed(lean_object* v_ref_4019_, lean_object* v_msgData_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_){
_start:
{
lean_object* v_res_4024_; 
v_res_4024_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1(v_ref_4019_, v_msgData_4020_, v___y_4021_, v___y_4022_);
lean_dec(v___y_4022_);
lean_dec_ref(v___y_4021_);
lean_dec(v_ref_4019_);
return v_res_4024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0(size_t v_sz_4026_, size_t v_i_4027_, lean_object* v_bs_4028_){
_start:
{
uint8_t v___x_4029_; 
v___x_4029_ = lean_usize_dec_lt(v_i_4027_, v_sz_4026_);
if (v___x_4029_ == 0)
{
return v_bs_4028_;
}
else
{
lean_object* v_v_4030_; lean_object* v___x_4031_; lean_object* v_bs_x27_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; size_t v___x_4036_; size_t v___x_4037_; lean_object* v___x_4038_; 
v_v_4030_ = lean_array_uget(v_bs_4028_, v_i_4027_);
v___x_4031_ = lean_unsigned_to_nat(0u);
v_bs_x27_4032_ = lean_array_uset(v_bs_4028_, v_i_4027_, v___x_4031_);
v___x_4033_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___closed__0));
v___x_4034_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_v_4030_, v___x_4029_);
v___x_4035_ = lean_string_append(v___x_4033_, v___x_4034_);
lean_dec_ref(v___x_4034_);
v___x_4036_ = ((size_t)1ULL);
v___x_4037_ = lean_usize_add(v_i_4027_, v___x_4036_);
v___x_4038_ = lean_array_uset(v_bs_x27_4032_, v_i_4027_, v___x_4035_);
v_i_4027_ = v___x_4037_;
v_bs_4028_ = v___x_4038_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0___boxed(lean_object* v_sz_4040_, lean_object* v_i_4041_, lean_object* v_bs_4042_){
_start:
{
size_t v_sz_boxed_4043_; size_t v_i_boxed_4044_; lean_object* v_res_4045_; 
v_sz_boxed_4043_ = lean_unbox_usize(v_sz_4040_);
lean_dec(v_sz_4040_);
v_i_boxed_4044_ = lean_unbox_usize(v_i_4041_);
lean_dec(v_i_4041_);
v_res_4045_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0(v_sz_boxed_4043_, v_i_boxed_4044_, v_bs_4042_);
return v_res_4045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2(lean_object* v_as_4046_, size_t v_i_4047_, size_t v_stop_4048_, lean_object* v_b_4049_){
_start:
{
lean_object* v___y_4051_; uint8_t v___x_4055_; 
v___x_4055_ = lean_usize_dec_eq(v_i_4047_, v_stop_4048_);
if (v___x_4055_ == 0)
{
lean_object* v___x_4056_; uint8_t v___x_4057_; 
v___x_4056_ = lean_array_uget_borrowed(v_as_4046_, v_i_4047_);
v___x_4057_ = lp_mathlib_Mathlib_Command_MinImports_isInitImport(v___x_4056_);
if (v___x_4057_ == 0)
{
lean_object* v___x_4058_; 
lean_inc(v___x_4056_);
v___x_4058_ = lean_array_push(v_b_4049_, v___x_4056_);
v___y_4051_ = v___x_4058_;
goto v___jp_4050_;
}
else
{
v___y_4051_ = v_b_4049_;
goto v___jp_4050_;
}
}
else
{
return v_b_4049_;
}
v___jp_4050_:
{
size_t v___x_4052_; size_t v___x_4053_; 
v___x_4052_ = ((size_t)1ULL);
v___x_4053_ = lean_usize_add(v_i_4047_, v___x_4052_);
v_i_4047_ = v___x_4053_;
v_b_4049_ = v___y_4051_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2___boxed(lean_object* v_as_4059_, lean_object* v_i_4060_, lean_object* v_stop_4061_, lean_object* v_b_4062_){
_start:
{
size_t v_i_boxed_4063_; size_t v_stop_boxed_4064_; lean_object* v_res_4065_; 
v_i_boxed_4063_ = lean_unbox_usize(v_i_4060_);
lean_dec(v_i_4060_);
v_stop_boxed_4064_ = lean_unbox_usize(v_stop_4061_);
lean_dec(v_stop_4061_);
v_res_4065_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2(v_as_4059_, v_i_boxed_4063_, v_stop_boxed_4064_, v_b_4062_);
lean_dec_ref(v_as_4059_);
return v_res_4065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore(lean_object* v_stx_4069_, lean_object* v_id_4070_, lean_object* v_a_4071_, lean_object* v_a_4072_){
_start:
{
lean_object* v___x_4074_; uint8_t v___x_4075_; lean_object* v___x_4076_; 
v___x_4074_ = lean_st_ref_get(v_a_4072_);
v___x_4075_ = 0;
v___x_4076_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports(v_stx_4069_, v_id_4070_, v___x_4075_, v_a_4071_, v_a_4072_);
if (lean_obj_tag(v___x_4076_) == 0)
{
lean_object* v_a_4077_; lean_object* v___y_4079_; lean_object* v___y_4099_; lean_object* v___y_4100_; lean_object* v___y_4101_; lean_object* v___y_4102_; lean_object* v___y_4105_; lean_object* v___y_4106_; lean_object* v___y_4107_; lean_object* v___y_4108_; lean_object* v___y_4111_; lean_object* v___y_4112_; lean_object* v_env_4118_; lean_object* v___x_4119_; lean_object* v___y_4121_; 
v_a_4077_ = lean_ctor_get(v___x_4076_, 0);
lean_inc(v_a_4077_);
lean_dec_ref_known(v___x_4076_, 1);
v_env_4118_ = lean_ctor_get(v___x_4074_, 0);
lean_inc_ref(v_env_4118_);
lean_dec(v___x_4074_);
v___x_4119_ = lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(v_env_4118_, v_a_4077_);
lean_dec_ref(v_env_4118_);
if (lean_obj_tag(v___x_4119_) == 0)
{
lean_object* v_size_4135_; 
v_size_4135_ = lean_ctor_get(v___x_4119_, 0);
lean_inc(v_size_4135_);
v___y_4121_ = v_size_4135_;
goto v___jp_4120_;
}
else
{
lean_object* v___x_4136_; 
v___x_4136_ = lean_unsigned_to_nat(0u);
v___y_4121_ = v___x_4136_;
goto v___jp_4120_;
}
v___jp_4078_:
{
lean_object* v___x_4080_; 
v___x_4080_ = l_Lean_Elab_Command_getRef___redArg(v_a_4071_);
if (lean_obj_tag(v___x_4080_) == 0)
{
lean_object* v_a_4081_; lean_object* v___x_4082_; size_t v_sz_4083_; size_t v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; 
v_a_4081_ = lean_ctor_get(v___x_4080_, 0);
lean_inc(v_a_4081_);
lean_dec_ref_known(v___x_4080_, 1);
v___x_4082_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__0));
v_sz_4083_ = lean_array_size(v___y_4079_);
v___x_4084_ = ((size_t)0ULL);
v___x_4085_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_MinImports_minImpsCore_spec__0(v_sz_4083_, v___x_4084_, v___y_4079_);
v___x_4086_ = lean_array_to_list(v___x_4085_);
v___x_4087_ = l_String_intercalate(v___x_4082_, v___x_4086_);
v___x_4088_ = l_Lean_stringToMessageData(v___x_4087_);
v___x_4089_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Command_MinImports_minImpsCore_spec__1(v_a_4081_, v___x_4088_, v_a_4071_, v_a_4072_);
lean_dec(v_a_4081_);
return v___x_4089_;
}
else
{
lean_object* v_a_4090_; lean_object* v___x_4092_; uint8_t v_isShared_4093_; uint8_t v_isSharedCheck_4097_; 
lean_dec_ref(v___y_4079_);
v_a_4090_ = lean_ctor_get(v___x_4080_, 0);
v_isSharedCheck_4097_ = !lean_is_exclusive(v___x_4080_);
if (v_isSharedCheck_4097_ == 0)
{
v___x_4092_ = v___x_4080_;
v_isShared_4093_ = v_isSharedCheck_4097_;
goto v_resetjp_4091_;
}
else
{
lean_inc(v_a_4090_);
lean_dec(v___x_4080_);
v___x_4092_ = lean_box(0);
v_isShared_4093_ = v_isSharedCheck_4097_;
goto v_resetjp_4091_;
}
v_resetjp_4091_:
{
lean_object* v___x_4095_; 
if (v_isShared_4093_ == 0)
{
v___x_4095_ = v___x_4092_;
goto v_reusejp_4094_;
}
else
{
lean_object* v_reuseFailAlloc_4096_; 
v_reuseFailAlloc_4096_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4096_, 0, v_a_4090_);
v___x_4095_ = v_reuseFailAlloc_4096_;
goto v_reusejp_4094_;
}
v_reusejp_4094_:
{
return v___x_4095_;
}
}
}
}
v___jp_4098_:
{
lean_object* v___x_4103_; 
v___x_4103_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Command_MinImports_getAllImports_spec__8___redArg(v___y_4100_, v___y_4101_, v___y_4099_, v___y_4102_);
lean_dec(v___y_4102_);
lean_dec(v___y_4100_);
v___y_4079_ = v___x_4103_;
goto v___jp_4078_;
}
v___jp_4104_:
{
uint8_t v___x_4109_; 
v___x_4109_ = lean_nat_dec_le(v___y_4108_, v___y_4105_);
if (v___x_4109_ == 0)
{
lean_dec(v___y_4105_);
lean_inc(v___y_4108_);
v___y_4099_ = v___y_4108_;
v___y_4100_ = v___y_4106_;
v___y_4101_ = v___y_4107_;
v___y_4102_ = v___y_4108_;
goto v___jp_4098_;
}
else
{
v___y_4099_ = v___y_4108_;
v___y_4100_ = v___y_4106_;
v___y_4101_ = v___y_4107_;
v___y_4102_ = v___y_4105_;
goto v___jp_4098_;
}
}
v___jp_4110_:
{
lean_object* v___x_4113_; uint8_t v___x_4114_; 
v___x_4113_ = lean_array_get_size(v___y_4112_);
v___x_4114_ = lean_nat_dec_eq(v___x_4113_, v___y_4111_);
if (v___x_4114_ == 0)
{
lean_object* v___x_4115_; lean_object* v___x_4116_; uint8_t v___x_4117_; 
v___x_4115_ = lean_unsigned_to_nat(1u);
v___x_4116_ = lean_nat_sub(v___x_4113_, v___x_4115_);
v___x_4117_ = lean_nat_dec_le(v___y_4111_, v___x_4116_);
if (v___x_4117_ == 0)
{
lean_dec(v___y_4111_);
lean_inc(v___x_4116_);
v___y_4105_ = v___x_4116_;
v___y_4106_ = v___x_4113_;
v___y_4107_ = v___y_4112_;
v___y_4108_ = v___x_4116_;
goto v___jp_4104_;
}
else
{
v___y_4105_ = v___x_4116_;
v___y_4106_ = v___x_4113_;
v___y_4107_ = v___y_4112_;
v___y_4108_ = v___y_4111_;
goto v___jp_4104_;
}
}
else
{
lean_dec(v___y_4111_);
v___y_4079_ = v___y_4112_;
goto v___jp_4078_;
}
}
v___jp_4120_:
{
lean_object* v___x_4122_; lean_object* v___x_4123_; lean_object* v___x_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; uint8_t v___x_4127_; 
v___x_4122_ = lean_mk_empty_array_with_capacity(v___y_4121_);
lean_dec(v___y_4121_);
v___x_4123_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Command_MinImports_getAllImports_spec__7_spec__12(v___x_4122_, v___x_4119_);
v___x_4124_ = lean_unsigned_to_nat(0u);
v___x_4125_ = lean_array_get_size(v___x_4123_);
v___x_4126_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_minImpsCore___closed__1));
v___x_4127_ = lean_nat_dec_lt(v___x_4124_, v___x_4125_);
if (v___x_4127_ == 0)
{
lean_dec_ref(v___x_4123_);
v___y_4111_ = v___x_4124_;
v___y_4112_ = v___x_4126_;
goto v___jp_4110_;
}
else
{
uint8_t v___x_4128_; 
v___x_4128_ = lean_nat_dec_le(v___x_4125_, v___x_4125_);
if (v___x_4128_ == 0)
{
if (v___x_4127_ == 0)
{
lean_dec_ref(v___x_4123_);
v___y_4111_ = v___x_4124_;
v___y_4112_ = v___x_4126_;
goto v___jp_4110_;
}
else
{
size_t v___x_4129_; size_t v___x_4130_; lean_object* v___x_4131_; 
v___x_4129_ = ((size_t)0ULL);
v___x_4130_ = lean_usize_of_nat(v___x_4125_);
v___x_4131_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2(v___x_4123_, v___x_4129_, v___x_4130_, v___x_4126_);
lean_dec_ref(v___x_4123_);
v___y_4111_ = v___x_4124_;
v___y_4112_ = v___x_4131_;
goto v___jp_4110_;
}
}
else
{
size_t v___x_4132_; size_t v___x_4133_; lean_object* v___x_4134_; 
v___x_4132_ = ((size_t)0ULL);
v___x_4133_ = lean_usize_of_nat(v___x_4125_);
v___x_4134_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_MinImports_minImpsCore_spec__2(v___x_4123_, v___x_4132_, v___x_4133_, v___x_4126_);
lean_dec_ref(v___x_4123_);
v___y_4111_ = v___x_4124_;
v___y_4112_ = v___x_4134_;
goto v___jp_4110_;
}
}
}
}
else
{
lean_object* v_a_4137_; lean_object* v___x_4139_; uint8_t v_isShared_4140_; uint8_t v_isSharedCheck_4144_; 
lean_dec(v___x_4074_);
v_a_4137_ = lean_ctor_get(v___x_4076_, 0);
v_isSharedCheck_4144_ = !lean_is_exclusive(v___x_4076_);
if (v_isSharedCheck_4144_ == 0)
{
v___x_4139_ = v___x_4076_;
v_isShared_4140_ = v_isSharedCheck_4144_;
goto v_resetjp_4138_;
}
else
{
lean_inc(v_a_4137_);
lean_dec(v___x_4076_);
v___x_4139_ = lean_box(0);
v_isShared_4140_ = v_isSharedCheck_4144_;
goto v_resetjp_4138_;
}
v_resetjp_4138_:
{
lean_object* v___x_4142_; 
if (v_isShared_4140_ == 0)
{
v___x_4142_ = v___x_4139_;
goto v_reusejp_4141_;
}
else
{
lean_object* v_reuseFailAlloc_4143_; 
v_reuseFailAlloc_4143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4143_, 0, v_a_4137_);
v___x_4142_ = v_reuseFailAlloc_4143_;
goto v_reusejp_4141_;
}
v_reusejp_4141_:
{
return v___x_4142_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports_minImpsCore___boxed(lean_object* v_stx_4145_, lean_object* v_id_4146_, lean_object* v_a_4147_, lean_object* v_a_4148_, lean_object* v_a_4149_){
_start:
{
lean_object* v_res_4150_; 
v_res_4150_ = lp_mathlib_Mathlib_Command_MinImports_minImpsCore(v_stx_4145_, v_id_4146_, v_a_4147_, v_a_4148_);
lean_dec(v_a_4148_);
lean_dec_ref(v_a_4147_);
lean_dec(v_id_4146_);
return v_res_4150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__minImpsStx__1(lean_object* v_x_4208_, lean_object* v_a_4209_, lean_object* v_a_4210_){
_start:
{
lean_object* v___x_4212_; uint8_t v___x_4213_; 
v___x_4212_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_minImpsStx___closed__3));
lean_inc(v_x_4208_);
v___x_4213_ = l_Lean_Syntax_isOfKind(v_x_4208_, v___x_4212_);
if (v___x_4213_ == 0)
{
lean_object* v___x_4214_; 
lean_dec(v_x_4208_);
v___x_4214_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
return v___x_4214_;
}
else
{
lean_object* v___x_4215_; lean_object* v___x_4216_; lean_object* v___x_4217_; 
v___x_4215_ = lean_unsigned_to_nat(2u);
v___x_4216_ = l_Lean_Syntax_getArg(v_x_4208_, v___x_4215_);
lean_dec(v_x_4208_);
lean_inc(v___x_4216_);
v___x_4217_ = lp_mathlib_Mathlib_Command_MinImports_getId(v___x_4216_, v_a_4209_, v_a_4210_);
if (lean_obj_tag(v___x_4217_) == 0)
{
lean_object* v_a_4218_; lean_object* v___y_4220_; lean_object* v___x_4222_; 
v_a_4218_ = lean_ctor_get(v___x_4217_, 0);
lean_inc(v_a_4218_);
lean_dec_ref_known(v___x_4217_, 1);
lean_inc(v___x_4216_);
v___x_4222_ = l_Lean_Elab_Command_elabCommand(v___x_4216_, v_a_4209_, v_a_4210_);
if (lean_obj_tag(v___x_4222_) == 0)
{
v___y_4220_ = v___x_4222_;
goto v___jp_4219_;
}
else
{
lean_object* v_a_4223_; uint8_t v___x_4224_; 
v_a_4223_ = lean_ctor_get(v___x_4222_, 0);
lean_inc(v_a_4223_);
v___x_4224_ = l_Lean_Exception_isInterrupt(v_a_4223_);
lean_dec(v_a_4223_);
if (v___x_4224_ == 0)
{
lean_object* v___x_4225_; 
lean_dec_ref_known(v___x_4222_, 1);
v___x_4225_ = lp_mathlib_Mathlib_Command_MinImports_minImpsCore(v___x_4216_, v_a_4218_, v_a_4209_, v_a_4210_);
lean_dec(v_a_4218_);
return v___x_4225_;
}
else
{
v___y_4220_ = v___x_4222_;
goto v___jp_4219_;
}
}
v___jp_4219_:
{
if (lean_obj_tag(v___y_4220_) == 0)
{
lean_object* v___x_4221_; 
lean_dec_ref_known(v___y_4220_, 1);
v___x_4221_ = lp_mathlib_Mathlib_Command_MinImports_minImpsCore(v___x_4216_, v_a_4218_, v_a_4209_, v_a_4210_);
lean_dec(v_a_4218_);
return v___x_4221_;
}
else
{
lean_dec(v_a_4218_);
lean_dec(v___x_4216_);
return v___y_4220_;
}
}
}
else
{
lean_object* v_a_4226_; lean_object* v___x_4228_; uint8_t v_isShared_4229_; uint8_t v_isSharedCheck_4233_; 
lean_dec(v___x_4216_);
v_a_4226_ = lean_ctor_get(v___x_4217_, 0);
v_isSharedCheck_4233_ = !lean_is_exclusive(v___x_4217_);
if (v_isSharedCheck_4233_ == 0)
{
v___x_4228_ = v___x_4217_;
v_isShared_4229_ = v_isSharedCheck_4233_;
goto v_resetjp_4227_;
}
else
{
lean_inc(v_a_4226_);
lean_dec(v___x_4217_);
v___x_4228_ = lean_box(0);
v_isShared_4229_ = v_isSharedCheck_4233_;
goto v_resetjp_4227_;
}
v_resetjp_4227_:
{
lean_object* v___x_4231_; 
if (v_isShared_4229_ == 0)
{
v___x_4231_ = v___x_4228_;
goto v_reusejp_4230_;
}
else
{
lean_object* v_reuseFailAlloc_4232_; 
v_reuseFailAlloc_4232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4232_, 0, v_a_4226_);
v___x_4231_ = v_reuseFailAlloc_4232_;
goto v_reusejp_4230_;
}
v_reusejp_4230_:
{
return v___x_4231_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__minImpsStx__1___boxed(lean_object* v_x_4234_, lean_object* v_a_4235_, lean_object* v_a_4236_, lean_object* v_a_4237_){
_start:
{
lean_object* v_res_4238_; 
v_res_4238_ = lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__minImpsStx__1(v_x_4234_, v_a_4235_, v_a_4236_);
lean_dec(v_a_4236_);
lean_dec_ref(v_a_4235_);
return v_res_4238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__command_x23min__importsIn____1(lean_object* v_x_4239_, lean_object* v_a_4240_, lean_object* v_a_4241_){
_start:
{
lean_object* v___x_4243_; uint8_t v___x_4244_; 
v___x_4243_ = ((lean_object*)(lp_mathlib_Mathlib_Command_MinImports_command_x23min__importsIn___00__closed__1));
lean_inc(v_x_4239_);
v___x_4244_ = l_Lean_Syntax_isOfKind(v_x_4239_, v___x_4243_);
if (v___x_4244_ == 0)
{
lean_object* v___x_4245_; 
lean_dec(v_x_4239_);
v___x_4245_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Lean_Elab_elabDeclAttrs___at___00Lean_Elab_elabModifiers___at___00Mathlib_Command_MinImports_getDeclName_spec__0_spec__1_spec__4_spec__9_spec__16_spec__26___redArg();
return v___x_4245_;
}
else
{
lean_object* v___x_4246_; lean_object* v___x_4247_; lean_object* v___x_4248_; 
v___x_4246_ = lean_unsigned_to_nat(2u);
v___x_4247_ = l_Lean_Syntax_getArg(v_x_4239_, v___x_4246_);
lean_dec(v_x_4239_);
lean_inc(v___x_4247_);
v___x_4248_ = lp_mathlib_Mathlib_Command_MinImports_minImpsCore(v___x_4247_, v___x_4247_, v_a_4240_, v_a_4241_);
lean_dec(v___x_4247_);
return v___x_4248_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__command_x23min__importsIn____1___boxed(lean_object* v_x_4249_, lean_object* v_a_4250_, lean_object* v_a_4251_, lean_object* v_a_4252_){
_start:
{
lean_object* v_res_4253_; 
v_res_4253_ = lp_mathlib_Mathlib_Command_MinImports___aux__Mathlib__Tactic__MinImports______elabRules__Mathlib__Command__MinImports__command_x23min__importsIn____1(v_x_4249_, v_a_4250_, v_a_4251_);
lean_dec(v_a_4251_);
lean_dec_ref(v_a_4250_);
return v_res_4253_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_DeclModifiers(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_DeclModifiers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_DefView(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_CollectAxioms(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_DefView(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_CollectAxioms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_DefView(uint8_t builtin);
lean_object* initialize_Lean_Util_CollectAxioms(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Elab_DeclModifiers(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_DefView(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_CollectAxioms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_DeclModifiers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
}
#ifdef __cplusplus
}
#endif
