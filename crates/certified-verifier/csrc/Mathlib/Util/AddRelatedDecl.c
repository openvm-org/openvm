// Lean compiler output
// Module: Mathlib.Util.AddRelatedDecl
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.DeclarationRange
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_privateToUserName_x3f(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_Elab_expandMacroImpl_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkPrivateName(lean_object*, lean_object*);
lean_object* l_Lean_privateToUserName(lean_object*);
lean_object* l_Lean_ResolveName_resolveNamespace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* l_Lean_instInhabitedEffectiveImport_default;
lean_object* l_Lean_instHashableExtraModUse_hash___boxed(lean_object*);
lean_object* l_Lean_instBEqExtraModUse_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_extraModUses;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableExtraModUse_hash(lean_object*);
uint8_t l_Lean_instBEqExtraModUse_beq(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Std_HashMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_indirectModUseExt;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
uint8_t l_Lean_isMarkedMeta(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t lean_is_reserved_name(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
extern lean_object* l_Lean_declRangeExt;
lean_object* l_Lean_MapDeclarationExtension_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_toAttributeKind___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_expandMacros(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getAttributeImpl(lean_object*, lean_object*);
extern lean_object* l_Lean_regularInitAttr;
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_InternalExceptionId_getName(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
uint8_t l_Lean_Elab_isAbortExceptionId(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Term_applyAttributes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_DeclarationRange_ofStringPositions(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Environment_hasUnsafe(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t l_Lean_getReducibilityStatusCore(lean_object*, lean_object*);
uint8_t l_Lean_Meta_isInstanceCore(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Meta_check(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_inferDefEqAttr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_docStringExt;
lean_object* l_String_removeLeadingSpaces(lean_object*);
lean_object* l_Lean_findDocString_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_levelParams(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addDecl(lean_object*, uint8_t, lean_object*, lean_object*);
uint8_t l_Lean_isProtected(lean_object*, lean_object*);
lean_object* l_Lean_addProtected(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optAttrArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 54, 53, 112, 124, 81, 61, 225)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "attr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "attrInstance"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__21_value),LEAN_SCALAR_PTR_LITERAL(241, 75, 242, 110, 47, 5, 20, 104)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__28_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_optAttrArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__34_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_optAttrArg = (const lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__34_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqExtraModUse_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__0_value;
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableExtraModUse_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__3 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__3_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__4_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " extra mod use "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__5 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " of "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__7 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__10 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recording "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__13 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__15 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "regular"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__17 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__17_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__18 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__18_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__19 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__19_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__20 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2;
static const lean_array_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 158, .m_capacity = 158, .m_length = 157, .m_data = "maximum recursion depth has been reached\nuse `set_option maxRecDepth <num>` to increase limit\nuse `set_option diagnostics true` to get diagnostic information"};
static const lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Cannot use attribute `["};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "]`: module `"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = "` is loaded for IR only (reached as a private `meta` dependency). Add an import of `"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Unknown attribute `["};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "]`"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(107, 67, 254, 234, 65, 174, 209, 53)}};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Unknown attribute"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_optAttrArg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 150, 238, 148, 228, 221, 116, 224)}};
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "internal exception: "};
static const lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_elabOptAttrArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabOptAttrArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabOptAttrArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabOptAttrArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabOptAttrArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "tacticCheckInstances"};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 15, 63, 147, 29, 186, 208, 53)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "generated lemma "};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 117, .m_capacity = 117, .m_length = 116, .m_data = " is not type-correct at `.implicit` transparency; consider marking some of the following as `@[implicit_reducible]`:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_addRelatedDecl_spec__4(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "invalid doc string, declaration `"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__0 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1;
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is in an imported module"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__2 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "` has already been declared"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "a non-private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "a private declaration `"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "` is a reserved name"};
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___boxed, .m_arity = 7, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\n\n---\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Related declaration is not a proposition: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2(uint8_t v___x_77_, lean_object* v_as_78_, size_t v_i_79_, size_t v_stop_80_, lean_object* v_b_81_){
_start:
{
lean_object* v___y_83_; uint8_t v___x_87_; 
v___x_87_ = lean_usize_dec_eq(v_i_79_, v_stop_80_);
if (v___x_87_ == 0)
{
lean_object* v_fst_88_; uint8_t v___x_89_; 
v_fst_88_ = lean_ctor_get(v_b_81_, 0);
v___x_89_ = lean_unbox(v_fst_88_);
if (v___x_89_ == 0)
{
lean_object* v_snd_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_98_; 
v_snd_90_ = lean_ctor_get(v_b_81_, 1);
v_isSharedCheck_98_ = !lean_is_exclusive(v_b_81_);
if (v_isSharedCheck_98_ == 0)
{
lean_object* v_unused_99_; 
v_unused_99_ = lean_ctor_get(v_b_81_, 0);
lean_dec(v_unused_99_);
v___x_92_ = v_b_81_;
v_isShared_93_ = v_isSharedCheck_98_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_snd_90_);
lean_dec(v_b_81_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_98_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_94_; lean_object* v___x_96_; 
v___x_94_ = lean_box(v___x_77_);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 0, v___x_94_);
v___x_96_ = v___x_92_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v___x_94_);
lean_ctor_set(v_reuseFailAlloc_97_, 1, v_snd_90_);
v___x_96_ = v_reuseFailAlloc_97_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
v___y_83_ = v___x_96_;
goto v___jp_82_;
}
}
}
else
{
lean_object* v_snd_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_110_; 
v_snd_100_ = lean_ctor_get(v_b_81_, 1);
v_isSharedCheck_110_ = !lean_is_exclusive(v_b_81_);
if (v_isSharedCheck_110_ == 0)
{
lean_object* v_unused_111_; 
v_unused_111_ = lean_ctor_get(v_b_81_, 0);
lean_dec(v_unused_111_);
v___x_102_ = v_b_81_;
v_isShared_103_ = v_isSharedCheck_110_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_snd_100_);
lean_dec(v_b_81_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_110_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_108_; 
v___x_104_ = lean_array_uget_borrowed(v_as_78_, v_i_79_);
lean_inc(v___x_104_);
v___x_105_ = lean_array_push(v_snd_100_, v___x_104_);
v___x_106_ = lean_box(v___x_87_);
if (v_isShared_103_ == 0)
{
lean_ctor_set(v___x_102_, 1, v___x_105_);
lean_ctor_set(v___x_102_, 0, v___x_106_);
v___x_108_ = v___x_102_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v___x_106_);
lean_ctor_set(v_reuseFailAlloc_109_, 1, v___x_105_);
v___x_108_ = v_reuseFailAlloc_109_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
v___y_83_ = v___x_108_;
goto v___jp_82_;
}
}
}
}
else
{
return v_b_81_;
}
v___jp_82_:
{
size_t v___x_84_; size_t v___x_85_; 
v___x_84_ = ((size_t)1ULL);
v___x_85_ = lean_usize_add(v_i_79_, v___x_84_);
v_i_79_ = v___x_85_;
v_b_81_ = v___y_83_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2___boxed(lean_object* v___x_112_, lean_object* v_as_113_, lean_object* v_i_114_, lean_object* v_stop_115_, lean_object* v_b_116_){
_start:
{
uint8_t v___x_28187__boxed_117_; size_t v_i_boxed_118_; size_t v_stop_boxed_119_; lean_object* v_res_120_; 
v___x_28187__boxed_117_ = lean_unbox(v___x_112_);
v_i_boxed_118_ = lean_unbox_usize(v_i_114_);
lean_dec(v_i_114_);
v_stop_boxed_119_ = lean_unbox_usize(v_stop_115_);
lean_dec(v_stop_115_);
v_res_120_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2(v___x_28187__boxed_117_, v_as_113_, v_i_boxed_118_, v_stop_boxed_119_, v_b_116_);
lean_dec_ref(v_as_113_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(lean_object* v___y_121_, uint8_t v_isExporting_122_, lean_object* v___x_123_, lean_object* v___y_124_, lean_object* v___x_125_, lean_object* v_a_x3f_126_){
_start:
{
lean_object* v___x_128_; lean_object* v_env_129_; lean_object* v_nextMacroScope_130_; lean_object* v_ngen_131_; lean_object* v_auxDeclNGen_132_; lean_object* v_traceState_133_; lean_object* v_messages_134_; lean_object* v_infoState_135_; lean_object* v_snapshotTasks_136_; lean_object* v___x_138_; uint8_t v_isShared_139_; uint8_t v_isSharedCheck_161_; 
v___x_128_ = lean_st_ref_take(v___y_121_);
v_env_129_ = lean_ctor_get(v___x_128_, 0);
v_nextMacroScope_130_ = lean_ctor_get(v___x_128_, 1);
v_ngen_131_ = lean_ctor_get(v___x_128_, 2);
v_auxDeclNGen_132_ = lean_ctor_get(v___x_128_, 3);
v_traceState_133_ = lean_ctor_get(v___x_128_, 4);
v_messages_134_ = lean_ctor_get(v___x_128_, 6);
v_infoState_135_ = lean_ctor_get(v___x_128_, 7);
v_snapshotTasks_136_ = lean_ctor_get(v___x_128_, 8);
v_isSharedCheck_161_ = !lean_is_exclusive(v___x_128_);
if (v_isSharedCheck_161_ == 0)
{
lean_object* v_unused_162_; 
v_unused_162_ = lean_ctor_get(v___x_128_, 5);
lean_dec(v_unused_162_);
v___x_138_ = v___x_128_;
v_isShared_139_ = v_isSharedCheck_161_;
goto v_resetjp_137_;
}
else
{
lean_inc(v_snapshotTasks_136_);
lean_inc(v_infoState_135_);
lean_inc(v_messages_134_);
lean_inc(v_traceState_133_);
lean_inc(v_auxDeclNGen_132_);
lean_inc(v_ngen_131_);
lean_inc(v_nextMacroScope_130_);
lean_inc(v_env_129_);
lean_dec(v___x_128_);
v___x_138_ = lean_box(0);
v_isShared_139_ = v_isSharedCheck_161_;
goto v_resetjp_137_;
}
v_resetjp_137_:
{
lean_object* v___x_140_; lean_object* v___x_142_; 
v___x_140_ = l_Lean_Environment_setExporting(v_env_129_, v_isExporting_122_);
if (v_isShared_139_ == 0)
{
lean_ctor_set(v___x_138_, 5, v___x_123_);
lean_ctor_set(v___x_138_, 0, v___x_140_);
v___x_142_ = v___x_138_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v___x_140_);
lean_ctor_set(v_reuseFailAlloc_160_, 1, v_nextMacroScope_130_);
lean_ctor_set(v_reuseFailAlloc_160_, 2, v_ngen_131_);
lean_ctor_set(v_reuseFailAlloc_160_, 3, v_auxDeclNGen_132_);
lean_ctor_set(v_reuseFailAlloc_160_, 4, v_traceState_133_);
lean_ctor_set(v_reuseFailAlloc_160_, 5, v___x_123_);
lean_ctor_set(v_reuseFailAlloc_160_, 6, v_messages_134_);
lean_ctor_set(v_reuseFailAlloc_160_, 7, v_infoState_135_);
lean_ctor_set(v_reuseFailAlloc_160_, 8, v_snapshotTasks_136_);
v___x_142_ = v_reuseFailAlloc_160_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v_mctx_145_; lean_object* v_zetaDeltaFVarIds_146_; lean_object* v_postponed_147_; lean_object* v_diag_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_158_; 
v___x_143_ = lean_st_ref_set(v___y_121_, v___x_142_);
v___x_144_ = lean_st_ref_take(v___y_124_);
v_mctx_145_ = lean_ctor_get(v___x_144_, 0);
v_zetaDeltaFVarIds_146_ = lean_ctor_get(v___x_144_, 2);
v_postponed_147_ = lean_ctor_get(v___x_144_, 3);
v_diag_148_ = lean_ctor_get(v___x_144_, 4);
v_isSharedCheck_158_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_158_ == 0)
{
lean_object* v_unused_159_; 
v_unused_159_ = lean_ctor_get(v___x_144_, 1);
lean_dec(v_unused_159_);
v___x_150_ = v___x_144_;
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_diag_148_);
lean_inc(v_postponed_147_);
lean_inc(v_zetaDeltaFVarIds_146_);
lean_inc(v_mctx_145_);
lean_dec(v___x_144_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 1, v___x_125_);
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_mctx_145_);
lean_ctor_set(v_reuseFailAlloc_157_, 1, v___x_125_);
lean_ctor_set(v_reuseFailAlloc_157_, 2, v_zetaDeltaFVarIds_146_);
lean_ctor_set(v_reuseFailAlloc_157_, 3, v_postponed_147_);
lean_ctor_set(v_reuseFailAlloc_157_, 4, v_diag_148_);
v___x_153_ = v_reuseFailAlloc_157_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lean_st_ref_set(v___y_124_, v___x_153_);
v___x_155_ = lean_box(0);
v___x_156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
return v___x_156_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0___boxed(lean_object* v___y_163_, lean_object* v_isExporting_164_, lean_object* v___x_165_, lean_object* v___y_166_, lean_object* v___x_167_, lean_object* v_a_x3f_168_, lean_object* v___y_169_){
_start:
{
uint8_t v_isExporting_boxed_170_; lean_object* v_res_171_; 
v_isExporting_boxed_170_ = lean_unbox(v_isExporting_164_);
v_res_171_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(v___y_163_, v_isExporting_boxed_170_, v___x_165_, v___y_166_, v___x_167_, v_a_x3f_168_);
lean_dec(v_a_x3f_168_);
lean_dec(v___y_166_);
lean_dec(v___y_163_);
return v_res_171_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0(void){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_172_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__0);
v___x_174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_175_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1);
v___x_176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
return v___x_176_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3(void){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__1);
v___x_178_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_177_);
lean_ctor_set(v___x_178_, 2, v___x_177_);
lean_ctor_set(v___x_178_, 3, v___x_177_);
lean_ctor_set(v___x_178_, 4, v___x_177_);
lean_ctor_set(v___x_178_, 5, v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg(lean_object* v_x_179_, uint8_t v_isExporting_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_){
_start:
{
lean_object* v___x_188_; lean_object* v_env_189_; uint8_t v_isExporting_190_; lean_object* v___x_256_; uint8_t v_isModule_257_; 
v___x_188_ = lean_st_ref_get(v___y_186_);
v_env_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc_ref(v_env_189_);
lean_dec(v___x_188_);
v_isExporting_190_ = lean_ctor_get_uint8(v_env_189_, sizeof(void*)*8);
v___x_256_ = l_Lean_Environment_header(v_env_189_);
lean_dec_ref(v_env_189_);
v_isModule_257_ = lean_ctor_get_uint8(v___x_256_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_256_);
if (v_isModule_257_ == 0)
{
lean_object* v___x_258_; 
lean_inc(v___y_186_);
lean_inc_ref(v___y_185_);
lean_inc(v___y_184_);
lean_inc_ref(v___y_183_);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
v___x_258_ = lean_apply_7(v_x_179_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, lean_box(0));
return v___x_258_;
}
else
{
if (v_isExporting_190_ == 0)
{
if (v_isExporting_180_ == 0)
{
lean_object* v___x_259_; 
lean_inc(v___y_186_);
lean_inc_ref(v___y_185_);
lean_inc(v___y_184_);
lean_inc_ref(v___y_183_);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
v___x_259_ = lean_apply_7(v_x_179_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, lean_box(0));
return v___x_259_;
}
else
{
goto v___jp_191_;
}
}
else
{
if (v_isExporting_180_ == 0)
{
goto v___jp_191_;
}
else
{
lean_object* v___x_260_; 
lean_inc(v___y_186_);
lean_inc_ref(v___y_185_);
lean_inc(v___y_184_);
lean_inc_ref(v___y_183_);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
v___x_260_ = lean_apply_7(v_x_179_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, lean_box(0));
return v___x_260_;
}
}
}
v___jp_191_:
{
lean_object* v___x_192_; lean_object* v_env_193_; lean_object* v_nextMacroScope_194_; lean_object* v_ngen_195_; lean_object* v_auxDeclNGen_196_; lean_object* v_traceState_197_; lean_object* v_messages_198_; lean_object* v_infoState_199_; lean_object* v_snapshotTasks_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_254_; 
v___x_192_ = lean_st_ref_take(v___y_186_);
v_env_193_ = lean_ctor_get(v___x_192_, 0);
v_nextMacroScope_194_ = lean_ctor_get(v___x_192_, 1);
v_ngen_195_ = lean_ctor_get(v___x_192_, 2);
v_auxDeclNGen_196_ = lean_ctor_get(v___x_192_, 3);
v_traceState_197_ = lean_ctor_get(v___x_192_, 4);
v_messages_198_ = lean_ctor_get(v___x_192_, 6);
v_infoState_199_ = lean_ctor_get(v___x_192_, 7);
v_snapshotTasks_200_ = lean_ctor_get(v___x_192_, 8);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_254_ == 0)
{
lean_object* v_unused_255_; 
v_unused_255_ = lean_ctor_get(v___x_192_, 5);
lean_dec(v_unused_255_);
v___x_202_ = v___x_192_;
v_isShared_203_ = v_isSharedCheck_254_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_snapshotTasks_200_);
lean_inc(v_infoState_199_);
lean_inc(v_messages_198_);
lean_inc(v_traceState_197_);
lean_inc(v_auxDeclNGen_196_);
lean_inc(v_ngen_195_);
lean_inc(v_nextMacroScope_194_);
lean_inc(v_env_193_);
lean_dec(v___x_192_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_254_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_207_; 
v___x_204_ = l_Lean_Environment_setExporting(v_env_193_, v_isExporting_180_);
v___x_205_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 5, v___x_205_);
lean_ctor_set(v___x_202_, 0, v___x_204_);
v___x_207_ = v___x_202_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_204_);
lean_ctor_set(v_reuseFailAlloc_253_, 1, v_nextMacroScope_194_);
lean_ctor_set(v_reuseFailAlloc_253_, 2, v_ngen_195_);
lean_ctor_set(v_reuseFailAlloc_253_, 3, v_auxDeclNGen_196_);
lean_ctor_set(v_reuseFailAlloc_253_, 4, v_traceState_197_);
lean_ctor_set(v_reuseFailAlloc_253_, 5, v___x_205_);
lean_ctor_set(v_reuseFailAlloc_253_, 6, v_messages_198_);
lean_ctor_set(v_reuseFailAlloc_253_, 7, v_infoState_199_);
lean_ctor_set(v_reuseFailAlloc_253_, 8, v_snapshotTasks_200_);
v___x_207_ = v_reuseFailAlloc_253_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v_mctx_210_; lean_object* v_zetaDeltaFVarIds_211_; lean_object* v_postponed_212_; lean_object* v_diag_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_251_; 
v___x_208_ = lean_st_ref_set(v___y_186_, v___x_207_);
v___x_209_ = lean_st_ref_take(v___y_184_);
v_mctx_210_ = lean_ctor_get(v___x_209_, 0);
v_zetaDeltaFVarIds_211_ = lean_ctor_get(v___x_209_, 2);
v_postponed_212_ = lean_ctor_get(v___x_209_, 3);
v_diag_213_ = lean_ctor_get(v___x_209_, 4);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_209_);
if (v_isSharedCheck_251_ == 0)
{
lean_object* v_unused_252_; 
v_unused_252_ = lean_ctor_get(v___x_209_, 1);
lean_dec(v_unused_252_);
v___x_215_ = v___x_209_;
v_isShared_216_ = v_isSharedCheck_251_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_diag_213_);
lean_inc(v_postponed_212_);
lean_inc(v_zetaDeltaFVarIds_211_);
lean_inc(v_mctx_210_);
lean_dec(v___x_209_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_251_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v___x_217_; lean_object* v___x_219_; 
v___x_217_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_216_ == 0)
{
lean_ctor_set(v___x_215_, 1, v___x_217_);
v___x_219_ = v___x_215_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_mctx_210_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v___x_217_);
lean_ctor_set(v_reuseFailAlloc_250_, 2, v_zetaDeltaFVarIds_211_);
lean_ctor_set(v_reuseFailAlloc_250_, 3, v_postponed_212_);
lean_ctor_set(v_reuseFailAlloc_250_, 4, v_diag_213_);
v___x_219_ = v_reuseFailAlloc_250_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
lean_object* v___x_220_; lean_object* v_r_221_; 
v___x_220_ = lean_st_ref_set(v___y_184_, v___x_219_);
lean_inc(v___y_186_);
lean_inc_ref(v___y_185_);
lean_inc(v___y_184_);
lean_inc_ref(v___y_183_);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
v_r_221_ = lean_apply_7(v_x_179_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, lean_box(0));
if (lean_obj_tag(v_r_221_) == 0)
{
lean_object* v_a_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_238_; 
v_a_222_ = lean_ctor_get(v_r_221_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v_r_221_);
if (v_isSharedCheck_238_ == 0)
{
v___x_224_ = v_r_221_;
v_isShared_225_ = v_isSharedCheck_238_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_a_222_);
lean_dec(v_r_221_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_238_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
lean_inc(v_a_222_);
if (v_isShared_225_ == 0)
{
lean_ctor_set_tag(v___x_224_, 1);
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_a_222_);
v___x_227_ = v_reuseFailAlloc_237_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
lean_object* v___x_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
v___x_228_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(v___y_186_, v_isExporting_190_, v___x_205_, v___y_184_, v___x_217_, v___x_227_);
lean_dec_ref(v___x_227_);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_228_);
if (v_isSharedCheck_235_ == 0)
{
lean_object* v_unused_236_; 
v_unused_236_ = lean_ctor_get(v___x_228_, 0);
lean_dec(v_unused_236_);
v___x_230_ = v___x_228_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_dec(v___x_228_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v_a_222_);
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_222_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
else
{
lean_object* v_a_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
v_a_239_ = lean_ctor_get(v_r_221_, 0);
lean_inc(v_a_239_);
lean_dec_ref_known(v_r_221_, 1);
v___x_240_ = lean_box(0);
v___x_241_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(v___y_186_, v_isExporting_190_, v___x_205_, v___y_184_, v___x_217_, v___x_240_);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; 
v_unused_249_ = lean_ctor_get(v___x_241_, 0);
lean_dec(v_unused_249_);
v___x_243_ = v___x_241_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_dec(v___x_241_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_244_ == 0)
{
lean_ctor_set_tag(v___x_243_, 1);
lean_ctor_set(v___x_243_, 0, v_a_239_);
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_239_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___boxed(lean_object* v_x_261_, lean_object* v_isExporting_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
uint8_t v_isExporting_boxed_270_; lean_object* v_res_271_; 
v_isExporting_boxed_270_ = lean_unbox(v_isExporting_262_);
v_res_271_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg(v_x_261_, v_isExporting_boxed_270_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg(lean_object* v_x_272_, uint8_t v_when_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
if (v_when_273_ == 0)
{
lean_object* v___x_281_; 
lean_inc(v___y_279_);
lean_inc_ref(v___y_278_);
lean_inc(v___y_277_);
lean_inc_ref(v___y_276_);
lean_inc(v___y_275_);
lean_inc_ref(v___y_274_);
v___x_281_ = lean_apply_7(v_x_272_, v___y_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, lean_box(0));
return v___x_281_;
}
else
{
uint8_t v___x_282_; lean_object* v___x_283_; 
v___x_282_ = 0;
v___x_283_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg(v_x_272_, v___x_282_, v___y_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
return v___x_283_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg___boxed(lean_object* v_x_284_, lean_object* v_when_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_){
_start:
{
uint8_t v_when_boxed_293_; lean_object* v_res_294_; 
v_when_boxed_293_ = lean_unbox(v_when_285_);
v_res_294_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg(v_x_284_, v_when_boxed_293_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1(lean_object* v_env_295_, lean_object* v_declName_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
uint8_t v___x_299_; lean_object* v_env_300_; lean_object* v___x_301_; uint8_t v___x_302_; uint8_t v___x_303_; 
v___x_299_ = 0;
v_env_300_ = l_Lean_Environment_setExporting(v_env_295_, v___x_299_);
lean_inc(v_declName_296_);
v___x_301_ = l_Lean_mkPrivateName(v_env_300_, v_declName_296_);
v___x_302_ = 1;
lean_inc_ref(v_env_300_);
v___x_303_ = l_Lean_Environment_contains(v_env_300_, v___x_301_, v___x_302_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; uint8_t v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_304_ = l_Lean_privateToUserName(v_declName_296_);
v___x_305_ = l_Lean_Environment_contains(v_env_300_, v___x_304_, v___x_302_);
v___x_306_ = lean_box(v___x_305_);
v___x_307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_306_);
lean_ctor_set(v___x_307_, 1, v___y_298_);
return v___x_307_;
}
else
{
lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec_ref(v_env_300_);
lean_dec(v_declName_296_);
v___x_308_ = lean_box(v___x_303_);
v___x_309_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v___y_298_);
return v___x_309_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1___boxed(lean_object* v_env_310_, lean_object* v_declName_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1(v_env_310_, v_declName_311_, v___y_312_, v___y_313_);
lean_dec_ref(v___y_312_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(lean_object* v_msgData_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v___x_321_; lean_object* v_env_322_; lean_object* v___x_323_; lean_object* v_mctx_324_; lean_object* v_lctx_325_; lean_object* v_options_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_321_ = lean_st_ref_get(v___y_319_);
v_env_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc_ref(v_env_322_);
lean_dec(v___x_321_);
v___x_323_ = lean_st_ref_get(v___y_317_);
v_mctx_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc_ref(v_mctx_324_);
lean_dec(v___x_323_);
v_lctx_325_ = lean_ctor_get(v___y_316_, 2);
v_options_326_ = lean_ctor_get(v___y_318_, 2);
lean_inc_ref(v_options_326_);
lean_inc_ref(v_lctx_325_);
v___x_327_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_327_, 0, v_env_322_);
lean_ctor_set(v___x_327_, 1, v_mctx_324_);
lean_ctor_set(v___x_327_, 2, v_lctx_325_);
lean_ctor_set(v___x_327_, 3, v_options_326_);
v___x_328_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
lean_ctor_set(v___x_328_, 1, v_msgData_315_);
v___x_329_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15___boxed(lean_object* v_msgData_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v_msgData_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
lean_dec(v___y_332_);
lean_dec_ref(v___y_331_);
return v_res_336_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_337_; double v___x_338_; 
v___x_337_ = lean_unsigned_to_nat(0u);
v___x_338_ = lean_float_of_nat(v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(lean_object* v_cls_342_, lean_object* v_msg_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v_ref_349_; lean_object* v___x_350_; lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_395_; 
v_ref_349_ = lean_ctor_get(v___y_346_, 5);
v___x_350_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v_msg_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
v_a_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_395_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_395_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_395_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_355_; lean_object* v_traceState_356_; lean_object* v_env_357_; lean_object* v_nextMacroScope_358_; lean_object* v_ngen_359_; lean_object* v_auxDeclNGen_360_; lean_object* v_cache_361_; lean_object* v_messages_362_; lean_object* v_infoState_363_; lean_object* v_snapshotTasks_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_394_; 
v___x_355_ = lean_st_ref_take(v___y_347_);
v_traceState_356_ = lean_ctor_get(v___x_355_, 4);
v_env_357_ = lean_ctor_get(v___x_355_, 0);
v_nextMacroScope_358_ = lean_ctor_get(v___x_355_, 1);
v_ngen_359_ = lean_ctor_get(v___x_355_, 2);
v_auxDeclNGen_360_ = lean_ctor_get(v___x_355_, 3);
v_cache_361_ = lean_ctor_get(v___x_355_, 5);
v_messages_362_ = lean_ctor_get(v___x_355_, 6);
v_infoState_363_ = lean_ctor_get(v___x_355_, 7);
v_snapshotTasks_364_ = lean_ctor_get(v___x_355_, 8);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_394_ == 0)
{
v___x_366_ = v___x_355_;
v_isShared_367_ = v_isSharedCheck_394_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_snapshotTasks_364_);
lean_inc(v_infoState_363_);
lean_inc(v_messages_362_);
lean_inc(v_cache_361_);
lean_inc(v_traceState_356_);
lean_inc(v_auxDeclNGen_360_);
lean_inc(v_ngen_359_);
lean_inc(v_nextMacroScope_358_);
lean_inc(v_env_357_);
lean_dec(v___x_355_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_394_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
uint64_t v_tid_368_; lean_object* v_traces_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_393_; 
v_tid_368_ = lean_ctor_get_uint64(v_traceState_356_, sizeof(void*)*1);
v_traces_369_ = lean_ctor_get(v_traceState_356_, 0);
v_isSharedCheck_393_ = !lean_is_exclusive(v_traceState_356_);
if (v_isSharedCheck_393_ == 0)
{
v___x_371_ = v_traceState_356_;
v_isShared_372_ = v_isSharedCheck_393_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_traces_369_);
lean_dec(v_traceState_356_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_393_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_373_; double v___x_374_; uint8_t v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_383_; 
v___x_373_ = lean_box(0);
v___x_374_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__0);
v___x_375_ = 0;
v___x_376_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1));
v___x_377_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_377_, 0, v_cls_342_);
lean_ctor_set(v___x_377_, 1, v___x_373_);
lean_ctor_set(v___x_377_, 2, v___x_376_);
lean_ctor_set_float(v___x_377_, sizeof(void*)*3, v___x_374_);
lean_ctor_set_float(v___x_377_, sizeof(void*)*3 + 8, v___x_374_);
lean_ctor_set_uint8(v___x_377_, sizeof(void*)*3 + 16, v___x_375_);
v___x_378_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__2));
v___x_379_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_379_, 0, v___x_377_);
lean_ctor_set(v___x_379_, 1, v_a_351_);
lean_ctor_set(v___x_379_, 2, v___x_378_);
lean_inc(v_ref_349_);
v___x_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_380_, 0, v_ref_349_);
lean_ctor_set(v___x_380_, 1, v___x_379_);
v___x_381_ = l_Lean_PersistentArray_push___redArg(v_traces_369_, v___x_380_);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 0, v___x_381_);
v___x_383_ = v___x_371_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v___x_381_);
lean_ctor_set_uint64(v_reuseFailAlloc_392_, sizeof(void*)*1, v_tid_368_);
v___x_383_ = v_reuseFailAlloc_392_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
lean_object* v___x_385_; 
if (v_isShared_367_ == 0)
{
lean_ctor_set(v___x_366_, 4, v___x_383_);
v___x_385_ = v___x_366_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_env_357_);
lean_ctor_set(v_reuseFailAlloc_391_, 1, v_nextMacroScope_358_);
lean_ctor_set(v_reuseFailAlloc_391_, 2, v_ngen_359_);
lean_ctor_set(v_reuseFailAlloc_391_, 3, v_auxDeclNGen_360_);
lean_ctor_set(v_reuseFailAlloc_391_, 4, v___x_383_);
lean_ctor_set(v_reuseFailAlloc_391_, 5, v_cache_361_);
lean_ctor_set(v_reuseFailAlloc_391_, 6, v_messages_362_);
lean_ctor_set(v_reuseFailAlloc_391_, 7, v_infoState_363_);
lean_ctor_set(v_reuseFailAlloc_391_, 8, v_snapshotTasks_364_);
v___x_385_ = v_reuseFailAlloc_391_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_389_; 
v___x_386_ = lean_st_ref_set(v___y_347_, v___x_385_);
v___x_387_ = lean_box(0);
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 0, v___x_387_);
v___x_389_ = v___x_353_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_387_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_cls_396_, lean_object* v_msg_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(v_cls_396_, v_msg_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
return v_res_403_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg(lean_object* v_keys_404_, lean_object* v_i_405_, lean_object* v_k_406_){
_start:
{
lean_object* v___x_407_; uint8_t v___x_408_; 
v___x_407_ = lean_array_get_size(v_keys_404_);
v___x_408_ = lean_nat_dec_lt(v_i_405_, v___x_407_);
if (v___x_408_ == 0)
{
lean_dec(v_i_405_);
return v___x_408_;
}
else
{
lean_object* v_k_x27_409_; uint8_t v___x_410_; 
v_k_x27_409_ = lean_array_fget_borrowed(v_keys_404_, v_i_405_);
v___x_410_ = l_Lean_instBEqExtraModUse_beq(v_k_406_, v_k_x27_409_);
if (v___x_410_ == 0)
{
lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_411_ = lean_unsigned_to_nat(1u);
v___x_412_ = lean_nat_add(v_i_405_, v___x_411_);
lean_dec(v_i_405_);
v_i_405_ = v___x_412_;
goto _start;
}
else
{
lean_dec(v_i_405_);
return v___x_410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg___boxed(lean_object* v_keys_414_, lean_object* v_i_415_, lean_object* v_k_416_){
_start:
{
uint8_t v_res_417_; lean_object* v_r_418_; 
v_res_417_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg(v_keys_414_, v_i_415_, v_k_416_);
lean_dec_ref(v_k_416_);
lean_dec_ref(v_keys_414_);
v_r_418_ = lean_box(v_res_417_);
return v_r_418_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg(lean_object* v_x_419_, size_t v_x_420_, lean_object* v_x_421_){
_start:
{
if (lean_obj_tag(v_x_419_) == 0)
{
lean_object* v_es_422_; lean_object* v___x_423_; size_t v___x_424_; size_t v___x_425_; lean_object* v_j_426_; lean_object* v___x_427_; 
v_es_422_ = lean_ctor_get(v_x_419_, 0);
v___x_423_ = lean_box(2);
v___x_424_ = ((size_t)31ULL);
v___x_425_ = lean_usize_land(v_x_420_, v___x_424_);
v_j_426_ = lean_usize_to_nat(v___x_425_);
v___x_427_ = lean_array_get_borrowed(v___x_423_, v_es_422_, v_j_426_);
lean_dec(v_j_426_);
switch(lean_obj_tag(v___x_427_))
{
case 0:
{
lean_object* v_key_428_; uint8_t v___x_429_; 
v_key_428_ = lean_ctor_get(v___x_427_, 0);
v___x_429_ = l_Lean_instBEqExtraModUse_beq(v_x_421_, v_key_428_);
return v___x_429_;
}
case 1:
{
lean_object* v_node_430_; size_t v___x_431_; size_t v___x_432_; 
v_node_430_ = lean_ctor_get(v___x_427_, 0);
v___x_431_ = ((size_t)5ULL);
v___x_432_ = lean_usize_shift_right(v_x_420_, v___x_431_);
v_x_419_ = v_node_430_;
v_x_420_ = v___x_432_;
goto _start;
}
default: 
{
uint8_t v___x_434_; 
v___x_434_ = 0;
return v___x_434_;
}
}
}
else
{
lean_object* v_ks_435_; lean_object* v___x_436_; uint8_t v___x_437_; 
v_ks_435_ = lean_ctor_get(v_x_419_, 0);
v___x_436_ = lean_unsigned_to_nat(0u);
v___x_437_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg(v_ks_435_, v___x_436_, v_x_421_);
return v___x_437_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg___boxed(lean_object* v_x_438_, lean_object* v_x_439_, lean_object* v_x_440_){
_start:
{
size_t v_x_28675__boxed_441_; uint8_t v_res_442_; lean_object* v_r_443_; 
v_x_28675__boxed_441_ = lean_unbox_usize(v_x_439_);
lean_dec(v_x_439_);
v_res_442_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg(v_x_438_, v_x_28675__boxed_441_, v_x_440_);
lean_dec_ref(v_x_440_);
lean_dec_ref(v_x_438_);
v_r_443_ = lean_box(v_res_442_);
return v_r_443_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg(lean_object* v_x_444_, lean_object* v_x_445_){
_start:
{
uint64_t v___x_446_; size_t v___x_447_; uint8_t v___x_448_; 
v___x_446_ = l_Lean_instHashableExtraModUse_hash(v_x_445_);
v___x_447_ = lean_uint64_to_usize(v___x_446_);
v___x_448_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg(v_x_444_, v___x_447_, v_x_445_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg___boxed(lean_object* v_x_449_, lean_object* v_x_450_){
_start:
{
uint8_t v_res_451_; lean_object* v_r_452_; 
v_res_451_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg(v_x_449_, v_x_450_);
lean_dec_ref(v_x_450_);
lean_dec_ref(v_x_449_);
v_r_452_ = lean_box(v_res_451_);
return v_r_452_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_455_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__1));
v___x_456_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__0));
v___x_457_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_456_, v___x_455_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__5));
v___x_463_ = l_Lean_stringToMessageData(v___x_462_);
return v___x_463_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_465_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__7));
v___x_466_ = l_Lean_stringToMessageData(v___x_465_);
return v___x_466_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9(void){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_467_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1));
v___x_468_ = l_Lean_stringToMessageData(v___x_467_);
return v___x_468_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12(void){
_start:
{
lean_object* v_cls_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v_cls_472_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__4));
v___x_473_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11));
v___x_474_ = l_Lean_Name_append(v___x_473_, v_cls_472_);
return v___x_474_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_476_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__13));
v___x_477_ = l_Lean_stringToMessageData(v___x_476_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__15));
v___x_480_ = l_Lean_stringToMessageData(v___x_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11(lean_object* v_mod_485_, uint8_t v_isMeta_486_, lean_object* v_hint_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_){
_start:
{
lean_object* v___x_495_; lean_object* v_env_496_; uint8_t v_isExporting_497_; lean_object* v___x_498_; lean_object* v_env_499_; lean_object* v___x_500_; lean_object* v_entry_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___x_547_; uint8_t v___x_548_; 
v___x_495_ = lean_st_ref_get(v___y_493_);
v_env_496_ = lean_ctor_get(v___x_495_, 0);
lean_inc_ref(v_env_496_);
lean_dec(v___x_495_);
v_isExporting_497_ = lean_ctor_get_uint8(v_env_496_, sizeof(void*)*8);
lean_dec_ref(v_env_496_);
v___x_498_ = lean_st_ref_get(v___y_493_);
v_env_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc_ref(v_env_499_);
lean_dec(v___x_498_);
v___x_500_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__2);
lean_inc(v_mod_485_);
v_entry_501_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_entry_501_, 0, v_mod_485_);
lean_ctor_set_uint8(v_entry_501_, sizeof(void*)*1, v_isExporting_497_);
lean_ctor_set_uint8(v_entry_501_, sizeof(void*)*1 + 1, v_isMeta_486_);
v___x_502_ = l___private_Lean_ExtraModUses_0__Lean_extraModUses;
v___x_503_ = lean_box(1);
v___x_504_ = lean_box(0);
v___x_547_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_500_, v___x_502_, v_env_499_, v___x_503_, v___x_504_);
v___x_548_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg(v___x_547_, v_entry_501_);
lean_dec(v___x_547_);
if (v___x_548_ == 0)
{
lean_object* v_options_549_; uint8_t v_hasTrace_550_; 
v_options_549_ = lean_ctor_get(v___y_492_, 2);
v_hasTrace_550_ = lean_ctor_get_uint8(v_options_549_, sizeof(void*)*1);
if (v_hasTrace_550_ == 0)
{
lean_dec(v_hint_487_);
lean_dec(v_mod_485_);
v___y_506_ = v___y_491_;
v___y_507_ = v___y_493_;
goto v___jp_505_;
}
else
{
lean_object* v_inheritedTraceOptions_551_; lean_object* v_cls_552_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_559_; lean_object* v___y_560_; lean_object* v___x_572_; uint8_t v___x_573_; 
v_inheritedTraceOptions_551_ = lean_ctor_get(v___y_492_, 13);
v_cls_552_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__4));
v___x_572_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__12);
v___x_573_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_551_, v_options_549_, v___x_572_);
if (v___x_573_ == 0)
{
lean_dec(v_hint_487_);
lean_dec(v_mod_485_);
v___y_506_ = v___y_491_;
v___y_507_ = v___y_493_;
goto v___jp_505_;
}
else
{
lean_object* v___x_574_; lean_object* v___y_576_; 
v___x_574_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__14);
if (v_isExporting_497_ == 0)
{
lean_object* v___x_583_; 
v___x_583_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__19));
v___y_576_ = v___x_583_;
goto v___jp_575_;
}
else
{
lean_object* v___x_584_; 
v___x_584_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__20));
v___y_576_ = v___x_584_;
goto v___jp_575_;
}
v___jp_575_:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
lean_inc_ref(v___y_576_);
v___x_577_ = l_Lean_stringToMessageData(v___y_576_);
v___x_578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_574_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__16);
v___x_580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_580_, 0, v___x_578_);
lean_ctor_set(v___x_580_, 1, v___x_579_);
if (v_isMeta_486_ == 0)
{
lean_object* v___x_581_; 
v___x_581_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__17));
v___y_559_ = v___x_580_;
v___y_560_ = v___x_581_;
goto v___jp_558_;
}
else
{
lean_object* v___x_582_; 
v___x_582_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__18));
v___y_559_ = v___x_580_;
v___y_560_ = v___x_582_;
goto v___jp_558_;
}
}
}
v___jp_553_:
{
lean_object* v___x_556_; lean_object* v___x_557_; 
v___x_556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_556_, 0, v___y_554_);
lean_ctor_set(v___x_556_, 1, v___y_555_);
v___x_557_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(v_cls_552_, v___x_556_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_dec_ref_known(v___x_557_, 1);
v___y_506_ = v___y_491_;
v___y_507_ = v___y_493_;
goto v___jp_505_;
}
else
{
lean_dec_ref_known(v_entry_501_, 1);
return v___x_557_;
}
}
v___jp_558_:
{
lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; uint8_t v___x_567_; 
lean_inc_ref(v___y_560_);
v___x_561_ = l_Lean_stringToMessageData(v___y_560_);
v___x_562_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_562_, 0, v___y_559_);
lean_ctor_set(v___x_562_, 1, v___x_561_);
v___x_563_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__6);
v___x_564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_564_, 0, v___x_562_);
lean_ctor_set(v___x_564_, 1, v___x_563_);
v___x_565_ = l_Lean_MessageData_ofName(v_mod_485_);
v___x_566_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_564_);
lean_ctor_set(v___x_566_, 1, v___x_565_);
v___x_567_ = l_Lean_Name_isAnonymous(v_hint_487_);
if (v___x_567_ == 0)
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_568_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__8);
v___x_569_ = l_Lean_MessageData_ofName(v_hint_487_);
v___x_570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_568_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___y_554_ = v___x_566_;
v___y_555_ = v___x_570_;
goto v___jp_553_;
}
else
{
lean_object* v___x_571_; 
lean_dec(v_hint_487_);
v___x_571_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__9);
v___y_554_ = v___x_566_;
v___y_555_ = v___x_571_;
goto v___jp_553_;
}
}
}
}
else
{
lean_object* v___x_585_; lean_object* v___x_586_; 
lean_dec_ref_known(v_entry_501_, 1);
lean_dec(v_hint_487_);
lean_dec(v_mod_485_);
v___x_585_ = lean_box(0);
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v___x_585_);
return v___x_586_;
}
v___jp_505_:
{
lean_object* v___x_508_; lean_object* v_toEnvExtension_509_; lean_object* v_env_510_; lean_object* v_nextMacroScope_511_; lean_object* v_ngen_512_; lean_object* v_auxDeclNGen_513_; lean_object* v_traceState_514_; lean_object* v_messages_515_; lean_object* v_infoState_516_; lean_object* v_snapshotTasks_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_545_; 
v___x_508_ = lean_st_ref_take(v___y_507_);
v_toEnvExtension_509_ = lean_ctor_get(v___x_502_, 0);
v_env_510_ = lean_ctor_get(v___x_508_, 0);
v_nextMacroScope_511_ = lean_ctor_get(v___x_508_, 1);
v_ngen_512_ = lean_ctor_get(v___x_508_, 2);
v_auxDeclNGen_513_ = lean_ctor_get(v___x_508_, 3);
v_traceState_514_ = lean_ctor_get(v___x_508_, 4);
v_messages_515_ = lean_ctor_get(v___x_508_, 6);
v_infoState_516_ = lean_ctor_get(v___x_508_, 7);
v_snapshotTasks_517_ = lean_ctor_get(v___x_508_, 8);
v_isSharedCheck_545_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_545_ == 0)
{
lean_object* v_unused_546_; 
v_unused_546_ = lean_ctor_get(v___x_508_, 5);
lean_dec(v_unused_546_);
v___x_519_ = v___x_508_;
v_isShared_520_ = v_isSharedCheck_545_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_snapshotTasks_517_);
lean_inc(v_infoState_516_);
lean_inc(v_messages_515_);
lean_inc(v_traceState_514_);
lean_inc(v_auxDeclNGen_513_);
lean_inc(v_ngen_512_);
lean_inc(v_nextMacroScope_511_);
lean_inc(v_env_510_);
lean_dec(v___x_508_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_545_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v_asyncMode_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_525_; 
v_asyncMode_521_ = lean_ctor_get(v_toEnvExtension_509_, 2);
v___x_522_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_502_, v_env_510_, v_entry_501_, v_asyncMode_521_, v___x_504_);
v___x_523_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_520_ == 0)
{
lean_ctor_set(v___x_519_, 5, v___x_523_);
lean_ctor_set(v___x_519_, 0, v___x_522_);
v___x_525_ = v___x_519_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v___x_522_);
lean_ctor_set(v_reuseFailAlloc_544_, 1, v_nextMacroScope_511_);
lean_ctor_set(v_reuseFailAlloc_544_, 2, v_ngen_512_);
lean_ctor_set(v_reuseFailAlloc_544_, 3, v_auxDeclNGen_513_);
lean_ctor_set(v_reuseFailAlloc_544_, 4, v_traceState_514_);
lean_ctor_set(v_reuseFailAlloc_544_, 5, v___x_523_);
lean_ctor_set(v_reuseFailAlloc_544_, 6, v_messages_515_);
lean_ctor_set(v_reuseFailAlloc_544_, 7, v_infoState_516_);
lean_ctor_set(v_reuseFailAlloc_544_, 8, v_snapshotTasks_517_);
v___x_525_ = v_reuseFailAlloc_544_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v_mctx_528_; lean_object* v_zetaDeltaFVarIds_529_; lean_object* v_postponed_530_; lean_object* v_diag_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_542_; 
v___x_526_ = lean_st_ref_set(v___y_507_, v___x_525_);
v___x_527_ = lean_st_ref_take(v___y_506_);
v_mctx_528_ = lean_ctor_get(v___x_527_, 0);
v_zetaDeltaFVarIds_529_ = lean_ctor_get(v___x_527_, 2);
v_postponed_530_ = lean_ctor_get(v___x_527_, 3);
v_diag_531_ = lean_ctor_get(v___x_527_, 4);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_527_);
if (v_isSharedCheck_542_ == 0)
{
lean_object* v_unused_543_; 
v_unused_543_ = lean_ctor_get(v___x_527_, 1);
lean_dec(v_unused_543_);
v___x_533_ = v___x_527_;
v_isShared_534_ = v_isSharedCheck_542_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_diag_531_);
lean_inc(v_postponed_530_);
lean_inc(v_zetaDeltaFVarIds_529_);
lean_inc(v_mctx_528_);
lean_dec(v___x_527_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_542_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_535_; lean_object* v___x_537_; 
v___x_535_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 1, v___x_535_);
v___x_537_ = v___x_533_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_mctx_528_);
lean_ctor_set(v_reuseFailAlloc_541_, 1, v___x_535_);
lean_ctor_set(v_reuseFailAlloc_541_, 2, v_zetaDeltaFVarIds_529_);
lean_ctor_set(v_reuseFailAlloc_541_, 3, v_postponed_530_);
lean_ctor_set(v_reuseFailAlloc_541_, 4, v_diag_531_);
v___x_537_ = v_reuseFailAlloc_541_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_538_ = lean_st_ref_set(v___y_506_, v___x_537_);
v___x_539_ = lean_box(0);
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
return v___x_540_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___boxed(lean_object* v_mod_587_, lean_object* v_isMeta_588_, lean_object* v_hint_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
uint8_t v_isMeta_boxed_597_; lean_object* v_res_598_; 
v_isMeta_boxed_597_ = lean_unbox(v_isMeta_588_);
v_res_598_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11(v_mod_587_, v_isMeta_boxed_597_, v_hint_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12(lean_object* v___x_599_, lean_object* v_declName_600_, lean_object* v_as_601_, size_t v_sz_602_, size_t v_i_603_, lean_object* v_b_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_){
_start:
{
uint8_t v___x_612_; 
v___x_612_ = lean_usize_dec_lt(v_i_603_, v_sz_602_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; 
lean_dec(v_declName_600_);
v___x_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_613_, 0, v_b_604_);
return v___x_613_;
}
else
{
lean_object* v___x_614_; lean_object* v_modules_615_; lean_object* v___x_616_; lean_object* v_a_617_; lean_object* v___x_618_; lean_object* v_toImport_619_; lean_object* v_module_620_; uint8_t v___x_621_; lean_object* v___x_622_; 
v___x_614_ = l_Lean_Environment_header(v___x_599_);
v_modules_615_ = lean_ctor_get(v___x_614_, 3);
lean_inc_ref(v_modules_615_);
lean_dec_ref(v___x_614_);
v___x_616_ = l_Lean_instInhabitedEffectiveImport_default;
v_a_617_ = lean_array_uget_borrowed(v_as_601_, v_i_603_);
v___x_618_ = lean_array_get(v___x_616_, v_modules_615_, v_a_617_);
lean_dec_ref(v_modules_615_);
v_toImport_619_ = lean_ctor_get(v___x_618_, 0);
lean_inc_ref(v_toImport_619_);
lean_dec(v___x_618_);
v_module_620_ = lean_ctor_get(v_toImport_619_, 0);
lean_inc(v_module_620_);
lean_dec_ref(v_toImport_619_);
v___x_621_ = 0;
lean_inc(v_declName_600_);
v___x_622_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11(v_module_620_, v___x_621_, v_declName_600_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_);
if (lean_obj_tag(v___x_622_) == 0)
{
lean_object* v___x_623_; size_t v___x_624_; size_t v___x_625_; 
lean_dec_ref_known(v___x_622_, 1);
v___x_623_ = lean_box(0);
v___x_624_ = ((size_t)1ULL);
v___x_625_ = lean_usize_add(v_i_603_, v___x_624_);
v_i_603_ = v___x_625_;
v_b_604_ = v___x_623_;
goto _start;
}
else
{
lean_dec(v_declName_600_);
return v___x_622_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12___boxed(lean_object* v___x_627_, lean_object* v_declName_628_, lean_object* v_as_629_, lean_object* v_sz_630_, lean_object* v_i_631_, lean_object* v_b_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
size_t v_sz_boxed_640_; size_t v_i_boxed_641_; lean_object* v_res_642_; 
v_sz_boxed_640_ = lean_unbox_usize(v_sz_630_);
lean_dec(v_sz_630_);
v_i_boxed_641_ = lean_unbox_usize(v_i_631_);
lean_dec(v_i_631_);
v_res_642_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12(v___x_627_, v_declName_628_, v_as_629_, v_sz_boxed_640_, v_i_boxed_641_, v_b_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_, v___y_638_);
lean_dec(v___y_638_);
lean_dec_ref(v___y_637_);
lean_dec(v___y_636_);
lean_dec_ref(v___y_635_);
lean_dec(v___y_634_);
lean_dec_ref(v___y_633_);
lean_dec_ref(v_as_629_);
lean_dec_ref(v___x_627_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg(lean_object* v_a_643_, lean_object* v_x_644_){
_start:
{
if (lean_obj_tag(v_x_644_) == 0)
{
lean_object* v___x_645_; 
v___x_645_ = lean_box(0);
return v___x_645_;
}
else
{
lean_object* v_key_646_; lean_object* v_value_647_; lean_object* v_tail_648_; uint8_t v___x_649_; 
v_key_646_ = lean_ctor_get(v_x_644_, 0);
v_value_647_ = lean_ctor_get(v_x_644_, 1);
v_tail_648_ = lean_ctor_get(v_x_644_, 2);
v___x_649_ = lean_name_eq(v_key_646_, v_a_643_);
if (v___x_649_ == 0)
{
v_x_644_ = v_tail_648_;
goto _start;
}
else
{
lean_object* v___x_651_; 
lean_inc(v_value_647_);
v___x_651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_651_, 0, v_value_647_);
return v___x_651_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg___boxed(lean_object* v_a_652_, lean_object* v_x_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg(v_a_652_, v_x_653_);
lean_dec(v_x_653_);
lean_dec(v_a_652_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg(lean_object* v_m_655_, lean_object* v_a_656_){
_start:
{
lean_object* v_buckets_657_; lean_object* v___x_658_; uint64_t v___y_660_; 
v_buckets_657_ = lean_ctor_get(v_m_655_, 1);
v___x_658_ = lean_array_get_size(v_buckets_657_);
if (lean_obj_tag(v_a_656_) == 0)
{
uint64_t v___x_674_; 
v___x_674_ = 1723ULL;
v___y_660_ = v___x_674_;
goto v___jp_659_;
}
else
{
uint64_t v_hash_675_; 
v_hash_675_ = lean_ctor_get_uint64(v_a_656_, sizeof(void*)*2);
v___y_660_ = v_hash_675_;
goto v___jp_659_;
}
v___jp_659_:
{
uint64_t v___x_661_; uint64_t v___x_662_; uint64_t v_fold_663_; uint64_t v___x_664_; uint64_t v___x_665_; uint64_t v___x_666_; size_t v___x_667_; size_t v___x_668_; size_t v___x_669_; size_t v___x_670_; size_t v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_661_ = 32ULL;
v___x_662_ = lean_uint64_shift_right(v___y_660_, v___x_661_);
v_fold_663_ = lean_uint64_xor(v___y_660_, v___x_662_);
v___x_664_ = 16ULL;
v___x_665_ = lean_uint64_shift_right(v_fold_663_, v___x_664_);
v___x_666_ = lean_uint64_xor(v_fold_663_, v___x_665_);
v___x_667_ = lean_uint64_to_usize(v___x_666_);
v___x_668_ = lean_usize_of_nat(v___x_658_);
v___x_669_ = ((size_t)1ULL);
v___x_670_ = lean_usize_sub(v___x_668_, v___x_669_);
v___x_671_ = lean_usize_land(v___x_667_, v___x_670_);
v___x_672_ = lean_array_uget_borrowed(v_buckets_657_, v___x_671_);
v___x_673_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg(v_a_656_, v___x_672_);
return v___x_673_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg___boxed(lean_object* v_m_676_, lean_object* v_a_677_){
_start:
{
lean_object* v_res_678_; 
v_res_678_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg(v_m_676_, v_a_677_);
lean_dec(v_a_677_);
lean_dec_ref(v_m_676_);
return v_res_678_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2(void){
_start:
{
lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; 
v___x_681_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__1));
v___x_682_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__0));
v___x_683_ = l_Std_HashMap_instInhabited(lean_box(0), lean_box(0), v___x_682_, v___x_681_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3(lean_object* v_declName_686_, uint8_t v_isMeta_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_){
_start:
{
lean_object* v___x_695_; lean_object* v_env_699_; lean_object* v___y_701_; lean_object* v___x_714_; 
v___x_695_ = lean_st_ref_get(v___y_693_);
v_env_699_ = lean_ctor_get(v___x_695_, 0);
lean_inc_ref(v_env_699_);
lean_dec(v___x_695_);
v___x_714_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_699_, v_declName_686_);
if (lean_obj_tag(v___x_714_) == 0)
{
lean_dec_ref(v_env_699_);
lean_dec(v_declName_686_);
goto v___jp_696_;
}
else
{
lean_object* v_val_715_; lean_object* v___x_716_; lean_object* v_modules_717_; lean_object* v___x_718_; uint8_t v___x_719_; 
v_val_715_ = lean_ctor_get(v___x_714_, 0);
lean_inc(v_val_715_);
lean_dec_ref_known(v___x_714_, 1);
v___x_716_ = l_Lean_Environment_header(v_env_699_);
v_modules_717_ = lean_ctor_get(v___x_716_, 3);
lean_inc_ref(v_modules_717_);
lean_dec_ref(v___x_716_);
v___x_718_ = lean_array_get_size(v_modules_717_);
v___x_719_ = lean_nat_dec_lt(v_val_715_, v___x_718_);
if (v___x_719_ == 0)
{
lean_dec_ref(v_modules_717_);
lean_dec(v_val_715_);
lean_dec_ref(v_env_699_);
lean_dec(v_declName_686_);
goto v___jp_696_;
}
else
{
lean_object* v___x_720_; lean_object* v_env_721_; lean_object* v___x_722_; lean_object* v___x_723_; uint8_t v___y_725_; 
v___x_720_ = lean_st_ref_get(v___y_693_);
v_env_721_ = lean_ctor_get(v___x_720_, 0);
lean_inc_ref(v_env_721_);
lean_dec(v___x_720_);
v___x_722_ = lean_obj_once(&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2, &lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2_once, _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__2);
v___x_723_ = lean_array_fget(v_modules_717_, v_val_715_);
lean_dec(v_val_715_);
lean_dec_ref(v_modules_717_);
if (v_isMeta_687_ == 0)
{
lean_dec_ref(v_env_721_);
v___y_725_ = v_isMeta_687_;
goto v___jp_724_;
}
else
{
uint8_t v___x_736_; 
lean_inc(v_declName_686_);
v___x_736_ = l_Lean_isMarkedMeta(v_env_721_, v_declName_686_);
if (v___x_736_ == 0)
{
v___y_725_ = v_isMeta_687_;
goto v___jp_724_;
}
else
{
uint8_t v___x_737_; 
v___x_737_ = 0;
v___y_725_ = v___x_737_;
goto v___jp_724_;
}
}
v___jp_724_:
{
lean_object* v_toImport_726_; lean_object* v_module_727_; lean_object* v___x_728_; 
v_toImport_726_ = lean_ctor_get(v___x_723_, 0);
lean_inc_ref(v_toImport_726_);
lean_dec(v___x_723_);
v_module_727_ = lean_ctor_get(v_toImport_726_, 0);
lean_inc(v_module_727_);
lean_dec_ref(v_toImport_726_);
lean_inc(v_declName_686_);
v___x_728_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11(v_module_727_, v___y_725_, v_declName_686_, v___y_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_);
if (lean_obj_tag(v___x_728_) == 0)
{
lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
lean_dec_ref_known(v___x_728_, 1);
v___x_729_ = l_Lean_indirectModUseExt;
v___x_730_ = lean_box(1);
v___x_731_ = lean_box(0);
lean_inc_ref(v_env_699_);
v___x_732_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_722_, v___x_729_, v_env_699_, v___x_730_, v___x_731_);
v___x_733_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg(v___x_732_, v_declName_686_);
lean_dec(v___x_732_);
if (lean_obj_tag(v___x_733_) == 0)
{
lean_object* v___x_734_; 
v___x_734_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___closed__3));
v___y_701_ = v___x_734_;
goto v___jp_700_;
}
else
{
lean_object* v_val_735_; 
v_val_735_ = lean_ctor_get(v___x_733_, 0);
lean_inc(v_val_735_);
lean_dec_ref_known(v___x_733_, 1);
v___y_701_ = v_val_735_;
goto v___jp_700_;
}
}
else
{
lean_dec_ref(v_env_699_);
lean_dec(v_declName_686_);
return v___x_728_;
}
}
}
}
v___jp_696_:
{
lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_697_ = lean_box(0);
v___x_698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_698_, 0, v___x_697_);
return v___x_698_;
}
v___jp_700_:
{
lean_object* v___x_702_; size_t v_sz_703_; size_t v___x_704_; lean_object* v___x_705_; 
v___x_702_ = lean_box(0);
v_sz_703_ = lean_array_size(v___y_701_);
v___x_704_ = ((size_t)0ULL);
v___x_705_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__12(v_env_699_, v_declName_686_, v___y_701_, v_sz_703_, v___x_704_, v___x_702_, v___y_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_);
lean_dec_ref(v___y_701_);
lean_dec_ref(v_env_699_);
if (lean_obj_tag(v___x_705_) == 0)
{
lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_712_; 
v_isSharedCheck_712_ = !lean_is_exclusive(v___x_705_);
if (v_isSharedCheck_712_ == 0)
{
lean_object* v_unused_713_; 
v_unused_713_ = lean_ctor_get(v___x_705_, 0);
lean_dec(v_unused_713_);
v___x_707_ = v___x_705_;
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
else
{
lean_dec(v___x_705_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_712_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v___x_710_; 
if (v_isShared_708_ == 0)
{
lean_ctor_set(v___x_707_, 0, v___x_702_);
v___x_710_ = v___x_707_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_711_; 
v_reuseFailAlloc_711_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_711_, 0, v___x_702_);
v___x_710_ = v_reuseFailAlloc_711_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
return v___x_710_;
}
}
}
else
{
return v___x_705_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3___boxed(lean_object* v_declName_738_, lean_object* v_isMeta_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
uint8_t v_isMeta_boxed_747_; lean_object* v_res_748_; 
v_isMeta_boxed_747_ = lean_unbox(v_isMeta_739_);
v_res_748_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3(v_declName_738_, v_isMeta_boxed_747_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
lean_dec(v___y_745_);
lean_dec_ref(v___y_744_);
lean_dec(v___y_743_);
lean_dec_ref(v___y_742_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
return v_res_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg(lean_object* v_as_x27_749_, lean_object* v_b_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
if (lean_obj_tag(v_as_x27_749_) == 0)
{
lean_object* v___x_758_; 
v___x_758_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_758_, 0, v_b_750_);
return v___x_758_;
}
else
{
lean_object* v_head_759_; lean_object* v_tail_760_; uint8_t v___x_761_; lean_object* v___x_762_; 
v_head_759_ = lean_ctor_get(v_as_x27_749_, 0);
v_tail_760_ = lean_ctor_get(v_as_x27_749_, 1);
v___x_761_ = 1;
lean_inc(v_head_759_);
v___x_762_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3(v_head_759_, v___x_761_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
if (lean_obj_tag(v___x_762_) == 0)
{
lean_object* v___x_763_; 
lean_dec_ref_known(v___x_762_, 1);
v___x_763_ = lean_box(0);
v_as_x27_749_ = v_tail_760_;
v_b_750_ = v___x_763_;
goto _start;
}
else
{
return v___x_762_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg___boxed(lean_object* v_as_x27_765_, lean_object* v_b_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_){
_start:
{
lean_object* v_res_774_; 
v_res_774_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg(v_as_x27_765_, v_b_766_, v___y_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_);
lean_dec(v___y_772_);
lean_dec_ref(v___y_771_);
lean_dec(v___y_770_);
lean_dec_ref(v___y_769_);
lean_dec(v___y_768_);
lean_dec_ref(v___y_767_);
lean_dec(v_as_x27_765_);
return v_res_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3(lean_object* v_currNamespace_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_778_, 0, v_currNamespace_775_);
lean_ctor_set(v___x_778_, 1, v___y_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3___boxed(lean_object* v_currNamespace_779_, lean_object* v___y_780_, lean_object* v___y_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3(v_currNamespace_779_, v___y_780_, v___y_781_);
lean_dec_ref(v___y_780_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7(lean_object* v_as_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
if (lean_obj_tag(v_as_783_) == 0)
{
lean_object* v___x_791_; lean_object* v___x_792_; 
v___x_791_ = lean_box(0);
v___x_792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_792_, 0, v___x_791_);
return v___x_792_;
}
else
{
lean_object* v_options_793_; uint8_t v_hasTrace_794_; 
v_options_793_ = lean_ctor_get(v___y_788_, 2);
v_hasTrace_794_ = lean_ctor_get_uint8(v_options_793_, sizeof(void*)*1);
if (v_hasTrace_794_ == 0)
{
lean_object* v_tail_795_; 
v_tail_795_ = lean_ctor_get(v_as_783_, 1);
lean_inc(v_tail_795_);
lean_dec_ref_known(v_as_783_, 2);
v_as_783_ = v_tail_795_;
goto _start;
}
else
{
lean_object* v_head_797_; lean_object* v_tail_798_; lean_object* v_fst_799_; lean_object* v_snd_800_; lean_object* v_inheritedTraceOptions_801_; lean_object* v___x_802_; lean_object* v___x_803_; uint8_t v___x_804_; 
v_head_797_ = lean_ctor_get(v_as_783_, 0);
lean_inc(v_head_797_);
v_tail_798_ = lean_ctor_get(v_as_783_, 1);
lean_inc(v_tail_798_);
lean_dec_ref_known(v_as_783_, 2);
v_fst_799_ = lean_ctor_get(v_head_797_, 0);
lean_inc_n(v_fst_799_, 2);
v_snd_800_ = lean_ctor_get(v_head_797_, 1);
lean_inc(v_snd_800_);
lean_dec(v_head_797_);
v_inheritedTraceOptions_801_ = lean_ctor_get(v___y_788_, 13);
v___x_802_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11));
v___x_803_ = l_Lean_Name_append(v___x_802_, v_fst_799_);
v___x_804_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_801_, v_options_793_, v___x_803_);
lean_dec(v___x_803_);
if (v___x_804_ == 0)
{
lean_dec(v_snd_800_);
lean_dec(v_fst_799_);
v_as_783_ = v_tail_798_;
goto _start;
}
else
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_806_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_806_, 0, v_snd_800_);
v___x_807_ = l_Lean_MessageData_ofFormat(v___x_806_);
v___x_808_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(v_fst_799_, v___x_807_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
if (lean_obj_tag(v___x_808_) == 0)
{
lean_dec_ref_known(v___x_808_, 1);
v_as_783_ = v_tail_798_;
goto _start;
}
else
{
lean_dec(v_tail_798_);
return v___x_808_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7___boxed(lean_object* v_as_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
lean_object* v_res_818_; 
v_res_818_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7(v_as_810_, v___y_811_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec(v___y_814_);
lean_dec_ref(v___y_813_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
return v_res_818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4(lean_object* v_env_819_, lean_object* v_options_820_, lean_object* v_currNamespace_821_, lean_object* v_openDecls_822_, lean_object* v_n_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_826_ = l_Lean_ResolveName_resolveGlobalName(v_env_819_, v_options_820_, v_currNamespace_821_, v_openDecls_822_, v_n_823_);
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set(v___x_827_, 1, v___y_825_);
return v___x_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4___boxed(lean_object* v_env_828_, lean_object* v_options_829_, lean_object* v_currNamespace_830_, lean_object* v_openDecls_831_, lean_object* v_n_832_, lean_object* v___y_833_, lean_object* v___y_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4(v_env_828_, v_options_829_, v_currNamespace_830_, v_openDecls_831_, v_n_832_, v___y_833_, v___y_834_);
lean_dec_ref(v___y_833_);
lean_dec_ref(v_options_829_);
return v_res_835_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0(void){
_start:
{
lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_836_ = lean_box(0);
v___x_837_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_838_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
lean_ctor_set(v___x_838_, 1, v___x_836_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg(){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_840_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___closed__0);
v___x_841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_841_, 0, v___x_840_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg___boxed(lean_object* v___y_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg();
return v_res_843_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3(void){
_start:
{
lean_object* v___x_849_; lean_object* v___x_850_; 
v___x_849_ = l_Lean_maxRecDepthErrorMessage;
v___x_850_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_850_, 0, v___x_849_);
return v___x_850_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4(void){
_start:
{
lean_object* v___x_851_; lean_object* v___x_852_; 
v___x_851_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__3);
v___x_852_ = l_Lean_MessageData_ofFormat(v___x_851_);
return v___x_852_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5(void){
_start:
{
lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_853_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__4);
v___x_854_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__2));
v___x_855_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_855_, 0, v___x_854_);
lean_ctor_set(v___x_855_, 1, v___x_853_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg(lean_object* v_ref_856_){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_858_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___closed__5);
v___x_859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_859_, 0, v_ref_856_);
lean_ctor_set(v___x_859_, 1, v___x_858_);
v___x_860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_860_, 0, v___x_859_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg___boxed(lean_object* v_ref_861_, lean_object* v___y_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg(v_ref_861_);
return v_res_863_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; 
v___x_864_ = lean_box(1);
v___x_865_ = l_Lean_MessageData_ofFormat(v___x_864_);
return v___x_865_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3(void){
_start:
{
lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_869_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__2));
v___x_870_ = l_Lean_MessageData_ofFormat(v___x_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22(lean_object* v_x_871_, lean_object* v_x_872_){
_start:
{
if (lean_obj_tag(v_x_872_) == 0)
{
return v_x_871_;
}
else
{
lean_object* v_head_873_; lean_object* v_tail_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_896_; 
v_head_873_ = lean_ctor_get(v_x_872_, 0);
v_tail_874_ = lean_ctor_get(v_x_872_, 1);
v_isSharedCheck_896_ = !lean_is_exclusive(v_x_872_);
if (v_isSharedCheck_896_ == 0)
{
v___x_876_ = v_x_872_;
v_isShared_877_ = v_isSharedCheck_896_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_tail_874_);
lean_inc(v_head_873_);
lean_dec(v_x_872_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_896_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v_before_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_894_; 
v_before_878_ = lean_ctor_get(v_head_873_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v_head_873_);
if (v_isSharedCheck_894_ == 0)
{
lean_object* v_unused_895_; 
v_unused_895_ = lean_ctor_get(v_head_873_, 1);
lean_dec(v_unused_895_);
v___x_880_ = v_head_873_;
v_isShared_881_ = v_isSharedCheck_894_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_before_878_);
lean_dec(v_head_873_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_894_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v___x_882_; lean_object* v___x_884_; 
v___x_882_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0);
if (v_isShared_881_ == 0)
{
lean_ctor_set_tag(v___x_880_, 7);
lean_ctor_set(v___x_880_, 1, v___x_882_);
lean_ctor_set(v___x_880_, 0, v_x_871_);
v___x_884_ = v___x_880_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_x_871_);
lean_ctor_set(v_reuseFailAlloc_893_, 1, v___x_882_);
v___x_884_ = v_reuseFailAlloc_893_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
lean_object* v___x_885_; lean_object* v___x_887_; 
v___x_885_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__3);
if (v_isShared_877_ == 0)
{
lean_ctor_set_tag(v___x_876_, 7);
lean_ctor_set(v___x_876_, 1, v___x_885_);
lean_ctor_set(v___x_876_, 0, v___x_884_);
v___x_887_ = v___x_876_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_884_);
lean_ctor_set(v_reuseFailAlloc_892_, 1, v___x_885_);
v___x_887_ = v_reuseFailAlloc_892_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v___x_888_ = l_Lean_MessageData_ofSyntax(v_before_878_);
v___x_889_ = l_Lean_indentD(v___x_888_);
v___x_890_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_890_, 0, v___x_887_);
lean_ctor_set(v___x_890_, 1, v___x_889_);
v_x_871_ = v___x_890_;
v_x_872_ = v_tail_874_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(lean_object* v_opts_897_, lean_object* v_opt_898_){
_start:
{
lean_object* v_name_899_; lean_object* v_defValue_900_; lean_object* v_map_901_; lean_object* v___x_902_; 
v_name_899_ = lean_ctor_get(v_opt_898_, 0);
v_defValue_900_ = lean_ctor_get(v_opt_898_, 1);
v_map_901_ = lean_ctor_get(v_opts_897_, 0);
v___x_902_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_901_, v_name_899_);
if (lean_obj_tag(v___x_902_) == 0)
{
uint8_t v___x_903_; 
v___x_903_ = lean_unbox(v_defValue_900_);
return v___x_903_;
}
else
{
lean_object* v_val_904_; 
v_val_904_ = lean_ctor_get(v___x_902_, 0);
lean_inc(v_val_904_);
lean_dec_ref_known(v___x_902_, 1);
if (lean_obj_tag(v_val_904_) == 1)
{
uint8_t v_v_905_; 
v_v_905_ = lean_ctor_get_uint8(v_val_904_, 0);
lean_dec_ref_known(v_val_904_, 0);
return v_v_905_;
}
else
{
uint8_t v___x_906_; 
lean_dec(v_val_904_);
v___x_906_ = lean_unbox(v_defValue_900_);
return v___x_906_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21___boxed(lean_object* v_opts_907_, lean_object* v_opt_908_){
_start:
{
uint8_t v_res_909_; lean_object* v_r_910_; 
v_res_909_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v_opts_907_, v_opt_908_);
lean_dec_ref(v_opt_908_);
lean_dec_ref(v_opts_907_);
v_r_910_ = lean_box(v_res_909_);
return v_r_910_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2(void){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; 
v___x_914_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__1));
v___x_915_ = l_Lean_MessageData_ofFormat(v___x_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg(lean_object* v_msgData_916_, lean_object* v_macroStack_917_, lean_object* v___y_918_){
_start:
{
lean_object* v_options_920_; lean_object* v___x_921_; uint8_t v___x_922_; 
v_options_920_ = lean_ctor_get(v___y_918_, 2);
v___x_921_ = l_Lean_Elab_pp_macroStack;
v___x_922_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v_options_920_, v___x_921_);
if (v___x_922_ == 0)
{
lean_object* v___x_923_; 
lean_dec(v_macroStack_917_);
v___x_923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_923_, 0, v_msgData_916_);
return v___x_923_;
}
else
{
if (lean_obj_tag(v_macroStack_917_) == 0)
{
lean_object* v___x_924_; 
v___x_924_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_924_, 0, v_msgData_916_);
return v___x_924_;
}
else
{
lean_object* v_head_925_; lean_object* v_after_926_; lean_object* v___x_928_; uint8_t v_isShared_929_; uint8_t v_isSharedCheck_941_; 
v_head_925_ = lean_ctor_get(v_macroStack_917_, 0);
lean_inc(v_head_925_);
v_after_926_ = lean_ctor_get(v_head_925_, 1);
v_isSharedCheck_941_ = !lean_is_exclusive(v_head_925_);
if (v_isSharedCheck_941_ == 0)
{
lean_object* v_unused_942_; 
v_unused_942_ = lean_ctor_get(v_head_925_, 0);
lean_dec(v_unused_942_);
v___x_928_ = v_head_925_;
v_isShared_929_ = v_isSharedCheck_941_;
goto v_resetjp_927_;
}
else
{
lean_inc(v_after_926_);
lean_dec(v_head_925_);
v___x_928_ = lean_box(0);
v_isShared_929_ = v_isSharedCheck_941_;
goto v_resetjp_927_;
}
v_resetjp_927_:
{
lean_object* v___x_930_; lean_object* v___x_932_; 
v___x_930_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0);
if (v_isShared_929_ == 0)
{
lean_ctor_set_tag(v___x_928_, 7);
lean_ctor_set(v___x_928_, 1, v___x_930_);
lean_ctor_set(v___x_928_, 0, v_msgData_916_);
v___x_932_ = v___x_928_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_msgData_916_);
lean_ctor_set(v_reuseFailAlloc_940_, 1, v___x_930_);
v___x_932_ = v_reuseFailAlloc_940_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v_msgData_937_; lean_object* v___x_938_; lean_object* v___x_939_; 
v___x_933_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___closed__2);
v___x_934_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_934_, 0, v___x_932_);
lean_ctor_set(v___x_934_, 1, v___x_933_);
v___x_935_ = l_Lean_MessageData_ofSyntax(v_after_926_);
v___x_936_ = l_Lean_indentD(v___x_935_);
v_msgData_937_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_937_, 0, v___x_934_);
lean_ctor_set(v_msgData_937_, 1, v___x_936_);
v___x_938_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22(v_msgData_937_, v_macroStack_917_);
v___x_939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_939_, 0, v___x_938_);
return v___x_939_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg___boxed(lean_object* v_msgData_943_, lean_object* v_macroStack_944_, lean_object* v___y_945_, lean_object* v___y_946_){
_start:
{
lean_object* v_res_947_; 
v_res_947_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg(v_msgData_943_, v_macroStack_944_, v___y_945_);
lean_dec_ref(v___y_945_);
return v_res_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(lean_object* v_msg_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_){
_start:
{
lean_object* v_ref_956_; lean_object* v___x_957_; lean_object* v_a_958_; lean_object* v_macroStack_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_970_; 
v_ref_956_ = lean_ctor_get(v___y_953_, 5);
v___x_957_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v_msg_948_, v___y_951_, v___y_952_, v___y_953_, v___y_954_);
v_a_958_ = lean_ctor_get(v___x_957_, 0);
lean_inc(v_a_958_);
lean_dec_ref(v___x_957_);
v_macroStack_959_ = lean_ctor_get(v___y_949_, 1);
v___x_960_ = l_Lean_Elab_getBetterRef(v_ref_956_, v_macroStack_959_);
lean_inc(v_macroStack_959_);
v___x_961_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg(v_a_958_, v_macroStack_959_, v___y_953_);
v_a_962_ = lean_ctor_get(v___x_961_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_961_);
if (v_isSharedCheck_970_ == 0)
{
v___x_964_ = v___x_961_;
v_isShared_965_ = v_isSharedCheck_970_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_961_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_970_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___x_966_; lean_object* v___x_968_; 
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v___x_960_);
lean_ctor_set(v___x_966_, 1, v_a_962_);
if (v_isShared_965_ == 0)
{
lean_ctor_set_tag(v___x_964_, 1);
lean_ctor_set(v___x_964_, 0, v___x_966_);
v___x_968_ = v___x_964_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_msg_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(v_msg_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(lean_object* v_ref_980_, lean_object* v_msg_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_){
_start:
{
lean_object* v_fileName_989_; lean_object* v_fileMap_990_; lean_object* v_options_991_; lean_object* v_currRecDepth_992_; lean_object* v_maxRecDepth_993_; lean_object* v_ref_994_; lean_object* v_currNamespace_995_; lean_object* v_openDecls_996_; lean_object* v_initHeartbeats_997_; lean_object* v_maxHeartbeats_998_; lean_object* v_quotContext_999_; lean_object* v_currMacroScope_1000_; uint8_t v_diag_1001_; lean_object* v_cancelTk_x3f_1002_; uint8_t v_suppressElabErrors_1003_; lean_object* v_inheritedTraceOptions_1004_; lean_object* v_ref_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; 
v_fileName_989_ = lean_ctor_get(v___y_986_, 0);
v_fileMap_990_ = lean_ctor_get(v___y_986_, 1);
v_options_991_ = lean_ctor_get(v___y_986_, 2);
v_currRecDepth_992_ = lean_ctor_get(v___y_986_, 3);
v_maxRecDepth_993_ = lean_ctor_get(v___y_986_, 4);
v_ref_994_ = lean_ctor_get(v___y_986_, 5);
v_currNamespace_995_ = lean_ctor_get(v___y_986_, 6);
v_openDecls_996_ = lean_ctor_get(v___y_986_, 7);
v_initHeartbeats_997_ = lean_ctor_get(v___y_986_, 8);
v_maxHeartbeats_998_ = lean_ctor_get(v___y_986_, 9);
v_quotContext_999_ = lean_ctor_get(v___y_986_, 10);
v_currMacroScope_1000_ = lean_ctor_get(v___y_986_, 11);
v_diag_1001_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*14);
v_cancelTk_x3f_1002_ = lean_ctor_get(v___y_986_, 12);
v_suppressElabErrors_1003_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1004_ = lean_ctor_get(v___y_986_, 13);
v_ref_1005_ = l_Lean_replaceRef(v_ref_980_, v_ref_994_);
lean_inc_ref(v_inheritedTraceOptions_1004_);
lean_inc(v_cancelTk_x3f_1002_);
lean_inc(v_currMacroScope_1000_);
lean_inc(v_quotContext_999_);
lean_inc(v_maxHeartbeats_998_);
lean_inc(v_initHeartbeats_997_);
lean_inc(v_openDecls_996_);
lean_inc(v_currNamespace_995_);
lean_inc(v_maxRecDepth_993_);
lean_inc(v_currRecDepth_992_);
lean_inc_ref(v_options_991_);
lean_inc_ref(v_fileMap_990_);
lean_inc_ref(v_fileName_989_);
v___x_1006_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1006_, 0, v_fileName_989_);
lean_ctor_set(v___x_1006_, 1, v_fileMap_990_);
lean_ctor_set(v___x_1006_, 2, v_options_991_);
lean_ctor_set(v___x_1006_, 3, v_currRecDepth_992_);
lean_ctor_set(v___x_1006_, 4, v_maxRecDepth_993_);
lean_ctor_set(v___x_1006_, 5, v_ref_1005_);
lean_ctor_set(v___x_1006_, 6, v_currNamespace_995_);
lean_ctor_set(v___x_1006_, 7, v_openDecls_996_);
lean_ctor_set(v___x_1006_, 8, v_initHeartbeats_997_);
lean_ctor_set(v___x_1006_, 9, v_maxHeartbeats_998_);
lean_ctor_set(v___x_1006_, 10, v_quotContext_999_);
lean_ctor_set(v___x_1006_, 11, v_currMacroScope_1000_);
lean_ctor_set(v___x_1006_, 12, v_cancelTk_x3f_1002_);
lean_ctor_set(v___x_1006_, 13, v_inheritedTraceOptions_1004_);
lean_ctor_set_uint8(v___x_1006_, sizeof(void*)*14, v_diag_1001_);
lean_ctor_set_uint8(v___x_1006_, sizeof(void*)*14 + 1, v_suppressElabErrors_1003_);
v___x_1007_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(v_msg_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___x_1006_, v___y_987_);
lean_dec_ref_known(v___x_1006_, 14);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg___boxed(lean_object* v_ref_1008_, lean_object* v_msg_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_){
_start:
{
lean_object* v_res_1017_; 
v_res_1017_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(v_ref_1008_, v_msg_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_, v___y_1014_, v___y_1015_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec_ref(v___y_1010_);
lean_dec(v_ref_1008_);
return v_res_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2(lean_object* v_env_1018_, lean_object* v_currNamespace_1019_, lean_object* v_openDecls_1020_, lean_object* v_n_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = l_Lean_ResolveName_resolveNamespace(v_env_1018_, v_currNamespace_1019_, v_openDecls_1020_, v_n_1021_);
v___x_1025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1024_);
lean_ctor_set(v___x_1025_, 1, v___y_1023_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2___boxed(lean_object* v_env_1026_, lean_object* v_currNamespace_1027_, lean_object* v_openDecls_1028_, lean_object* v_n_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_){
_start:
{
lean_object* v_res_1032_; 
v_res_1032_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2(v_env_1026_, v_currNamespace_1027_, v_openDecls_1028_, v_n_1029_, v___y_1030_, v___y_1031_);
lean_dec_ref(v___y_1030_);
return v_res_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(lean_object* v_x_1033_, lean_object* v___y_1034_){
_start:
{
if (lean_obj_tag(v_x_1033_) == 0)
{
lean_object* v_a_1035_; lean_object* v___x_1036_; 
v_a_1035_ = lean_ctor_get(v_x_1033_, 0);
lean_inc(v_a_1035_);
v___x_1036_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1036_, 0, v_a_1035_);
lean_ctor_set(v___x_1036_, 1, v___y_1034_);
return v___x_1036_;
}
else
{
lean_object* v_a_1037_; lean_object* v___x_1038_; 
v_a_1037_ = lean_ctor_get(v_x_1033_, 0);
lean_inc(v_a_1037_);
v___x_1038_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1038_, 0, v_a_1037_);
lean_ctor_set(v___x_1038_, 1, v___y_1034_);
return v___x_1038_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_x_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_res_1041_; 
v_res_1041_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(v_x_1039_, v___y_1040_);
lean_dec_ref(v_x_1039_);
return v_res_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0(lean_object* v_env_1042_, lean_object* v_stx_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_){
_start:
{
lean_object* v___x_1046_; 
v___x_1046_ = l_Lean_Elab_expandMacroImpl_x3f(v_env_1042_, v_stx_1043_, v___y_1044_, v___y_1045_);
if (lean_obj_tag(v___x_1046_) == 0)
{
lean_object* v_a_1047_; 
v_a_1047_ = lean_ctor_get(v___x_1046_, 0);
lean_inc(v_a_1047_);
if (lean_obj_tag(v_a_1047_) == 0)
{
lean_object* v_a_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1056_; 
v_a_1048_ = lean_ctor_get(v___x_1046_, 1);
v_isSharedCheck_1056_ = !lean_is_exclusive(v___x_1046_);
if (v_isSharedCheck_1056_ == 0)
{
lean_object* v_unused_1057_; 
v_unused_1057_ = lean_ctor_get(v___x_1046_, 0);
lean_dec(v_unused_1057_);
v___x_1050_ = v___x_1046_;
v_isShared_1051_ = v_isSharedCheck_1056_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_a_1048_);
lean_dec(v___x_1046_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1056_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
lean_object* v___x_1052_; lean_object* v___x_1054_; 
v___x_1052_ = lean_box(0);
if (v_isShared_1051_ == 0)
{
lean_ctor_set(v___x_1050_, 0, v___x_1052_);
v___x_1054_ = v___x_1050_;
goto v_reusejp_1053_;
}
else
{
lean_object* v_reuseFailAlloc_1055_; 
v_reuseFailAlloc_1055_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1055_, 0, v___x_1052_);
lean_ctor_set(v_reuseFailAlloc_1055_, 1, v_a_1048_);
v___x_1054_ = v_reuseFailAlloc_1055_;
goto v_reusejp_1053_;
}
v_reusejp_1053_:
{
return v___x_1054_;
}
}
}
else
{
lean_object* v_val_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1086_; 
v_val_1058_ = lean_ctor_get(v_a_1047_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v_a_1047_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1060_ = v_a_1047_;
v_isShared_1061_ = v_isSharedCheck_1086_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_val_1058_);
lean_dec(v_a_1047_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1086_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v_snd_1062_; 
v_snd_1062_ = lean_ctor_get(v_val_1058_, 1);
lean_inc(v_snd_1062_);
lean_dec(v_val_1058_);
if (lean_obj_tag(v_snd_1062_) == 0)
{
lean_object* v_a_1063_; lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1072_; 
lean_del_object(v___x_1060_);
v_a_1063_ = lean_ctor_get(v___x_1046_, 1);
lean_inc(v_a_1063_);
lean_dec_ref_known(v___x_1046_, 2);
v_a_1064_ = lean_ctor_get(v_snd_1062_, 0);
v_isSharedCheck_1072_ = !lean_is_exclusive(v_snd_1062_);
if (v_isSharedCheck_1072_ == 0)
{
v___x_1066_ = v_snd_1062_;
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v_snd_1062_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1069_; 
if (v_isShared_1067_ == 0)
{
v___x_1069_ = v___x_1066_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v_a_1064_);
v___x_1069_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
lean_object* v___x_1070_; 
v___x_1070_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(v___x_1069_, v_a_1063_);
lean_dec_ref(v___x_1069_);
return v___x_1070_;
}
}
}
else
{
lean_object* v_a_1073_; lean_object* v_a_1074_; lean_object* v___x_1076_; uint8_t v_isShared_1077_; uint8_t v_isSharedCheck_1085_; 
v_a_1073_ = lean_ctor_get(v___x_1046_, 1);
lean_inc(v_a_1073_);
lean_dec_ref_known(v___x_1046_, 2);
v_a_1074_ = lean_ctor_get(v_snd_1062_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v_snd_1062_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1076_ = v_snd_1062_;
v_isShared_1077_ = v_isSharedCheck_1085_;
goto v_resetjp_1075_;
}
else
{
lean_inc(v_a_1074_);
lean_dec(v_snd_1062_);
v___x_1076_ = lean_box(0);
v_isShared_1077_ = v_isSharedCheck_1085_;
goto v_resetjp_1075_;
}
v_resetjp_1075_:
{
lean_object* v___x_1079_; 
if (v_isShared_1061_ == 0)
{
lean_ctor_set(v___x_1060_, 0, v_a_1074_);
v___x_1079_ = v___x_1060_;
goto v_reusejp_1078_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1074_);
v___x_1079_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1078_;
}
v_reusejp_1078_:
{
lean_object* v___x_1081_; 
if (v_isShared_1077_ == 0)
{
lean_ctor_set(v___x_1076_, 0, v___x_1079_);
v___x_1081_ = v___x_1076_;
goto v_reusejp_1080_;
}
else
{
lean_object* v_reuseFailAlloc_1083_; 
v_reuseFailAlloc_1083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1083_, 0, v___x_1079_);
v___x_1081_ = v_reuseFailAlloc_1083_;
goto v_reusejp_1080_;
}
v_reusejp_1080_:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(v___x_1081_, v_a_1073_);
lean_dec_ref(v___x_1081_);
return v___x_1082_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1087_; lean_object* v_a_1088_; lean_object* v___x_1090_; uint8_t v_isShared_1091_; uint8_t v_isSharedCheck_1095_; 
v_a_1087_ = lean_ctor_get(v___x_1046_, 0);
v_a_1088_ = lean_ctor_get(v___x_1046_, 1);
v_isSharedCheck_1095_ = !lean_is_exclusive(v___x_1046_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1090_ = v___x_1046_;
v_isShared_1091_ = v_isSharedCheck_1095_;
goto v_resetjp_1089_;
}
else
{
lean_inc(v_a_1088_);
lean_inc(v_a_1087_);
lean_dec(v___x_1046_);
v___x_1090_ = lean_box(0);
v_isShared_1091_ = v_isSharedCheck_1095_;
goto v_resetjp_1089_;
}
v_resetjp_1089_:
{
lean_object* v___x_1093_; 
if (v_isShared_1091_ == 0)
{
v___x_1093_ = v___x_1090_;
goto v_reusejp_1092_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v_a_1087_);
lean_ctor_set(v_reuseFailAlloc_1094_, 1, v_a_1088_);
v___x_1093_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1092_;
}
v_reusejp_1092_:
{
return v___x_1093_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object* v_env_1096_, lean_object* v_stx_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0(v_env_1096_, v_stx_1097_, v___y_1098_, v___y_1099_);
lean_dec_ref(v___y_1098_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(lean_object* v_x_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_){
_start:
{
lean_object* v___x_1110_; lean_object* v_env_1111_; lean_object* v_options_1112_; lean_object* v_currRecDepth_1113_; lean_object* v_maxRecDepth_1114_; lean_object* v_ref_1115_; lean_object* v_currNamespace_1116_; lean_object* v_openDecls_1117_; lean_object* v_quotContext_1118_; lean_object* v_currMacroScope_1119_; lean_object* v___x_1120_; lean_object* v_nextMacroScope_1121_; lean_object* v___f_1122_; lean_object* v___f_1123_; lean_object* v___f_1124_; lean_object* v___f_1125_; lean_object* v___f_1126_; lean_object* v_methods_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1110_ = lean_st_ref_get(v___y_1108_);
v_env_1111_ = lean_ctor_get(v___x_1110_, 0);
lean_inc_ref_n(v_env_1111_, 4);
lean_dec(v___x_1110_);
v_options_1112_ = lean_ctor_get(v___y_1107_, 2);
v_currRecDepth_1113_ = lean_ctor_get(v___y_1107_, 3);
v_maxRecDepth_1114_ = lean_ctor_get(v___y_1107_, 4);
v_ref_1115_ = lean_ctor_get(v___y_1107_, 5);
v_currNamespace_1116_ = lean_ctor_get(v___y_1107_, 6);
v_openDecls_1117_ = lean_ctor_get(v___y_1107_, 7);
v_quotContext_1118_ = lean_ctor_get(v___y_1107_, 10);
v_currMacroScope_1119_ = lean_ctor_get(v___y_1107_, 11);
v___x_1120_ = lean_st_ref_get(v___y_1108_);
v_nextMacroScope_1121_ = lean_ctor_get(v___x_1120_, 1);
lean_inc(v_nextMacroScope_1121_);
lean_dec(v___x_1120_);
v___f_1122_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1122_, 0, v_env_1111_);
v___f_1123_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_1123_, 0, v_env_1111_);
lean_inc_n(v_openDecls_1117_, 2);
lean_inc_n(v_currNamespace_1116_, 3);
v___f_1124_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__2___boxed), 6, 3);
lean_closure_set(v___f_1124_, 0, v_env_1111_);
lean_closure_set(v___f_1124_, 1, v_currNamespace_1116_);
lean_closure_set(v___f_1124_, 2, v_openDecls_1117_);
v___f_1125_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_1125_, 0, v_currNamespace_1116_);
lean_inc_ref(v_options_1112_);
v___f_1126_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___lam__4___boxed), 7, 4);
lean_closure_set(v___f_1126_, 0, v_env_1111_);
lean_closure_set(v___f_1126_, 1, v_options_1112_);
lean_closure_set(v___f_1126_, 2, v_currNamespace_1116_);
lean_closure_set(v___f_1126_, 3, v_openDecls_1117_);
v_methods_1127_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_methods_1127_, 0, v___f_1122_);
lean_ctor_set(v_methods_1127_, 1, v___f_1125_);
lean_ctor_set(v_methods_1127_, 2, v___f_1123_);
lean_ctor_set(v_methods_1127_, 3, v___f_1124_);
lean_ctor_set(v_methods_1127_, 4, v___f_1126_);
lean_inc(v_ref_1115_);
lean_inc(v_maxRecDepth_1114_);
lean_inc(v_currRecDepth_1113_);
lean_inc(v_currMacroScope_1119_);
lean_inc(v_quotContext_1118_);
v___x_1128_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1128_, 0, v_methods_1127_);
lean_ctor_set(v___x_1128_, 1, v_quotContext_1118_);
lean_ctor_set(v___x_1128_, 2, v_currMacroScope_1119_);
lean_ctor_set(v___x_1128_, 3, v_currRecDepth_1113_);
lean_ctor_set(v___x_1128_, 4, v_maxRecDepth_1114_);
lean_ctor_set(v___x_1128_, 5, v_ref_1115_);
v___x_1129_ = lean_box(0);
v___x_1130_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1130_, 0, v_nextMacroScope_1121_);
lean_ctor_set(v___x_1130_, 1, v___x_1129_);
lean_ctor_set(v___x_1130_, 2, v___x_1129_);
v___x_1131_ = lean_apply_2(v_x_1102_, v___x_1128_, v___x_1130_);
if (lean_obj_tag(v___x_1131_) == 0)
{
lean_object* v_a_1132_; lean_object* v_a_1133_; lean_object* v_macroScope_1134_; lean_object* v_traceMsgs_1135_; lean_object* v_expandedMacroDecls_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; 
v_a_1132_ = lean_ctor_get(v___x_1131_, 1);
lean_inc(v_a_1132_);
v_a_1133_ = lean_ctor_get(v___x_1131_, 0);
lean_inc(v_a_1133_);
lean_dec_ref_known(v___x_1131_, 2);
v_macroScope_1134_ = lean_ctor_get(v_a_1132_, 0);
lean_inc(v_macroScope_1134_);
v_traceMsgs_1135_ = lean_ctor_get(v_a_1132_, 1);
lean_inc(v_traceMsgs_1135_);
v_expandedMacroDecls_1136_ = lean_ctor_get(v_a_1132_, 2);
lean_inc(v_expandedMacroDecls_1136_);
lean_dec(v_a_1132_);
v___x_1137_ = lean_box(0);
v___x_1138_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg(v_expandedMacroDecls_1136_, v___x_1137_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_);
lean_dec(v_expandedMacroDecls_1136_);
if (lean_obj_tag(v___x_1138_) == 0)
{
lean_object* v___x_1139_; lean_object* v_env_1140_; lean_object* v_ngen_1141_; lean_object* v_auxDeclNGen_1142_; lean_object* v_traceState_1143_; lean_object* v_cache_1144_; lean_object* v_messages_1145_; lean_object* v_infoState_1146_; lean_object* v_snapshotTasks_1147_; lean_object* v___x_1149_; uint8_t v_isShared_1150_; uint8_t v_isSharedCheck_1173_; 
lean_dec_ref_known(v___x_1138_, 1);
v___x_1139_ = lean_st_ref_take(v___y_1108_);
v_env_1140_ = lean_ctor_get(v___x_1139_, 0);
v_ngen_1141_ = lean_ctor_get(v___x_1139_, 2);
v_auxDeclNGen_1142_ = lean_ctor_get(v___x_1139_, 3);
v_traceState_1143_ = lean_ctor_get(v___x_1139_, 4);
v_cache_1144_ = lean_ctor_get(v___x_1139_, 5);
v_messages_1145_ = lean_ctor_get(v___x_1139_, 6);
v_infoState_1146_ = lean_ctor_get(v___x_1139_, 7);
v_snapshotTasks_1147_ = lean_ctor_get(v___x_1139_, 8);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1173_ == 0)
{
lean_object* v_unused_1174_; 
v_unused_1174_ = lean_ctor_get(v___x_1139_, 1);
lean_dec(v_unused_1174_);
v___x_1149_ = v___x_1139_;
v_isShared_1150_ = v_isSharedCheck_1173_;
goto v_resetjp_1148_;
}
else
{
lean_inc(v_snapshotTasks_1147_);
lean_inc(v_infoState_1146_);
lean_inc(v_messages_1145_);
lean_inc(v_cache_1144_);
lean_inc(v_traceState_1143_);
lean_inc(v_auxDeclNGen_1142_);
lean_inc(v_ngen_1141_);
lean_inc(v_env_1140_);
lean_dec(v___x_1139_);
v___x_1149_ = lean_box(0);
v_isShared_1150_ = v_isSharedCheck_1173_;
goto v_resetjp_1148_;
}
v_resetjp_1148_:
{
lean_object* v___x_1152_; 
if (v_isShared_1150_ == 0)
{
lean_ctor_set(v___x_1149_, 1, v_macroScope_1134_);
v___x_1152_ = v___x_1149_;
goto v_reusejp_1151_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v_env_1140_);
lean_ctor_set(v_reuseFailAlloc_1172_, 1, v_macroScope_1134_);
lean_ctor_set(v_reuseFailAlloc_1172_, 2, v_ngen_1141_);
lean_ctor_set(v_reuseFailAlloc_1172_, 3, v_auxDeclNGen_1142_);
lean_ctor_set(v_reuseFailAlloc_1172_, 4, v_traceState_1143_);
lean_ctor_set(v_reuseFailAlloc_1172_, 5, v_cache_1144_);
lean_ctor_set(v_reuseFailAlloc_1172_, 6, v_messages_1145_);
lean_ctor_set(v_reuseFailAlloc_1172_, 7, v_infoState_1146_);
lean_ctor_set(v_reuseFailAlloc_1172_, 8, v_snapshotTasks_1147_);
v___x_1152_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1151_;
}
v_reusejp_1151_:
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; 
v___x_1153_ = lean_st_ref_set(v___y_1108_, v___x_1152_);
v___x_1154_ = l_List_reverse___redArg(v_traceMsgs_1135_);
v___x_1155_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__7(v___x_1154_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_);
if (lean_obj_tag(v___x_1155_) == 0)
{
lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1162_ == 0)
{
lean_object* v_unused_1163_; 
v_unused_1163_ = lean_ctor_get(v___x_1155_, 0);
lean_dec(v_unused_1163_);
v___x_1157_ = v___x_1155_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_dec(v___x_1155_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
lean_ctor_set(v___x_1157_, 0, v_a_1133_);
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_a_1133_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
else
{
lean_object* v_a_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1171_; 
lean_dec(v_a_1133_);
v_a_1164_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1171_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1171_ == 0)
{
v___x_1166_ = v___x_1155_;
v_isShared_1167_ = v_isSharedCheck_1171_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_a_1164_);
lean_dec(v___x_1155_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1171_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1169_; 
if (v_isShared_1167_ == 0)
{
v___x_1169_ = v___x_1166_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v_a_1164_);
v___x_1169_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
return v___x_1169_;
}
}
}
}
}
}
else
{
lean_object* v_a_1175_; lean_object* v___x_1177_; uint8_t v_isShared_1178_; uint8_t v_isSharedCheck_1182_; 
lean_dec(v_traceMsgs_1135_);
lean_dec(v_macroScope_1134_);
lean_dec(v_a_1133_);
v_a_1175_ = lean_ctor_get(v___x_1138_, 0);
v_isSharedCheck_1182_ = !lean_is_exclusive(v___x_1138_);
if (v_isSharedCheck_1182_ == 0)
{
v___x_1177_ = v___x_1138_;
v_isShared_1178_ = v_isSharedCheck_1182_;
goto v_resetjp_1176_;
}
else
{
lean_inc(v_a_1175_);
lean_dec(v___x_1138_);
v___x_1177_ = lean_box(0);
v_isShared_1178_ = v_isSharedCheck_1182_;
goto v_resetjp_1176_;
}
v_resetjp_1176_:
{
lean_object* v___x_1180_; 
if (v_isShared_1178_ == 0)
{
v___x_1180_ = v___x_1177_;
goto v_reusejp_1179_;
}
else
{
lean_object* v_reuseFailAlloc_1181_; 
v_reuseFailAlloc_1181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1181_, 0, v_a_1175_);
v___x_1180_ = v_reuseFailAlloc_1181_;
goto v_reusejp_1179_;
}
v_reusejp_1179_:
{
return v___x_1180_;
}
}
}
}
else
{
lean_object* v_a_1183_; 
v_a_1183_ = lean_ctor_get(v___x_1131_, 0);
lean_inc(v_a_1183_);
lean_dec_ref_known(v___x_1131_, 2);
if (lean_obj_tag(v_a_1183_) == 0)
{
lean_object* v_a_1184_; lean_object* v_a_1185_; lean_object* v___x_1186_; uint8_t v___x_1187_; 
v_a_1184_ = lean_ctor_get(v_a_1183_, 0);
lean_inc(v_a_1184_);
v_a_1185_ = lean_ctor_get(v_a_1183_, 1);
lean_inc_ref(v_a_1185_);
lean_dec_ref_known(v_a_1183_, 2);
v___x_1186_ = ((lean_object*)(lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___closed__0));
v___x_1187_ = lean_string_dec_eq(v_a_1185_, v___x_1186_);
if (v___x_1187_ == 0)
{
lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; 
v___x_1188_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1188_, 0, v_a_1185_);
v___x_1189_ = l_Lean_MessageData_ofFormat(v___x_1188_);
v___x_1190_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(v_a_1184_, v___x_1189_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_);
lean_dec(v_a_1184_);
return v___x_1190_;
}
else
{
lean_object* v___x_1191_; 
lean_dec_ref(v_a_1185_);
v___x_1191_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg(v_a_1184_);
return v___x_1191_;
}
}
else
{
lean_object* v___x_1192_; 
v___x_1192_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg();
return v___x_1192_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_){
_start:
{
lean_object* v_res_1201_; 
v_res_1201_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(v_x_1193_, v___y_1194_, v___y_1195_, v___y_1196_, v___y_1197_, v___y_1198_, v___y_1199_);
lean_dec(v___y_1199_);
lean_dec_ref(v___y_1198_);
lean_dec(v___y_1197_);
lean_dec_ref(v___y_1196_);
lean_dec(v___y_1195_);
lean_dec_ref(v___y_1194_);
return v_res_1201_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1203_; lean_object* v___x_1204_; 
v___x_1203_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__0));
v___x_1204_ = l_Lean_stringToMessageData(v___x_1203_);
return v___x_1204_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; 
v___x_1206_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__2));
v___x_1207_ = l_Lean_stringToMessageData(v___x_1206_);
return v___x_1207_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5(void){
_start:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; 
v___x_1209_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__4));
v___x_1210_ = l_Lean_stringToMessageData(v___x_1209_);
return v___x_1210_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1212_; lean_object* v___x_1213_; 
v___x_1212_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__6));
v___x_1213_ = l_Lean_stringToMessageData(v___x_1212_);
return v___x_1213_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1215_; lean_object* v___x_1216_; 
v___x_1215_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__8));
v___x_1216_ = l_Lean_stringToMessageData(v___x_1215_);
return v___x_1216_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1218_; lean_object* v___x_1219_; 
v___x_1218_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__10));
v___x_1219_ = l_Lean_stringToMessageData(v___x_1218_);
return v___x_1219_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16(void){
_start:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; 
v___x_1228_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__15));
v___x_1229_ = l_Lean_stringToMessageData(v___x_1228_);
return v___x_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1(lean_object* v___x_1230_, lean_object* v_attrInstance_1231_, lean_object* v___f_1232_, lean_object* v___x_1233_, lean_object* v___x_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_){
_start:
{
lean_object* v___x_1242_; 
v___x_1242_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(v___x_1230_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_);
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_object* v_a_1243_; lean_object* v___x_1244_; lean_object* v_attr_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; 
v_a_1243_ = lean_ctor_get(v___x_1242_, 0);
lean_inc(v_a_1243_);
lean_dec_ref_known(v___x_1242_, 1);
v___x_1244_ = lean_unsigned_to_nat(1u);
v_attr_1245_ = l_Lean_Syntax_getArg(v_attrInstance_1231_, v___x_1244_);
v___x_1246_ = lean_alloc_closure((void*)(l_Lean_expandMacros), 4, 2);
lean_closure_set(v___x_1246_, 0, v_attr_1245_);
lean_closure_set(v___x_1246_, 1, v___f_1232_);
v___x_1247_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(v___x_1246_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_);
if (lean_obj_tag(v___x_1247_) == 0)
{
lean_object* v_a_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1362_; 
v_a_1248_ = lean_ctor_get(v___x_1247_, 0);
v_isSharedCheck_1362_ = !lean_is_exclusive(v___x_1247_);
if (v_isSharedCheck_1362_ == 0)
{
v___x_1250_ = v___x_1247_;
v_isShared_1251_ = v_isSharedCheck_1362_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_a_1248_);
lean_dec(v___x_1247_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1362_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v___y_1253_; uint8_t v___y_1260_; lean_object* v___y_1261_; lean_object* v___y_1262_; lean_object* v___y_1263_; lean_object* v___y_1264_; lean_object* v___y_1265_; lean_object* v___y_1266_; lean_object* v___y_1267_; lean_object* v___y_1268_; lean_object* v_attrName_1279_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1282_; lean_object* v___y_1283_; lean_object* v___y_1284_; lean_object* v___y_1285_; lean_object* v___x_1343_; lean_object* v___x_1344_; uint8_t v___x_1345_; 
lean_inc(v_a_1248_);
v___x_1343_ = l_Lean_Syntax_getKind(v_a_1248_);
v___x_1344_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__14));
v___x_1345_ = lean_name_eq(v___x_1343_, v___x_1344_);
if (v___x_1345_ == 0)
{
if (lean_obj_tag(v___x_1343_) == 1)
{
lean_object* v_str_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; 
v_str_1346_ = lean_ctor_get(v___x_1343_, 1);
lean_inc_ref(v_str_1346_);
lean_dec_ref_known(v___x_1343_, 2);
v___x_1347_ = lean_box(0);
v___x_1348_ = l_Lean_Name_str___override(v___x_1347_, v_str_1346_);
v_attrName_1279_ = v___x_1348_;
v___y_1280_ = v___y_1235_;
v___y_1281_ = v___y_1236_;
v___y_1282_ = v___y_1237_;
v___y_1283_ = v___y_1238_;
v___y_1284_ = v___y_1239_;
v___y_1285_ = v___y_1240_;
goto v___jp_1278_;
}
else
{
lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v_a_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1358_; 
lean_dec(v___x_1343_);
lean_del_object(v___x_1250_);
lean_dec(v_a_1243_);
lean_dec(v___x_1233_);
v___x_1349_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__16);
v___x_1350_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(v_a_1248_, v___x_1349_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_);
lean_dec(v_a_1248_);
v_a_1351_ = lean_ctor_get(v___x_1350_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1350_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1353_ = v___x_1350_;
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_a_1351_);
lean_dec(v___x_1350_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_a_1351_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
}
else
{
lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
lean_dec(v___x_1343_);
v___x_1359_ = l_Lean_Syntax_getArg(v_a_1248_, v___x_1234_);
v___x_1360_ = l_Lean_Syntax_getId(v___x_1359_);
lean_dec(v___x_1359_);
v___x_1361_ = l_Lean_Name_eraseMacroScopes(v___x_1360_);
lean_dec(v___x_1360_);
v_attrName_1279_ = v___x_1361_;
v___y_1280_ = v___y_1235_;
v___y_1281_ = v___y_1236_;
v___y_1282_ = v___y_1237_;
v___y_1283_ = v___y_1238_;
v___y_1284_ = v___y_1239_;
v___y_1285_ = v___y_1240_;
goto v___jp_1278_;
}
v___jp_1252_:
{
lean_object* v___x_1254_; uint8_t v___x_1255_; lean_object* v___x_1257_; 
v___x_1254_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1254_, 0, v___y_1253_);
lean_ctor_set(v___x_1254_, 1, v_a_1248_);
v___x_1255_ = lean_unbox(v_a_1243_);
lean_dec(v_a_1243_);
lean_ctor_set_uint8(v___x_1254_, sizeof(void*)*2, v___x_1255_);
if (v_isShared_1251_ == 0)
{
lean_ctor_set(v___x_1250_, 0, v___x_1254_);
v___x_1257_ = v___x_1250_;
goto v_reusejp_1256_;
}
else
{
lean_object* v_reuseFailAlloc_1258_; 
v_reuseFailAlloc_1258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1258_, 0, v___x_1254_);
v___x_1257_ = v_reuseFailAlloc_1258_;
goto v_reusejp_1256_;
}
v_reusejp_1256_:
{
return v___x_1257_;
}
}
v___jp_1259_:
{
lean_object* v___x_1269_; 
v___x_1269_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3(v___y_1262_, v___y_1260_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_);
if (lean_obj_tag(v___x_1269_) == 0)
{
lean_dec_ref_known(v___x_1269_, 1);
v___y_1253_ = v___y_1261_;
goto v___jp_1252_;
}
else
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1277_; 
lean_dec(v___y_1261_);
lean_del_object(v___x_1250_);
lean_dec(v_a_1248_);
lean_dec(v_a_1243_);
v_a_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1277_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1277_ == 0)
{
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1275_; 
if (v_isShared_1273_ == 0)
{
v___x_1275_ = v___x_1272_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v_a_1270_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
v___jp_1278_:
{
lean_object* v___x_1286_; lean_object* v_env_1287_; lean_object* v___x_1288_; 
v___x_1286_ = lean_st_ref_get(v___y_1285_);
v_env_1287_ = lean_ctor_get(v___x_1286_, 0);
lean_inc_ref(v_env_1287_);
lean_dec(v___x_1286_);
lean_inc(v_attrName_1279_);
v___x_1288_ = l_Lean_getAttributeImpl(v_env_1287_, v_attrName_1279_);
if (lean_obj_tag(v___x_1288_) == 1)
{
lean_object* v___x_1289_; lean_object* v_env_1290_; lean_object* v___x_1291_; 
lean_dec_ref_known(v___x_1288_, 1);
v___x_1289_ = lean_st_ref_get(v___y_1285_);
v_env_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc_ref(v_env_1290_);
lean_dec(v___x_1289_);
lean_inc(v_attrName_1279_);
v___x_1291_ = l_Lean_getAttributeImpl(v_env_1290_, v_attrName_1279_);
if (lean_obj_tag(v___x_1291_) == 1)
{
lean_object* v_a_1292_; lean_object* v___x_1293_; lean_object* v_toAttributeImplCore_1294_; lean_object* v_env_1295_; lean_object* v_ref_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; 
v_a_1292_ = lean_ctor_get(v___x_1291_, 0);
lean_inc(v_a_1292_);
lean_dec_ref_known(v___x_1291_, 1);
v___x_1293_ = lean_st_ref_get(v___y_1285_);
v_toAttributeImplCore_1294_ = lean_ctor_get(v_a_1292_, 0);
lean_inc_ref(v_toAttributeImplCore_1294_);
lean_dec(v_a_1292_);
v_env_1295_ = lean_ctor_get(v___x_1293_, 0);
lean_inc_ref(v_env_1295_);
lean_dec(v___x_1293_);
v_ref_1296_ = lean_ctor_get(v_toAttributeImplCore_1294_, 0);
lean_inc_n(v_ref_1296_, 2);
lean_dec_ref(v_toAttributeImplCore_1294_);
v___x_1297_ = l_Lean_regularInitAttr;
v___x_1298_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_1233_, v___x_1297_, v_env_1295_, v_ref_1296_);
if (lean_obj_tag(v___x_1298_) == 0)
{
lean_dec(v_ref_1296_);
v___y_1253_ = v_attrName_1279_;
goto v___jp_1252_;
}
else
{
lean_object* v___x_1299_; lean_object* v_env_1300_; uint8_t v___x_1301_; lean_object* v___x_1302_; 
lean_dec_ref_known(v___x_1298_, 1);
v___x_1299_ = lean_st_ref_get(v___y_1285_);
v_env_1300_ = lean_ctor_get(v___x_1299_, 0);
lean_inc_ref(v_env_1300_);
lean_dec(v___x_1299_);
v___x_1301_ = 1;
v___x_1302_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1300_, v_ref_1296_);
lean_dec_ref(v_env_1300_);
if (lean_obj_tag(v___x_1302_) == 1)
{
lean_object* v_val_1303_; lean_object* v___x_1304_; lean_object* v_env_1305_; lean_object* v___x_1306_; lean_object* v_modules_1307_; lean_object* v___x_1308_; uint8_t v___x_1309_; 
v_val_1303_ = lean_ctor_get(v___x_1302_, 0);
lean_inc(v_val_1303_);
lean_dec_ref_known(v___x_1302_, 1);
v___x_1304_ = lean_st_ref_get(v___y_1285_);
v_env_1305_ = lean_ctor_get(v___x_1304_, 0);
lean_inc_ref(v_env_1305_);
lean_dec(v___x_1304_);
v___x_1306_ = l_Lean_Environment_header(v_env_1305_);
lean_dec_ref(v_env_1305_);
v_modules_1307_ = lean_ctor_get(v___x_1306_, 3);
lean_inc_ref(v_modules_1307_);
lean_dec_ref(v___x_1306_);
v___x_1308_ = lean_array_get_size(v_modules_1307_);
v___x_1309_ = lean_nat_dec_lt(v_val_1303_, v___x_1308_);
if (v___x_1309_ == 0)
{
lean_dec_ref(v_modules_1307_);
lean_dec(v_val_1303_);
v___y_1260_ = v___x_1301_;
v___y_1261_ = v_attrName_1279_;
v___y_1262_ = v_ref_1296_;
v___y_1263_ = v___y_1280_;
v___y_1264_ = v___y_1281_;
v___y_1265_ = v___y_1282_;
v___y_1266_ = v___y_1283_;
v___y_1267_ = v___y_1284_;
v___y_1268_ = v___y_1285_;
goto v___jp_1259_;
}
else
{
lean_object* v___x_1310_; uint8_t v_hasData_1311_; 
v___x_1310_ = lean_array_fget_borrowed(v_modules_1307_, v_val_1303_);
v_hasData_1311_ = lean_ctor_get_uint8(v___x_1310_, sizeof(void*)*1 + 1);
if (v_hasData_1311_ == 0)
{
lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v_toImport_1314_; lean_object* v_module_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1336_; 
lean_dec(v_ref_1296_);
lean_del_object(v___x_1250_);
lean_dec(v_a_1248_);
lean_dec(v_a_1243_);
v___x_1312_ = l_Lean_instInhabitedEffectiveImport_default;
v___x_1313_ = lean_array_get(v___x_1312_, v_modules_1307_, v_val_1303_);
lean_dec(v_val_1303_);
lean_dec_ref(v_modules_1307_);
v_toImport_1314_ = lean_ctor_get(v___x_1313_, 0);
lean_inc_ref(v_toImport_1314_);
lean_dec(v___x_1313_);
v_module_1315_ = lean_ctor_get(v_toImport_1314_, 0);
lean_inc(v_module_1315_);
lean_dec_ref(v_toImport_1314_);
v___x_1316_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__1);
v___x_1317_ = l_Lean_MessageData_ofName(v_attrName_1279_);
v___x_1318_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1316_);
lean_ctor_set(v___x_1318_, 1, v___x_1317_);
v___x_1319_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__3);
v___x_1320_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1320_, 0, v___x_1318_);
lean_ctor_set(v___x_1320_, 1, v___x_1319_);
v___x_1321_ = l_Lean_MessageData_ofName(v_module_1315_);
lean_inc_ref(v___x_1321_);
v___x_1322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1320_);
lean_ctor_set(v___x_1322_, 1, v___x_1321_);
v___x_1323_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__5);
v___x_1324_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1322_);
lean_ctor_set(v___x_1324_, 1, v___x_1323_);
v___x_1325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1325_, 0, v___x_1324_);
lean_ctor_set(v___x_1325_, 1, v___x_1321_);
v___x_1326_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7);
v___x_1327_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1325_);
lean_ctor_set(v___x_1327_, 1, v___x_1326_);
v___x_1328_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(v___x_1327_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
v_a_1329_ = lean_ctor_get(v___x_1328_, 0);
v_isSharedCheck_1336_ = !lean_is_exclusive(v___x_1328_);
if (v_isSharedCheck_1336_ == 0)
{
v___x_1331_ = v___x_1328_;
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1328_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1334_; 
if (v_isShared_1332_ == 0)
{
v___x_1334_ = v___x_1331_;
goto v_reusejp_1333_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v_a_1329_);
v___x_1334_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1333_;
}
v_reusejp_1333_:
{
return v___x_1334_;
}
}
}
else
{
lean_dec_ref(v_modules_1307_);
lean_dec(v_val_1303_);
v___y_1260_ = v___x_1301_;
v___y_1261_ = v_attrName_1279_;
v___y_1262_ = v_ref_1296_;
v___y_1263_ = v___y_1280_;
v___y_1264_ = v___y_1281_;
v___y_1265_ = v___y_1282_;
v___y_1266_ = v___y_1283_;
v___y_1267_ = v___y_1284_;
v___y_1268_ = v___y_1285_;
goto v___jp_1259_;
}
}
}
else
{
lean_dec(v___x_1302_);
v___y_1260_ = v___x_1301_;
v___y_1261_ = v_attrName_1279_;
v___y_1262_ = v_ref_1296_;
v___y_1263_ = v___y_1280_;
v___y_1264_ = v___y_1281_;
v___y_1265_ = v___y_1282_;
v___y_1266_ = v___y_1283_;
v___y_1267_ = v___y_1284_;
v___y_1268_ = v___y_1285_;
goto v___jp_1259_;
}
}
}
else
{
lean_dec_ref(v___x_1291_);
lean_dec(v___x_1233_);
v___y_1253_ = v_attrName_1279_;
goto v___jp_1252_;
}
}
else
{
lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; 
lean_dec_ref(v___x_1288_);
lean_del_object(v___x_1250_);
lean_dec(v_a_1248_);
lean_dec(v_a_1243_);
lean_dec(v___x_1233_);
v___x_1337_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__9);
v___x_1338_ = l_Lean_MessageData_ofName(v_attrName_1279_);
v___x_1339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1339_, 0, v___x_1337_);
lean_ctor_set(v___x_1339_, 1, v___x_1338_);
v___x_1340_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__11);
v___x_1341_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1341_, 0, v___x_1339_);
lean_ctor_set(v___x_1341_, 1, v___x_1340_);
v___x_1342_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(v___x_1341_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
return v___x_1342_;
}
}
}
}
else
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
lean_dec(v_a_1243_);
lean_dec(v___x_1233_);
v_a_1363_ = lean_ctor_get(v___x_1247_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1247_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1247_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1247_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_a_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
else
{
lean_object* v_a_1371_; lean_object* v___x_1373_; uint8_t v_isShared_1374_; uint8_t v_isSharedCheck_1378_; 
lean_dec(v___x_1233_);
lean_dec_ref(v___f_1232_);
v_a_1371_ = lean_ctor_get(v___x_1242_, 0);
v_isSharedCheck_1378_ = !lean_is_exclusive(v___x_1242_);
if (v_isSharedCheck_1378_ == 0)
{
v___x_1373_ = v___x_1242_;
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
else
{
lean_inc(v_a_1371_);
lean_dec(v___x_1242_);
v___x_1373_ = lean_box(0);
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
v_resetjp_1372_:
{
lean_object* v___x_1376_; 
if (v_isShared_1374_ == 0)
{
v___x_1376_ = v___x_1373_;
goto v_reusejp_1375_;
}
else
{
lean_object* v_reuseFailAlloc_1377_; 
v_reuseFailAlloc_1377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1377_, 0, v_a_1371_);
v___x_1376_ = v_reuseFailAlloc_1377_;
goto v_reusejp_1375_;
}
v_reusejp_1375_:
{
return v___x_1376_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___boxed(lean_object* v___x_1379_, lean_object* v_attrInstance_1380_, lean_object* v___f_1381_, lean_object* v___x_1382_, lean_object* v___x_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_){
_start:
{
lean_object* v_res_1391_; 
v_res_1391_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1(v___x_1379_, v_attrInstance_1380_, v___f_1381_, v___x_1382_, v___x_1383_, v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
lean_dec(v___y_1389_);
lean_dec_ref(v___y_1388_);
lean_dec(v___y_1387_);
lean_dec_ref(v___y_1386_);
lean_dec(v___y_1385_);
lean_dec_ref(v___y_1384_);
lean_dec(v___x_1383_);
lean_dec(v_attrInstance_1380_);
return v_res_1391_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0(lean_object* v_k_1398_){
_start:
{
lean_object* v___x_1399_; uint8_t v___x_1400_; 
v___x_1399_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___closed__1));
v___x_1400_ = lean_name_eq(v_k_1398_, v___x_1399_);
if (v___x_1400_ == 0)
{
uint8_t v___x_1401_; 
v___x_1401_ = 1;
return v___x_1401_;
}
else
{
uint8_t v___x_1402_; 
v___x_1402_ = 0;
return v___x_1402_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0___boxed(lean_object* v_k_1403_){
_start:
{
uint8_t v_res_1404_; lean_object* v_r_1405_; 
v_res_1404_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__0(v_k_1403_);
lean_dec(v_k_1403_);
v_r_1405_ = lean_box(v_res_1404_);
return v_r_1405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1(lean_object* v_attrInstance_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_){
_start:
{
lean_object* v___f_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___f_1420_; uint8_t v___x_1421_; lean_object* v___x_1422_; 
v___f_1415_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___closed__0));
v___x_1416_ = lean_box(0);
v___x_1417_ = lean_unsigned_to_nat(0u);
v___x_1418_ = l_Lean_Syntax_getArg(v_attrInstance_1407_, v___x_1417_);
v___x_1419_ = lean_alloc_closure((void*)(l_Lean_Elab_toAttributeKind___boxed), 3, 1);
lean_closure_set(v___x_1419_, 0, v___x_1418_);
v___f_1420_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___boxed), 12, 5);
lean_closure_set(v___f_1420_, 0, v___x_1419_);
lean_closure_set(v___f_1420_, 1, v_attrInstance_1407_);
lean_closure_set(v___f_1420_, 2, v___f_1415_);
lean_closure_set(v___f_1420_, 3, v___x_1416_);
lean_closure_set(v___f_1420_, 4, v___x_1417_);
v___x_1421_ = 1;
v___x_1422_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg(v___f_1420_, v___x_1421_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_);
return v___x_1422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___boxed(lean_object* v_attrInstance_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_){
_start:
{
lean_object* v_res_1431_; 
v_res_1431_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1(v_attrInstance_1423_, v___y_1424_, v___y_1425_, v___y_1426_, v___y_1427_, v___y_1428_, v___y_1429_);
lean_dec(v___y_1429_);
lean_dec_ref(v___y_1428_);
lean_dec(v___y_1427_);
lean_dec_ref(v___y_1426_);
lean_dec(v___y_1425_);
lean_dec_ref(v___y_1424_);
return v_res_1431_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0(uint8_t v___y_1438_, uint8_t v_suppressElabErrors_1439_, lean_object* v_x_1440_){
_start:
{
if (lean_obj_tag(v_x_1440_) == 1)
{
lean_object* v_pre_1441_; 
v_pre_1441_ = lean_ctor_get(v_x_1440_, 0);
switch(lean_obj_tag(v_pre_1441_))
{
case 1:
{
lean_object* v_pre_1442_; 
v_pre_1442_ = lean_ctor_get(v_pre_1441_, 0);
switch(lean_obj_tag(v_pre_1442_))
{
case 0:
{
lean_object* v_str_1443_; lean_object* v_str_1444_; lean_object* v___x_1445_; uint8_t v___x_1446_; 
v_str_1443_ = lean_ctor_get(v_x_1440_, 1);
v_str_1444_ = lean_ctor_get(v_pre_1441_, 1);
v___x_1445_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__0));
v___x_1446_ = lean_string_dec_eq(v_str_1444_, v___x_1445_);
if (v___x_1446_ == 0)
{
lean_object* v___x_1447_; uint8_t v___x_1448_; 
v___x_1447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_optAttrArg___closed__2));
v___x_1448_ = lean_string_dec_eq(v_str_1444_, v___x_1447_);
if (v___x_1448_ == 0)
{
return v___y_1438_;
}
else
{
lean_object* v___x_1449_; uint8_t v___x_1450_; 
v___x_1449_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__1));
v___x_1450_ = lean_string_dec_eq(v_str_1443_, v___x_1449_);
if (v___x_1450_ == 0)
{
return v___y_1438_;
}
else
{
return v_suppressElabErrors_1439_;
}
}
}
else
{
lean_object* v___x_1451_; uint8_t v___x_1452_; 
v___x_1451_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__2));
v___x_1452_ = lean_string_dec_eq(v_str_1443_, v___x_1451_);
if (v___x_1452_ == 0)
{
return v___y_1438_;
}
else
{
return v_suppressElabErrors_1439_;
}
}
}
case 1:
{
lean_object* v_pre_1453_; 
v_pre_1453_ = lean_ctor_get(v_pre_1442_, 0);
if (lean_obj_tag(v_pre_1453_) == 0)
{
lean_object* v_str_1454_; lean_object* v_str_1455_; lean_object* v_str_1456_; lean_object* v___x_1457_; uint8_t v___x_1458_; 
v_str_1454_ = lean_ctor_get(v_x_1440_, 1);
v_str_1455_ = lean_ctor_get(v_pre_1441_, 1);
v_str_1456_ = lean_ctor_get(v_pre_1442_, 1);
v___x_1457_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__3));
v___x_1458_ = lean_string_dec_eq(v_str_1456_, v___x_1457_);
if (v___x_1458_ == 0)
{
return v___y_1438_;
}
else
{
lean_object* v___x_1459_; uint8_t v___x_1460_; 
v___x_1459_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__4));
v___x_1460_ = lean_string_dec_eq(v_str_1455_, v___x_1459_);
if (v___x_1460_ == 0)
{
return v___y_1438_;
}
else
{
lean_object* v___x_1461_; uint8_t v___x_1462_; 
v___x_1461_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___closed__5));
v___x_1462_ = lean_string_dec_eq(v_str_1454_, v___x_1461_);
if (v___x_1462_ == 0)
{
return v___y_1438_;
}
else
{
return v_suppressElabErrors_1439_;
}
}
}
}
else
{
return v___y_1438_;
}
}
default: 
{
return v___y_1438_;
}
}
}
case 0:
{
lean_object* v_str_1463_; lean_object* v___x_1464_; uint8_t v___x_1465_; 
v_str_1463_ = lean_ctor_get(v_x_1440_, 1);
v___x_1464_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__10));
v___x_1465_ = lean_string_dec_eq(v_str_1463_, v___x_1464_);
if (v___x_1465_ == 0)
{
return v___y_1438_;
}
else
{
return v_suppressElabErrors_1439_;
}
}
default: 
{
return v___y_1438_;
}
}
}
else
{
return v___y_1438_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___boxed(lean_object* v___y_1466_, lean_object* v_suppressElabErrors_1467_, lean_object* v_x_1468_){
_start:
{
uint8_t v___y_30349__boxed_1469_; uint8_t v_suppressElabErrors_boxed_1470_; uint8_t v_res_1471_; lean_object* v_r_1472_; 
v___y_30349__boxed_1469_ = lean_unbox(v___y_1466_);
v_suppressElabErrors_boxed_1470_ = lean_unbox(v_suppressElabErrors_1467_);
v_res_1471_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0(v___y_30349__boxed_1469_, v_suppressElabErrors_boxed_1470_, v_x_1468_);
lean_dec(v_x_1468_);
v_r_1472_ = lean_box(v_res_1471_);
return v_r_1472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(lean_object* v_ref_1473_, lean_object* v_msgData_1474_, uint8_t v_severity_1475_, uint8_t v_isSilent_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_){
_start:
{
uint8_t v___y_1483_; lean_object* v___y_1484_; lean_object* v___y_1485_; lean_object* v___y_1486_; lean_object* v___y_1487_; lean_object* v___y_1488_; uint8_t v___y_1489_; lean_object* v___y_1490_; lean_object* v___y_1491_; lean_object* v___y_1519_; lean_object* v___y_1520_; lean_object* v___y_1521_; uint8_t v___y_1522_; uint8_t v___y_1523_; lean_object* v___y_1524_; uint8_t v___y_1525_; lean_object* v___y_1526_; lean_object* v___y_1544_; uint8_t v___y_1545_; lean_object* v___y_1546_; uint8_t v___y_1547_; lean_object* v___y_1548_; lean_object* v___y_1549_; uint8_t v___y_1550_; lean_object* v___y_1551_; lean_object* v___y_1555_; lean_object* v___y_1556_; uint8_t v___y_1557_; uint8_t v___y_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; uint8_t v___y_1561_; uint8_t v___x_1566_; lean_object* v___y_1568_; uint8_t v___y_1569_; lean_object* v___y_1570_; lean_object* v___y_1571_; lean_object* v___y_1572_; uint8_t v___y_1573_; uint8_t v___y_1574_; uint8_t v___y_1576_; uint8_t v___x_1591_; 
v___x_1566_ = 2;
v___x_1591_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1475_, v___x_1566_);
if (v___x_1591_ == 0)
{
v___y_1576_ = v___x_1591_;
goto v___jp_1575_;
}
else
{
uint8_t v___x_1592_; 
lean_inc_ref(v_msgData_1474_);
v___x_1592_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1474_);
v___y_1576_ = v___x_1592_;
goto v___jp_1575_;
}
v___jp_1482_:
{
lean_object* v___x_1492_; lean_object* v_currNamespace_1493_; lean_object* v_openDecls_1494_; lean_object* v_env_1495_; lean_object* v_nextMacroScope_1496_; lean_object* v_ngen_1497_; lean_object* v_auxDeclNGen_1498_; lean_object* v_traceState_1499_; lean_object* v_cache_1500_; lean_object* v_messages_1501_; lean_object* v_infoState_1502_; lean_object* v_snapshotTasks_1503_; lean_object* v___x_1505_; uint8_t v_isShared_1506_; uint8_t v_isSharedCheck_1517_; 
v___x_1492_ = lean_st_ref_take(v___y_1491_);
v_currNamespace_1493_ = lean_ctor_get(v___y_1490_, 6);
v_openDecls_1494_ = lean_ctor_get(v___y_1490_, 7);
v_env_1495_ = lean_ctor_get(v___x_1492_, 0);
v_nextMacroScope_1496_ = lean_ctor_get(v___x_1492_, 1);
v_ngen_1497_ = lean_ctor_get(v___x_1492_, 2);
v_auxDeclNGen_1498_ = lean_ctor_get(v___x_1492_, 3);
v_traceState_1499_ = lean_ctor_get(v___x_1492_, 4);
v_cache_1500_ = lean_ctor_get(v___x_1492_, 5);
v_messages_1501_ = lean_ctor_get(v___x_1492_, 6);
v_infoState_1502_ = lean_ctor_get(v___x_1492_, 7);
v_snapshotTasks_1503_ = lean_ctor_get(v___x_1492_, 8);
v_isSharedCheck_1517_ = !lean_is_exclusive(v___x_1492_);
if (v_isSharedCheck_1517_ == 0)
{
v___x_1505_ = v___x_1492_;
v_isShared_1506_ = v_isSharedCheck_1517_;
goto v_resetjp_1504_;
}
else
{
lean_inc(v_snapshotTasks_1503_);
lean_inc(v_infoState_1502_);
lean_inc(v_messages_1501_);
lean_inc(v_cache_1500_);
lean_inc(v_traceState_1499_);
lean_inc(v_auxDeclNGen_1498_);
lean_inc(v_ngen_1497_);
lean_inc(v_nextMacroScope_1496_);
lean_inc(v_env_1495_);
lean_dec(v___x_1492_);
v___x_1505_ = lean_box(0);
v_isShared_1506_ = v_isSharedCheck_1517_;
goto v_resetjp_1504_;
}
v_resetjp_1504_:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1512_; 
lean_inc(v_openDecls_1494_);
lean_inc(v_currNamespace_1493_);
v___x_1507_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1507_, 0, v_currNamespace_1493_);
lean_ctor_set(v___x_1507_, 1, v_openDecls_1494_);
v___x_1508_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1508_, 0, v___x_1507_);
lean_ctor_set(v___x_1508_, 1, v___y_1485_);
lean_inc_ref(v___y_1486_);
lean_inc_ref(v___y_1484_);
v___x_1509_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1509_, 0, v___y_1484_);
lean_ctor_set(v___x_1509_, 1, v___y_1488_);
lean_ctor_set(v___x_1509_, 2, v___y_1487_);
lean_ctor_set(v___x_1509_, 3, v___y_1486_);
lean_ctor_set(v___x_1509_, 4, v___x_1508_);
lean_ctor_set_uint8(v___x_1509_, sizeof(void*)*5, v___y_1483_);
lean_ctor_set_uint8(v___x_1509_, sizeof(void*)*5 + 1, v___y_1489_);
lean_ctor_set_uint8(v___x_1509_, sizeof(void*)*5 + 2, v_isSilent_1476_);
v___x_1510_ = l_Lean_MessageLog_add(v___x_1509_, v_messages_1501_);
if (v_isShared_1506_ == 0)
{
lean_ctor_set(v___x_1505_, 6, v___x_1510_);
v___x_1512_ = v___x_1505_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1516_; 
v_reuseFailAlloc_1516_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1516_, 0, v_env_1495_);
lean_ctor_set(v_reuseFailAlloc_1516_, 1, v_nextMacroScope_1496_);
lean_ctor_set(v_reuseFailAlloc_1516_, 2, v_ngen_1497_);
lean_ctor_set(v_reuseFailAlloc_1516_, 3, v_auxDeclNGen_1498_);
lean_ctor_set(v_reuseFailAlloc_1516_, 4, v_traceState_1499_);
lean_ctor_set(v_reuseFailAlloc_1516_, 5, v_cache_1500_);
lean_ctor_set(v_reuseFailAlloc_1516_, 6, v___x_1510_);
lean_ctor_set(v_reuseFailAlloc_1516_, 7, v_infoState_1502_);
lean_ctor_set(v_reuseFailAlloc_1516_, 8, v_snapshotTasks_1503_);
v___x_1512_ = v_reuseFailAlloc_1516_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; 
v___x_1513_ = lean_st_ref_set(v___y_1491_, v___x_1512_);
v___x_1514_ = lean_box(0);
v___x_1515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1515_, 0, v___x_1514_);
return v___x_1515_;
}
}
}
v___jp_1518_:
{
lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v_a_1529_; lean_object* v___x_1531_; uint8_t v_isShared_1532_; uint8_t v_isSharedCheck_1542_; 
v___x_1527_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1474_);
v___x_1528_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v___x_1527_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_);
v_a_1529_ = lean_ctor_get(v___x_1528_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1528_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1531_ = v___x_1528_;
v_isShared_1532_ = v_isSharedCheck_1542_;
goto v_resetjp_1530_;
}
else
{
lean_inc(v_a_1529_);
lean_dec(v___x_1528_);
v___x_1531_ = lean_box(0);
v_isShared_1532_ = v_isSharedCheck_1542_;
goto v_resetjp_1530_;
}
v_resetjp_1530_:
{
lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; 
lean_inc_ref_n(v___y_1524_, 2);
v___x_1533_ = l_Lean_FileMap_toPosition(v___y_1524_, v___y_1520_);
lean_dec(v___y_1520_);
v___x_1534_ = l_Lean_FileMap_toPosition(v___y_1524_, v___y_1526_);
lean_dec(v___y_1526_);
v___x_1535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1535_, 0, v___x_1534_);
v___x_1536_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1));
if (v___y_1523_ == 0)
{
lean_del_object(v___x_1531_);
lean_dec_ref(v___y_1519_);
v___y_1483_ = v___y_1522_;
v___y_1484_ = v___y_1521_;
v___y_1485_ = v_a_1529_;
v___y_1486_ = v___x_1536_;
v___y_1487_ = v___x_1535_;
v___y_1488_ = v___x_1533_;
v___y_1489_ = v___y_1525_;
v___y_1490_ = v___y_1479_;
v___y_1491_ = v___y_1480_;
goto v___jp_1482_;
}
else
{
uint8_t v___x_1537_; 
lean_inc(v_a_1529_);
v___x_1537_ = l_Lean_MessageData_hasTag(v___y_1519_, v_a_1529_);
if (v___x_1537_ == 0)
{
lean_object* v___x_1538_; lean_object* v___x_1540_; 
lean_dec_ref_known(v___x_1535_, 1);
lean_dec_ref(v___x_1533_);
lean_dec(v_a_1529_);
v___x_1538_ = lean_box(0);
if (v_isShared_1532_ == 0)
{
lean_ctor_set(v___x_1531_, 0, v___x_1538_);
v___x_1540_ = v___x_1531_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v___x_1538_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
else
{
lean_del_object(v___x_1531_);
v___y_1483_ = v___y_1522_;
v___y_1484_ = v___y_1521_;
v___y_1485_ = v_a_1529_;
v___y_1486_ = v___x_1536_;
v___y_1487_ = v___x_1535_;
v___y_1488_ = v___x_1533_;
v___y_1489_ = v___y_1525_;
v___y_1490_ = v___y_1479_;
v___y_1491_ = v___y_1480_;
goto v___jp_1482_;
}
}
}
}
v___jp_1543_:
{
lean_object* v___x_1552_; 
v___x_1552_ = l_Lean_Syntax_getTailPos_x3f(v___y_1548_, v___y_1545_);
lean_dec(v___y_1548_);
if (lean_obj_tag(v___x_1552_) == 0)
{
lean_inc(v___y_1551_);
v___y_1519_ = v___y_1544_;
v___y_1520_ = v___y_1551_;
v___y_1521_ = v___y_1546_;
v___y_1522_ = v___y_1545_;
v___y_1523_ = v___y_1547_;
v___y_1524_ = v___y_1549_;
v___y_1525_ = v___y_1550_;
v___y_1526_ = v___y_1551_;
goto v___jp_1518_;
}
else
{
lean_object* v_val_1553_; 
v_val_1553_ = lean_ctor_get(v___x_1552_, 0);
lean_inc(v_val_1553_);
lean_dec_ref_known(v___x_1552_, 1);
v___y_1519_ = v___y_1544_;
v___y_1520_ = v___y_1551_;
v___y_1521_ = v___y_1546_;
v___y_1522_ = v___y_1545_;
v___y_1523_ = v___y_1547_;
v___y_1524_ = v___y_1549_;
v___y_1525_ = v___y_1550_;
v___y_1526_ = v_val_1553_;
goto v___jp_1518_;
}
}
v___jp_1554_:
{
lean_object* v_ref_1562_; lean_object* v___x_1563_; 
v_ref_1562_ = l_Lean_replaceRef(v_ref_1473_, v___y_1560_);
v___x_1563_ = l_Lean_Syntax_getPos_x3f(v_ref_1562_, v___y_1557_);
if (lean_obj_tag(v___x_1563_) == 0)
{
lean_object* v___x_1564_; 
v___x_1564_ = lean_unsigned_to_nat(0u);
v___y_1544_ = v___y_1555_;
v___y_1545_ = v___y_1557_;
v___y_1546_ = v___y_1556_;
v___y_1547_ = v___y_1558_;
v___y_1548_ = v_ref_1562_;
v___y_1549_ = v___y_1559_;
v___y_1550_ = v___y_1561_;
v___y_1551_ = v___x_1564_;
goto v___jp_1543_;
}
else
{
lean_object* v_val_1565_; 
v_val_1565_ = lean_ctor_get(v___x_1563_, 0);
lean_inc(v_val_1565_);
lean_dec_ref_known(v___x_1563_, 1);
v___y_1544_ = v___y_1555_;
v___y_1545_ = v___y_1557_;
v___y_1546_ = v___y_1556_;
v___y_1547_ = v___y_1558_;
v___y_1548_ = v_ref_1562_;
v___y_1549_ = v___y_1559_;
v___y_1550_ = v___y_1561_;
v___y_1551_ = v_val_1565_;
goto v___jp_1543_;
}
}
v___jp_1567_:
{
if (v___y_1574_ == 0)
{
v___y_1555_ = v___y_1570_;
v___y_1556_ = v___y_1568_;
v___y_1557_ = v___y_1573_;
v___y_1558_ = v___y_1569_;
v___y_1559_ = v___y_1571_;
v___y_1560_ = v___y_1572_;
v___y_1561_ = v_severity_1475_;
goto v___jp_1554_;
}
else
{
v___y_1555_ = v___y_1570_;
v___y_1556_ = v___y_1568_;
v___y_1557_ = v___y_1573_;
v___y_1558_ = v___y_1569_;
v___y_1559_ = v___y_1571_;
v___y_1560_ = v___y_1572_;
v___y_1561_ = v___x_1566_;
goto v___jp_1554_;
}
}
v___jp_1575_:
{
if (v___y_1576_ == 0)
{
lean_object* v_fileName_1577_; lean_object* v_fileMap_1578_; lean_object* v_options_1579_; lean_object* v_ref_1580_; uint8_t v_suppressElabErrors_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___f_1584_; uint8_t v___x_1585_; uint8_t v___x_1586_; 
v_fileName_1577_ = lean_ctor_get(v___y_1479_, 0);
v_fileMap_1578_ = lean_ctor_get(v___y_1479_, 1);
v_options_1579_ = lean_ctor_get(v___y_1479_, 2);
v_ref_1580_ = lean_ctor_get(v___y_1479_, 5);
v_suppressElabErrors_1581_ = lean_ctor_get_uint8(v___y_1479_, sizeof(void*)*14 + 1);
v___x_1582_ = lean_box(v___y_1576_);
v___x_1583_ = lean_box(v_suppressElabErrors_1581_);
v___f_1584_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1584_, 0, v___x_1582_);
lean_closure_set(v___f_1584_, 1, v___x_1583_);
v___x_1585_ = 1;
v___x_1586_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1475_, v___x_1585_);
if (v___x_1586_ == 0)
{
v___y_1568_ = v_fileName_1577_;
v___y_1569_ = v_suppressElabErrors_1581_;
v___y_1570_ = v___f_1584_;
v___y_1571_ = v_fileMap_1578_;
v___y_1572_ = v_ref_1580_;
v___y_1573_ = v___y_1576_;
v___y_1574_ = v___x_1586_;
goto v___jp_1567_;
}
else
{
lean_object* v___x_1587_; uint8_t v___x_1588_; 
v___x_1587_ = l_Lean_warningAsError;
v___x_1588_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v_options_1579_, v___x_1587_);
v___y_1568_ = v_fileName_1577_;
v___y_1569_ = v_suppressElabErrors_1581_;
v___y_1570_ = v___f_1584_;
v___y_1571_ = v_fileMap_1578_;
v___y_1572_ = v_ref_1580_;
v___y_1573_ = v___y_1576_;
v___y_1574_ = v___x_1588_;
goto v___jp_1567_;
}
}
else
{
lean_object* v___x_1589_; lean_object* v___x_1590_; 
lean_dec_ref(v_msgData_1474_);
v___x_1589_ = lean_box(0);
v___x_1590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1590_, 0, v___x_1589_);
return v___x_1590_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___boxed(lean_object* v_ref_1593_, lean_object* v_msgData_1594_, lean_object* v_severity_1595_, lean_object* v_isSilent_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
uint8_t v_severity_boxed_1602_; uint8_t v_isSilent_boxed_1603_; lean_object* v_res_1604_; 
v_severity_boxed_1602_ = lean_unbox(v_severity_1595_);
v_isSilent_boxed_1603_ = lean_unbox(v_isSilent_1596_);
v_res_1604_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(v_ref_1593_, v_msgData_1594_, v_severity_boxed_1602_, v_isSilent_boxed_1603_, v___y_1597_, v___y_1598_, v___y_1599_, v___y_1600_);
lean_dec(v___y_1600_);
lean_dec_ref(v___y_1599_);
lean_dec(v___y_1598_);
lean_dec_ref(v___y_1597_);
lean_dec(v_ref_1593_);
return v_res_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8(lean_object* v_ref_1605_, lean_object* v_msgData_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
uint8_t v___x_1614_; uint8_t v___x_1615_; lean_object* v___x_1616_; 
v___x_1614_ = 2;
v___x_1615_ = 0;
v___x_1616_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(v_ref_1605_, v_msgData_1606_, v___x_1614_, v___x_1615_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_);
return v___x_1616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8___boxed(lean_object* v_ref_1617_, lean_object* v_msgData_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v_res_1626_; 
v_res_1626_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8(v_ref_1617_, v_msgData_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_);
lean_dec(v___y_1624_);
lean_dec_ref(v___y_1623_);
lean_dec(v___y_1622_);
lean_dec_ref(v___y_1621_);
lean_dec(v___y_1620_);
lean_dec_ref(v___y_1619_);
lean_dec(v_ref_1617_);
return v_res_1626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24(lean_object* v_msgData_1627_, uint8_t v_severity_1628_, uint8_t v_isSilent_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
lean_object* v_ref_1637_; lean_object* v___x_1638_; 
v_ref_1637_ = lean_ctor_get(v___y_1634_, 5);
v___x_1638_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(v_ref_1637_, v_msgData_1627_, v_severity_1628_, v_isSilent_1629_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24___boxed(lean_object* v_msgData_1639_, lean_object* v_severity_1640_, lean_object* v_isSilent_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_){
_start:
{
uint8_t v_severity_boxed_1649_; uint8_t v_isSilent_boxed_1650_; lean_object* v_res_1651_; 
v_severity_boxed_1649_ = lean_unbox(v_severity_1640_);
v_isSilent_boxed_1650_ = lean_unbox(v_isSilent_1641_);
v_res_1651_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24(v_msgData_1639_, v_severity_boxed_1649_, v_isSilent_boxed_1650_, v___y_1642_, v___y_1643_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
lean_dec(v___y_1647_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v___y_1643_);
lean_dec_ref(v___y_1642_);
return v_res_1651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9(lean_object* v_msgData_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_){
_start:
{
uint8_t v___x_1660_; uint8_t v___x_1661_; lean_object* v___x_1662_; 
v___x_1660_ = 2;
v___x_1661_ = 0;
v___x_1662_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9_spec__24(v_msgData_1652_, v___x_1660_, v___x_1661_, v___y_1653_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_);
return v___x_1662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9___boxed(lean_object* v_msgData_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_){
_start:
{
lean_object* v_res_1671_; 
v_res_1671_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9(v_msgData_1663_, v___y_1664_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_, v___y_1669_);
lean_dec(v___y_1669_);
lean_dec_ref(v___y_1668_);
lean_dec(v___y_1667_);
lean_dec_ref(v___y_1666_);
lean_dec(v___y_1665_);
lean_dec_ref(v___y_1664_);
return v_res_1671_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1673_ = ((lean_object*)(lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__0));
v___x_1674_ = l_Lean_stringToMessageData(v___x_1673_);
return v___x_1674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2(lean_object* v_ex_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_){
_start:
{
if (lean_obj_tag(v_ex_1675_) == 0)
{
lean_object* v_ref_1683_; lean_object* v_msg_1684_; lean_object* v___x_1685_; 
v_ref_1683_ = lean_ctor_get(v_ex_1675_, 0);
lean_inc(v_ref_1683_);
v_msg_1684_ = lean_ctor_get(v_ex_1675_, 1);
lean_inc_ref(v_msg_1684_);
lean_dec_ref_known(v_ex_1675_, 2);
v___x_1685_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8(v_ref_1683_, v_msg_1684_, v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
lean_dec(v_ref_1683_);
return v___x_1685_;
}
else
{
lean_object* v_id_1686_; uint8_t v___y_1688_; uint8_t v___x_1710_; 
v_id_1686_ = lean_ctor_get(v_ex_1675_, 0);
lean_inc(v_id_1686_);
v___x_1710_ = l_Lean_Elab_isAbortExceptionId(v_id_1686_);
if (v___x_1710_ == 0)
{
uint8_t v___x_1711_; 
v___x_1711_ = l_Lean_Exception_isInterrupt(v_ex_1675_);
lean_dec_ref_known(v_ex_1675_, 2);
v___y_1688_ = v___x_1711_;
goto v___jp_1687_;
}
else
{
lean_dec_ref_known(v_ex_1675_, 2);
v___y_1688_ = v___x_1710_;
goto v___jp_1687_;
}
v___jp_1687_:
{
if (v___y_1688_ == 0)
{
lean_object* v___x_1689_; 
v___x_1689_ = l_Lean_InternalExceptionId_getName(v_id_1686_);
lean_dec(v_id_1686_);
if (lean_obj_tag(v___x_1689_) == 0)
{
lean_object* v_a_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; 
v_a_1690_ = lean_ctor_get(v___x_1689_, 0);
lean_inc(v_a_1690_);
lean_dec_ref_known(v___x_1689_, 1);
v___x_1691_ = lean_obj_once(&lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___closed__1);
v___x_1692_ = l_Lean_MessageData_ofName(v_a_1690_);
v___x_1693_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1693_, 0, v___x_1691_);
lean_ctor_set(v___x_1693_, 1, v___x_1692_);
v___x_1694_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__9(v___x_1693_, v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
return v___x_1694_;
}
else
{
lean_object* v_a_1695_; lean_object* v___x_1697_; uint8_t v_isShared_1698_; uint8_t v_isSharedCheck_1707_; 
v_a_1695_ = lean_ctor_get(v___x_1689_, 0);
v_isSharedCheck_1707_ = !lean_is_exclusive(v___x_1689_);
if (v_isSharedCheck_1707_ == 0)
{
v___x_1697_ = v___x_1689_;
v_isShared_1698_ = v_isSharedCheck_1707_;
goto v_resetjp_1696_;
}
else
{
lean_inc(v_a_1695_);
lean_dec(v___x_1689_);
v___x_1697_ = lean_box(0);
v_isShared_1698_ = v_isSharedCheck_1707_;
goto v_resetjp_1696_;
}
v_resetjp_1696_:
{
lean_object* v_ref_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1705_; 
v_ref_1699_ = lean_ctor_get(v___y_1680_, 5);
v___x_1700_ = lean_io_error_to_string(v_a_1695_);
v___x_1701_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1701_, 0, v___x_1700_);
v___x_1702_ = l_Lean_MessageData_ofFormat(v___x_1701_);
lean_inc(v_ref_1699_);
v___x_1703_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1703_, 0, v_ref_1699_);
lean_ctor_set(v___x_1703_, 1, v___x_1702_);
if (v_isShared_1698_ == 0)
{
lean_ctor_set(v___x_1697_, 0, v___x_1703_);
v___x_1705_ = v___x_1697_;
goto v_reusejp_1704_;
}
else
{
lean_object* v_reuseFailAlloc_1706_; 
v_reuseFailAlloc_1706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1706_, 0, v___x_1703_);
v___x_1705_ = v_reuseFailAlloc_1706_;
goto v_reusejp_1704_;
}
v_reusejp_1704_:
{
return v___x_1705_;
}
}
}
}
else
{
lean_object* v___x_1708_; lean_object* v___x_1709_; 
lean_dec(v_id_1686_);
v___x_1708_ = lean_box(0);
v___x_1709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1709_, 0, v___x_1708_);
return v___x_1709_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2___boxed(lean_object* v_ex_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_){
_start:
{
lean_object* v_res_1720_; 
v_res_1720_ = lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2(v_ex_1712_, v___y_1713_, v___y_1714_, v___y_1715_, v___y_1716_, v___y_1717_, v___y_1718_);
lean_dec(v___y_1718_);
lean_dec_ref(v___y_1717_);
lean_dec(v___y_1716_);
lean_dec_ref(v___y_1715_);
lean_dec(v___y_1714_);
lean_dec_ref(v___y_1713_);
return v_res_1720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3(lean_object* v_as_1721_, size_t v_sz_1722_, size_t v_i_1723_, lean_object* v_b_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_){
_start:
{
lean_object* v_snd_1733_; uint8_t v___x_1737_; 
v___x_1737_ = lean_usize_dec_lt(v_i_1723_, v_sz_1722_);
if (v___x_1737_ == 0)
{
lean_object* v___x_1738_; 
v___x_1738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1738_, 0, v_b_1724_);
return v___x_1738_;
}
else
{
lean_object* v_fileName_1739_; lean_object* v_fileMap_1740_; lean_object* v_options_1741_; lean_object* v_currRecDepth_1742_; lean_object* v_maxRecDepth_1743_; lean_object* v_ref_1744_; lean_object* v_currNamespace_1745_; lean_object* v_openDecls_1746_; lean_object* v_initHeartbeats_1747_; lean_object* v_maxHeartbeats_1748_; lean_object* v_quotContext_1749_; lean_object* v_currMacroScope_1750_; uint8_t v_diag_1751_; lean_object* v_cancelTk_x3f_1752_; uint8_t v_suppressElabErrors_1753_; lean_object* v_inheritedTraceOptions_1754_; lean_object* v_a_1755_; lean_object* v_ref_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; 
v_fileName_1739_ = lean_ctor_get(v___y_1729_, 0);
v_fileMap_1740_ = lean_ctor_get(v___y_1729_, 1);
v_options_1741_ = lean_ctor_get(v___y_1729_, 2);
v_currRecDepth_1742_ = lean_ctor_get(v___y_1729_, 3);
v_maxRecDepth_1743_ = lean_ctor_get(v___y_1729_, 4);
v_ref_1744_ = lean_ctor_get(v___y_1729_, 5);
v_currNamespace_1745_ = lean_ctor_get(v___y_1729_, 6);
v_openDecls_1746_ = lean_ctor_get(v___y_1729_, 7);
v_initHeartbeats_1747_ = lean_ctor_get(v___y_1729_, 8);
v_maxHeartbeats_1748_ = lean_ctor_get(v___y_1729_, 9);
v_quotContext_1749_ = lean_ctor_get(v___y_1729_, 10);
v_currMacroScope_1750_ = lean_ctor_get(v___y_1729_, 11);
v_diag_1751_ = lean_ctor_get_uint8(v___y_1729_, sizeof(void*)*14);
v_cancelTk_x3f_1752_ = lean_ctor_get(v___y_1729_, 12);
v_suppressElabErrors_1753_ = lean_ctor_get_uint8(v___y_1729_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1754_ = lean_ctor_get(v___y_1729_, 13);
v_a_1755_ = lean_array_uget_borrowed(v_as_1721_, v_i_1723_);
v_ref_1756_ = l_Lean_replaceRef(v_a_1755_, v_ref_1744_);
lean_inc_ref(v_inheritedTraceOptions_1754_);
lean_inc(v_cancelTk_x3f_1752_);
lean_inc(v_currMacroScope_1750_);
lean_inc(v_quotContext_1749_);
lean_inc(v_maxHeartbeats_1748_);
lean_inc(v_initHeartbeats_1747_);
lean_inc(v_openDecls_1746_);
lean_inc(v_currNamespace_1745_);
lean_inc(v_maxRecDepth_1743_);
lean_inc(v_currRecDepth_1742_);
lean_inc_ref(v_options_1741_);
lean_inc_ref(v_fileMap_1740_);
lean_inc_ref(v_fileName_1739_);
v___x_1757_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1757_, 0, v_fileName_1739_);
lean_ctor_set(v___x_1757_, 1, v_fileMap_1740_);
lean_ctor_set(v___x_1757_, 2, v_options_1741_);
lean_ctor_set(v___x_1757_, 3, v_currRecDepth_1742_);
lean_ctor_set(v___x_1757_, 4, v_maxRecDepth_1743_);
lean_ctor_set(v___x_1757_, 5, v_ref_1756_);
lean_ctor_set(v___x_1757_, 6, v_currNamespace_1745_);
lean_ctor_set(v___x_1757_, 7, v_openDecls_1746_);
lean_ctor_set(v___x_1757_, 8, v_initHeartbeats_1747_);
lean_ctor_set(v___x_1757_, 9, v_maxHeartbeats_1748_);
lean_ctor_set(v___x_1757_, 10, v_quotContext_1749_);
lean_ctor_set(v___x_1757_, 11, v_currMacroScope_1750_);
lean_ctor_set(v___x_1757_, 12, v_cancelTk_x3f_1752_);
lean_ctor_set(v___x_1757_, 13, v_inheritedTraceOptions_1754_);
lean_ctor_set_uint8(v___x_1757_, sizeof(void*)*14, v_diag_1751_);
lean_ctor_set_uint8(v___x_1757_, sizeof(void*)*14 + 1, v_suppressElabErrors_1753_);
lean_inc(v_a_1755_);
v___x_1758_ = lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1(v_a_1755_, v___y_1725_, v___y_1726_, v___y_1727_, v___y_1728_, v___x_1757_, v___y_1730_);
lean_dec_ref_known(v___x_1757_, 14);
if (lean_obj_tag(v___x_1758_) == 0)
{
lean_object* v_a_1759_; lean_object* v___x_1760_; 
v_a_1759_ = lean_ctor_get(v___x_1758_, 0);
lean_inc(v_a_1759_);
lean_dec_ref_known(v___x_1758_, 1);
v___x_1760_ = lean_array_push(v_b_1724_, v_a_1759_);
v_snd_1733_ = v___x_1760_;
goto v___jp_1732_;
}
else
{
lean_object* v_a_1761_; lean_object* v___x_1763_; uint8_t v_isShared_1764_; uint8_t v_isSharedCheck_1781_; 
v_a_1761_ = lean_ctor_get(v___x_1758_, 0);
v_isSharedCheck_1781_ = !lean_is_exclusive(v___x_1758_);
if (v_isSharedCheck_1781_ == 0)
{
v___x_1763_ = v___x_1758_;
v_isShared_1764_ = v_isSharedCheck_1781_;
goto v_resetjp_1762_;
}
else
{
lean_inc(v_a_1761_);
lean_dec(v___x_1758_);
v___x_1763_ = lean_box(0);
v_isShared_1764_ = v_isSharedCheck_1781_;
goto v_resetjp_1762_;
}
v_resetjp_1762_:
{
uint8_t v___y_1766_; uint8_t v___x_1779_; 
v___x_1779_ = l_Lean_Exception_isInterrupt(v_a_1761_);
if (v___x_1779_ == 0)
{
uint8_t v___x_1780_; 
lean_inc(v_a_1761_);
v___x_1780_ = l_Lean_Exception_isRuntime(v_a_1761_);
v___y_1766_ = v___x_1780_;
goto v___jp_1765_;
}
else
{
v___y_1766_ = v___x_1779_;
goto v___jp_1765_;
}
v___jp_1765_:
{
if (v___y_1766_ == 0)
{
lean_object* v___x_1767_; 
lean_del_object(v___x_1763_);
v___x_1767_ = lp_mathlib_Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2(v_a_1761_, v___y_1725_, v___y_1726_, v___y_1727_, v___y_1728_, v___y_1729_, v___y_1730_);
if (lean_obj_tag(v___x_1767_) == 0)
{
lean_dec_ref_known(v___x_1767_, 1);
v_snd_1733_ = v_b_1724_;
goto v___jp_1732_;
}
else
{
lean_object* v_a_1768_; lean_object* v___x_1770_; uint8_t v_isShared_1771_; uint8_t v_isSharedCheck_1775_; 
lean_dec_ref(v_b_1724_);
v_a_1768_ = lean_ctor_get(v___x_1767_, 0);
v_isSharedCheck_1775_ = !lean_is_exclusive(v___x_1767_);
if (v_isSharedCheck_1775_ == 0)
{
v___x_1770_ = v___x_1767_;
v_isShared_1771_ = v_isSharedCheck_1775_;
goto v_resetjp_1769_;
}
else
{
lean_inc(v_a_1768_);
lean_dec(v___x_1767_);
v___x_1770_ = lean_box(0);
v_isShared_1771_ = v_isSharedCheck_1775_;
goto v_resetjp_1769_;
}
v_resetjp_1769_:
{
lean_object* v___x_1773_; 
if (v_isShared_1771_ == 0)
{
v___x_1773_ = v___x_1770_;
goto v_reusejp_1772_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v_a_1768_);
v___x_1773_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1772_;
}
v_reusejp_1772_:
{
return v___x_1773_;
}
}
}
}
else
{
lean_object* v___x_1777_; 
lean_dec_ref(v_b_1724_);
if (v_isShared_1764_ == 0)
{
v___x_1777_ = v___x_1763_;
goto v_reusejp_1776_;
}
else
{
lean_object* v_reuseFailAlloc_1778_; 
v_reuseFailAlloc_1778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1778_, 0, v_a_1761_);
v___x_1777_ = v_reuseFailAlloc_1778_;
goto v_reusejp_1776_;
}
v_reusejp_1776_:
{
return v___x_1777_;
}
}
}
}
}
}
v___jp_1732_:
{
size_t v___x_1734_; size_t v___x_1735_; 
v___x_1734_ = ((size_t)1ULL);
v___x_1735_ = lean_usize_add(v_i_1723_, v___x_1734_);
v_i_1723_ = v___x_1735_;
v_b_1724_ = v_snd_1733_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3___boxed(lean_object* v_as_1782_, lean_object* v_sz_1783_, lean_object* v_i_1784_, lean_object* v_b_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_){
_start:
{
size_t v_sz_boxed_1793_; size_t v_i_boxed_1794_; lean_object* v_res_1795_; 
v_sz_boxed_1793_ = lean_unbox_usize(v_sz_1783_);
lean_dec(v_sz_1783_);
v_i_boxed_1794_ = lean_unbox_usize(v_i_1784_);
lean_dec(v_i_1784_);
v_res_1795_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3(v_as_1782_, v_sz_boxed_1793_, v_i_boxed_1794_, v_b_1785_, v___y_1786_, v___y_1787_, v___y_1788_, v___y_1789_, v___y_1790_, v___y_1791_);
lean_dec(v___y_1791_);
lean_dec_ref(v___y_1790_);
lean_dec(v___y_1789_);
lean_dec_ref(v___y_1788_);
lean_dec(v___y_1787_);
lean_dec_ref(v___y_1786_);
lean_dec_ref(v_as_1782_);
return v_res_1795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1(lean_object* v_attrInstances_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_){
_start:
{
lean_object* v_attrs_1806_; size_t v_sz_1807_; size_t v___x_1808_; lean_object* v___x_1809_; 
v_attrs_1806_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0));
v_sz_1807_ = lean_array_size(v_attrInstances_1798_);
v___x_1808_ = ((size_t)0ULL);
v___x_1809_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__3(v_attrInstances_1798_, v_sz_1807_, v___x_1808_, v_attrs_1806_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_);
return v___x_1809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___boxed(lean_object* v_attrInstances_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_){
_start:
{
lean_object* v_res_1818_; 
v_res_1818_ = lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1(v_attrInstances_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_);
lean_dec(v___y_1816_);
lean_dec_ref(v___y_1815_);
lean_dec(v___y_1814_);
lean_dec_ref(v___y_1813_);
lean_dec(v___y_1812_);
lean_dec_ref(v___y_1811_);
lean_dec_ref(v_attrInstances_1810_);
return v_res_1818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0(size_t v_sz_1819_, size_t v_i_1820_, lean_object* v_bs_1821_){
_start:
{
uint8_t v___x_1822_; 
v___x_1822_ = lean_usize_dec_lt(v_i_1820_, v_sz_1819_);
if (v___x_1822_ == 0)
{
lean_object* v___x_1823_; 
v___x_1823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1823_, 0, v_bs_1821_);
return v___x_1823_;
}
else
{
lean_object* v_v_1824_; lean_object* v___x_1825_; lean_object* v_bs_x27_1826_; size_t v___x_1827_; size_t v___x_1828_; lean_object* v___x_1829_; 
v_v_1824_ = lean_array_uget(v_bs_1821_, v_i_1820_);
v___x_1825_ = lean_unsigned_to_nat(0u);
v_bs_x27_1826_ = lean_array_uset(v_bs_1821_, v_i_1820_, v___x_1825_);
v___x_1827_ = ((size_t)1ULL);
v___x_1828_ = lean_usize_add(v_i_1820_, v___x_1827_);
v___x_1829_ = lean_array_uset(v_bs_x27_1826_, v_i_1820_, v_v_1824_);
v_i_1820_ = v___x_1828_;
v_bs_1821_ = v___x_1829_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0___boxed(lean_object* v_sz_1831_, lean_object* v_i_1832_, lean_object* v_bs_1833_){
_start:
{
size_t v_sz_boxed_1834_; size_t v_i_boxed_1835_; lean_object* v_res_1836_; 
v_sz_boxed_1834_ = lean_unbox_usize(v_sz_1831_);
lean_dec(v_sz_1831_);
v_i_boxed_1835_ = lean_unbox_usize(v_i_1832_);
lean_dec(v_i_1832_);
v_res_1836_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0(v_sz_boxed_1834_, v_i_boxed_1835_, v_bs_1833_);
return v_res_1836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabOptAttrArg(lean_object* v_x_1839_, lean_object* v_a_1840_, lean_object* v_a_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_, lean_object* v_a_1844_, lean_object* v_a_1845_){
_start:
{
lean_object* v___x_1847_; uint8_t v___x_1848_; 
v___x_1847_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_optAttrArg___closed__3));
lean_inc(v_x_1839_);
v___x_1848_ = l_Lean_Syntax_isOfKind(v_x_1839_, v___x_1847_);
if (v___x_1848_ == 0)
{
lean_object* v___x_1849_; lean_object* v___x_1850_; 
lean_dec(v_x_1839_);
v___x_1849_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0));
v___x_1850_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1850_, 0, v___x_1849_);
return v___x_1850_;
}
else
{
lean_object* v___x_1851_; lean_object* v___y_1853_; lean_object* v___x_1861_; lean_object* v___x_1862_; uint8_t v___x_1863_; 
v___x_1851_ = lean_unsigned_to_nat(0u);
v___x_1861_ = l_Lean_Syntax_getArg(v_x_1839_, v___x_1851_);
lean_dec(v_x_1839_);
v___x_1862_ = lean_unsigned_to_nat(5u);
lean_inc(v___x_1861_);
v___x_1863_ = l_Lean_Syntax_matchesNull(v___x_1861_, v___x_1862_);
if (v___x_1863_ == 0)
{
lean_object* v___x_1864_; lean_object* v___x_1865_; 
lean_dec(v___x_1861_);
v___x_1864_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0));
v___x_1865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1865_, 0, v___x_1864_);
return v___x_1865_;
}
else
{
lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; uint8_t v___x_1871_; 
v___x_1866_ = lean_unsigned_to_nat(3u);
v___x_1867_ = l_Lean_Syntax_getArg(v___x_1861_, v___x_1866_);
lean_dec(v___x_1861_);
v___x_1868_ = l_Lean_Syntax_getArgs(v___x_1867_);
lean_dec(v___x_1867_);
v___x_1869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabOptAttrArg___closed__0));
v___x_1870_ = lean_array_get_size(v___x_1868_);
v___x_1871_ = lean_nat_dec_lt(v___x_1851_, v___x_1870_);
if (v___x_1871_ == 0)
{
lean_dec_ref(v___x_1868_);
v___y_1853_ = v___x_1869_;
goto v___jp_1852_;
}
else
{
lean_object* v___x_1872_; lean_object* v___x_1873_; uint8_t v___x_1874_; 
v___x_1872_ = lean_box(v___x_1863_);
v___x_1873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1873_, 0, v___x_1872_);
lean_ctor_set(v___x_1873_, 1, v___x_1869_);
v___x_1874_ = lean_nat_dec_le(v___x_1870_, v___x_1870_);
if (v___x_1874_ == 0)
{
if (v___x_1871_ == 0)
{
lean_dec_ref_known(v___x_1873_, 2);
lean_dec_ref(v___x_1868_);
v___y_1853_ = v___x_1869_;
goto v___jp_1852_;
}
else
{
size_t v___x_1875_; size_t v___x_1876_; lean_object* v___x_1877_; lean_object* v_snd_1878_; 
v___x_1875_ = ((size_t)0ULL);
v___x_1876_ = lean_usize_of_nat(v___x_1870_);
v___x_1877_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2(v___x_1863_, v___x_1868_, v___x_1875_, v___x_1876_, v___x_1873_);
lean_dec_ref(v___x_1868_);
v_snd_1878_ = lean_ctor_get(v___x_1877_, 1);
lean_inc(v_snd_1878_);
lean_dec_ref(v___x_1877_);
v___y_1853_ = v_snd_1878_;
goto v___jp_1852_;
}
}
else
{
size_t v___x_1879_; size_t v___x_1880_; lean_object* v___x_1881_; lean_object* v_snd_1882_; 
v___x_1879_ = ((size_t)0ULL);
v___x_1880_ = lean_usize_of_nat(v___x_1870_);
v___x_1881_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_elabOptAttrArg_spec__2(v___x_1863_, v___x_1868_, v___x_1879_, v___x_1880_, v___x_1873_);
lean_dec_ref(v___x_1868_);
v_snd_1882_ = lean_ctor_get(v___x_1881_, 1);
lean_inc(v_snd_1882_);
lean_dec_ref(v___x_1881_);
v___y_1853_ = v_snd_1882_;
goto v___jp_1852_;
}
}
}
v___jp_1852_:
{
size_t v_sz_1854_; size_t v___x_1855_; lean_object* v___x_1856_; 
v_sz_1854_ = lean_array_size(v___y_1853_);
v___x_1855_ = ((size_t)0ULL);
v___x_1856_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_elabOptAttrArg_spec__0(v_sz_1854_, v___x_1855_, v___y_1853_);
if (lean_obj_tag(v___x_1856_) == 0)
{
lean_object* v___x_1857_; lean_object* v___x_1858_; 
v___x_1857_ = ((lean_object*)(lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1___closed__0));
v___x_1858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1858_, 0, v___x_1857_);
return v___x_1858_;
}
else
{
lean_object* v_val_1859_; lean_object* v___x_1860_; 
v_val_1859_ = lean_ctor_get(v___x_1856_, 0);
lean_inc(v_val_1859_);
lean_dec_ref_known(v___x_1856_, 1);
v___x_1860_ = lp_mathlib_Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1(v_val_1859_, v_a_1840_, v_a_1841_, v_a_1842_, v_a_1843_, v_a_1844_, v_a_1845_);
lean_dec(v_val_1859_);
return v___x_1860_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabOptAttrArg___boxed(lean_object* v_x_1883_, lean_object* v_a_1884_, lean_object* v_a_1885_, lean_object* v_a_1886_, lean_object* v_a_1887_, lean_object* v_a_1888_, lean_object* v_a_1889_, lean_object* v_a_1890_){
_start:
{
lean_object* v_res_1891_; 
v_res_1891_ = lp_mathlib_Mathlib_Tactic_elabOptAttrArg(v_x_1883_, v_a_1884_, v_a_1885_, v_a_1886_, v_a_1887_, v_a_1888_, v_a_1889_);
lean_dec(v_a_1889_);
lean_dec_ref(v_a_1888_);
lean_dec(v_a_1887_);
lean_dec_ref(v_a_1886_);
lean_dec(v_a_1885_);
lean_dec_ref(v_a_1884_);
return v_res_1891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5(lean_object* v_00_u03b1_1892_, lean_object* v_x_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_){
_start:
{
lean_object* v___x_1896_; 
v___x_1896_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___redArg(v_x_1893_, v___y_1895_);
return v___x_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b1_1897_, lean_object* v_x_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_){
_start:
{
lean_object* v_res_1901_; 
v_res_1901_ = lp_mathlib_liftExcept___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__5(v_00_u03b1_1897_, v_x_1898_, v___y_1899_, v___y_1900_);
lean_dec_ref(v___y_1899_);
lean_dec_ref(v_x_1898_);
return v_res_1901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8(lean_object* v_00_u03b1_1902_, lean_object* v_ref_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
lean_object* v___x_1911_; 
v___x_1911_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___redArg(v_ref_1903_);
return v___x_1911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8___boxed(lean_object* v_00_u03b1_1912_, lean_object* v_ref_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_){
_start:
{
lean_object* v_res_1921_; 
v_res_1921_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__8(v_00_u03b1_1912_, v_ref_1913_, v___y_1914_, v___y_1915_, v___y_1916_, v___y_1917_, v___y_1918_, v___y_1919_);
lean_dec(v___y_1919_);
lean_dec_ref(v___y_1918_);
lean_dec(v___y_1917_);
lean_dec_ref(v___y_1916_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
return v_res_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9(lean_object* v_00_u03b1_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_){
_start:
{
lean_object* v___x_1930_; 
v___x_1930_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___redArg();
return v___x_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9___boxed(lean_object* v_00_u03b1_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_){
_start:
{
lean_object* v_res_1939_; 
v_res_1939_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__9(v_00_u03b1_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_, v___y_1937_);
lean_dec(v___y_1937_);
lean_dec_ref(v___y_1936_);
lean_dec(v___y_1935_);
lean_dec_ref(v___y_1934_);
lean_dec(v___y_1933_);
lean_dec_ref(v___y_1932_);
return v_res_1939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2(lean_object* v_00_u03b1_1940_, lean_object* v_x_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_){
_start:
{
lean_object* v___x_1949_; 
v___x_1949_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___redArg(v_x_1941_, v___y_1942_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_);
return v___x_1949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b1_1950_, lean_object* v_x_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_){
_start:
{
lean_object* v_res_1959_; 
v_res_1959_ = lp_mathlib_Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2(v_00_u03b1_1950_, v_x_1951_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_);
lean_dec(v___y_1957_);
lean_dec_ref(v___y_1956_);
lean_dec(v___y_1955_);
lean_dec_ref(v___y_1954_);
lean_dec(v___y_1953_);
lean_dec_ref(v___y_1952_);
return v_res_1959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4(lean_object* v_00_u03b1_1960_, lean_object* v_msg_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_){
_start:
{
lean_object* v___x_1969_; 
v___x_1969_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___redArg(v_msg_1961_, v___y_1962_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_);
return v___x_1969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1970_, lean_object* v_msg_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v_res_1979_; 
v_res_1979_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4(v_00_u03b1_1970_, v_msg_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_);
lean_dec(v___y_1977_);
lean_dec_ref(v___y_1976_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5(lean_object* v_00_u03b1_1980_, lean_object* v_ref_1981_, lean_object* v_msg_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_){
_start:
{
lean_object* v___x_1990_; 
v___x_1990_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(v_ref_1981_, v_msg_1982_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___boxed(lean_object* v_00_u03b1_1991_, lean_object* v_ref_1992_, lean_object* v_msg_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_){
_start:
{
lean_object* v_res_2001_; 
v_res_2001_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5(v_00_u03b1_1991_, v_ref_1992_, v_msg_1993_, v___y_1994_, v___y_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_);
lean_dec(v___y_1999_);
lean_dec_ref(v___y_1998_);
lean_dec(v___y_1997_);
lean_dec_ref(v___y_1996_);
lean_dec(v___y_1995_);
lean_dec_ref(v___y_1994_);
lean_dec(v_ref_1992_);
return v_res_2001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19(lean_object* v_00_u03b1_2002_, lean_object* v_x_2003_, uint8_t v_isExporting_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_){
_start:
{
lean_object* v___x_2012_; 
v___x_2012_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg(v_x_2003_, v_isExporting_2004_, v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_);
return v___x_2012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___boxed(lean_object* v_00_u03b1_2013_, lean_object* v_x_2014_, lean_object* v_isExporting_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_){
_start:
{
uint8_t v_isExporting_boxed_2023_; lean_object* v_res_2024_; 
v_isExporting_boxed_2023_ = lean_unbox(v_isExporting_2015_);
v_res_2024_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19(v_00_u03b1_2013_, v_x_2014_, v_isExporting_boxed_2023_, v___y_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_);
lean_dec(v___y_2021_);
lean_dec_ref(v___y_2020_);
lean_dec(v___y_2019_);
lean_dec_ref(v___y_2018_);
lean_dec(v___y_2017_);
lean_dec_ref(v___y_2016_);
return v_res_2024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6(lean_object* v_00_u03b1_2025_, lean_object* v_x_2026_, uint8_t v_when_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_){
_start:
{
lean_object* v___x_2035_; 
v___x_2035_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___redArg(v_x_2026_, v_when_2027_, v___y_2028_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_);
return v___x_2035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6___boxed(lean_object* v_00_u03b1_2036_, lean_object* v_x_2037_, lean_object* v_when_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_){
_start:
{
uint8_t v_when_boxed_2046_; lean_object* v_res_2047_; 
v_when_boxed_2046_ = lean_unbox(v_when_2038_);
v_res_2047_ = lp_mathlib_Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6(v_00_u03b1_2036_, v_x_2037_, v_when_boxed_2046_, v___y_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_);
lean_dec(v___y_2044_);
lean_dec_ref(v___y_2043_);
lean_dec(v___y_2042_);
lean_dec_ref(v___y_2041_);
lean_dec(v___y_2040_);
lean_dec_ref(v___y_2039_);
return v_res_2047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4(lean_object* v_cls_2048_, lean_object* v_msg_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_){
_start:
{
lean_object* v___x_2057_; 
v___x_2057_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg(v_cls_2048_, v_msg_2049_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_);
return v___x_2057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___boxed(lean_object* v_cls_2058_, lean_object* v_msg_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
lean_object* v_res_2067_; 
v_res_2067_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4(v_cls_2058_, v_msg_2059_, v___y_2060_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_, v___y_2065_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
lean_dec(v___y_2063_);
lean_dec_ref(v___y_2062_);
lean_dec(v___y_2061_);
lean_dec_ref(v___y_2060_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6(lean_object* v_as_2068_, lean_object* v_as_x27_2069_, lean_object* v_b_2070_, lean_object* v_a_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_, lean_object* v___y_2077_){
_start:
{
lean_object* v___x_2079_; 
v___x_2079_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___redArg(v_as_x27_2069_, v_b_2070_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_, v___y_2077_);
return v___x_2079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6___boxed(lean_object* v_as_2080_, lean_object* v_as_x27_2081_, lean_object* v_b_2082_, lean_object* v_a_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_){
_start:
{
lean_object* v_res_2091_; 
v_res_2091_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__6(v_as_2080_, v_as_x27_2081_, v_b_2082_, v_a_2083_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_);
lean_dec(v___y_2089_);
lean_dec_ref(v___y_2088_);
lean_dec(v___y_2087_);
lean_dec_ref(v___y_2086_);
lean_dec(v___y_2085_);
lean_dec_ref(v___y_2084_);
lean_dec(v_as_x27_2081_);
lean_dec(v_as_2080_);
return v_res_2091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13(lean_object* v_00_u03b2_2092_, lean_object* v_m_2093_, lean_object* v_a_2094_){
_start:
{
lean_object* v___x_2095_; 
v___x_2095_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___redArg(v_m_2093_, v_a_2094_);
return v___x_2095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13___boxed(lean_object* v_00_u03b2_2096_, lean_object* v_m_2097_, lean_object* v_a_2098_){
_start:
{
lean_object* v_res_2099_; 
v_res_2099_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13(v_00_u03b2_2096_, v_m_2097_, v_a_2098_);
lean_dec(v_a_2098_);
lean_dec_ref(v_m_2097_);
return v_res_2099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16(lean_object* v_msgData_2100_, lean_object* v_macroStack_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_){
_start:
{
lean_object* v___x_2109_; 
v___x_2109_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___redArg(v_msgData_2100_, v_macroStack_2101_, v___y_2106_);
return v___x_2109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16___boxed(lean_object* v_msgData_2110_, lean_object* v_macroStack_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_){
_start:
{
lean_object* v_res_2119_; 
v_res_2119_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16(v_msgData_2110_, v_macroStack_2111_, v___y_2112_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_);
lean_dec(v___y_2117_);
lean_dec_ref(v___y_2116_);
lean_dec(v___y_2115_);
lean_dec_ref(v___y_2114_);
lean_dec(v___y_2113_);
lean_dec_ref(v___y_2112_);
return v_res_2119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22(lean_object* v_ref_2120_, lean_object* v_msgData_2121_, uint8_t v_severity_2122_, uint8_t v_isSilent_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_){
_start:
{
lean_object* v___x_2131_; 
v___x_2131_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg(v_ref_2120_, v_msgData_2121_, v_severity_2122_, v_isSilent_2123_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_);
return v___x_2131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___boxed(lean_object* v_ref_2132_, lean_object* v_msgData_2133_, lean_object* v_severity_2134_, lean_object* v_isSilent_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_, lean_object* v___y_2141_, lean_object* v___y_2142_){
_start:
{
uint8_t v_severity_boxed_2143_; uint8_t v_isSilent_boxed_2144_; lean_object* v_res_2145_; 
v_severity_boxed_2143_ = lean_unbox(v_severity_2134_);
v_isSilent_boxed_2144_ = lean_unbox(v_isSilent_2135_);
v_res_2145_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22(v_ref_2132_, v_msgData_2133_, v_severity_boxed_2143_, v_isSilent_boxed_2144_, v___y_2136_, v___y_2137_, v___y_2138_, v___y_2139_, v___y_2140_, v___y_2141_);
lean_dec(v___y_2141_);
lean_dec_ref(v___y_2140_);
lean_dec(v___y_2139_);
lean_dec_ref(v___y_2138_);
lean_dec(v___y_2137_);
lean_dec_ref(v___y_2136_);
lean_dec(v_ref_2132_);
return v_res_2145_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14(lean_object* v_00_u03b2_2146_, lean_object* v_x_2147_, lean_object* v_x_2148_){
_start:
{
uint8_t v___x_2149_; 
v___x_2149_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___redArg(v_x_2147_, v_x_2148_);
return v___x_2149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14___boxed(lean_object* v_00_u03b2_2150_, lean_object* v_x_2151_, lean_object* v_x_2152_){
_start:
{
uint8_t v_res_2153_; lean_object* v_r_2154_; 
v_res_2153_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14(v_00_u03b2_2150_, v_x_2151_, v_x_2152_);
lean_dec_ref(v_x_2152_);
lean_dec_ref(v_x_2151_);
v_r_2154_ = lean_box(v_res_2153_);
return v_r_2154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17(lean_object* v_00_u03b2_2155_, lean_object* v_a_2156_, lean_object* v_x_2157_){
_start:
{
lean_object* v___x_2158_; 
v___x_2158_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___redArg(v_a_2156_, v_x_2157_);
return v___x_2158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17___boxed(lean_object* v_00_u03b2_2159_, lean_object* v_a_2160_, lean_object* v_x_2161_){
_start:
{
lean_object* v_res_2162_; 
v_res_2162_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__13_spec__17(v_00_u03b2_2159_, v_a_2160_, v_x_2161_);
lean_dec(v_x_2161_);
lean_dec(v_a_2160_);
return v_res_2162_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22(lean_object* v_00_u03b2_2163_, lean_object* v_x_2164_, size_t v_x_2165_, lean_object* v_x_2166_){
_start:
{
uint8_t v___x_2167_; 
v___x_2167_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___redArg(v_x_2164_, v_x_2165_, v_x_2166_);
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22___boxed(lean_object* v_00_u03b2_2168_, lean_object* v_x_2169_, lean_object* v_x_2170_, lean_object* v_x_2171_){
_start:
{
size_t v_x_31288__boxed_2172_; uint8_t v_res_2173_; lean_object* v_r_2174_; 
v_x_31288__boxed_2172_ = lean_unbox_usize(v_x_2170_);
lean_dec(v_x_2170_);
v_res_2173_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22(v_00_u03b2_2168_, v_x_2169_, v_x_31288__boxed_2172_, v_x_2171_);
lean_dec_ref(v_x_2171_);
lean_dec_ref(v_x_2169_);
v_r_2174_ = lean_box(v_res_2173_);
return v_r_2174_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29(lean_object* v_00_u03b2_2175_, lean_object* v_keys_2176_, lean_object* v_vals_2177_, lean_object* v_heq_2178_, lean_object* v_i_2179_, lean_object* v_k_2180_){
_start:
{
uint8_t v___x_2181_; 
v___x_2181_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___redArg(v_keys_2176_, v_i_2179_, v_k_2180_);
return v___x_2181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29___boxed(lean_object* v_00_u03b2_2182_, lean_object* v_keys_2183_, lean_object* v_vals_2184_, lean_object* v_heq_2185_, lean_object* v_i_2186_, lean_object* v_k_2187_){
_start:
{
uint8_t v_res_2188_; lean_object* v_r_2189_; 
v_res_2188_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11_spec__14_spec__22_spec__29(v_00_u03b2_2182_, v_keys_2183_, v_vals_2184_, v_heq_2185_, v_i_2186_, v_k_2187_);
lean_dec_ref(v_k_2187_);
lean_dec_ref(v_vals_2184_);
lean_dec_ref(v_keys_2183_);
v_r_2189_ = lean_box(v_res_2188_);
return v_r_2189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1(lean_object* v_opts_2190_, lean_object* v_opt_2191_){
_start:
{
lean_object* v_name_2192_; lean_object* v_defValue_2193_; lean_object* v_map_2194_; lean_object* v___x_2195_; 
v_name_2192_ = lean_ctor_get(v_opt_2191_, 0);
v_defValue_2193_ = lean_ctor_get(v_opt_2191_, 1);
v_map_2194_ = lean_ctor_get(v_opts_2190_, 0);
v___x_2195_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2194_, v_name_2192_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_inc(v_defValue_2193_);
return v_defValue_2193_;
}
else
{
lean_object* v_val_2196_; 
v_val_2196_ = lean_ctor_get(v___x_2195_, 0);
lean_inc(v_val_2196_);
lean_dec_ref_known(v___x_2195_, 1);
if (lean_obj_tag(v_val_2196_) == 3)
{
lean_object* v_v_2197_; 
v_v_2197_ = lean_ctor_get(v_val_2196_, 0);
lean_inc(v_v_2197_);
lean_dec_ref_known(v_val_2196_, 1);
return v_v_2197_;
}
else
{
lean_dec(v_val_2196_);
lean_inc(v_defValue_2193_);
return v_defValue_2193_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1___boxed(lean_object* v_opts_2198_, lean_object* v_opt_2199_){
_start:
{
lean_object* v_res_2200_; 
v_res_2200_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1(v_opts_2198_, v_opt_2199_);
lean_dec_ref(v_opt_2199_);
lean_dec_ref(v_opts_2198_);
return v_res_2200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___lam__0(lean_object* v_ps_2201_, lean_object* v_k_2202_, lean_object* v_v_2203_){
_start:
{
lean_object* v___x_2204_; lean_object* v___x_2205_; 
v___x_2204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2204_, 0, v_k_2202_);
lean_ctor_set(v___x_2204_, 1, v_v_2203_);
v___x_2205_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2205_, 0, v___x_2204_);
lean_ctor_set(v___x_2205_, 1, v_ps_2201_);
return v___x_2205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg(lean_object* v_f_2206_, lean_object* v_keys_2207_, lean_object* v_vals_2208_, lean_object* v_i_2209_, lean_object* v_acc_2210_){
_start:
{
lean_object* v___x_2211_; uint8_t v___x_2212_; 
v___x_2211_ = lean_array_get_size(v_keys_2207_);
v___x_2212_ = lean_nat_dec_lt(v_i_2209_, v___x_2211_);
if (v___x_2212_ == 0)
{
lean_dec(v_i_2209_);
lean_dec(v_f_2206_);
return v_acc_2210_;
}
else
{
lean_object* v_k_2213_; lean_object* v_v_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; 
v_k_2213_ = lean_array_fget_borrowed(v_keys_2207_, v_i_2209_);
v_v_2214_ = lean_array_fget_borrowed(v_vals_2208_, v_i_2209_);
lean_inc(v_f_2206_);
lean_inc(v_v_2214_);
lean_inc(v_k_2213_);
v___x_2215_ = lean_apply_3(v_f_2206_, v_acc_2210_, v_k_2213_, v_v_2214_);
v___x_2216_ = lean_unsigned_to_nat(1u);
v___x_2217_ = lean_nat_add(v_i_2209_, v___x_2216_);
lean_dec(v_i_2209_);
v_i_2209_ = v___x_2217_;
v_acc_2210_ = v___x_2215_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg___boxed(lean_object* v_f_2219_, lean_object* v_keys_2220_, lean_object* v_vals_2221_, lean_object* v_i_2222_, lean_object* v_acc_2223_){
_start:
{
lean_object* v_res_2224_; 
v_res_2224_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg(v_f_2219_, v_keys_2220_, v_vals_2221_, v_i_2222_, v_acc_2223_);
lean_dec_ref(v_vals_2221_);
lean_dec_ref(v_keys_2220_);
return v_res_2224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(lean_object* v_f_2225_, lean_object* v_x_2226_, lean_object* v_x_2227_){
_start:
{
if (lean_obj_tag(v_x_2226_) == 0)
{
lean_object* v_es_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; uint8_t v___x_2231_; 
v_es_2228_ = lean_ctor_get(v_x_2226_, 0);
v___x_2229_ = lean_unsigned_to_nat(0u);
v___x_2230_ = lean_array_get_size(v_es_2228_);
v___x_2231_ = lean_nat_dec_lt(v___x_2229_, v___x_2230_);
if (v___x_2231_ == 0)
{
lean_dec(v_f_2225_);
return v_x_2227_;
}
else
{
uint8_t v___x_2232_; 
v___x_2232_ = lean_nat_dec_le(v___x_2230_, v___x_2230_);
if (v___x_2232_ == 0)
{
if (v___x_2231_ == 0)
{
lean_dec(v_f_2225_);
return v_x_2227_;
}
else
{
size_t v___x_2233_; size_t v___x_2234_; lean_object* v___x_2235_; 
v___x_2233_ = ((size_t)0ULL);
v___x_2234_ = lean_usize_of_nat(v___x_2230_);
v___x_2235_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(v_f_2225_, v_es_2228_, v___x_2233_, v___x_2234_, v_x_2227_);
return v___x_2235_;
}
}
else
{
size_t v___x_2236_; size_t v___x_2237_; lean_object* v___x_2238_; 
v___x_2236_ = ((size_t)0ULL);
v___x_2237_ = lean_usize_of_nat(v___x_2230_);
v___x_2238_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(v_f_2225_, v_es_2228_, v___x_2236_, v___x_2237_, v_x_2227_);
return v___x_2238_;
}
}
}
else
{
lean_object* v_ks_2239_; lean_object* v_vs_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; 
v_ks_2239_ = lean_ctor_get(v_x_2226_, 0);
v_vs_2240_ = lean_ctor_get(v_x_2226_, 1);
v___x_2241_ = lean_unsigned_to_nat(0u);
v___x_2242_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg(v_f_2225_, v_ks_2239_, v_vs_2240_, v___x_2241_, v_x_2227_);
return v___x_2242_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(lean_object* v_f_2243_, lean_object* v_as_2244_, size_t v_i_2245_, size_t v_stop_2246_, lean_object* v_b_2247_){
_start:
{
lean_object* v___y_2249_; uint8_t v___x_2253_; 
v___x_2253_ = lean_usize_dec_eq(v_i_2245_, v_stop_2246_);
if (v___x_2253_ == 0)
{
lean_object* v___x_2254_; 
v___x_2254_ = lean_array_uget_borrowed(v_as_2244_, v_i_2245_);
switch(lean_obj_tag(v___x_2254_))
{
case 0:
{
lean_object* v_key_2255_; lean_object* v_val_2256_; lean_object* v___x_2257_; 
v_key_2255_ = lean_ctor_get(v___x_2254_, 0);
v_val_2256_ = lean_ctor_get(v___x_2254_, 1);
lean_inc(v_f_2243_);
lean_inc(v_val_2256_);
lean_inc(v_key_2255_);
v___x_2257_ = lean_apply_3(v_f_2243_, v_b_2247_, v_key_2255_, v_val_2256_);
v___y_2249_ = v___x_2257_;
goto v___jp_2248_;
}
case 1:
{
lean_object* v_node_2258_; lean_object* v___x_2259_; 
v_node_2258_ = lean_ctor_get(v___x_2254_, 0);
lean_inc(v_f_2243_);
v___x_2259_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v_f_2243_, v_node_2258_, v_b_2247_);
v___y_2249_ = v___x_2259_;
goto v___jp_2248_;
}
default: 
{
v___y_2249_ = v_b_2247_;
goto v___jp_2248_;
}
}
}
else
{
lean_dec(v_f_2243_);
return v_b_2247_;
}
v___jp_2248_:
{
size_t v___x_2250_; size_t v___x_2251_; 
v___x_2250_ = ((size_t)1ULL);
v___x_2251_ = lean_usize_add(v_i_2245_, v___x_2250_);
v_i_2245_ = v___x_2251_;
v_b_2247_ = v___y_2249_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg___boxed(lean_object* v_f_2260_, lean_object* v_as_2261_, lean_object* v_i_2262_, lean_object* v_stop_2263_, lean_object* v_b_2264_){
_start:
{
size_t v_i_boxed_2265_; size_t v_stop_boxed_2266_; lean_object* v_res_2267_; 
v_i_boxed_2265_ = lean_unbox_usize(v_i_2262_);
lean_dec(v_i_2262_);
v_stop_boxed_2266_ = lean_unbox_usize(v_stop_2263_);
lean_dec(v_stop_2263_);
v_res_2267_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(v_f_2260_, v_as_2261_, v_i_boxed_2265_, v_stop_boxed_2266_, v_b_2264_);
lean_dec_ref(v_as_2261_);
return v_res_2267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg___boxed(lean_object* v_f_2268_, lean_object* v_x_2269_, lean_object* v_x_2270_){
_start:
{
lean_object* v_res_2271_; 
v_res_2271_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v_f_2268_, v_x_2269_, v_x_2270_);
lean_dec_ref(v_x_2269_);
return v_res_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg___lam__0(lean_object* v_f_2272_, lean_object* v_x1_2273_, lean_object* v_x2_2274_, lean_object* v_x3_2275_){
_start:
{
lean_object* v___x_2276_; 
v___x_2276_ = lean_apply_3(v_f_2272_, v_x1_2273_, v_x2_2274_, v_x3_2275_);
return v___x_2276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg(lean_object* v_map_2277_, lean_object* v_f_2278_, lean_object* v_init_2279_){
_start:
{
lean_object* v___f_2280_; lean_object* v___x_2281_; 
v___f_2280_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2280_, 0, v_f_2278_);
v___x_2281_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v___f_2280_, v_map_2277_, v_init_2279_);
return v___x_2281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg___boxed(lean_object* v_map_2282_, lean_object* v_f_2283_, lean_object* v_init_2284_){
_start:
{
lean_object* v_res_2285_; 
v_res_2285_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg(v_map_2282_, v_f_2283_, v_init_2284_);
lean_dec_ref(v_map_2282_);
return v_res_2285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg(lean_object* v_m_2287_){
_start:
{
lean_object* v___f_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; 
v___f_2288_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___closed__0));
v___x_2289_ = lean_box(0);
v___x_2290_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg(v_m_2287_, v___f_2288_, v___x_2289_);
return v___x_2290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg___boxed(lean_object* v_m_2291_){
_start:
{
lean_object* v_res_2292_; 
v_res_2292_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg(v_m_2291_);
lean_dec_ref(v_m_2291_);
return v_res_2292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0(lean_object* v_o_2293_, lean_object* v_k_2294_, uint8_t v_v_2295_){
_start:
{
lean_object* v_map_2296_; uint8_t v_hasTrace_2297_; lean_object* v___x_2299_; uint8_t v_isShared_2300_; uint8_t v_isSharedCheck_2311_; 
v_map_2296_ = lean_ctor_get(v_o_2293_, 0);
v_hasTrace_2297_ = lean_ctor_get_uint8(v_o_2293_, sizeof(void*)*1);
v_isSharedCheck_2311_ = !lean_is_exclusive(v_o_2293_);
if (v_isSharedCheck_2311_ == 0)
{
v___x_2299_ = v_o_2293_;
v_isShared_2300_ = v_isSharedCheck_2311_;
goto v_resetjp_2298_;
}
else
{
lean_inc(v_map_2296_);
lean_dec(v_o_2293_);
v___x_2299_ = lean_box(0);
v_isShared_2300_ = v_isSharedCheck_2311_;
goto v_resetjp_2298_;
}
v_resetjp_2298_:
{
lean_object* v___x_2301_; lean_object* v___x_2302_; 
v___x_2301_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2301_, 0, v_v_2295_);
lean_inc(v_k_2294_);
v___x_2302_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_2294_, v___x_2301_, v_map_2296_);
if (v_hasTrace_2297_ == 0)
{
lean_object* v___x_2303_; uint8_t v___x_2304_; lean_object* v___x_2306_; 
v___x_2303_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__3_spec__11___closed__11));
v___x_2304_ = l_Lean_Name_isPrefixOf(v___x_2303_, v_k_2294_);
lean_dec(v_k_2294_);
if (v_isShared_2300_ == 0)
{
lean_ctor_set(v___x_2299_, 0, v___x_2302_);
v___x_2306_ = v___x_2299_;
goto v_reusejp_2305_;
}
else
{
lean_object* v_reuseFailAlloc_2307_; 
v_reuseFailAlloc_2307_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2307_, 0, v___x_2302_);
v___x_2306_ = v_reuseFailAlloc_2307_;
goto v_reusejp_2305_;
}
v_reusejp_2305_:
{
lean_ctor_set_uint8(v___x_2306_, sizeof(void*)*1, v___x_2304_);
return v___x_2306_;
}
}
else
{
lean_object* v___x_2309_; 
lean_dec(v_k_2294_);
if (v_isShared_2300_ == 0)
{
lean_ctor_set(v___x_2299_, 0, v___x_2302_);
v___x_2309_ = v___x_2299_;
goto v_reusejp_2308_;
}
else
{
lean_object* v_reuseFailAlloc_2310_; 
v_reuseFailAlloc_2310_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2310_, 0, v___x_2302_);
lean_ctor_set_uint8(v_reuseFailAlloc_2310_, sizeof(void*)*1, v_hasTrace_2297_);
v___x_2309_ = v_reuseFailAlloc_2310_;
goto v_reusejp_2308_;
}
v_reusejp_2308_:
{
return v___x_2309_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0___boxed(lean_object* v_o_2312_, lean_object* v_k_2313_, lean_object* v_v_2314_){
_start:
{
uint8_t v_v_boxed_2315_; lean_object* v_res_2316_; 
v_v_boxed_2315_ = lean_unbox(v_v_2314_);
v_res_2316_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0(v_o_2312_, v_k_2313_, v_v_boxed_2315_);
return v_res_2316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0(lean_object* v_opts_2317_, lean_object* v_opt_2318_, uint8_t v_val_2319_){
_start:
{
lean_object* v_name_2320_; lean_object* v___x_2321_; 
v_name_2320_ = lean_ctor_get(v_opt_2318_, 0);
lean_inc(v_name_2320_);
lean_dec_ref(v_opt_2318_);
v___x_2321_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0_spec__0(v_opts_2317_, v_name_2320_, v_val_2319_);
return v___x_2321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0___boxed(lean_object* v_opts_2322_, lean_object* v_opt_2323_, lean_object* v_val_2324_){
_start:
{
uint8_t v_val_boxed_2325_; lean_object* v_res_2326_; 
v_val_boxed_2325_ = lean_unbox(v_val_2324_);
v_res_2326_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0(v_opts_2322_, v_opt_2323_, v_val_boxed_2325_);
return v_res_2326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__4(lean_object* v___x_2327_, lean_object* v_a_2328_, lean_object* v_a_2329_){
_start:
{
if (lean_obj_tag(v_a_2328_) == 0)
{
lean_object* v___x_2330_; 
lean_dec_ref(v___x_2327_);
v___x_2330_ = lean_array_to_list(v_a_2329_);
return v___x_2330_;
}
else
{
lean_object* v_head_2331_; lean_object* v_tail_2332_; lean_object* v_fst_2333_; lean_object* v_snd_2334_; lean_object* v___x_2335_; uint8_t v___x_2336_; 
v_head_2331_ = lean_ctor_get(v_a_2328_, 0);
lean_inc(v_head_2331_);
v_tail_2332_ = lean_ctor_get(v_a_2328_, 1);
lean_inc(v_tail_2332_);
lean_dec_ref_known(v_a_2328_, 2);
v_fst_2333_ = lean_ctor_get(v_head_2331_, 0);
lean_inc(v_fst_2333_);
v_snd_2334_ = lean_ctor_get(v_head_2331_, 1);
lean_inc(v_snd_2334_);
lean_dec(v_head_2331_);
v___x_2335_ = lean_unsigned_to_nat(0u);
v___x_2336_ = lean_nat_dec_lt(v___x_2335_, v_snd_2334_);
lean_dec(v_snd_2334_);
if (v___x_2336_ == 0)
{
lean_dec(v_fst_2333_);
v_a_2328_ = v_tail_2332_;
goto _start;
}
else
{
uint8_t v___x_2338_; 
lean_inc(v_fst_2333_);
lean_inc_ref(v___x_2327_);
v___x_2338_ = l_Lean_getReducibilityStatusCore(v___x_2327_, v_fst_2333_);
if (v___x_2338_ == 1)
{
uint8_t v___x_2339_; 
lean_inc_ref(v___x_2327_);
v___x_2339_ = l_Lean_Meta_isInstanceCore(v___x_2327_, v_fst_2333_);
if (v___x_2339_ == 0)
{
lean_object* v___x_2340_; 
v___x_2340_ = lean_array_push(v_a_2329_, v_fst_2333_);
v_a_2328_ = v_tail_2332_;
v_a_2329_ = v___x_2340_;
goto _start;
}
else
{
lean_dec(v_fst_2333_);
v_a_2328_ = v_tail_2332_;
goto _start;
}
}
else
{
lean_dec(v_fst_2333_);
v_a_2328_ = v_tail_2332_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13___redArg(lean_object* v_x_2344_, lean_object* v_x_2345_, lean_object* v_x_2346_, lean_object* v_x_2347_){
_start:
{
lean_object* v_ks_2348_; lean_object* v_vs_2349_; lean_object* v___x_2351_; uint8_t v_isShared_2352_; uint8_t v_isSharedCheck_2373_; 
v_ks_2348_ = lean_ctor_get(v_x_2344_, 0);
v_vs_2349_ = lean_ctor_get(v_x_2344_, 1);
v_isSharedCheck_2373_ = !lean_is_exclusive(v_x_2344_);
if (v_isSharedCheck_2373_ == 0)
{
v___x_2351_ = v_x_2344_;
v_isShared_2352_ = v_isSharedCheck_2373_;
goto v_resetjp_2350_;
}
else
{
lean_inc(v_vs_2349_);
lean_inc(v_ks_2348_);
lean_dec(v_x_2344_);
v___x_2351_ = lean_box(0);
v_isShared_2352_ = v_isSharedCheck_2373_;
goto v_resetjp_2350_;
}
v_resetjp_2350_:
{
lean_object* v___x_2353_; uint8_t v___x_2354_; 
v___x_2353_ = lean_array_get_size(v_ks_2348_);
v___x_2354_ = lean_nat_dec_lt(v_x_2345_, v___x_2353_);
if (v___x_2354_ == 0)
{
lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2358_; 
lean_dec(v_x_2345_);
v___x_2355_ = lean_array_push(v_ks_2348_, v_x_2346_);
v___x_2356_ = lean_array_push(v_vs_2349_, v_x_2347_);
if (v_isShared_2352_ == 0)
{
lean_ctor_set(v___x_2351_, 1, v___x_2356_);
lean_ctor_set(v___x_2351_, 0, v___x_2355_);
v___x_2358_ = v___x_2351_;
goto v_reusejp_2357_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v___x_2355_);
lean_ctor_set(v_reuseFailAlloc_2359_, 1, v___x_2356_);
v___x_2358_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2357_;
}
v_reusejp_2357_:
{
return v___x_2358_;
}
}
else
{
lean_object* v_k_x27_2360_; uint8_t v___x_2361_; 
v_k_x27_2360_ = lean_array_fget_borrowed(v_ks_2348_, v_x_2345_);
v___x_2361_ = lean_name_eq(v_x_2346_, v_k_x27_2360_);
if (v___x_2361_ == 0)
{
lean_object* v___x_2363_; 
if (v_isShared_2352_ == 0)
{
v___x_2363_ = v___x_2351_;
goto v_reusejp_2362_;
}
else
{
lean_object* v_reuseFailAlloc_2367_; 
v_reuseFailAlloc_2367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2367_, 0, v_ks_2348_);
lean_ctor_set(v_reuseFailAlloc_2367_, 1, v_vs_2349_);
v___x_2363_ = v_reuseFailAlloc_2367_;
goto v_reusejp_2362_;
}
v_reusejp_2362_:
{
lean_object* v___x_2364_; lean_object* v___x_2365_; 
v___x_2364_ = lean_unsigned_to_nat(1u);
v___x_2365_ = lean_nat_add(v_x_2345_, v___x_2364_);
lean_dec(v_x_2345_);
v_x_2344_ = v___x_2363_;
v_x_2345_ = v___x_2365_;
goto _start;
}
}
else
{
lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2371_; 
v___x_2368_ = lean_array_fset(v_ks_2348_, v_x_2345_, v_x_2346_);
v___x_2369_ = lean_array_fset(v_vs_2349_, v_x_2345_, v_x_2347_);
lean_dec(v_x_2345_);
if (v_isShared_2352_ == 0)
{
lean_ctor_set(v___x_2351_, 1, v___x_2369_);
lean_ctor_set(v___x_2351_, 0, v___x_2368_);
v___x_2371_ = v___x_2351_;
goto v_reusejp_2370_;
}
else
{
lean_object* v_reuseFailAlloc_2372_; 
v_reuseFailAlloc_2372_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2372_, 0, v___x_2368_);
lean_ctor_set(v_reuseFailAlloc_2372_, 1, v___x_2369_);
v___x_2371_ = v_reuseFailAlloc_2372_;
goto v_reusejp_2370_;
}
v_reusejp_2370_:
{
return v___x_2371_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10___redArg(lean_object* v_n_2374_, lean_object* v_k_2375_, lean_object* v_v_2376_){
_start:
{
lean_object* v___x_2377_; lean_object* v___x_2378_; 
v___x_2377_ = lean_unsigned_to_nat(0u);
v___x_2378_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13___redArg(v_n_2374_, v___x_2377_, v_k_2375_, v_v_2376_);
return v___x_2378_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_2379_; 
v___x_2379_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_2379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(lean_object* v_x_2380_, size_t v_x_2381_, size_t v_x_2382_, lean_object* v_x_2383_, lean_object* v_x_2384_){
_start:
{
if (lean_obj_tag(v_x_2380_) == 0)
{
lean_object* v_es_2385_; size_t v___x_2386_; size_t v___x_2387_; lean_object* v_j_2388_; lean_object* v___x_2389_; uint8_t v___x_2390_; 
v_es_2385_ = lean_ctor_get(v_x_2380_, 0);
v___x_2386_ = ((size_t)31ULL);
v___x_2387_ = lean_usize_land(v_x_2381_, v___x_2386_);
v_j_2388_ = lean_usize_to_nat(v___x_2387_);
v___x_2389_ = lean_array_get_size(v_es_2385_);
v___x_2390_ = lean_nat_dec_lt(v_j_2388_, v___x_2389_);
if (v___x_2390_ == 0)
{
lean_dec(v_j_2388_);
lean_dec(v_x_2384_);
lean_dec(v_x_2383_);
return v_x_2380_;
}
else
{
lean_object* v___x_2392_; uint8_t v_isShared_2393_; uint8_t v_isSharedCheck_2429_; 
lean_inc_ref(v_es_2385_);
v_isSharedCheck_2429_ = !lean_is_exclusive(v_x_2380_);
if (v_isSharedCheck_2429_ == 0)
{
lean_object* v_unused_2430_; 
v_unused_2430_ = lean_ctor_get(v_x_2380_, 0);
lean_dec(v_unused_2430_);
v___x_2392_ = v_x_2380_;
v_isShared_2393_ = v_isSharedCheck_2429_;
goto v_resetjp_2391_;
}
else
{
lean_dec(v_x_2380_);
v___x_2392_ = lean_box(0);
v_isShared_2393_ = v_isSharedCheck_2429_;
goto v_resetjp_2391_;
}
v_resetjp_2391_:
{
lean_object* v_v_2394_; lean_object* v___x_2395_; lean_object* v_xs_x27_2396_; lean_object* v___y_2398_; 
v_v_2394_ = lean_array_fget(v_es_2385_, v_j_2388_);
v___x_2395_ = lean_box(0);
v_xs_x27_2396_ = lean_array_fset(v_es_2385_, v_j_2388_, v___x_2395_);
switch(lean_obj_tag(v_v_2394_))
{
case 0:
{
lean_object* v_key_2403_; lean_object* v_val_2404_; lean_object* v___x_2406_; uint8_t v_isShared_2407_; uint8_t v_isSharedCheck_2414_; 
v_key_2403_ = lean_ctor_get(v_v_2394_, 0);
v_val_2404_ = lean_ctor_get(v_v_2394_, 1);
v_isSharedCheck_2414_ = !lean_is_exclusive(v_v_2394_);
if (v_isSharedCheck_2414_ == 0)
{
v___x_2406_ = v_v_2394_;
v_isShared_2407_ = v_isSharedCheck_2414_;
goto v_resetjp_2405_;
}
else
{
lean_inc(v_val_2404_);
lean_inc(v_key_2403_);
lean_dec(v_v_2394_);
v___x_2406_ = lean_box(0);
v_isShared_2407_ = v_isSharedCheck_2414_;
goto v_resetjp_2405_;
}
v_resetjp_2405_:
{
uint8_t v___x_2408_; 
v___x_2408_ = lean_name_eq(v_x_2383_, v_key_2403_);
if (v___x_2408_ == 0)
{
lean_object* v___x_2409_; lean_object* v___x_2410_; 
lean_del_object(v___x_2406_);
v___x_2409_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2403_, v_val_2404_, v_x_2383_, v_x_2384_);
v___x_2410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2410_, 0, v___x_2409_);
v___y_2398_ = v___x_2410_;
goto v___jp_2397_;
}
else
{
lean_object* v___x_2412_; 
lean_dec(v_val_2404_);
lean_dec(v_key_2403_);
if (v_isShared_2407_ == 0)
{
lean_ctor_set(v___x_2406_, 1, v_x_2384_);
lean_ctor_set(v___x_2406_, 0, v_x_2383_);
v___x_2412_ = v___x_2406_;
goto v_reusejp_2411_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v_x_2383_);
lean_ctor_set(v_reuseFailAlloc_2413_, 1, v_x_2384_);
v___x_2412_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2411_;
}
v_reusejp_2411_:
{
v___y_2398_ = v___x_2412_;
goto v___jp_2397_;
}
}
}
}
case 1:
{
lean_object* v_node_2415_; lean_object* v___x_2417_; uint8_t v_isShared_2418_; uint8_t v_isSharedCheck_2427_; 
v_node_2415_ = lean_ctor_get(v_v_2394_, 0);
v_isSharedCheck_2427_ = !lean_is_exclusive(v_v_2394_);
if (v_isSharedCheck_2427_ == 0)
{
v___x_2417_ = v_v_2394_;
v_isShared_2418_ = v_isSharedCheck_2427_;
goto v_resetjp_2416_;
}
else
{
lean_inc(v_node_2415_);
lean_dec(v_v_2394_);
v___x_2417_ = lean_box(0);
v_isShared_2418_ = v_isSharedCheck_2427_;
goto v_resetjp_2416_;
}
v_resetjp_2416_:
{
size_t v___x_2419_; size_t v___x_2420_; size_t v___x_2421_; size_t v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2425_; 
v___x_2419_ = ((size_t)5ULL);
v___x_2420_ = lean_usize_shift_right(v_x_2381_, v___x_2419_);
v___x_2421_ = ((size_t)1ULL);
v___x_2422_ = lean_usize_add(v_x_2382_, v___x_2421_);
v___x_2423_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(v_node_2415_, v___x_2420_, v___x_2422_, v_x_2383_, v_x_2384_);
if (v_isShared_2418_ == 0)
{
lean_ctor_set(v___x_2417_, 0, v___x_2423_);
v___x_2425_ = v___x_2417_;
goto v_reusejp_2424_;
}
else
{
lean_object* v_reuseFailAlloc_2426_; 
v_reuseFailAlloc_2426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2426_, 0, v___x_2423_);
v___x_2425_ = v_reuseFailAlloc_2426_;
goto v_reusejp_2424_;
}
v_reusejp_2424_:
{
v___y_2398_ = v___x_2425_;
goto v___jp_2397_;
}
}
}
default: 
{
lean_object* v___x_2428_; 
v___x_2428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2428_, 0, v_x_2383_);
lean_ctor_set(v___x_2428_, 1, v_x_2384_);
v___y_2398_ = v___x_2428_;
goto v___jp_2397_;
}
}
v___jp_2397_:
{
lean_object* v___x_2399_; lean_object* v___x_2401_; 
v___x_2399_ = lean_array_fset(v_xs_x27_2396_, v_j_2388_, v___y_2398_);
lean_dec(v_j_2388_);
if (v_isShared_2393_ == 0)
{
lean_ctor_set(v___x_2392_, 0, v___x_2399_);
v___x_2401_ = v___x_2392_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v___x_2399_);
v___x_2401_ = v_reuseFailAlloc_2402_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
return v___x_2401_;
}
}
}
}
}
else
{
lean_object* v_ks_2431_; lean_object* v_vs_2432_; lean_object* v___x_2434_; uint8_t v_isShared_2435_; uint8_t v_isSharedCheck_2452_; 
v_ks_2431_ = lean_ctor_get(v_x_2380_, 0);
v_vs_2432_ = lean_ctor_get(v_x_2380_, 1);
v_isSharedCheck_2452_ = !lean_is_exclusive(v_x_2380_);
if (v_isSharedCheck_2452_ == 0)
{
v___x_2434_ = v_x_2380_;
v_isShared_2435_ = v_isSharedCheck_2452_;
goto v_resetjp_2433_;
}
else
{
lean_inc(v_vs_2432_);
lean_inc(v_ks_2431_);
lean_dec(v_x_2380_);
v___x_2434_ = lean_box(0);
v_isShared_2435_ = v_isSharedCheck_2452_;
goto v_resetjp_2433_;
}
v_resetjp_2433_:
{
lean_object* v___x_2437_; 
if (v_isShared_2435_ == 0)
{
v___x_2437_ = v___x_2434_;
goto v_reusejp_2436_;
}
else
{
lean_object* v_reuseFailAlloc_2451_; 
v_reuseFailAlloc_2451_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2451_, 0, v_ks_2431_);
lean_ctor_set(v_reuseFailAlloc_2451_, 1, v_vs_2432_);
v___x_2437_ = v_reuseFailAlloc_2451_;
goto v_reusejp_2436_;
}
v_reusejp_2436_:
{
lean_object* v_newNode_2438_; uint8_t v___y_2440_; size_t v___x_2446_; uint8_t v___x_2447_; 
v_newNode_2438_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10___redArg(v___x_2437_, v_x_2383_, v_x_2384_);
v___x_2446_ = ((size_t)7ULL);
v___x_2447_ = lean_usize_dec_le(v___x_2446_, v_x_2382_);
if (v___x_2447_ == 0)
{
lean_object* v___x_2448_; lean_object* v___x_2449_; uint8_t v___x_2450_; 
v___x_2448_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_2438_);
v___x_2449_ = lean_unsigned_to_nat(4u);
v___x_2450_ = lean_nat_dec_lt(v___x_2448_, v___x_2449_);
lean_dec(v___x_2448_);
v___y_2440_ = v___x_2450_;
goto v___jp_2439_;
}
else
{
v___y_2440_ = v___x_2447_;
goto v___jp_2439_;
}
v___jp_2439_:
{
if (v___y_2440_ == 0)
{
lean_object* v_ks_2441_; lean_object* v_vs_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; 
v_ks_2441_ = lean_ctor_get(v_newNode_2438_, 0);
lean_inc_ref(v_ks_2441_);
v_vs_2442_ = lean_ctor_get(v_newNode_2438_, 1);
lean_inc_ref(v_vs_2442_);
lean_dec_ref(v_newNode_2438_);
v___x_2443_ = lean_unsigned_to_nat(0u);
v___x_2444_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___closed__0);
v___x_2445_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg(v_x_2382_, v_ks_2441_, v_vs_2442_, v___x_2443_, v___x_2444_);
lean_dec_ref(v_vs_2442_);
lean_dec_ref(v_ks_2441_);
return v___x_2445_;
}
else
{
return v_newNode_2438_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg(size_t v_depth_2453_, lean_object* v_keys_2454_, lean_object* v_vals_2455_, lean_object* v_i_2456_, lean_object* v_entries_2457_){
_start:
{
lean_object* v___x_2458_; uint8_t v___x_2459_; 
v___x_2458_ = lean_array_get_size(v_keys_2454_);
v___x_2459_ = lean_nat_dec_lt(v_i_2456_, v___x_2458_);
if (v___x_2459_ == 0)
{
lean_dec(v_i_2456_);
return v_entries_2457_;
}
else
{
lean_object* v_k_2460_; lean_object* v_v_2461_; uint64_t v___y_2463_; 
v_k_2460_ = lean_array_fget_borrowed(v_keys_2454_, v_i_2456_);
v_v_2461_ = lean_array_fget_borrowed(v_vals_2455_, v_i_2456_);
if (lean_obj_tag(v_k_2460_) == 0)
{
uint64_t v___x_2474_; 
v___x_2474_ = 1723ULL;
v___y_2463_ = v___x_2474_;
goto v___jp_2462_;
}
else
{
uint64_t v_hash_2475_; 
v_hash_2475_ = lean_ctor_get_uint64(v_k_2460_, sizeof(void*)*2);
v___y_2463_ = v_hash_2475_;
goto v___jp_2462_;
}
v___jp_2462_:
{
size_t v_h_2464_; size_t v___x_2465_; lean_object* v___x_2466_; size_t v___x_2467_; size_t v___x_2468_; size_t v___x_2469_; size_t v_h_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; 
v_h_2464_ = lean_uint64_to_usize(v___y_2463_);
v___x_2465_ = ((size_t)5ULL);
v___x_2466_ = lean_unsigned_to_nat(1u);
v___x_2467_ = ((size_t)1ULL);
v___x_2468_ = lean_usize_sub(v_depth_2453_, v___x_2467_);
v___x_2469_ = lean_usize_mul(v___x_2465_, v___x_2468_);
v_h_2470_ = lean_usize_shift_right(v_h_2464_, v___x_2469_);
v___x_2471_ = lean_nat_add(v_i_2456_, v___x_2466_);
lean_dec(v_i_2456_);
lean_inc(v_v_2461_);
lean_inc(v_k_2460_);
v___x_2472_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(v_entries_2457_, v_h_2470_, v_depth_2453_, v_k_2460_, v_v_2461_);
v_i_2456_ = v___x_2471_;
v_entries_2457_ = v___x_2472_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg___boxed(lean_object* v_depth_2476_, lean_object* v_keys_2477_, lean_object* v_vals_2478_, lean_object* v_i_2479_, lean_object* v_entries_2480_){
_start:
{
size_t v_depth_boxed_2481_; lean_object* v_res_2482_; 
v_depth_boxed_2481_ = lean_unbox_usize(v_depth_2476_);
lean_dec(v_depth_2476_);
v_res_2482_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg(v_depth_boxed_2481_, v_keys_2477_, v_vals_2478_, v_i_2479_, v_entries_2480_);
lean_dec_ref(v_vals_2478_);
lean_dec_ref(v_keys_2477_);
return v_res_2482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg___boxed(lean_object* v_x_2483_, lean_object* v_x_2484_, lean_object* v_x_2485_, lean_object* v_x_2486_, lean_object* v_x_2487_){
_start:
{
size_t v_x_10321__boxed_2488_; size_t v_x_10322__boxed_2489_; lean_object* v_res_2490_; 
v_x_10321__boxed_2488_ = lean_unbox_usize(v_x_2484_);
lean_dec(v_x_2484_);
v_x_10322__boxed_2489_ = lean_unbox_usize(v_x_2485_);
lean_dec(v_x_2485_);
v_res_2490_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(v_x_2483_, v_x_10321__boxed_2488_, v_x_10322__boxed_2489_, v_x_2486_, v_x_2487_);
return v_res_2490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4___redArg(lean_object* v_x_2491_, lean_object* v_x_2492_, lean_object* v_x_2493_){
_start:
{
uint64_t v___y_2495_; 
if (lean_obj_tag(v_x_2492_) == 0)
{
uint64_t v___x_2499_; 
v___x_2499_ = 1723ULL;
v___y_2495_ = v___x_2499_;
goto v___jp_2494_;
}
else
{
uint64_t v_hash_2500_; 
v_hash_2500_ = lean_ctor_get_uint64(v_x_2492_, sizeof(void*)*2);
v___y_2495_ = v_hash_2500_;
goto v___jp_2494_;
}
v___jp_2494_:
{
size_t v___x_2496_; size_t v___x_2497_; lean_object* v___x_2498_; 
v___x_2496_ = lean_uint64_to_usize(v___y_2495_);
v___x_2497_ = ((size_t)1ULL);
v___x_2498_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(v_x_2491_, v___x_2496_, v___x_2497_, v_x_2492_, v_x_2493_);
return v___x_2498_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg(lean_object* v_keys_2501_, lean_object* v_vals_2502_, lean_object* v_i_2503_, lean_object* v_k_2504_){
_start:
{
lean_object* v___x_2505_; uint8_t v___x_2506_; 
v___x_2505_ = lean_array_get_size(v_keys_2501_);
v___x_2506_ = lean_nat_dec_lt(v_i_2503_, v___x_2505_);
if (v___x_2506_ == 0)
{
lean_object* v___x_2507_; 
lean_dec(v_i_2503_);
v___x_2507_ = lean_box(0);
return v___x_2507_;
}
else
{
lean_object* v_k_x27_2508_; uint8_t v___x_2509_; 
v_k_x27_2508_ = lean_array_fget_borrowed(v_keys_2501_, v_i_2503_);
v___x_2509_ = lean_name_eq(v_k_2504_, v_k_x27_2508_);
if (v___x_2509_ == 0)
{
lean_object* v___x_2510_; lean_object* v___x_2511_; 
v___x_2510_ = lean_unsigned_to_nat(1u);
v___x_2511_ = lean_nat_add(v_i_2503_, v___x_2510_);
lean_dec(v_i_2503_);
v_i_2503_ = v___x_2511_;
goto _start;
}
else
{
lean_object* v___x_2513_; lean_object* v___x_2514_; 
v___x_2513_ = lean_array_fget_borrowed(v_vals_2502_, v_i_2503_);
lean_dec(v_i_2503_);
lean_inc(v___x_2513_);
v___x_2514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2514_, 0, v___x_2513_);
return v___x_2514_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg___boxed(lean_object* v_keys_2515_, lean_object* v_vals_2516_, lean_object* v_i_2517_, lean_object* v_k_2518_){
_start:
{
lean_object* v_res_2519_; 
v_res_2519_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg(v_keys_2515_, v_vals_2516_, v_i_2517_, v_k_2518_);
lean_dec(v_k_2518_);
lean_dec_ref(v_vals_2516_);
lean_dec_ref(v_keys_2515_);
return v_res_2519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg(lean_object* v_x_2520_, size_t v_x_2521_, lean_object* v_x_2522_){
_start:
{
if (lean_obj_tag(v_x_2520_) == 0)
{
lean_object* v_es_2523_; lean_object* v___x_2524_; size_t v___x_2525_; size_t v___x_2526_; lean_object* v_j_2527_; lean_object* v___x_2528_; 
v_es_2523_ = lean_ctor_get(v_x_2520_, 0);
v___x_2524_ = lean_box(2);
v___x_2525_ = ((size_t)31ULL);
v___x_2526_ = lean_usize_land(v_x_2521_, v___x_2525_);
v_j_2527_ = lean_usize_to_nat(v___x_2526_);
v___x_2528_ = lean_array_get_borrowed(v___x_2524_, v_es_2523_, v_j_2527_);
lean_dec(v_j_2527_);
switch(lean_obj_tag(v___x_2528_))
{
case 0:
{
lean_object* v_key_2529_; lean_object* v_val_2530_; uint8_t v___x_2531_; 
v_key_2529_ = lean_ctor_get(v___x_2528_, 0);
v_val_2530_ = lean_ctor_get(v___x_2528_, 1);
v___x_2531_ = lean_name_eq(v_x_2522_, v_key_2529_);
if (v___x_2531_ == 0)
{
lean_object* v___x_2532_; 
v___x_2532_ = lean_box(0);
return v___x_2532_;
}
else
{
lean_object* v___x_2533_; 
lean_inc(v_val_2530_);
v___x_2533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2533_, 0, v_val_2530_);
return v___x_2533_;
}
}
case 1:
{
lean_object* v_node_2534_; size_t v___x_2535_; size_t v___x_2536_; 
v_node_2534_ = lean_ctor_get(v___x_2528_, 0);
v___x_2535_ = ((size_t)5ULL);
v___x_2536_ = lean_usize_shift_right(v_x_2521_, v___x_2535_);
v_x_2520_ = v_node_2534_;
v_x_2521_ = v___x_2536_;
goto _start;
}
default: 
{
lean_object* v___x_2538_; 
v___x_2538_ = lean_box(0);
return v___x_2538_;
}
}
}
else
{
lean_object* v_ks_2539_; lean_object* v_vs_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; 
v_ks_2539_ = lean_ctor_get(v_x_2520_, 0);
v_vs_2540_ = lean_ctor_get(v_x_2520_, 1);
v___x_2541_ = lean_unsigned_to_nat(0u);
v___x_2542_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg(v_ks_2539_, v_vs_2540_, v___x_2541_, v_x_2522_);
return v___x_2542_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_x_2543_, lean_object* v_x_2544_, lean_object* v_x_2545_){
_start:
{
size_t v_x_10515__boxed_2546_; lean_object* v_res_2547_; 
v_x_10515__boxed_2546_ = lean_unbox_usize(v_x_2544_);
lean_dec(v_x_2544_);
v_res_2547_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg(v_x_2543_, v_x_10515__boxed_2546_, v_x_2545_);
lean_dec(v_x_2545_);
lean_dec_ref(v_x_2543_);
return v_res_2547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg(lean_object* v_x_2548_, lean_object* v_x_2549_){
_start:
{
uint64_t v___y_2551_; 
if (lean_obj_tag(v_x_2549_) == 0)
{
uint64_t v___x_2554_; 
v___x_2554_ = 1723ULL;
v___y_2551_ = v___x_2554_;
goto v___jp_2550_;
}
else
{
uint64_t v_hash_2555_; 
v_hash_2555_ = lean_ctor_get_uint64(v_x_2549_, sizeof(void*)*2);
v___y_2551_ = v_hash_2555_;
goto v___jp_2550_;
}
v___jp_2550_:
{
size_t v___x_2552_; lean_object* v___x_2553_; 
v___x_2552_ = lean_uint64_to_usize(v___y_2551_);
v___x_2553_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg(v_x_2548_, v___x_2552_, v_x_2549_);
return v___x_2553_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg___boxed(lean_object* v_x_2556_, lean_object* v_x_2557_){
_start:
{
lean_object* v_res_2558_; 
v_res_2558_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg(v_x_2556_, v_x_2557_);
lean_dec(v_x_2557_);
lean_dec_ref(v_x_2556_);
return v_res_2558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0(lean_object* v_oldCounters_2559_, lean_object* v_x_2560_, lean_object* v_____s_2561_){
_start:
{
lean_object* v_fst_2562_; lean_object* v_snd_2563_; lean_object* v___x_2564_; 
v_fst_2562_ = lean_ctor_get(v_x_2560_, 0);
lean_inc(v_fst_2562_);
v_snd_2563_ = lean_ctor_get(v_x_2560_, 1);
lean_inc(v_snd_2563_);
lean_dec_ref(v_x_2560_);
v___x_2564_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg(v_oldCounters_2559_, v_fst_2562_);
if (lean_obj_tag(v___x_2564_) == 1)
{
lean_object* v_val_2565_; lean_object* v___x_2567_; uint8_t v_isShared_2568_; uint8_t v_isSharedCheck_2574_; 
v_val_2565_ = lean_ctor_get(v___x_2564_, 0);
v_isSharedCheck_2574_ = !lean_is_exclusive(v___x_2564_);
if (v_isSharedCheck_2574_ == 0)
{
v___x_2567_ = v___x_2564_;
v_isShared_2568_ = v_isSharedCheck_2574_;
goto v_resetjp_2566_;
}
else
{
lean_inc(v_val_2565_);
lean_dec(v___x_2564_);
v___x_2567_ = lean_box(0);
v_isShared_2568_ = v_isSharedCheck_2574_;
goto v_resetjp_2566_;
}
v_resetjp_2566_:
{
lean_object* v___x_2569_; lean_object* v_result_2570_; lean_object* v___x_2572_; 
v___x_2569_ = lean_nat_sub(v_snd_2563_, v_val_2565_);
lean_dec(v_val_2565_);
lean_dec(v_snd_2563_);
v_result_2570_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4___redArg(v_____s_2561_, v_fst_2562_, v___x_2569_);
if (v_isShared_2568_ == 0)
{
lean_ctor_set(v___x_2567_, 0, v_result_2570_);
v___x_2572_ = v___x_2567_;
goto v_reusejp_2571_;
}
else
{
lean_object* v_reuseFailAlloc_2573_; 
v_reuseFailAlloc_2573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2573_, 0, v_result_2570_);
v___x_2572_ = v_reuseFailAlloc_2573_;
goto v_reusejp_2571_;
}
v_reusejp_2571_:
{
return v___x_2572_;
}
}
}
else
{
lean_object* v_result_2575_; lean_object* v___x_2576_; 
lean_dec(v___x_2564_);
v_result_2575_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4___redArg(v_____s_2561_, v_fst_2562_, v_snd_2563_);
v___x_2576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2576_, 0, v_result_2575_);
return v___x_2576_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0___boxed(lean_object* v_oldCounters_2577_, lean_object* v_x_2578_, lean_object* v_____s_2579_){
_start:
{
lean_object* v_res_2580_; 
v_res_2580_ = lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0(v_oldCounters_2577_, v_x_2578_, v_____s_2579_);
lean_dec_ref(v_oldCounters_2577_);
return v_res_2580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg___lam__0(lean_object* v_f_2581_, lean_object* v_s_2582_, lean_object* v_a_2583_, lean_object* v_b_2584_){
_start:
{
lean_object* v___x_2585_; lean_object* v___x_2586_; 
v___x_2585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2585_, 0, v_a_2583_);
lean_ctor_set(v___x_2585_, 1, v_b_2584_);
v___x_2586_ = lean_apply_2(v_f_2581_, v___x_2585_, v_s_2582_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2587_; lean_object* v___x_2589_; uint8_t v_isShared_2590_; uint8_t v_isSharedCheck_2594_; 
v_a_2587_ = lean_ctor_get(v___x_2586_, 0);
v_isSharedCheck_2594_ = !lean_is_exclusive(v___x_2586_);
if (v_isSharedCheck_2594_ == 0)
{
v___x_2589_ = v___x_2586_;
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
else
{
lean_inc(v_a_2587_);
lean_dec(v___x_2586_);
v___x_2589_ = lean_box(0);
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
v_resetjp_2588_:
{
lean_object* v___x_2592_; 
if (v_isShared_2590_ == 0)
{
v___x_2592_ = v___x_2589_;
goto v_reusejp_2591_;
}
else
{
lean_object* v_reuseFailAlloc_2593_; 
v_reuseFailAlloc_2593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2593_, 0, v_a_2587_);
v___x_2592_ = v_reuseFailAlloc_2593_;
goto v_reusejp_2591_;
}
v_reusejp_2591_:
{
return v___x_2592_;
}
}
}
else
{
lean_object* v_a_2595_; lean_object* v___x_2597_; uint8_t v_isShared_2598_; uint8_t v_isSharedCheck_2602_; 
v_a_2595_ = lean_ctor_get(v___x_2586_, 0);
v_isSharedCheck_2602_ = !lean_is_exclusive(v___x_2586_);
if (v_isSharedCheck_2602_ == 0)
{
v___x_2597_ = v___x_2586_;
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
else
{
lean_inc(v_a_2595_);
lean_dec(v___x_2586_);
v___x_2597_ = lean_box(0);
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
v_resetjp_2596_:
{
lean_object* v___x_2600_; 
if (v_isShared_2598_ == 0)
{
v___x_2600_ = v___x_2597_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2601_; 
v_reuseFailAlloc_2601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2601_, 0, v_a_2595_);
v___x_2600_ = v_reuseFailAlloc_2601_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
return v___x_2600_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg(lean_object* v_f_2603_, lean_object* v_keys_2604_, lean_object* v_vals_2605_, lean_object* v_i_2606_, lean_object* v_acc_2607_){
_start:
{
lean_object* v___x_2608_; uint8_t v___x_2609_; 
v___x_2608_ = lean_array_get_size(v_keys_2604_);
v___x_2609_ = lean_nat_dec_lt(v_i_2606_, v___x_2608_);
if (v___x_2609_ == 0)
{
lean_object* v___x_2610_; 
lean_dec(v_i_2606_);
lean_dec_ref(v_f_2603_);
v___x_2610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2610_, 0, v_acc_2607_);
return v___x_2610_;
}
else
{
lean_object* v_k_2611_; lean_object* v_v_2612_; lean_object* v___x_2613_; 
v_k_2611_ = lean_array_fget_borrowed(v_keys_2604_, v_i_2606_);
v_v_2612_ = lean_array_fget_borrowed(v_vals_2605_, v_i_2606_);
lean_inc_ref(v_f_2603_);
lean_inc(v_v_2612_);
lean_inc(v_k_2611_);
v___x_2613_ = lean_apply_3(v_f_2603_, v_acc_2607_, v_k_2611_, v_v_2612_);
if (lean_obj_tag(v___x_2613_) == 0)
{
lean_dec(v_i_2606_);
lean_dec_ref(v_f_2603_);
return v___x_2613_;
}
else
{
lean_object* v_a_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; 
v_a_2614_ = lean_ctor_get(v___x_2613_, 0);
lean_inc(v_a_2614_);
lean_dec_ref_known(v___x_2613_, 1);
v___x_2615_ = lean_unsigned_to_nat(1u);
v___x_2616_ = lean_nat_add(v_i_2606_, v___x_2615_);
lean_dec(v_i_2606_);
v_i_2606_ = v___x_2616_;
v_acc_2607_ = v_a_2614_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg___boxed(lean_object* v_f_2618_, lean_object* v_keys_2619_, lean_object* v_vals_2620_, lean_object* v_i_2621_, lean_object* v_acc_2622_){
_start:
{
lean_object* v_res_2623_; 
v_res_2623_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg(v_f_2618_, v_keys_2619_, v_vals_2620_, v_i_2621_, v_acc_2622_);
lean_dec_ref(v_vals_2620_);
lean_dec_ref(v_keys_2619_);
return v_res_2623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(lean_object* v_f_2624_, lean_object* v_x_2625_, lean_object* v_x_2626_){
_start:
{
if (lean_obj_tag(v_x_2625_) == 0)
{
lean_object* v_es_2627_; lean_object* v___x_2629_; uint8_t v_isShared_2630_; uint8_t v_isSharedCheck_2647_; 
v_es_2627_ = lean_ctor_get(v_x_2625_, 0);
v_isSharedCheck_2647_ = !lean_is_exclusive(v_x_2625_);
if (v_isSharedCheck_2647_ == 0)
{
v___x_2629_ = v_x_2625_;
v_isShared_2630_ = v_isSharedCheck_2647_;
goto v_resetjp_2628_;
}
else
{
lean_inc(v_es_2627_);
lean_dec(v_x_2625_);
v___x_2629_ = lean_box(0);
v_isShared_2630_ = v_isSharedCheck_2647_;
goto v_resetjp_2628_;
}
v_resetjp_2628_:
{
lean_object* v___x_2631_; lean_object* v___x_2632_; uint8_t v___x_2633_; 
v___x_2631_ = lean_unsigned_to_nat(0u);
v___x_2632_ = lean_array_get_size(v_es_2627_);
v___x_2633_ = lean_nat_dec_lt(v___x_2631_, v___x_2632_);
if (v___x_2633_ == 0)
{
lean_object* v___x_2635_; 
lean_dec_ref(v_es_2627_);
lean_dec_ref(v_f_2624_);
if (v_isShared_2630_ == 0)
{
lean_ctor_set_tag(v___x_2629_, 1);
lean_ctor_set(v___x_2629_, 0, v_x_2626_);
v___x_2635_ = v___x_2629_;
goto v_reusejp_2634_;
}
else
{
lean_object* v_reuseFailAlloc_2636_; 
v_reuseFailAlloc_2636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2636_, 0, v_x_2626_);
v___x_2635_ = v_reuseFailAlloc_2636_;
goto v_reusejp_2634_;
}
v_reusejp_2634_:
{
return v___x_2635_;
}
}
else
{
uint8_t v___x_2637_; 
v___x_2637_ = lean_nat_dec_le(v___x_2632_, v___x_2632_);
if (v___x_2637_ == 0)
{
if (v___x_2633_ == 0)
{
lean_object* v___x_2639_; 
lean_dec_ref(v_es_2627_);
lean_dec_ref(v_f_2624_);
if (v_isShared_2630_ == 0)
{
lean_ctor_set_tag(v___x_2629_, 1);
lean_ctor_set(v___x_2629_, 0, v_x_2626_);
v___x_2639_ = v___x_2629_;
goto v_reusejp_2638_;
}
else
{
lean_object* v_reuseFailAlloc_2640_; 
v_reuseFailAlloc_2640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2640_, 0, v_x_2626_);
v___x_2639_ = v_reuseFailAlloc_2640_;
goto v_reusejp_2638_;
}
v_reusejp_2638_:
{
return v___x_2639_;
}
}
else
{
size_t v___x_2641_; size_t v___x_2642_; lean_object* v___x_2643_; 
lean_del_object(v___x_2629_);
v___x_2641_ = ((size_t)0ULL);
v___x_2642_ = lean_usize_of_nat(v___x_2632_);
v___x_2643_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(v_f_2624_, v_es_2627_, v___x_2641_, v___x_2642_, v_x_2626_);
lean_dec_ref(v_es_2627_);
return v___x_2643_;
}
}
else
{
size_t v___x_2644_; size_t v___x_2645_; lean_object* v___x_2646_; 
lean_del_object(v___x_2629_);
v___x_2644_ = ((size_t)0ULL);
v___x_2645_ = lean_usize_of_nat(v___x_2632_);
v___x_2646_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(v_f_2624_, v_es_2627_, v___x_2644_, v___x_2645_, v_x_2626_);
lean_dec_ref(v_es_2627_);
return v___x_2646_;
}
}
}
}
else
{
lean_object* v_ks_2648_; lean_object* v_vs_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; 
v_ks_2648_ = lean_ctor_get(v_x_2625_, 0);
lean_inc_ref(v_ks_2648_);
v_vs_2649_ = lean_ctor_get(v_x_2625_, 1);
lean_inc_ref(v_vs_2649_);
lean_dec_ref_known(v_x_2625_, 2);
v___x_2650_ = lean_unsigned_to_nat(0u);
v___x_2651_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg(v_f_2624_, v_ks_2648_, v_vs_2649_, v___x_2650_, v_x_2626_);
lean_dec_ref(v_vs_2649_);
lean_dec_ref(v_ks_2648_);
return v___x_2651_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(lean_object* v_f_2652_, lean_object* v_as_2653_, size_t v_i_2654_, size_t v_stop_2655_, lean_object* v_b_2656_){
_start:
{
lean_object* v_a_2658_; lean_object* v___y_2663_; uint8_t v___x_2665_; 
v___x_2665_ = lean_usize_dec_eq(v_i_2654_, v_stop_2655_);
if (v___x_2665_ == 0)
{
lean_object* v___x_2666_; 
v___x_2666_ = lean_array_uget_borrowed(v_as_2653_, v_i_2654_);
switch(lean_obj_tag(v___x_2666_))
{
case 0:
{
lean_object* v_key_2667_; lean_object* v_val_2668_; lean_object* v___x_2669_; 
v_key_2667_ = lean_ctor_get(v___x_2666_, 0);
v_val_2668_ = lean_ctor_get(v___x_2666_, 1);
lean_inc_ref(v_f_2652_);
lean_inc(v_val_2668_);
lean_inc(v_key_2667_);
v___x_2669_ = lean_apply_3(v_f_2652_, v_b_2656_, v_key_2667_, v_val_2668_);
v___y_2663_ = v___x_2669_;
goto v___jp_2662_;
}
case 1:
{
lean_object* v_node_2670_; lean_object* v___x_2671_; 
v_node_2670_ = lean_ctor_get(v___x_2666_, 0);
lean_inc(v_node_2670_);
lean_inc_ref(v_f_2652_);
v___x_2671_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(v_f_2652_, v_node_2670_, v_b_2656_);
v___y_2663_ = v___x_2671_;
goto v___jp_2662_;
}
default: 
{
v_a_2658_ = v_b_2656_;
goto v___jp_2657_;
}
}
}
else
{
lean_object* v___x_2672_; 
lean_dec_ref(v_f_2652_);
v___x_2672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2672_, 0, v_b_2656_);
return v___x_2672_;
}
v___jp_2657_:
{
size_t v___x_2659_; size_t v___x_2660_; 
v___x_2659_ = ((size_t)1ULL);
v___x_2660_ = lean_usize_add(v_i_2654_, v___x_2659_);
v_i_2654_ = v___x_2660_;
v_b_2656_ = v_a_2658_;
goto _start;
}
v___jp_2662_:
{
if (lean_obj_tag(v___y_2663_) == 0)
{
lean_dec_ref(v_f_2652_);
return v___y_2663_;
}
else
{
lean_object* v_a_2664_; 
v_a_2664_ = lean_ctor_get(v___y_2663_, 0);
lean_inc(v_a_2664_);
lean_dec_ref_known(v___y_2663_, 1);
v_a_2658_ = v_a_2664_;
goto v___jp_2657_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg___boxed(lean_object* v_f_2673_, lean_object* v_as_2674_, lean_object* v_i_2675_, lean_object* v_stop_2676_, lean_object* v_b_2677_){
_start:
{
size_t v_i_boxed_2678_; size_t v_stop_boxed_2679_; lean_object* v_res_2680_; 
v_i_boxed_2678_ = lean_unbox_usize(v_i_2675_);
lean_dec(v_i_2675_);
v_stop_boxed_2679_ = lean_unbox_usize(v_stop_2676_);
lean_dec(v_stop_2676_);
v_res_2680_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(v_f_2673_, v_as_2674_, v_i_boxed_2678_, v_stop_boxed_2679_, v_b_2677_);
lean_dec_ref(v_as_2674_);
return v_res_2680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg(lean_object* v_map_2681_, lean_object* v_init_2682_, lean_object* v_f_2683_){
_start:
{
lean_object* v___f_2684_; lean_object* v___x_2685_; lean_object* v_a_2686_; 
v___f_2684_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2684_, 0, v_f_2683_);
lean_inc_ref(v_map_2681_);
v___x_2685_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(v___f_2684_, v_map_2681_, v_init_2682_);
v_a_2686_ = lean_ctor_get(v___x_2685_, 0);
lean_inc(v_a_2686_);
lean_dec_ref(v___x_2685_);
return v_a_2686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg___boxed(lean_object* v_map_2687_, lean_object* v_init_2688_, lean_object* v_f_2689_){
_start:
{
lean_object* v_res_2690_; 
v_res_2690_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg(v_map_2687_, v_init_2688_, v_f_2689_);
lean_dec_ref(v_map_2687_);
return v_res_2690_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0(void){
_start:
{
lean_object* v___x_2691_; 
v___x_2691_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2691_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1(void){
_start:
{
lean_object* v___x_2692_; lean_object* v_result_2693_; 
v___x_2692_ = lean_obj_once(&lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0, &lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0_once, _init_lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__0);
v_result_2693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_result_2693_, 0, v___x_2692_);
return v_result_2693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2(lean_object* v_newCounters_2694_, lean_object* v_oldCounters_2695_){
_start:
{
lean_object* v___f_2696_; lean_object* v_result_2697_; lean_object* v___x_2698_; 
v___f_2696_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2696_, 0, v_oldCounters_2695_);
v_result_2697_ = lean_obj_once(&lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1, &lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1_once, _init_lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___closed__1);
v___x_2698_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg(v_newCounters_2694_, v_result_2697_, v___f_2696_);
return v___x_2698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2___boxed(lean_object* v_newCounters_2699_, lean_object* v_oldCounters_2700_){
_start:
{
lean_object* v_res_2701_; 
v_res_2701_ = lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2(v_newCounters_2699_, v_oldCounters_2700_);
lean_dec_ref(v_newCounters_2699_);
return v_res_2701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency(lean_object* v_declType_2704_, lean_object* v_a_2705_, lean_object* v_a_2706_, lean_object* v_a_2707_, lean_object* v_a_2708_){
_start:
{
lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v_diag_2712_; lean_object* v_a_2714_; lean_object* v___y_2731_; lean_object* v___y_2732_; lean_object* v___y_2733_; uint8_t v___y_2734_; lean_object* v___y_2747_; uint8_t v___y_2748_; lean_object* v_fileName_2751_; lean_object* v_fileMap_2752_; lean_object* v_options_2753_; lean_object* v_currRecDepth_2754_; lean_object* v_ref_2755_; lean_object* v_currNamespace_2756_; lean_object* v_openDecls_2757_; lean_object* v_initHeartbeats_2758_; lean_object* v_maxHeartbeats_2759_; lean_object* v_quotContext_2760_; lean_object* v_currMacroScope_2761_; lean_object* v_cancelTk_x3f_2762_; uint8_t v_suppressElabErrors_2763_; lean_object* v_inheritedTraceOptions_2764_; lean_object* v_env_2765_; uint8_t v___x_2766_; lean_object* v___x_2767_; uint8_t v___x_2768_; lean_object* v___x_2769_; uint8_t v___x_2770_; lean_object* v_fileName_2772_; lean_object* v_fileMap_2773_; lean_object* v_currRecDepth_2774_; lean_object* v_ref_2775_; lean_object* v_currNamespace_2776_; lean_object* v_openDecls_2777_; lean_object* v_initHeartbeats_2778_; lean_object* v_maxHeartbeats_2779_; lean_object* v_quotContext_2780_; lean_object* v_currMacroScope_2781_; lean_object* v_cancelTk_x3f_2782_; uint8_t v_suppressElabErrors_2783_; lean_object* v_inheritedTraceOptions_2784_; lean_object* v___y_2785_; uint8_t v___y_2817_; uint8_t v___x_2838_; 
v___x_2710_ = lean_st_ref_get(v_a_2706_);
v___x_2711_ = lean_st_ref_get(v_a_2708_);
v_diag_2712_ = lean_ctor_get(v___x_2710_, 4);
lean_inc_ref(v_diag_2712_);
lean_dec(v___x_2710_);
v_fileName_2751_ = lean_ctor_get(v_a_2707_, 0);
v_fileMap_2752_ = lean_ctor_get(v_a_2707_, 1);
v_options_2753_ = lean_ctor_get(v_a_2707_, 2);
v_currRecDepth_2754_ = lean_ctor_get(v_a_2707_, 3);
v_ref_2755_ = lean_ctor_get(v_a_2707_, 5);
v_currNamespace_2756_ = lean_ctor_get(v_a_2707_, 6);
v_openDecls_2757_ = lean_ctor_get(v_a_2707_, 7);
v_initHeartbeats_2758_ = lean_ctor_get(v_a_2707_, 8);
v_maxHeartbeats_2759_ = lean_ctor_get(v_a_2707_, 9);
v_quotContext_2760_ = lean_ctor_get(v_a_2707_, 10);
v_currMacroScope_2761_ = lean_ctor_get(v_a_2707_, 11);
v_cancelTk_x3f_2762_ = lean_ctor_get(v_a_2707_, 12);
v_suppressElabErrors_2763_ = lean_ctor_get_uint8(v_a_2707_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2764_ = lean_ctor_get(v_a_2707_, 13);
v_env_2765_ = lean_ctor_get(v___x_2711_, 0);
lean_inc_ref(v_env_2765_);
lean_dec(v___x_2711_);
v___x_2766_ = 1;
v___x_2767_ = l_Lean_diagnostics;
v___x_2768_ = 1;
lean_inc_ref(v_options_2753_);
v___x_2769_ = lp_mathlib_Lean_Option_set___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__0(v_options_2753_, v___x_2767_, v___x_2768_);
v___x_2770_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v___x_2769_, v___x_2767_);
v___x_2838_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_2765_);
lean_dec_ref(v_env_2765_);
if (v___x_2838_ == 0)
{
if (v___x_2770_ == 0)
{
v_fileName_2772_ = v_fileName_2751_;
v_fileMap_2773_ = v_fileMap_2752_;
v_currRecDepth_2774_ = v_currRecDepth_2754_;
v_ref_2775_ = v_ref_2755_;
v_currNamespace_2776_ = v_currNamespace_2756_;
v_openDecls_2777_ = v_openDecls_2757_;
v_initHeartbeats_2778_ = v_initHeartbeats_2758_;
v_maxHeartbeats_2779_ = v_maxHeartbeats_2759_;
v_quotContext_2780_ = v_quotContext_2760_;
v_currMacroScope_2781_ = v_currMacroScope_2761_;
v_cancelTk_x3f_2782_ = v_cancelTk_x3f_2762_;
v_suppressElabErrors_2783_ = v_suppressElabErrors_2763_;
v_inheritedTraceOptions_2784_ = v_inheritedTraceOptions_2764_;
v___y_2785_ = v_a_2708_;
goto v___jp_2771_;
}
else
{
v___y_2817_ = v___x_2838_;
goto v___jp_2816_;
}
}
else
{
v___y_2817_ = v___x_2770_;
goto v___jp_2816_;
}
v___jp_2713_:
{
lean_object* v___x_2715_; lean_object* v_mctx_2716_; lean_object* v_cache_2717_; lean_object* v_zetaDeltaFVarIds_2718_; lean_object* v_postponed_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2728_; 
v___x_2715_ = lean_st_ref_take(v_a_2706_);
v_mctx_2716_ = lean_ctor_get(v___x_2715_, 0);
v_cache_2717_ = lean_ctor_get(v___x_2715_, 1);
v_zetaDeltaFVarIds_2718_ = lean_ctor_get(v___x_2715_, 2);
v_postponed_2719_ = lean_ctor_get(v___x_2715_, 3);
v_isSharedCheck_2728_ = !lean_is_exclusive(v___x_2715_);
if (v_isSharedCheck_2728_ == 0)
{
lean_object* v_unused_2729_; 
v_unused_2729_ = lean_ctor_get(v___x_2715_, 4);
lean_dec(v_unused_2729_);
v___x_2721_ = v___x_2715_;
v_isShared_2722_ = v_isSharedCheck_2728_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_postponed_2719_);
lean_inc(v_zetaDeltaFVarIds_2718_);
lean_inc(v_cache_2717_);
lean_inc(v_mctx_2716_);
lean_dec(v___x_2715_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2728_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
lean_object* v___x_2724_; 
if (v_isShared_2722_ == 0)
{
lean_ctor_set(v___x_2721_, 4, v_diag_2712_);
v___x_2724_ = v___x_2721_;
goto v_reusejp_2723_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v_mctx_2716_);
lean_ctor_set(v_reuseFailAlloc_2727_, 1, v_cache_2717_);
lean_ctor_set(v_reuseFailAlloc_2727_, 2, v_zetaDeltaFVarIds_2718_);
lean_ctor_set(v_reuseFailAlloc_2727_, 3, v_postponed_2719_);
lean_ctor_set(v_reuseFailAlloc_2727_, 4, v_diag_2712_);
v___x_2724_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2723_;
}
v_reusejp_2723_:
{
lean_object* v___x_2725_; lean_object* v___x_2726_; 
v___x_2725_ = lean_st_ref_set(v_a_2706_, v___x_2724_);
v___x_2726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2726_, 0, v_a_2714_);
return v___x_2726_;
}
}
}
v___jp_2730_:
{
if (v___y_2734_ == 0)
{
lean_object* v___x_2735_; lean_object* v___x_2736_; lean_object* v_diag_2737_; lean_object* v_unfoldCounter_2738_; lean_object* v_env_2739_; lean_object* v___x_2740_; lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; 
lean_dec_ref(v___y_2732_);
v___x_2735_ = lean_st_ref_get(v_a_2706_);
v___x_2736_ = lean_st_ref_get(v___y_2731_);
v_diag_2737_ = lean_ctor_get(v___x_2735_, 4);
lean_inc_ref(v_diag_2737_);
lean_dec(v___x_2735_);
v_unfoldCounter_2738_ = lean_ctor_get(v_diag_2737_, 0);
lean_inc_ref(v_unfoldCounter_2738_);
lean_dec_ref(v_diag_2737_);
v_env_2739_ = lean_ctor_get(v___x_2736_, 0);
lean_inc_ref(v_env_2739_);
lean_dec(v___x_2736_);
v___x_2740_ = lp_mathlib_Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2(v___y_2733_, v_unfoldCounter_2738_);
lean_dec_ref(v___y_2733_);
v___x_2741_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg(v___x_2740_);
lean_dec_ref(v___x_2740_);
v___x_2742_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___closed__0));
v___x_2743_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__4(v_env_2739_, v___x_2741_, v___x_2742_);
v___x_2744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2744_, 0, v___x_2743_);
v_a_2714_ = v___x_2744_;
goto v___jp_2713_;
}
else
{
lean_object* v___x_2745_; 
lean_dec_ref(v___y_2733_);
lean_dec_ref(v_diag_2712_);
v___x_2745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2745_, 0, v___y_2732_);
return v___x_2745_;
}
}
v___jp_2746_:
{
if (v___y_2748_ == 0)
{
lean_object* v___x_2749_; 
lean_dec_ref(v___y_2747_);
v___x_2749_ = lean_box(0);
v_a_2714_ = v___x_2749_;
goto v___jp_2713_;
}
else
{
lean_object* v___x_2750_; 
lean_dec_ref(v_diag_2712_);
v___x_2750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2750_, 0, v___y_2747_);
return v___x_2750_;
}
}
v___jp_2771_:
{
lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; 
v___x_2786_ = l_Lean_maxRecDepth;
v___x_2787_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__1(v___x_2769_, v___x_2786_);
lean_inc_ref(v_inheritedTraceOptions_2784_);
lean_inc(v_cancelTk_x3f_2782_);
lean_inc(v_currMacroScope_2781_);
lean_inc(v_quotContext_2780_);
lean_inc(v_maxHeartbeats_2779_);
lean_inc(v_initHeartbeats_2778_);
lean_inc(v_openDecls_2777_);
lean_inc(v_currNamespace_2776_);
lean_inc(v_ref_2775_);
lean_inc(v_currRecDepth_2774_);
lean_inc_ref(v_fileMap_2773_);
lean_inc_ref(v_fileName_2772_);
v___x_2788_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2788_, 0, v_fileName_2772_);
lean_ctor_set(v___x_2788_, 1, v_fileMap_2773_);
lean_ctor_set(v___x_2788_, 2, v___x_2769_);
lean_ctor_set(v___x_2788_, 3, v_currRecDepth_2774_);
lean_ctor_set(v___x_2788_, 4, v___x_2787_);
lean_ctor_set(v___x_2788_, 5, v_ref_2775_);
lean_ctor_set(v___x_2788_, 6, v_currNamespace_2776_);
lean_ctor_set(v___x_2788_, 7, v_openDecls_2777_);
lean_ctor_set(v___x_2788_, 8, v_initHeartbeats_2778_);
lean_ctor_set(v___x_2788_, 9, v_maxHeartbeats_2779_);
lean_ctor_set(v___x_2788_, 10, v_quotContext_2780_);
lean_ctor_set(v___x_2788_, 11, v_currMacroScope_2781_);
lean_ctor_set(v___x_2788_, 12, v_cancelTk_x3f_2782_);
lean_ctor_set(v___x_2788_, 13, v_inheritedTraceOptions_2784_);
lean_ctor_set_uint8(v___x_2788_, sizeof(void*)*14, v___x_2770_);
lean_ctor_set_uint8(v___x_2788_, sizeof(void*)*14 + 1, v_suppressElabErrors_2783_);
lean_inc_ref(v_declType_2704_);
v___x_2789_ = l_Lean_Meta_check(v_declType_2704_, v___x_2766_, v_a_2705_, v_a_2706_, v___x_2788_, v___y_2785_);
if (lean_obj_tag(v___x_2789_) == 0)
{
lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v_mctx_2792_; lean_object* v_cache_2793_; lean_object* v_zetaDeltaFVarIds_2794_; lean_object* v_postponed_2795_; lean_object* v___x_2797_; uint8_t v_isShared_2798_; uint8_t v_isSharedCheck_2811_; 
lean_dec_ref_known(v___x_2789_, 1);
v___x_2790_ = lean_st_ref_get(v_a_2706_);
v___x_2791_ = lean_st_ref_take(v_a_2706_);
v_mctx_2792_ = lean_ctor_get(v___x_2791_, 0);
v_cache_2793_ = lean_ctor_get(v___x_2791_, 1);
v_zetaDeltaFVarIds_2794_ = lean_ctor_get(v___x_2791_, 2);
v_postponed_2795_ = lean_ctor_get(v___x_2791_, 3);
v_isSharedCheck_2811_ = !lean_is_exclusive(v___x_2791_);
if (v_isSharedCheck_2811_ == 0)
{
lean_object* v_unused_2812_; 
v_unused_2812_ = lean_ctor_get(v___x_2791_, 4);
lean_dec(v_unused_2812_);
v___x_2797_ = v___x_2791_;
v_isShared_2798_ = v_isSharedCheck_2811_;
goto v_resetjp_2796_;
}
else
{
lean_inc(v_postponed_2795_);
lean_inc(v_zetaDeltaFVarIds_2794_);
lean_inc(v_cache_2793_);
lean_inc(v_mctx_2792_);
lean_dec(v___x_2791_);
v___x_2797_ = lean_box(0);
v_isShared_2798_ = v_isSharedCheck_2811_;
goto v_resetjp_2796_;
}
v_resetjp_2796_:
{
lean_object* v___x_2800_; 
lean_inc_ref(v_diag_2712_);
if (v_isShared_2798_ == 0)
{
lean_ctor_set(v___x_2797_, 4, v_diag_2712_);
v___x_2800_ = v___x_2797_;
goto v_reusejp_2799_;
}
else
{
lean_object* v_reuseFailAlloc_2810_; 
v_reuseFailAlloc_2810_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2810_, 0, v_mctx_2792_);
lean_ctor_set(v_reuseFailAlloc_2810_, 1, v_cache_2793_);
lean_ctor_set(v_reuseFailAlloc_2810_, 2, v_zetaDeltaFVarIds_2794_);
lean_ctor_set(v_reuseFailAlloc_2810_, 3, v_postponed_2795_);
lean_ctor_set(v_reuseFailAlloc_2810_, 4, v_diag_2712_);
v___x_2800_ = v_reuseFailAlloc_2810_;
goto v_reusejp_2799_;
}
v_reusejp_2799_:
{
lean_object* v___x_2801_; uint8_t v___x_2802_; lean_object* v___x_2803_; 
v___x_2801_ = lean_st_ref_set(v_a_2706_, v___x_2800_);
v___x_2802_ = 5;
v___x_2803_ = l_Lean_Meta_check(v_declType_2704_, v___x_2802_, v_a_2705_, v_a_2706_, v___x_2788_, v___y_2785_);
lean_dec_ref_known(v___x_2788_, 14);
if (lean_obj_tag(v___x_2803_) == 0)
{
lean_object* v___x_2804_; 
lean_dec_ref_known(v___x_2803_, 1);
lean_dec(v___x_2790_);
v___x_2804_ = lean_box(0);
v_a_2714_ = v___x_2804_;
goto v___jp_2713_;
}
else
{
lean_object* v_diag_2805_; lean_object* v_a_2806_; lean_object* v_unfoldCounter_2807_; uint8_t v___x_2808_; 
v_diag_2805_ = lean_ctor_get(v___x_2790_, 4);
lean_inc_ref(v_diag_2805_);
lean_dec(v___x_2790_);
v_a_2806_ = lean_ctor_get(v___x_2803_, 0);
lean_inc(v_a_2806_);
lean_dec_ref_known(v___x_2803_, 1);
v_unfoldCounter_2807_ = lean_ctor_get(v_diag_2805_, 0);
lean_inc_ref(v_unfoldCounter_2807_);
lean_dec_ref(v_diag_2805_);
v___x_2808_ = l_Lean_Exception_isInterrupt(v_a_2806_);
if (v___x_2808_ == 0)
{
uint8_t v___x_2809_; 
lean_inc(v_a_2806_);
v___x_2809_ = l_Lean_Exception_isRuntime(v_a_2806_);
v___y_2731_ = v___y_2785_;
v___y_2732_ = v_a_2806_;
v___y_2733_ = v_unfoldCounter_2807_;
v___y_2734_ = v___x_2809_;
goto v___jp_2730_;
}
else
{
v___y_2731_ = v___y_2785_;
v___y_2732_ = v_a_2806_;
v___y_2733_ = v_unfoldCounter_2807_;
v___y_2734_ = v___x_2808_;
goto v___jp_2730_;
}
}
}
}
}
else
{
lean_object* v_a_2813_; uint8_t v___x_2814_; 
lean_dec_ref_known(v___x_2788_, 14);
lean_dec_ref(v_declType_2704_);
v_a_2813_ = lean_ctor_get(v___x_2789_, 0);
lean_inc(v_a_2813_);
lean_dec_ref_known(v___x_2789_, 1);
v___x_2814_ = l_Lean_Exception_isInterrupt(v_a_2813_);
if (v___x_2814_ == 0)
{
uint8_t v___x_2815_; 
lean_inc(v_a_2813_);
v___x_2815_ = l_Lean_Exception_isRuntime(v_a_2813_);
v___y_2747_ = v_a_2813_;
v___y_2748_ = v___x_2815_;
goto v___jp_2746_;
}
else
{
v___y_2747_ = v_a_2813_;
v___y_2748_ = v___x_2814_;
goto v___jp_2746_;
}
}
}
v___jp_2816_:
{
if (v___y_2817_ == 0)
{
lean_object* v___x_2818_; lean_object* v_env_2819_; lean_object* v_nextMacroScope_2820_; lean_object* v_ngen_2821_; lean_object* v_auxDeclNGen_2822_; lean_object* v_traceState_2823_; lean_object* v_messages_2824_; lean_object* v_infoState_2825_; lean_object* v_snapshotTasks_2826_; lean_object* v___x_2828_; uint8_t v_isShared_2829_; uint8_t v_isSharedCheck_2836_; 
v___x_2818_ = lean_st_ref_take(v_a_2708_);
v_env_2819_ = lean_ctor_get(v___x_2818_, 0);
v_nextMacroScope_2820_ = lean_ctor_get(v___x_2818_, 1);
v_ngen_2821_ = lean_ctor_get(v___x_2818_, 2);
v_auxDeclNGen_2822_ = lean_ctor_get(v___x_2818_, 3);
v_traceState_2823_ = lean_ctor_get(v___x_2818_, 4);
v_messages_2824_ = lean_ctor_get(v___x_2818_, 6);
v_infoState_2825_ = lean_ctor_get(v___x_2818_, 7);
v_snapshotTasks_2826_ = lean_ctor_get(v___x_2818_, 8);
v_isSharedCheck_2836_ = !lean_is_exclusive(v___x_2818_);
if (v_isSharedCheck_2836_ == 0)
{
lean_object* v_unused_2837_; 
v_unused_2837_ = lean_ctor_get(v___x_2818_, 5);
lean_dec(v_unused_2837_);
v___x_2828_ = v___x_2818_;
v_isShared_2829_ = v_isSharedCheck_2836_;
goto v_resetjp_2827_;
}
else
{
lean_inc(v_snapshotTasks_2826_);
lean_inc(v_infoState_2825_);
lean_inc(v_messages_2824_);
lean_inc(v_traceState_2823_);
lean_inc(v_auxDeclNGen_2822_);
lean_inc(v_ngen_2821_);
lean_inc(v_nextMacroScope_2820_);
lean_inc(v_env_2819_);
lean_dec(v___x_2818_);
v___x_2828_ = lean_box(0);
v_isShared_2829_ = v_isSharedCheck_2836_;
goto v_resetjp_2827_;
}
v_resetjp_2827_:
{
lean_object* v___x_2830_; lean_object* v___x_2831_; lean_object* v___x_2833_; 
v___x_2830_ = l_Lean_Kernel_enableDiag(v_env_2819_, v___x_2770_);
v___x_2831_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_2829_ == 0)
{
lean_ctor_set(v___x_2828_, 5, v___x_2831_);
lean_ctor_set(v___x_2828_, 0, v___x_2830_);
v___x_2833_ = v___x_2828_;
goto v_reusejp_2832_;
}
else
{
lean_object* v_reuseFailAlloc_2835_; 
v_reuseFailAlloc_2835_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2835_, 0, v___x_2830_);
lean_ctor_set(v_reuseFailAlloc_2835_, 1, v_nextMacroScope_2820_);
lean_ctor_set(v_reuseFailAlloc_2835_, 2, v_ngen_2821_);
lean_ctor_set(v_reuseFailAlloc_2835_, 3, v_auxDeclNGen_2822_);
lean_ctor_set(v_reuseFailAlloc_2835_, 4, v_traceState_2823_);
lean_ctor_set(v_reuseFailAlloc_2835_, 5, v___x_2831_);
lean_ctor_set(v_reuseFailAlloc_2835_, 6, v_messages_2824_);
lean_ctor_set(v_reuseFailAlloc_2835_, 7, v_infoState_2825_);
lean_ctor_set(v_reuseFailAlloc_2835_, 8, v_snapshotTasks_2826_);
v___x_2833_ = v_reuseFailAlloc_2835_;
goto v_reusejp_2832_;
}
v_reusejp_2832_:
{
lean_object* v___x_2834_; 
v___x_2834_ = lean_st_ref_set(v_a_2708_, v___x_2833_);
v_fileName_2772_ = v_fileName_2751_;
v_fileMap_2773_ = v_fileMap_2752_;
v_currRecDepth_2774_ = v_currRecDepth_2754_;
v_ref_2775_ = v_ref_2755_;
v_currNamespace_2776_ = v_currNamespace_2756_;
v_openDecls_2777_ = v_openDecls_2757_;
v_initHeartbeats_2778_ = v_initHeartbeats_2758_;
v_maxHeartbeats_2779_ = v_maxHeartbeats_2759_;
v_quotContext_2780_ = v_quotContext_2760_;
v_currMacroScope_2781_ = v_currMacroScope_2761_;
v_cancelTk_x3f_2782_ = v_cancelTk_x3f_2762_;
v_suppressElabErrors_2783_ = v_suppressElabErrors_2763_;
v_inheritedTraceOptions_2784_ = v_inheritedTraceOptions_2764_;
v___y_2785_ = v_a_2708_;
goto v___jp_2771_;
}
}
}
else
{
v_fileName_2772_ = v_fileName_2751_;
v_fileMap_2773_ = v_fileMap_2752_;
v_currRecDepth_2774_ = v_currRecDepth_2754_;
v_ref_2775_ = v_ref_2755_;
v_currNamespace_2776_ = v_currNamespace_2756_;
v_openDecls_2777_ = v_openDecls_2757_;
v_initHeartbeats_2778_ = v_initHeartbeats_2758_;
v_maxHeartbeats_2779_ = v_maxHeartbeats_2759_;
v_quotContext_2780_ = v_quotContext_2760_;
v_currMacroScope_2781_ = v_currMacroScope_2761_;
v_cancelTk_x3f_2782_ = v_cancelTk_x3f_2762_;
v_suppressElabErrors_2783_ = v_suppressElabErrors_2763_;
v_inheritedTraceOptions_2784_ = v_inheritedTraceOptions_2764_;
v___y_2785_ = v_a_2708_;
goto v___jp_2771_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency___boxed(lean_object* v_declType_2839_, lean_object* v_a_2840_, lean_object* v_a_2841_, lean_object* v_a_2842_, lean_object* v_a_2843_, lean_object* v_a_2844_){
_start:
{
lean_object* v_res_2845_; 
v_res_2845_ = lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency(v_declType_2839_, v_a_2840_, v_a_2841_, v_a_2842_, v_a_2843_);
lean_dec(v_a_2843_);
lean_dec_ref(v_a_2842_);
lean_dec(v_a_2841_);
lean_dec_ref(v_a_2840_);
return v_res_2845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3(lean_object* v_00_u03b2_2846_, lean_object* v_m_2847_){
_start:
{
lean_object* v___x_2848_; 
v___x_2848_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___redArg(v_m_2847_);
return v___x_2848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3___boxed(lean_object* v_00_u03b2_2849_, lean_object* v_m_2850_){
_start:
{
lean_object* v_res_2851_; 
v_res_2851_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3(v_00_u03b2_2849_, v_m_2850_);
lean_dec_ref(v_m_2850_);
return v_res_2851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3(lean_object* v_00_u03b2_2852_, lean_object* v_x_2853_, lean_object* v_x_2854_){
_start:
{
lean_object* v___x_2855_; 
v___x_2855_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___redArg(v_x_2853_, v_x_2854_);
return v___x_2855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3___boxed(lean_object* v_00_u03b2_2856_, lean_object* v_x_2857_, lean_object* v_x_2858_){
_start:
{
lean_object* v_res_2859_; 
v_res_2859_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3(v_00_u03b2_2856_, v_x_2857_, v_x_2858_);
lean_dec(v_x_2858_);
lean_dec_ref(v_x_2857_);
return v_res_2859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4(lean_object* v_00_u03b2_2860_, lean_object* v_x_2861_, lean_object* v_x_2862_, lean_object* v_x_2863_){
_start:
{
lean_object* v___x_2864_; 
v___x_2864_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4___redArg(v_x_2861_, v_x_2862_, v_x_2863_);
return v___x_2864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5(lean_object* v_00_u03c3_2865_, lean_object* v_00_u03b2_2866_, lean_object* v_map_2867_, lean_object* v_init_2868_, lean_object* v_f_2869_){
_start:
{
lean_object* v___x_2870_; 
v___x_2870_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___redArg(v_map_2867_, v_init_2868_, v_f_2869_);
return v___x_2870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5___boxed(lean_object* v_00_u03c3_2871_, lean_object* v_00_u03b2_2872_, lean_object* v_map_2873_, lean_object* v_init_2874_, lean_object* v_f_2875_){
_start:
{
lean_object* v_res_2876_; 
v_res_2876_ = lp_mathlib_Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5(v_00_u03c3_2871_, v_00_u03b2_2872_, v_map_2873_, v_init_2874_, v_f_2875_);
lean_dec_ref(v_map_2873_);
return v_res_2876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7(lean_object* v_00_u03c3_2877_, lean_object* v_00_u03b2_2878_, lean_object* v_map_2879_, lean_object* v_f_2880_, lean_object* v_init_2881_){
_start:
{
lean_object* v___x_2882_; 
v___x_2882_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___redArg(v_map_2879_, v_f_2880_, v_init_2881_);
return v___x_2882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7___boxed(lean_object* v_00_u03c3_2883_, lean_object* v_00_u03b2_2884_, lean_object* v_map_2885_, lean_object* v_f_2886_, lean_object* v_init_2887_){
_start:
{
lean_object* v_res_2888_; 
v_res_2888_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7(v_00_u03c3_2883_, v_00_u03b2_2884_, v_map_2885_, v_f_2886_, v_init_2887_);
lean_dec_ref(v_map_2885_);
return v_res_2888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_2889_, lean_object* v_x_2890_, size_t v_x_2891_, lean_object* v_x_2892_){
_start:
{
lean_object* v___x_2893_; 
v___x_2893_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___redArg(v_x_2890_, v_x_2891_, v_x_2892_);
return v___x_2893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b2_2894_, lean_object* v_x_2895_, lean_object* v_x_2896_, lean_object* v_x_2897_){
_start:
{
size_t v_x_10964__boxed_2898_; lean_object* v_res_2899_; 
v_x_10964__boxed_2898_ = lean_unbox_usize(v_x_2896_);
lean_dec(v_x_2896_);
v_res_2899_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4(v_00_u03b2_2894_, v_x_2895_, v_x_10964__boxed_2898_, v_x_2897_);
lean_dec(v_x_2897_);
lean_dec_ref(v_x_2895_);
return v_res_2899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6(lean_object* v_00_u03b2_2900_, lean_object* v_x_2901_, size_t v_x_2902_, size_t v_x_2903_, lean_object* v_x_2904_, lean_object* v_x_2905_){
_start:
{
lean_object* v___x_2906_; 
v___x_2906_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___redArg(v_x_2901_, v_x_2902_, v_x_2903_, v_x_2904_, v_x_2905_);
return v___x_2906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6___boxed(lean_object* v_00_u03b2_2907_, lean_object* v_x_2908_, lean_object* v_x_2909_, lean_object* v_x_2910_, lean_object* v_x_2911_, lean_object* v_x_2912_){
_start:
{
size_t v_x_10975__boxed_2913_; size_t v_x_10976__boxed_2914_; lean_object* v_res_2915_; 
v_x_10975__boxed_2913_ = lean_unbox_usize(v_x_2909_);
lean_dec(v_x_2909_);
v_x_10976__boxed_2914_ = lean_unbox_usize(v_x_2910_);
lean_dec(v_x_2910_);
v_res_2915_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6(v_00_u03b2_2907_, v_x_2908_, v_x_10975__boxed_2913_, v_x_10976__boxed_2914_, v_x_2911_, v_x_2912_);
return v_res_2915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8___redArg(lean_object* v_map_2916_, lean_object* v_f_2917_, lean_object* v_init_2918_){
_start:
{
lean_object* v___x_2919_; 
v___x_2919_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(v_f_2917_, v_map_2916_, v_init_2918_);
return v___x_2919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8(lean_object* v_00_u03c3_2920_, lean_object* v_00_u03c3_2921_, lean_object* v_00_u03b2_2922_, lean_object* v_map_2923_, lean_object* v_f_2924_, lean_object* v_init_2925_){
_start:
{
lean_object* v___x_2926_; 
v___x_2926_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(v_f_2924_, v_map_2923_, v_init_2925_);
return v___x_2926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___redArg(lean_object* v_map_2927_, lean_object* v_f_2928_, lean_object* v_init_2929_){
_start:
{
lean_object* v___x_2930_; 
v___x_2930_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v_f_2928_, v_map_2927_, v_init_2929_);
return v___x_2930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___redArg___boxed(lean_object* v_map_2931_, lean_object* v_f_2932_, lean_object* v_init_2933_){
_start:
{
lean_object* v_res_2934_; 
v_res_2934_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___redArg(v_map_2931_, v_f_2932_, v_init_2933_);
lean_dec_ref(v_map_2931_);
return v_res_2934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11(lean_object* v_00_u03c3_2935_, lean_object* v_00_u03b2_2936_, lean_object* v_map_2937_, lean_object* v_f_2938_, lean_object* v_init_2939_){
_start:
{
lean_object* v___x_2940_; 
v___x_2940_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v_f_2938_, v_map_2937_, v_init_2939_);
return v___x_2940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11___boxed(lean_object* v_00_u03c3_2941_, lean_object* v_00_u03b2_2942_, lean_object* v_map_2943_, lean_object* v_f_2944_, lean_object* v_init_2945_){
_start:
{
lean_object* v_res_2946_; 
v_res_2946_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11(v_00_u03c3_2941_, v_00_u03b2_2942_, v_map_2943_, v_f_2944_, v_init_2945_);
lean_dec_ref(v_map_2943_);
return v_res_2946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7(lean_object* v_00_u03b2_2947_, lean_object* v_keys_2948_, lean_object* v_vals_2949_, lean_object* v_heq_2950_, lean_object* v_i_2951_, lean_object* v_k_2952_){
_start:
{
lean_object* v___x_2953_; 
v___x_2953_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___redArg(v_keys_2948_, v_vals_2949_, v_i_2951_, v_k_2952_);
return v___x_2953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7___boxed(lean_object* v_00_u03b2_2954_, lean_object* v_keys_2955_, lean_object* v_vals_2956_, lean_object* v_heq_2957_, lean_object* v_i_2958_, lean_object* v_k_2959_){
_start:
{
lean_object* v_res_2960_; 
v_res_2960_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__3_spec__4_spec__7(v_00_u03b2_2954_, v_keys_2955_, v_vals_2956_, v_heq_2957_, v_i_2958_, v_k_2959_);
lean_dec(v_k_2959_);
lean_dec_ref(v_vals_2956_);
lean_dec_ref(v_keys_2955_);
return v_res_2960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10(lean_object* v_00_u03b2_2961_, lean_object* v_n_2962_, lean_object* v_k_2963_, lean_object* v_v_2964_){
_start:
{
lean_object* v___x_2965_; 
v___x_2965_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10___redArg(v_n_2962_, v_k_2963_, v_v_2964_);
return v___x_2965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11(lean_object* v_00_u03b2_2966_, size_t v_depth_2967_, lean_object* v_keys_2968_, lean_object* v_vals_2969_, lean_object* v_heq_2970_, lean_object* v_i_2971_, lean_object* v_entries_2972_){
_start:
{
lean_object* v___x_2973_; 
v___x_2973_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___redArg(v_depth_2967_, v_keys_2968_, v_vals_2969_, v_i_2971_, v_entries_2972_);
return v___x_2973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11___boxed(lean_object* v_00_u03b2_2974_, lean_object* v_depth_2975_, lean_object* v_keys_2976_, lean_object* v_vals_2977_, lean_object* v_heq_2978_, lean_object* v_i_2979_, lean_object* v_entries_2980_){
_start:
{
size_t v_depth_boxed_2981_; lean_object* v_res_2982_; 
v_depth_boxed_2981_ = lean_unbox_usize(v_depth_2975_);
lean_dec(v_depth_2975_);
v_res_2982_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__11(v_00_u03b2_2974_, v_depth_boxed_2981_, v_keys_2976_, v_vals_2977_, v_heq_2978_, v_i_2979_, v_entries_2980_);
lean_dec_ref(v_vals_2977_);
lean_dec_ref(v_keys_2976_);
return v_res_2982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14(lean_object* v_00_u03c3_2983_, lean_object* v_00_u03c3_2984_, lean_object* v_00_u03b1_2985_, lean_object* v_00_u03b2_2986_, lean_object* v_f_2987_, lean_object* v_x_2988_, lean_object* v_x_2989_){
_start:
{
lean_object* v___x_2990_; 
v___x_2990_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14___redArg(v_f_2987_, v_x_2988_, v_x_2989_);
return v___x_2990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17(lean_object* v_00_u03c3_2991_, lean_object* v_00_u03b1_2992_, lean_object* v_00_u03b2_2993_, lean_object* v_f_2994_, lean_object* v_x_2995_, lean_object* v_x_2996_){
_start:
{
lean_object* v___x_2997_; 
v___x_2997_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___redArg(v_f_2994_, v_x_2995_, v_x_2996_);
return v___x_2997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17___boxed(lean_object* v_00_u03c3_2998_, lean_object* v_00_u03b1_2999_, lean_object* v_00_u03b2_3000_, lean_object* v_f_3001_, lean_object* v_x_3002_, lean_object* v_x_3003_){
_start:
{
lean_object* v_res_3004_; 
v_res_3004_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17(v_00_u03c3_2998_, v_00_u03b1_2999_, v_00_u03b2_3000_, v_f_3001_, v_x_3002_, v_x_3003_);
lean_dec_ref(v_x_3002_);
return v_res_3004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13(lean_object* v_00_u03b2_3005_, lean_object* v_x_3006_, lean_object* v_x_3007_, lean_object* v_x_3008_, lean_object* v_x_3009_){
_start:
{
lean_object* v___x_3010_; 
v___x_3010_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__4_spec__6_spec__10_spec__13___redArg(v_x_3006_, v_x_3007_, v_x_3008_, v_x_3009_);
return v___x_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17(lean_object* v_00_u03b1_3011_, lean_object* v_00_u03b2_3012_, lean_object* v_00_u03c3_3013_, lean_object* v_00_u03c3_3014_, lean_object* v_f_3015_, lean_object* v_as_3016_, size_t v_i_3017_, size_t v_stop_3018_, lean_object* v_b_3019_){
_start:
{
lean_object* v___x_3020_; 
v___x_3020_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___redArg(v_f_3015_, v_as_3016_, v_i_3017_, v_stop_3018_, v_b_3019_);
return v___x_3020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17___boxed(lean_object* v_00_u03b1_3021_, lean_object* v_00_u03b2_3022_, lean_object* v_00_u03c3_3023_, lean_object* v_00_u03c3_3024_, lean_object* v_f_3025_, lean_object* v_as_3026_, lean_object* v_i_3027_, lean_object* v_stop_3028_, lean_object* v_b_3029_){
_start:
{
size_t v_i_boxed_3030_; size_t v_stop_boxed_3031_; lean_object* v_res_3032_; 
v_i_boxed_3030_ = lean_unbox_usize(v_i_3027_);
lean_dec(v_i_3027_);
v_stop_boxed_3031_ = lean_unbox_usize(v_stop_3028_);
lean_dec(v_stop_3028_);
v_res_3032_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__17(v_00_u03b1_3021_, v_00_u03b2_3022_, v_00_u03c3_3023_, v_00_u03c3_3024_, v_f_3025_, v_as_3026_, v_i_boxed_3030_, v_stop_boxed_3031_, v_b_3029_);
lean_dec_ref(v_as_3026_);
return v_res_3032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18(lean_object* v_00_u03c3_3033_, lean_object* v_00_u03c3_3034_, lean_object* v_00_u03b1_3035_, lean_object* v_00_u03b2_3036_, lean_object* v_f_3037_, lean_object* v_keys_3038_, lean_object* v_vals_3039_, lean_object* v_heq_3040_, lean_object* v_i_3041_, lean_object* v_acc_3042_){
_start:
{
lean_object* v___x_3043_; 
v___x_3043_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___redArg(v_f_3037_, v_keys_3038_, v_vals_3039_, v_i_3041_, v_acc_3042_);
return v___x_3043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18___boxed(lean_object* v_00_u03c3_3044_, lean_object* v_00_u03c3_3045_, lean_object* v_00_u03b1_3046_, lean_object* v_00_u03b2_3047_, lean_object* v_f_3048_, lean_object* v_keys_3049_, lean_object* v_vals_3050_, lean_object* v_heq_3051_, lean_object* v_i_3052_, lean_object* v_acc_3053_){
_start:
{
lean_object* v_res_3054_; 
v_res_3054_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_Meta_subCounters___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__2_spec__5_spec__8_spec__14_spec__18(v_00_u03c3_3044_, v_00_u03c3_3045_, v_00_u03b1_3046_, v_00_u03b2_3047_, v_f_3048_, v_keys_3049_, v_vals_3050_, v_heq_3051_, v_i_3052_, v_acc_3053_);
lean_dec_ref(v_vals_3050_);
lean_dec_ref(v_keys_3049_);
return v_res_3054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21(lean_object* v_00_u03b1_3055_, lean_object* v_00_u03b2_3056_, lean_object* v_00_u03c3_3057_, lean_object* v_f_3058_, lean_object* v_as_3059_, size_t v_i_3060_, size_t v_stop_3061_, lean_object* v_b_3062_){
_start:
{
lean_object* v___x_3063_; 
v___x_3063_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___redArg(v_f_3058_, v_as_3059_, v_i_3060_, v_stop_3061_, v_b_3062_);
return v___x_3063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21___boxed(lean_object* v_00_u03b1_3064_, lean_object* v_00_u03b2_3065_, lean_object* v_00_u03c3_3066_, lean_object* v_f_3067_, lean_object* v_as_3068_, lean_object* v_i_3069_, lean_object* v_stop_3070_, lean_object* v_b_3071_){
_start:
{
size_t v_i_boxed_3072_; size_t v_stop_boxed_3073_; lean_object* v_res_3074_; 
v_i_boxed_3072_ = lean_unbox_usize(v_i_3069_);
lean_dec(v_i_3069_);
v_stop_boxed_3073_ = lean_unbox_usize(v_stop_3070_);
lean_dec(v_stop_3070_);
v_res_3074_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__21(v_00_u03b1_3064_, v_00_u03b2_3065_, v_00_u03c3_3066_, v_f_3067_, v_as_3068_, v_i_boxed_3072_, v_stop_boxed_3073_, v_b_3071_);
lean_dec_ref(v_as_3068_);
return v_res_3074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22(lean_object* v_00_u03c3_3075_, lean_object* v_00_u03b1_3076_, lean_object* v_00_u03b2_3077_, lean_object* v_f_3078_, lean_object* v_keys_3079_, lean_object* v_vals_3080_, lean_object* v_heq_3081_, lean_object* v_i_3082_, lean_object* v_acc_3083_){
_start:
{
lean_object* v___x_3084_; 
v___x_3084_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___redArg(v_f_3078_, v_keys_3079_, v_vals_3080_, v_i_3082_, v_acc_3083_);
return v___x_3084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22___boxed(lean_object* v_00_u03c3_3085_, lean_object* v_00_u03b1_3086_, lean_object* v_00_u03b2_3087_, lean_object* v_f_3088_, lean_object* v_keys_3089_, lean_object* v_vals_3090_, lean_object* v_heq_3091_, lean_object* v_i_3092_, lean_object* v_acc_3093_){
_start:
{
lean_object* v_res_3094_; 
v_res_3094_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00__private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency_spec__3_spec__7_spec__11_spec__17_spec__22(v_00_u03c3_3085_, v_00_u03b1_3086_, v_00_u03b2_3087_, v_f_3088_, v_keys_3089_, v_vals_3090_, v_heq_3091_, v_i_3092_, v_acc_3093_);
lean_dec_ref(v_vals_3090_);
lean_dec_ref(v_keys_3089_);
return v_res_3094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2(lean_object* v_ref_3095_, lean_object* v_msgData_3096_, uint8_t v_severity_3097_, uint8_t v_isSilent_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_){
_start:
{
lean_object* v___y_3105_; lean_object* v___y_3106_; uint8_t v___y_3107_; lean_object* v___y_3108_; lean_object* v___y_3109_; uint8_t v___y_3110_; lean_object* v___y_3111_; lean_object* v___y_3112_; lean_object* v___y_3113_; lean_object* v___y_3141_; lean_object* v___y_3142_; lean_object* v___y_3143_; uint8_t v___y_3144_; lean_object* v___y_3145_; uint8_t v___y_3146_; uint8_t v___y_3147_; lean_object* v___y_3148_; lean_object* v___y_3166_; lean_object* v___y_3167_; lean_object* v___y_3168_; uint8_t v___y_3169_; lean_object* v___y_3170_; uint8_t v___y_3171_; uint8_t v___y_3172_; lean_object* v___y_3173_; lean_object* v___y_3177_; lean_object* v___y_3178_; lean_object* v___y_3179_; uint8_t v___y_3180_; lean_object* v___y_3181_; uint8_t v___y_3182_; uint8_t v___y_3183_; uint8_t v___x_3188_; lean_object* v___y_3190_; lean_object* v___y_3191_; lean_object* v___y_3192_; lean_object* v___y_3193_; uint8_t v___y_3194_; uint8_t v___y_3195_; uint8_t v___y_3196_; uint8_t v___y_3198_; uint8_t v___x_3213_; 
v___x_3188_ = 2;
v___x_3213_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3097_, v___x_3188_);
if (v___x_3213_ == 0)
{
v___y_3198_ = v___x_3213_;
goto v___jp_3197_;
}
else
{
uint8_t v___x_3214_; 
lean_inc_ref(v_msgData_3096_);
v___x_3214_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_3096_);
v___y_3198_ = v___x_3214_;
goto v___jp_3197_;
}
v___jp_3104_:
{
lean_object* v___x_3114_; lean_object* v_currNamespace_3115_; lean_object* v_openDecls_3116_; lean_object* v_env_3117_; lean_object* v_nextMacroScope_3118_; lean_object* v_ngen_3119_; lean_object* v_auxDeclNGen_3120_; lean_object* v_traceState_3121_; lean_object* v_cache_3122_; lean_object* v_messages_3123_; lean_object* v_infoState_3124_; lean_object* v_snapshotTasks_3125_; lean_object* v___x_3127_; uint8_t v_isShared_3128_; uint8_t v_isSharedCheck_3139_; 
v___x_3114_ = lean_st_ref_take(v___y_3113_);
v_currNamespace_3115_ = lean_ctor_get(v___y_3112_, 6);
v_openDecls_3116_ = lean_ctor_get(v___y_3112_, 7);
v_env_3117_ = lean_ctor_get(v___x_3114_, 0);
v_nextMacroScope_3118_ = lean_ctor_get(v___x_3114_, 1);
v_ngen_3119_ = lean_ctor_get(v___x_3114_, 2);
v_auxDeclNGen_3120_ = lean_ctor_get(v___x_3114_, 3);
v_traceState_3121_ = lean_ctor_get(v___x_3114_, 4);
v_cache_3122_ = lean_ctor_get(v___x_3114_, 5);
v_messages_3123_ = lean_ctor_get(v___x_3114_, 6);
v_infoState_3124_ = lean_ctor_get(v___x_3114_, 7);
v_snapshotTasks_3125_ = lean_ctor_get(v___x_3114_, 8);
v_isSharedCheck_3139_ = !lean_is_exclusive(v___x_3114_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3127_ = v___x_3114_;
v_isShared_3128_ = v_isSharedCheck_3139_;
goto v_resetjp_3126_;
}
else
{
lean_inc(v_snapshotTasks_3125_);
lean_inc(v_infoState_3124_);
lean_inc(v_messages_3123_);
lean_inc(v_cache_3122_);
lean_inc(v_traceState_3121_);
lean_inc(v_auxDeclNGen_3120_);
lean_inc(v_ngen_3119_);
lean_inc(v_nextMacroScope_3118_);
lean_inc(v_env_3117_);
lean_dec(v___x_3114_);
v___x_3127_ = lean_box(0);
v_isShared_3128_ = v_isSharedCheck_3139_;
goto v_resetjp_3126_;
}
v_resetjp_3126_:
{
lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3134_; 
lean_inc(v_openDecls_3116_);
lean_inc(v_currNamespace_3115_);
v___x_3129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3129_, 0, v_currNamespace_3115_);
lean_ctor_set(v___x_3129_, 1, v_openDecls_3116_);
v___x_3130_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3130_, 0, v___x_3129_);
lean_ctor_set(v___x_3130_, 1, v___y_3106_);
lean_inc_ref(v___y_3111_);
lean_inc_ref(v___y_3108_);
v___x_3131_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3131_, 0, v___y_3108_);
lean_ctor_set(v___x_3131_, 1, v___y_3109_);
lean_ctor_set(v___x_3131_, 2, v___y_3105_);
lean_ctor_set(v___x_3131_, 3, v___y_3111_);
lean_ctor_set(v___x_3131_, 4, v___x_3130_);
lean_ctor_set_uint8(v___x_3131_, sizeof(void*)*5, v___y_3110_);
lean_ctor_set_uint8(v___x_3131_, sizeof(void*)*5 + 1, v___y_3107_);
lean_ctor_set_uint8(v___x_3131_, sizeof(void*)*5 + 2, v_isSilent_3098_);
v___x_3132_ = l_Lean_MessageLog_add(v___x_3131_, v_messages_3123_);
if (v_isShared_3128_ == 0)
{
lean_ctor_set(v___x_3127_, 6, v___x_3132_);
v___x_3134_ = v___x_3127_;
goto v_reusejp_3133_;
}
else
{
lean_object* v_reuseFailAlloc_3138_; 
v_reuseFailAlloc_3138_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3138_, 0, v_env_3117_);
lean_ctor_set(v_reuseFailAlloc_3138_, 1, v_nextMacroScope_3118_);
lean_ctor_set(v_reuseFailAlloc_3138_, 2, v_ngen_3119_);
lean_ctor_set(v_reuseFailAlloc_3138_, 3, v_auxDeclNGen_3120_);
lean_ctor_set(v_reuseFailAlloc_3138_, 4, v_traceState_3121_);
lean_ctor_set(v_reuseFailAlloc_3138_, 5, v_cache_3122_);
lean_ctor_set(v_reuseFailAlloc_3138_, 6, v___x_3132_);
lean_ctor_set(v_reuseFailAlloc_3138_, 7, v_infoState_3124_);
lean_ctor_set(v_reuseFailAlloc_3138_, 8, v_snapshotTasks_3125_);
v___x_3134_ = v_reuseFailAlloc_3138_;
goto v_reusejp_3133_;
}
v_reusejp_3133_:
{
lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; 
v___x_3135_ = lean_st_ref_set(v___y_3113_, v___x_3134_);
v___x_3136_ = lean_box(0);
v___x_3137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3137_, 0, v___x_3136_);
return v___x_3137_;
}
}
}
v___jp_3140_:
{
lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v_a_3151_; lean_object* v___x_3153_; uint8_t v_isShared_3154_; uint8_t v_isSharedCheck_3164_; 
v___x_3149_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_3096_);
v___x_3150_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v___x_3149_, v___y_3099_, v___y_3100_, v___y_3101_, v___y_3102_);
v_a_3151_ = lean_ctor_get(v___x_3150_, 0);
v_isSharedCheck_3164_ = !lean_is_exclusive(v___x_3150_);
if (v_isSharedCheck_3164_ == 0)
{
v___x_3153_ = v___x_3150_;
v_isShared_3154_ = v_isSharedCheck_3164_;
goto v_resetjp_3152_;
}
else
{
lean_inc(v_a_3151_);
lean_dec(v___x_3150_);
v___x_3153_ = lean_box(0);
v_isShared_3154_ = v_isSharedCheck_3164_;
goto v_resetjp_3152_;
}
v_resetjp_3152_:
{
lean_object* v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; 
lean_inc_ref_n(v___y_3143_, 2);
v___x_3155_ = l_Lean_FileMap_toPosition(v___y_3143_, v___y_3142_);
lean_dec(v___y_3142_);
v___x_3156_ = l_Lean_FileMap_toPosition(v___y_3143_, v___y_3148_);
lean_dec(v___y_3148_);
v___x_3157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3157_, 0, v___x_3156_);
v___x_3158_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__2_spec__4___redArg___closed__1));
if (v___y_3147_ == 0)
{
lean_del_object(v___x_3153_);
lean_dec_ref(v___y_3141_);
v___y_3105_ = v___x_3157_;
v___y_3106_ = v_a_3151_;
v___y_3107_ = v___y_3144_;
v___y_3108_ = v___y_3145_;
v___y_3109_ = v___x_3155_;
v___y_3110_ = v___y_3146_;
v___y_3111_ = v___x_3158_;
v___y_3112_ = v___y_3101_;
v___y_3113_ = v___y_3102_;
goto v___jp_3104_;
}
else
{
uint8_t v___x_3159_; 
lean_inc(v_a_3151_);
v___x_3159_ = l_Lean_MessageData_hasTag(v___y_3141_, v_a_3151_);
if (v___x_3159_ == 0)
{
lean_object* v___x_3160_; lean_object* v___x_3162_; 
lean_dec_ref_known(v___x_3157_, 1);
lean_dec_ref(v___x_3155_);
lean_dec(v_a_3151_);
v___x_3160_ = lean_box(0);
if (v_isShared_3154_ == 0)
{
lean_ctor_set(v___x_3153_, 0, v___x_3160_);
v___x_3162_ = v___x_3153_;
goto v_reusejp_3161_;
}
else
{
lean_object* v_reuseFailAlloc_3163_; 
v_reuseFailAlloc_3163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3163_, 0, v___x_3160_);
v___x_3162_ = v_reuseFailAlloc_3163_;
goto v_reusejp_3161_;
}
v_reusejp_3161_:
{
return v___x_3162_;
}
}
else
{
lean_del_object(v___x_3153_);
v___y_3105_ = v___x_3157_;
v___y_3106_ = v_a_3151_;
v___y_3107_ = v___y_3144_;
v___y_3108_ = v___y_3145_;
v___y_3109_ = v___x_3155_;
v___y_3110_ = v___y_3146_;
v___y_3111_ = v___x_3158_;
v___y_3112_ = v___y_3101_;
v___y_3113_ = v___y_3102_;
goto v___jp_3104_;
}
}
}
}
v___jp_3165_:
{
lean_object* v___x_3174_; 
v___x_3174_ = l_Lean_Syntax_getTailPos_x3f(v___y_3168_, v___y_3171_);
lean_dec(v___y_3168_);
if (lean_obj_tag(v___x_3174_) == 0)
{
lean_inc(v___y_3173_);
v___y_3141_ = v___y_3166_;
v___y_3142_ = v___y_3173_;
v___y_3143_ = v___y_3167_;
v___y_3144_ = v___y_3169_;
v___y_3145_ = v___y_3170_;
v___y_3146_ = v___y_3171_;
v___y_3147_ = v___y_3172_;
v___y_3148_ = v___y_3173_;
goto v___jp_3140_;
}
else
{
lean_object* v_val_3175_; 
v_val_3175_ = lean_ctor_get(v___x_3174_, 0);
lean_inc(v_val_3175_);
lean_dec_ref_known(v___x_3174_, 1);
v___y_3141_ = v___y_3166_;
v___y_3142_ = v___y_3173_;
v___y_3143_ = v___y_3167_;
v___y_3144_ = v___y_3169_;
v___y_3145_ = v___y_3170_;
v___y_3146_ = v___y_3171_;
v___y_3147_ = v___y_3172_;
v___y_3148_ = v_val_3175_;
goto v___jp_3140_;
}
}
v___jp_3176_:
{
lean_object* v_ref_3184_; lean_object* v___x_3185_; 
v_ref_3184_ = l_Lean_replaceRef(v_ref_3095_, v___y_3181_);
v___x_3185_ = l_Lean_Syntax_getPos_x3f(v_ref_3184_, v___y_3180_);
if (lean_obj_tag(v___x_3185_) == 0)
{
lean_object* v___x_3186_; 
v___x_3186_ = lean_unsigned_to_nat(0u);
v___y_3166_ = v___y_3177_;
v___y_3167_ = v___y_3178_;
v___y_3168_ = v_ref_3184_;
v___y_3169_ = v___y_3183_;
v___y_3170_ = v___y_3179_;
v___y_3171_ = v___y_3180_;
v___y_3172_ = v___y_3182_;
v___y_3173_ = v___x_3186_;
goto v___jp_3165_;
}
else
{
lean_object* v_val_3187_; 
v_val_3187_ = lean_ctor_get(v___x_3185_, 0);
lean_inc(v_val_3187_);
lean_dec_ref_known(v___x_3185_, 1);
v___y_3166_ = v___y_3177_;
v___y_3167_ = v___y_3178_;
v___y_3168_ = v_ref_3184_;
v___y_3169_ = v___y_3183_;
v___y_3170_ = v___y_3179_;
v___y_3171_ = v___y_3180_;
v___y_3172_ = v___y_3182_;
v___y_3173_ = v_val_3187_;
goto v___jp_3165_;
}
}
v___jp_3189_:
{
if (v___y_3196_ == 0)
{
v___y_3177_ = v___y_3192_;
v___y_3178_ = v___y_3190_;
v___y_3179_ = v___y_3191_;
v___y_3180_ = v___y_3195_;
v___y_3181_ = v___y_3193_;
v___y_3182_ = v___y_3194_;
v___y_3183_ = v_severity_3097_;
goto v___jp_3176_;
}
else
{
v___y_3177_ = v___y_3192_;
v___y_3178_ = v___y_3190_;
v___y_3179_ = v___y_3191_;
v___y_3180_ = v___y_3195_;
v___y_3181_ = v___y_3193_;
v___y_3182_ = v___y_3194_;
v___y_3183_ = v___x_3188_;
goto v___jp_3176_;
}
}
v___jp_3197_:
{
if (v___y_3198_ == 0)
{
lean_object* v_fileName_3199_; lean_object* v_fileMap_3200_; lean_object* v_options_3201_; lean_object* v_ref_3202_; uint8_t v_suppressElabErrors_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___f_3206_; uint8_t v___x_3207_; uint8_t v___x_3208_; 
v_fileName_3199_ = lean_ctor_get(v___y_3101_, 0);
v_fileMap_3200_ = lean_ctor_get(v___y_3101_, 1);
v_options_3201_ = lean_ctor_get(v___y_3101_, 2);
v_ref_3202_ = lean_ctor_get(v___y_3101_, 5);
v_suppressElabErrors_3203_ = lean_ctor_get_uint8(v___y_3101_, sizeof(void*)*14 + 1);
v___x_3204_ = lean_box(v___y_3198_);
v___x_3205_ = lean_box(v_suppressElabErrors_3203_);
v___f_3206_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__2_spec__8_spec__22___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3206_, 0, v___x_3204_);
lean_closure_set(v___f_3206_, 1, v___x_3205_);
v___x_3207_ = 1;
v___x_3208_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3097_, v___x_3207_);
if (v___x_3208_ == 0)
{
v___y_3190_ = v_fileMap_3200_;
v___y_3191_ = v_fileName_3199_;
v___y_3192_ = v___f_3206_;
v___y_3193_ = v_ref_3202_;
v___y_3194_ = v_suppressElabErrors_3203_;
v___y_3195_ = v___y_3198_;
v___y_3196_ = v___x_3208_;
goto v___jp_3189_;
}
else
{
lean_object* v___x_3209_; uint8_t v___x_3210_; 
v___x_3209_ = l_Lean_warningAsError;
v___x_3210_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v_options_3201_, v___x_3209_);
v___y_3190_ = v_fileMap_3200_;
v___y_3191_ = v_fileName_3199_;
v___y_3192_ = v___f_3206_;
v___y_3193_ = v_ref_3202_;
v___y_3194_ = v_suppressElabErrors_3203_;
v___y_3195_ = v___y_3198_;
v___y_3196_ = v___x_3210_;
goto v___jp_3189_;
}
}
else
{
lean_object* v___x_3211_; lean_object* v___x_3212_; 
lean_dec_ref(v_msgData_3096_);
v___x_3211_ = lean_box(0);
v___x_3212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3212_, 0, v___x_3211_);
return v___x_3212_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_3215_, lean_object* v_msgData_3216_, lean_object* v_severity_3217_, lean_object* v_isSilent_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_, lean_object* v___y_3221_, lean_object* v___y_3222_, lean_object* v___y_3223_){
_start:
{
uint8_t v_severity_boxed_3224_; uint8_t v_isSilent_boxed_3225_; lean_object* v_res_3226_; 
v_severity_boxed_3224_ = lean_unbox(v_severity_3217_);
v_isSilent_boxed_3225_ = lean_unbox(v_isSilent_3218_);
v_res_3226_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2(v_ref_3215_, v_msgData_3216_, v_severity_boxed_3224_, v_isSilent_boxed_3225_, v___y_3219_, v___y_3220_, v___y_3221_, v___y_3222_);
lean_dec(v___y_3222_);
lean_dec_ref(v___y_3221_);
lean_dec(v___y_3220_);
lean_dec_ref(v___y_3219_);
lean_dec(v_ref_3215_);
return v_res_3226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1(lean_object* v_ref_3227_, lean_object* v_msgData_3228_, lean_object* v___y_3229_, lean_object* v___y_3230_, lean_object* v___y_3231_, lean_object* v___y_3232_){
_start:
{
uint8_t v___x_3234_; uint8_t v___x_3235_; lean_object* v___x_3236_; 
v___x_3234_ = 1;
v___x_3235_ = 0;
v___x_3236_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1_spec__2(v_ref_3227_, v_msgData_3228_, v___x_3234_, v___x_3235_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_);
return v___x_3236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1___boxed(lean_object* v_ref_3237_, lean_object* v_msgData_3238_, lean_object* v___y_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_){
_start:
{
lean_object* v_res_3244_; 
v_res_3244_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1(v_ref_3237_, v_msgData_3238_, v___y_3239_, v___y_3240_, v___y_3241_, v___y_3242_);
lean_dec(v___y_3242_);
lean_dec_ref(v___y_3241_);
lean_dec(v___y_3240_);
lean_dec_ref(v___y_3239_);
lean_dec(v_ref_3237_);
return v_res_3244_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1(void){
_start:
{
lean_object* v___x_3246_; lean_object* v___x_3247_; 
v___x_3246_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__0));
v___x_3247_ = l_Lean_stringToMessageData(v___x_3246_);
return v___x_3247_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3(void){
_start:
{
lean_object* v___x_3249_; lean_object* v___x_3250_; 
v___x_3249_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__2));
v___x_3250_ = l_Lean_stringToMessageData(v___x_3249_);
return v___x_3250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1(lean_object* v_linterOption_3251_, lean_object* v_stx_3252_, lean_object* v_msg_3253_, lean_object* v___y_3254_, lean_object* v___y_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_){
_start:
{
lean_object* v_name_3259_; lean_object* v___x_3261_; uint8_t v_isShared_3262_; uint8_t v_isSharedCheck_3277_; 
v_name_3259_ = lean_ctor_get(v_linterOption_3251_, 0);
v_isSharedCheck_3277_ = !lean_is_exclusive(v_linterOption_3251_);
if (v_isSharedCheck_3277_ == 0)
{
lean_object* v_unused_3278_; 
v_unused_3278_ = lean_ctor_get(v_linterOption_3251_, 1);
lean_dec(v_unused_3278_);
v___x_3261_ = v_linterOption_3251_;
v_isShared_3262_ = v_isSharedCheck_3277_;
goto v_resetjp_3260_;
}
else
{
lean_inc(v_name_3259_);
lean_dec(v_linterOption_3251_);
v___x_3261_ = lean_box(0);
v_isShared_3262_ = v_isSharedCheck_3277_;
goto v_resetjp_3260_;
}
v_resetjp_3260_:
{
lean_object* v___x_3263_; lean_object* v___x_3264_; lean_object* v___x_3266_; 
v___x_3263_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__1);
lean_inc(v_name_3259_);
v___x_3264_ = l_Lean_MessageData_ofName(v_name_3259_);
if (v_isShared_3262_ == 0)
{
lean_ctor_set_tag(v___x_3261_, 7);
lean_ctor_set(v___x_3261_, 1, v___x_3264_);
lean_ctor_set(v___x_3261_, 0, v___x_3263_);
v___x_3266_ = v___x_3261_;
goto v_reusejp_3265_;
}
else
{
lean_object* v_reuseFailAlloc_3276_; 
v_reuseFailAlloc_3276_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3276_, 0, v___x_3263_);
lean_ctor_set(v_reuseFailAlloc_3276_, 1, v___x_3264_);
v___x_3266_ = v_reuseFailAlloc_3276_;
goto v_reusejp_3265_;
}
v_reusejp_3265_:
{
lean_object* v___x_3267_; lean_object* v___x_3268_; lean_object* v_disable_3269_; lean_object* v___x_3270_; lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; lean_object* v___x_3275_; 
v___x_3267_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___closed__3);
v___x_3268_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3268_, 0, v___x_3266_);
lean_ctor_set(v___x_3268_, 1, v___x_3267_);
v_disable_3269_ = l_Lean_MessageData_note(v___x_3268_);
v___x_3270_ = l_Lean_Linter_linterMessageTag;
v___x_3271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3271_, 0, v_msg_3253_);
lean_ctor_set(v___x_3271_, 1, v_disable_3269_);
v___x_3272_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3272_, 0, v___x_3270_);
lean_ctor_set(v___x_3272_, 1, v___x_3271_);
v___x_3273_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3273_, 0, v_name_3259_);
lean_ctor_set(v___x_3273_, 1, v___x_3272_);
lean_inc(v_stx_3252_);
v___x_3274_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_3274_, 0, v_stx_3252_);
lean_ctor_set(v___x_3274_, 1, v___x_3273_);
v___x_3275_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1_spec__1(v_stx_3252_, v___x_3274_, v___y_3254_, v___y_3255_, v___y_3256_, v___y_3257_);
lean_dec(v_stx_3252_);
return v___x_3275_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1___boxed(lean_object* v_linterOption_3279_, lean_object* v_stx_3280_, lean_object* v_msg_3281_, lean_object* v___y_3282_, lean_object* v___y_3283_, lean_object* v___y_3284_, lean_object* v___y_3285_, lean_object* v___y_3286_){
_start:
{
lean_object* v_res_3287_; 
v_res_3287_ = lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1(v_linterOption_3279_, v_stx_3280_, v_msg_3281_, v___y_3282_, v___y_3283_, v___y_3284_, v___y_3285_);
lean_dec(v___y_3285_);
lean_dec_ref(v___y_3284_);
lean_dec(v___y_3283_);
lean_dec_ref(v___y_3282_);
return v_res_3287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__0(lean_object* v_a_3288_, lean_object* v_a_3289_){
_start:
{
if (lean_obj_tag(v_a_3288_) == 0)
{
lean_object* v___x_3290_; 
v___x_3290_ = l_List_reverse___redArg(v_a_3289_);
return v___x_3290_;
}
else
{
lean_object* v_head_3291_; lean_object* v_tail_3292_; lean_object* v___x_3294_; uint8_t v_isShared_3295_; uint8_t v_isSharedCheck_3302_; 
v_head_3291_ = lean_ctor_get(v_a_3288_, 0);
v_tail_3292_ = lean_ctor_get(v_a_3288_, 1);
v_isSharedCheck_3302_ = !lean_is_exclusive(v_a_3288_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3294_ = v_a_3288_;
v_isShared_3295_ = v_isSharedCheck_3302_;
goto v_resetjp_3293_;
}
else
{
lean_inc(v_tail_3292_);
lean_inc(v_head_3291_);
lean_dec(v_a_3288_);
v___x_3294_ = lean_box(0);
v_isShared_3295_ = v_isSharedCheck_3302_;
goto v_resetjp_3293_;
}
v_resetjp_3293_:
{
uint8_t v___x_3296_; lean_object* v___x_3297_; lean_object* v___x_3299_; 
v___x_3296_ = 0;
v___x_3297_ = l_Lean_MessageData_ofConstName(v_head_3291_, v___x_3296_);
if (v_isShared_3295_ == 0)
{
lean_ctor_set(v___x_3294_, 1, v_a_3289_);
lean_ctor_set(v___x_3294_, 0, v___x_3297_);
v___x_3299_ = v___x_3294_;
goto v_reusejp_3298_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v___x_3297_);
lean_ctor_set(v_reuseFailAlloc_3301_, 1, v_a_3289_);
v___x_3299_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3298_;
}
v_reusejp_3298_:
{
v_a_3288_ = v_tail_3292_;
v_a_3289_ = v___x_3299_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5(void){
_start:
{
lean_object* v___x_3313_; lean_object* v___x_3314_; 
v___x_3313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__4));
v___x_3314_ = l_Lean_stringToMessageData(v___x_3313_);
return v___x_3314_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7(void){
_start:
{
lean_object* v___x_3316_; lean_object* v___x_3317_; 
v___x_3316_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__6));
v___x_3317_ = l_Lean_stringToMessageData(v___x_3316_);
return v___x_3317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped(lean_object* v_ref_3318_, lean_object* v_declName_3319_, lean_object* v_declType_3320_, lean_object* v_a_3321_, lean_object* v_a_3322_, lean_object* v_a_3323_, lean_object* v_a_3324_){
_start:
{
lean_object* v_options_3326_; uint8_t v___x_3327_; lean_object* v_lintOpt_3328_; uint8_t v___x_3329_; 
v_options_3326_ = lean_ctor_get(v_a_3323_, 2);
v___x_3327_ = 0;
v_lintOpt_3328_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__3));
v___x_3329_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__21(v_options_3326_, v_lintOpt_3328_);
if (v___x_3329_ == 0)
{
lean_object* v___x_3330_; lean_object* v___x_3331_; 
lean_dec_ref(v_declType_3320_);
lean_dec(v_declName_3319_);
lean_dec(v_ref_3318_);
v___x_3330_ = lean_box(0);
v___x_3331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3331_, 0, v___x_3330_);
return v___x_3331_;
}
else
{
lean_object* v___x_3332_; 
v___x_3332_ = lp_mathlib___private_Mathlib_Util_AddRelatedDecl_0__Mathlib_Tactic_checkImplicitTransparency(v_declType_3320_, v_a_3321_, v_a_3322_, v_a_3323_, v_a_3324_);
if (lean_obj_tag(v___x_3332_) == 0)
{
lean_object* v_a_3333_; lean_object* v___x_3335_; uint8_t v_isShared_3336_; uint8_t v_isSharedCheck_3359_; 
v_a_3333_ = lean_ctor_get(v___x_3332_, 0);
v_isSharedCheck_3359_ = !lean_is_exclusive(v___x_3332_);
if (v_isSharedCheck_3359_ == 0)
{
v___x_3335_ = v___x_3332_;
v_isShared_3336_ = v_isSharedCheck_3359_;
goto v_resetjp_3334_;
}
else
{
lean_inc(v_a_3333_);
lean_dec(v___x_3332_);
v___x_3335_ = lean_box(0);
v_isShared_3336_ = v_isSharedCheck_3359_;
goto v_resetjp_3334_;
}
v_resetjp_3334_:
{
if (lean_obj_tag(v_a_3333_) == 1)
{
lean_object* v_val_3337_; uint8_t v___x_3338_; 
v_val_3337_ = lean_ctor_get(v_a_3333_, 0);
lean_inc(v_val_3337_);
lean_dec_ref_known(v_a_3333_, 1);
v___x_3338_ = l_List_isEmpty___redArg(v_val_3337_);
if (v___x_3338_ == 0)
{
lean_object* v___x_3339_; lean_object* v___x_3340_; lean_object* v___x_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; 
lean_del_object(v___x_3335_);
v___x_3339_ = lean_box(0);
v___x_3340_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__0(v_val_3337_, v___x_3339_);
v___x_3341_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__16_spec__22___closed__0);
v___x_3342_ = l_Lean_MessageData_joinSep(v___x_3340_, v___x_3341_);
v___x_3343_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5, &lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__5);
v___x_3344_ = l_Lean_MessageData_ofConstName(v_declName_3319_, v___x_3327_);
v___x_3345_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3345_, 0, v___x_3343_);
lean_ctor_set(v___x_3345_, 1, v___x_3344_);
v___x_3346_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7, &lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___closed__7);
v___x_3347_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3347_, 0, v___x_3345_);
lean_ctor_set(v___x_3347_, 1, v___x_3346_);
v___x_3348_ = l_Lean_indentD(v___x_3342_);
v___x_3349_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3349_, 0, v___x_3347_);
lean_ctor_set(v___x_3349_, 1, v___x_3348_);
v___x_3350_ = lp_mathlib_Lean_Linter_logLint___at___00Mathlib_Tactic_warnIfImplicitIllTyped_spec__1(v_lintOpt_3328_, v_ref_3318_, v___x_3349_, v_a_3321_, v_a_3322_, v_a_3323_, v_a_3324_);
return v___x_3350_;
}
else
{
lean_object* v___x_3351_; lean_object* v___x_3353_; 
lean_dec(v_val_3337_);
lean_dec(v_declName_3319_);
lean_dec(v_ref_3318_);
v___x_3351_ = lean_box(0);
if (v_isShared_3336_ == 0)
{
lean_ctor_set(v___x_3335_, 0, v___x_3351_);
v___x_3353_ = v___x_3335_;
goto v_reusejp_3352_;
}
else
{
lean_object* v_reuseFailAlloc_3354_; 
v_reuseFailAlloc_3354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3354_, 0, v___x_3351_);
v___x_3353_ = v_reuseFailAlloc_3354_;
goto v_reusejp_3352_;
}
v_reusejp_3352_:
{
return v___x_3353_;
}
}
}
else
{
lean_object* v___x_3355_; lean_object* v___x_3357_; 
lean_dec(v_a_3333_);
lean_dec(v_declName_3319_);
lean_dec(v_ref_3318_);
v___x_3355_ = lean_box(0);
if (v_isShared_3336_ == 0)
{
lean_ctor_set(v___x_3335_, 0, v___x_3355_);
v___x_3357_ = v___x_3335_;
goto v_reusejp_3356_;
}
else
{
lean_object* v_reuseFailAlloc_3358_; 
v_reuseFailAlloc_3358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3358_, 0, v___x_3355_);
v___x_3357_ = v_reuseFailAlloc_3358_;
goto v_reusejp_3356_;
}
v_reusejp_3356_:
{
return v___x_3357_;
}
}
}
}
else
{
lean_object* v_a_3360_; lean_object* v___x_3362_; uint8_t v_isShared_3363_; uint8_t v_isSharedCheck_3367_; 
lean_dec(v_declName_3319_);
lean_dec(v_ref_3318_);
v_a_3360_ = lean_ctor_get(v___x_3332_, 0);
v_isSharedCheck_3367_ = !lean_is_exclusive(v___x_3332_);
if (v_isSharedCheck_3367_ == 0)
{
v___x_3362_ = v___x_3332_;
v_isShared_3363_ = v_isSharedCheck_3367_;
goto v_resetjp_3361_;
}
else
{
lean_inc(v_a_3360_);
lean_dec(v___x_3332_);
v___x_3362_ = lean_box(0);
v_isShared_3363_ = v_isSharedCheck_3367_;
goto v_resetjp_3361_;
}
v_resetjp_3361_:
{
lean_object* v___x_3365_; 
if (v_isShared_3363_ == 0)
{
v___x_3365_ = v___x_3362_;
goto v_reusejp_3364_;
}
else
{
lean_object* v_reuseFailAlloc_3366_; 
v_reuseFailAlloc_3366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3366_, 0, v_a_3360_);
v___x_3365_ = v_reuseFailAlloc_3366_;
goto v_reusejp_3364_;
}
v_reusejp_3364_:
{
return v___x_3365_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped___boxed(lean_object* v_ref_3368_, lean_object* v_declName_3369_, lean_object* v_declType_3370_, lean_object* v_a_3371_, lean_object* v_a_3372_, lean_object* v_a_3373_, lean_object* v_a_3374_, lean_object* v_a_3375_){
_start:
{
lean_object* v_res_3376_; 
v_res_3376_ = lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped(v_ref_3368_, v_declName_3369_, v_declType_3370_, v_a_3371_, v_a_3372_, v_a_3373_, v_a_3374_);
lean_dec(v_a_3374_);
lean_dec_ref(v_a_3373_);
lean_dec(v_a_3372_);
lean_dec_ref(v_a_3371_);
return v_res_3376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(lean_object* v_e_3377_, lean_object* v___y_3378_){
_start:
{
uint8_t v___x_3380_; 
v___x_3380_ = l_Lean_Expr_hasMVar(v_e_3377_);
if (v___x_3380_ == 0)
{
lean_object* v___x_3381_; 
v___x_3381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3381_, 0, v_e_3377_);
return v___x_3381_;
}
else
{
lean_object* v___x_3382_; lean_object* v_mctx_3383_; lean_object* v___x_3384_; lean_object* v_fst_3385_; lean_object* v_snd_3386_; lean_object* v___x_3387_; lean_object* v_cache_3388_; lean_object* v_zetaDeltaFVarIds_3389_; lean_object* v_postponed_3390_; lean_object* v_diag_3391_; lean_object* v___x_3393_; uint8_t v_isShared_3394_; uint8_t v_isSharedCheck_3400_; 
v___x_3382_ = lean_st_ref_get(v___y_3378_);
v_mctx_3383_ = lean_ctor_get(v___x_3382_, 0);
lean_inc_ref(v_mctx_3383_);
lean_dec(v___x_3382_);
v___x_3384_ = l_Lean_instantiateMVarsCore(v_mctx_3383_, v_e_3377_);
v_fst_3385_ = lean_ctor_get(v___x_3384_, 0);
lean_inc(v_fst_3385_);
v_snd_3386_ = lean_ctor_get(v___x_3384_, 1);
lean_inc(v_snd_3386_);
lean_dec_ref(v___x_3384_);
v___x_3387_ = lean_st_ref_take(v___y_3378_);
v_cache_3388_ = lean_ctor_get(v___x_3387_, 1);
v_zetaDeltaFVarIds_3389_ = lean_ctor_get(v___x_3387_, 2);
v_postponed_3390_ = lean_ctor_get(v___x_3387_, 3);
v_diag_3391_ = lean_ctor_get(v___x_3387_, 4);
v_isSharedCheck_3400_ = !lean_is_exclusive(v___x_3387_);
if (v_isSharedCheck_3400_ == 0)
{
lean_object* v_unused_3401_; 
v_unused_3401_ = lean_ctor_get(v___x_3387_, 0);
lean_dec(v_unused_3401_);
v___x_3393_ = v___x_3387_;
v_isShared_3394_ = v_isSharedCheck_3400_;
goto v_resetjp_3392_;
}
else
{
lean_inc(v_diag_3391_);
lean_inc(v_postponed_3390_);
lean_inc(v_zetaDeltaFVarIds_3389_);
lean_inc(v_cache_3388_);
lean_dec(v___x_3387_);
v___x_3393_ = lean_box(0);
v_isShared_3394_ = v_isSharedCheck_3400_;
goto v_resetjp_3392_;
}
v_resetjp_3392_:
{
lean_object* v___x_3396_; 
if (v_isShared_3394_ == 0)
{
lean_ctor_set(v___x_3393_, 0, v_snd_3386_);
v___x_3396_ = v___x_3393_;
goto v_reusejp_3395_;
}
else
{
lean_object* v_reuseFailAlloc_3399_; 
v_reuseFailAlloc_3399_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3399_, 0, v_snd_3386_);
lean_ctor_set(v_reuseFailAlloc_3399_, 1, v_cache_3388_);
lean_ctor_set(v_reuseFailAlloc_3399_, 2, v_zetaDeltaFVarIds_3389_);
lean_ctor_set(v_reuseFailAlloc_3399_, 3, v_postponed_3390_);
lean_ctor_set(v_reuseFailAlloc_3399_, 4, v_diag_3391_);
v___x_3396_ = v_reuseFailAlloc_3399_;
goto v_reusejp_3395_;
}
v_reusejp_3395_:
{
lean_object* v___x_3397_; lean_object* v___x_3398_; 
v___x_3397_ = lean_st_ref_set(v___y_3378_, v___x_3396_);
v___x_3398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3398_, 0, v_fst_3385_);
return v___x_3398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg___boxed(lean_object* v_e_3402_, lean_object* v___y_3403_, lean_object* v___y_3404_){
_start:
{
lean_object* v_res_3405_; 
v_res_3405_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(v_e_3402_, v___y_3403_);
lean_dec(v___y_3403_);
return v_res_3405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5(lean_object* v_e_3406_, lean_object* v___y_3407_, lean_object* v___y_3408_, lean_object* v___y_3409_, lean_object* v___y_3410_){
_start:
{
lean_object* v___x_3412_; 
v___x_3412_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(v_e_3406_, v___y_3408_);
return v___x_3412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___boxed(lean_object* v_e_3413_, lean_object* v___y_3414_, lean_object* v___y_3415_, lean_object* v___y_3416_, lean_object* v___y_3417_, lean_object* v___y_3418_){
_start:
{
lean_object* v_res_3419_; 
v_res_3419_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5(v_e_3413_, v___y_3414_, v___y_3415_, v___y_3416_, v___y_3417_);
lean_dec(v___y_3417_);
lean_dec_ref(v___y_3416_);
lean_dec(v___y_3415_);
lean_dec_ref(v___y_3414_);
return v_res_3419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg(lean_object* v_thm_3420_, lean_object* v___y_3421_){
_start:
{
lean_object* v___x_3423_; lean_object* v_env_3424_; lean_object* v_toConstantVal_3425_; lean_object* v_value_3426_; lean_object* v_all_3427_; uint8_t v___y_3429_; lean_object* v_type_3437_; uint8_t v___x_3438_; 
v___x_3423_ = lean_st_ref_get(v___y_3421_);
v_env_3424_ = lean_ctor_get(v___x_3423_, 0);
lean_inc_ref_n(v_env_3424_, 2);
lean_dec(v___x_3423_);
v_toConstantVal_3425_ = lean_ctor_get(v_thm_3420_, 0);
v_value_3426_ = lean_ctor_get(v_thm_3420_, 1);
v_all_3427_ = lean_ctor_get(v_thm_3420_, 2);
v_type_3437_ = lean_ctor_get(v_toConstantVal_3425_, 2);
v___x_3438_ = l_Lean_Environment_hasUnsafe(v_env_3424_, v_type_3437_);
if (v___x_3438_ == 0)
{
uint8_t v___x_3439_; 
v___x_3439_ = l_Lean_Environment_hasUnsafe(v_env_3424_, v_value_3426_);
v___y_3429_ = v___x_3439_;
goto v___jp_3428_;
}
else
{
lean_dec_ref(v_env_3424_);
v___y_3429_ = v___x_3438_;
goto v___jp_3428_;
}
v___jp_3428_:
{
if (v___y_3429_ == 0)
{
lean_object* v___x_3430_; lean_object* v___x_3431_; 
v___x_3430_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_3430_, 0, v_thm_3420_);
v___x_3431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3431_, 0, v___x_3430_);
return v___x_3431_;
}
else
{
lean_object* v___x_3432_; uint8_t v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; 
lean_inc(v_all_3427_);
lean_inc_ref(v_value_3426_);
lean_inc_ref(v_toConstantVal_3425_);
lean_dec_ref(v_thm_3420_);
v___x_3432_ = lean_box(0);
v___x_3433_ = 0;
v___x_3434_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_3434_, 0, v_toConstantVal_3425_);
lean_ctor_set(v___x_3434_, 1, v_value_3426_);
lean_ctor_set(v___x_3434_, 2, v___x_3432_);
lean_ctor_set(v___x_3434_, 3, v_all_3427_);
lean_ctor_set_uint8(v___x_3434_, sizeof(void*)*4, v___x_3433_);
v___x_3435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3435_, 0, v___x_3434_);
v___x_3436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3436_, 0, v___x_3435_);
return v___x_3436_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg___boxed(lean_object* v_thm_3440_, lean_object* v___y_3441_, lean_object* v___y_3442_){
_start:
{
lean_object* v_res_3443_; 
v_res_3443_ = lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg(v_thm_3440_, v___y_3441_);
lean_dec(v___y_3441_);
return v_res_3443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6(lean_object* v_thm_3444_, lean_object* v___y_3445_, lean_object* v___y_3446_, lean_object* v___y_3447_, lean_object* v___y_3448_){
_start:
{
lean_object* v___x_3450_; 
v___x_3450_ = lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg(v_thm_3444_, v___y_3448_);
return v___x_3450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___boxed(lean_object* v_thm_3451_, lean_object* v___y_3452_, lean_object* v___y_3453_, lean_object* v___y_3454_, lean_object* v___y_3455_, lean_object* v___y_3456_){
_start:
{
lean_object* v_res_3457_; 
v_res_3457_ = lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6(v_thm_3451_, v___y_3452_, v___y_3453_, v___y_3454_, v___y_3455_);
lean_dec(v___y_3455_);
lean_dec_ref(v___y_3454_);
lean_dec(v___y_3453_);
lean_dec_ref(v___y_3452_);
return v_res_3457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(lean_object* v_env_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_){
_start:
{
lean_object* v___x_3462_; lean_object* v_nextMacroScope_3463_; lean_object* v_ngen_3464_; lean_object* v_auxDeclNGen_3465_; lean_object* v_traceState_3466_; lean_object* v_messages_3467_; lean_object* v_infoState_3468_; lean_object* v_snapshotTasks_3469_; lean_object* v___x_3471_; uint8_t v_isShared_3472_; uint8_t v_isSharedCheck_3495_; 
v___x_3462_ = lean_st_ref_take(v___y_3460_);
v_nextMacroScope_3463_ = lean_ctor_get(v___x_3462_, 1);
v_ngen_3464_ = lean_ctor_get(v___x_3462_, 2);
v_auxDeclNGen_3465_ = lean_ctor_get(v___x_3462_, 3);
v_traceState_3466_ = lean_ctor_get(v___x_3462_, 4);
v_messages_3467_ = lean_ctor_get(v___x_3462_, 6);
v_infoState_3468_ = lean_ctor_get(v___x_3462_, 7);
v_snapshotTasks_3469_ = lean_ctor_get(v___x_3462_, 8);
v_isSharedCheck_3495_ = !lean_is_exclusive(v___x_3462_);
if (v_isSharedCheck_3495_ == 0)
{
lean_object* v_unused_3496_; lean_object* v_unused_3497_; 
v_unused_3496_ = lean_ctor_get(v___x_3462_, 5);
lean_dec(v_unused_3496_);
v_unused_3497_ = lean_ctor_get(v___x_3462_, 0);
lean_dec(v_unused_3497_);
v___x_3471_ = v___x_3462_;
v_isShared_3472_ = v_isSharedCheck_3495_;
goto v_resetjp_3470_;
}
else
{
lean_inc(v_snapshotTasks_3469_);
lean_inc(v_infoState_3468_);
lean_inc(v_messages_3467_);
lean_inc(v_traceState_3466_);
lean_inc(v_auxDeclNGen_3465_);
lean_inc(v_ngen_3464_);
lean_inc(v_nextMacroScope_3463_);
lean_dec(v___x_3462_);
v___x_3471_ = lean_box(0);
v_isShared_3472_ = v_isSharedCheck_3495_;
goto v_resetjp_3470_;
}
v_resetjp_3470_:
{
lean_object* v___x_3473_; lean_object* v___x_3475_; 
v___x_3473_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_3472_ == 0)
{
lean_ctor_set(v___x_3471_, 5, v___x_3473_);
lean_ctor_set(v___x_3471_, 0, v_env_3458_);
v___x_3475_ = v___x_3471_;
goto v_reusejp_3474_;
}
else
{
lean_object* v_reuseFailAlloc_3494_; 
v_reuseFailAlloc_3494_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3494_, 0, v_env_3458_);
lean_ctor_set(v_reuseFailAlloc_3494_, 1, v_nextMacroScope_3463_);
lean_ctor_set(v_reuseFailAlloc_3494_, 2, v_ngen_3464_);
lean_ctor_set(v_reuseFailAlloc_3494_, 3, v_auxDeclNGen_3465_);
lean_ctor_set(v_reuseFailAlloc_3494_, 4, v_traceState_3466_);
lean_ctor_set(v_reuseFailAlloc_3494_, 5, v___x_3473_);
lean_ctor_set(v_reuseFailAlloc_3494_, 6, v_messages_3467_);
lean_ctor_set(v_reuseFailAlloc_3494_, 7, v_infoState_3468_);
lean_ctor_set(v_reuseFailAlloc_3494_, 8, v_snapshotTasks_3469_);
v___x_3475_ = v_reuseFailAlloc_3494_;
goto v_reusejp_3474_;
}
v_reusejp_3474_:
{
lean_object* v___x_3476_; lean_object* v___x_3477_; lean_object* v_mctx_3478_; lean_object* v_zetaDeltaFVarIds_3479_; lean_object* v_postponed_3480_; lean_object* v_diag_3481_; lean_object* v___x_3483_; uint8_t v_isShared_3484_; uint8_t v_isSharedCheck_3492_; 
v___x_3476_ = lean_st_ref_set(v___y_3460_, v___x_3475_);
v___x_3477_ = lean_st_ref_take(v___y_3459_);
v_mctx_3478_ = lean_ctor_get(v___x_3477_, 0);
v_zetaDeltaFVarIds_3479_ = lean_ctor_get(v___x_3477_, 2);
v_postponed_3480_ = lean_ctor_get(v___x_3477_, 3);
v_diag_3481_ = lean_ctor_get(v___x_3477_, 4);
v_isSharedCheck_3492_ = !lean_is_exclusive(v___x_3477_);
if (v_isSharedCheck_3492_ == 0)
{
lean_object* v_unused_3493_; 
v_unused_3493_ = lean_ctor_get(v___x_3477_, 1);
lean_dec(v_unused_3493_);
v___x_3483_ = v___x_3477_;
v_isShared_3484_ = v_isSharedCheck_3492_;
goto v_resetjp_3482_;
}
else
{
lean_inc(v_diag_3481_);
lean_inc(v_postponed_3480_);
lean_inc(v_zetaDeltaFVarIds_3479_);
lean_inc(v_mctx_3478_);
lean_dec(v___x_3477_);
v___x_3483_ = lean_box(0);
v_isShared_3484_ = v_isSharedCheck_3492_;
goto v_resetjp_3482_;
}
v_resetjp_3482_:
{
lean_object* v___x_3485_; lean_object* v___x_3487_; 
v___x_3485_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_3484_ == 0)
{
lean_ctor_set(v___x_3483_, 1, v___x_3485_);
v___x_3487_ = v___x_3483_;
goto v_reusejp_3486_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v_mctx_3478_);
lean_ctor_set(v_reuseFailAlloc_3491_, 1, v___x_3485_);
lean_ctor_set(v_reuseFailAlloc_3491_, 2, v_zetaDeltaFVarIds_3479_);
lean_ctor_set(v_reuseFailAlloc_3491_, 3, v_postponed_3480_);
lean_ctor_set(v_reuseFailAlloc_3491_, 4, v_diag_3481_);
v___x_3487_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3486_;
}
v_reusejp_3486_:
{
lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v___x_3490_; 
v___x_3488_ = lean_st_ref_set(v___y_3459_, v___x_3487_);
v___x_3489_ = lean_box(0);
v___x_3490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3490_, 0, v___x_3489_);
return v___x_3490_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg___boxed(lean_object* v_env_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_){
_start:
{
lean_object* v_res_3502_; 
v_res_3502_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v_env_3498_, v___y_3499_, v___y_3500_);
lean_dec(v___y_3500_);
lean_dec(v___y_3499_);
return v_res_3502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9(lean_object* v_env_3503_, lean_object* v___y_3504_, lean_object* v___y_3505_, lean_object* v___y_3506_, lean_object* v___y_3507_){
_start:
{
lean_object* v___x_3509_; 
v___x_3509_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v_env_3503_, v___y_3505_, v___y_3507_);
return v___x_3509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___boxed(lean_object* v_env_3510_, lean_object* v___y_3511_, lean_object* v___y_3512_, lean_object* v___y_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_){
_start:
{
lean_object* v_res_3516_; 
v_res_3516_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9(v_env_3510_, v___y_3511_, v___y_3512_, v___y_3513_, v___y_3514_);
lean_dec(v___y_3514_);
lean_dec_ref(v___y_3513_);
lean_dec(v___y_3512_);
lean_dec_ref(v___y_3511_);
return v_res_3516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_addRelatedDecl_spec__4(lean_object* v_a_3517_, lean_object* v_a_3518_){
_start:
{
if (lean_obj_tag(v_a_3517_) == 0)
{
lean_object* v___x_3519_; 
v___x_3519_ = l_List_reverse___redArg(v_a_3518_);
return v___x_3519_;
}
else
{
lean_object* v_head_3520_; lean_object* v_tail_3521_; lean_object* v___x_3523_; uint8_t v_isShared_3524_; uint8_t v_isSharedCheck_3530_; 
v_head_3520_ = lean_ctor_get(v_a_3517_, 0);
v_tail_3521_ = lean_ctor_get(v_a_3517_, 1);
v_isSharedCheck_3530_ = !lean_is_exclusive(v_a_3517_);
if (v_isSharedCheck_3530_ == 0)
{
v___x_3523_ = v_a_3517_;
v_isShared_3524_ = v_isSharedCheck_3530_;
goto v_resetjp_3522_;
}
else
{
lean_inc(v_tail_3521_);
lean_inc(v_head_3520_);
lean_dec(v_a_3517_);
v___x_3523_ = lean_box(0);
v_isShared_3524_ = v_isSharedCheck_3530_;
goto v_resetjp_3522_;
}
v_resetjp_3522_:
{
lean_object* v___x_3525_; lean_object* v___x_3527_; 
v___x_3525_ = l_Lean_mkLevelParam(v_head_3520_);
if (v_isShared_3524_ == 0)
{
lean_ctor_set(v___x_3523_, 1, v_a_3518_);
lean_ctor_set(v___x_3523_, 0, v___x_3525_);
v___x_3527_ = v___x_3523_;
goto v_reusejp_3526_;
}
else
{
lean_object* v_reuseFailAlloc_3529_; 
v_reuseFailAlloc_3529_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3529_, 0, v___x_3525_);
lean_ctor_set(v_reuseFailAlloc_3529_, 1, v_a_3518_);
v___x_3527_ = v_reuseFailAlloc_3529_;
goto v_reusejp_3526_;
}
v_reusejp_3526_:
{
v_a_3517_ = v_tail_3521_;
v_a_3518_ = v___x_3527_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0(void){
_start:
{
lean_object* v___x_3531_; 
v___x_3531_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3531_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1(void){
_start:
{
lean_object* v___x_3532_; lean_object* v___x_3533_; 
v___x_3532_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__0);
v___x_3533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3533_, 0, v___x_3532_);
return v___x_3533_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2(void){
_start:
{
lean_object* v___x_3534_; lean_object* v___x_3535_; lean_object* v___x_3536_; 
v___x_3534_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1);
v___x_3535_ = lean_unsigned_to_nat(0u);
v___x_3536_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3536_, 0, v___x_3535_);
lean_ctor_set(v___x_3536_, 1, v___x_3535_);
lean_ctor_set(v___x_3536_, 2, v___x_3535_);
lean_ctor_set(v___x_3536_, 3, v___x_3535_);
lean_ctor_set(v___x_3536_, 4, v___x_3534_);
lean_ctor_set(v___x_3536_, 5, v___x_3534_);
lean_ctor_set(v___x_3536_, 6, v___x_3534_);
lean_ctor_set(v___x_3536_, 7, v___x_3534_);
lean_ctor_set(v___x_3536_, 8, v___x_3534_);
lean_ctor_set(v___x_3536_, 9, v___x_3534_);
return v___x_3536_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3(void){
_start:
{
lean_object* v___x_3537_; lean_object* v___x_3538_; lean_object* v___x_3539_; 
v___x_3537_ = lean_unsigned_to_nat(32u);
v___x_3538_ = lean_mk_empty_array_with_capacity(v___x_3537_);
v___x_3539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3539_, 0, v___x_3538_);
return v___x_3539_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4(void){
_start:
{
size_t v___x_3540_; lean_object* v___x_3541_; lean_object* v___x_3542_; lean_object* v___x_3543_; lean_object* v___x_3544_; lean_object* v___x_3545_; 
v___x_3540_ = ((size_t)5ULL);
v___x_3541_ = lean_unsigned_to_nat(0u);
v___x_3542_ = lean_unsigned_to_nat(32u);
v___x_3543_ = lean_mk_empty_array_with_capacity(v___x_3542_);
v___x_3544_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__3);
v___x_3545_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3545_, 0, v___x_3544_);
lean_ctor_set(v___x_3545_, 1, v___x_3543_);
lean_ctor_set(v___x_3545_, 2, v___x_3541_);
lean_ctor_set(v___x_3545_, 3, v___x_3541_);
lean_ctor_set_usize(v___x_3545_, 4, v___x_3540_);
return v___x_3545_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5(void){
_start:
{
lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; 
v___x_3546_ = lean_box(1);
v___x_3547_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4);
v___x_3548_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__1);
v___x_3549_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3549_, 0, v___x_3548_);
lean_ctor_set(v___x_3549_, 1, v___x_3547_);
lean_ctor_set(v___x_3549_, 2, v___x_3546_);
return v___x_3549_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7(void){
_start:
{
lean_object* v___x_3551_; lean_object* v___x_3552_; 
v___x_3551_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__6));
v___x_3552_ = l_Lean_stringToMessageData(v___x_3551_);
return v___x_3552_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9(void){
_start:
{
lean_object* v___x_3554_; lean_object* v___x_3555_; 
v___x_3554_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__8));
v___x_3555_ = l_Lean_stringToMessageData(v___x_3554_);
return v___x_3555_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11(void){
_start:
{
lean_object* v___x_3557_; lean_object* v___x_3558_; 
v___x_3557_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__10));
v___x_3558_ = l_Lean_stringToMessageData(v___x_3557_);
return v___x_3558_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13(void){
_start:
{
lean_object* v___x_3560_; lean_object* v___x_3561_; 
v___x_3560_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__12));
v___x_3561_ = l_Lean_stringToMessageData(v___x_3560_);
return v___x_3561_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15(void){
_start:
{
lean_object* v___x_3563_; lean_object* v___x_3564_; 
v___x_3563_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__14));
v___x_3564_ = l_Lean_stringToMessageData(v___x_3563_);
return v___x_3564_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17(void){
_start:
{
lean_object* v___x_3566_; lean_object* v___x_3567_; 
v___x_3566_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__16));
v___x_3567_ = l_Lean_stringToMessageData(v___x_3566_);
return v___x_3567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg(lean_object* v_msg_3568_, lean_object* v_declHint_3569_, lean_object* v___y_3570_){
_start:
{
lean_object* v___x_3572_; lean_object* v_env_3573_; uint8_t v___x_3574_; 
v___x_3572_ = lean_st_ref_get(v___y_3570_);
v_env_3573_ = lean_ctor_get(v___x_3572_, 0);
lean_inc_ref(v_env_3573_);
lean_dec(v___x_3572_);
v___x_3574_ = l_Lean_Name_isAnonymous(v_declHint_3569_);
if (v___x_3574_ == 0)
{
uint8_t v_isExporting_3575_; 
v_isExporting_3575_ = lean_ctor_get_uint8(v_env_3573_, sizeof(void*)*8);
if (v_isExporting_3575_ == 0)
{
lean_object* v___x_3576_; 
lean_dec_ref(v_env_3573_);
lean_dec(v_declHint_3569_);
v___x_3576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3576_, 0, v_msg_3568_);
return v___x_3576_;
}
else
{
lean_object* v___x_3577_; uint8_t v___x_3578_; 
lean_inc_ref(v_env_3573_);
v___x_3577_ = l_Lean_Environment_setExporting(v_env_3573_, v___x_3574_);
lean_inc(v_declHint_3569_);
lean_inc_ref(v___x_3577_);
v___x_3578_ = l_Lean_Environment_contains(v___x_3577_, v_declHint_3569_, v_isExporting_3575_);
if (v___x_3578_ == 0)
{
lean_object* v___x_3579_; 
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_env_3573_);
lean_dec(v_declHint_3569_);
v___x_3579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3579_, 0, v_msg_3568_);
return v___x_3579_;
}
else
{
lean_object* v___x_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v_c_3585_; lean_object* v___x_3586_; 
v___x_3580_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2);
v___x_3581_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5);
v___x_3582_ = l_Lean_Options_empty;
v___x_3583_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3583_, 0, v___x_3577_);
lean_ctor_set(v___x_3583_, 1, v___x_3580_);
lean_ctor_set(v___x_3583_, 2, v___x_3581_);
lean_ctor_set(v___x_3583_, 3, v___x_3582_);
lean_inc(v_declHint_3569_);
v___x_3584_ = l_Lean_MessageData_ofConstName(v_declHint_3569_, v___x_3574_);
v_c_3585_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_3585_, 0, v___x_3583_);
lean_ctor_set(v_c_3585_, 1, v___x_3584_);
v___x_3586_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_3573_, v_declHint_3569_);
if (lean_obj_tag(v___x_3586_) == 0)
{
lean_object* v___x_3587_; lean_object* v___x_3588_; lean_object* v___x_3589_; lean_object* v___x_3590_; lean_object* v___x_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; 
lean_dec_ref(v_env_3573_);
lean_dec(v_declHint_3569_);
v___x_3587_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7);
v___x_3588_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3588_, 0, v___x_3587_);
lean_ctor_set(v___x_3588_, 1, v_c_3585_);
v___x_3589_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9);
v___x_3590_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3590_, 0, v___x_3588_);
lean_ctor_set(v___x_3590_, 1, v___x_3589_);
v___x_3591_ = l_Lean_MessageData_note(v___x_3590_);
v___x_3592_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3592_, 0, v_msg_3568_);
lean_ctor_set(v___x_3592_, 1, v___x_3591_);
v___x_3593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3593_, 0, v___x_3592_);
return v___x_3593_;
}
else
{
lean_object* v_val_3594_; lean_object* v___x_3596_; uint8_t v_isShared_3597_; uint8_t v_isSharedCheck_3629_; 
v_val_3594_ = lean_ctor_get(v___x_3586_, 0);
v_isSharedCheck_3629_ = !lean_is_exclusive(v___x_3586_);
if (v_isSharedCheck_3629_ == 0)
{
v___x_3596_ = v___x_3586_;
v_isShared_3597_ = v_isSharedCheck_3629_;
goto v_resetjp_3595_;
}
else
{
lean_inc(v_val_3594_);
lean_dec(v___x_3586_);
v___x_3596_ = lean_box(0);
v_isShared_3597_ = v_isSharedCheck_3629_;
goto v_resetjp_3595_;
}
v_resetjp_3595_:
{
lean_object* v___x_3598_; lean_object* v___x_3599_; lean_object* v___x_3600_; lean_object* v_mod_3601_; uint8_t v___x_3602_; 
v___x_3598_ = lean_box(0);
v___x_3599_ = l_Lean_Environment_header(v_env_3573_);
lean_dec_ref(v_env_3573_);
v___x_3600_ = l_Lean_EnvironmentHeader_moduleNames(v___x_3599_);
v_mod_3601_ = lean_array_get(v___x_3598_, v___x_3600_, v_val_3594_);
lean_dec(v_val_3594_);
lean_dec_ref(v___x_3600_);
v___x_3602_ = l_Lean_isPrivateName(v_declHint_3569_);
lean_dec(v_declHint_3569_);
if (v___x_3602_ == 0)
{
lean_object* v___x_3603_; lean_object* v___x_3604_; lean_object* v___x_3605_; lean_object* v___x_3606_; lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v___x_3611_; lean_object* v___x_3612_; lean_object* v___x_3614_; 
v___x_3603_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11);
v___x_3604_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3604_, 0, v___x_3603_);
lean_ctor_set(v___x_3604_, 1, v_c_3585_);
v___x_3605_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13);
v___x_3606_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3606_, 0, v___x_3604_);
lean_ctor_set(v___x_3606_, 1, v___x_3605_);
v___x_3607_ = l_Lean_MessageData_ofName(v_mod_3601_);
v___x_3608_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3608_, 0, v___x_3606_);
lean_ctor_set(v___x_3608_, 1, v___x_3607_);
v___x_3609_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7);
v___x_3610_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3610_, 0, v___x_3608_);
lean_ctor_set(v___x_3610_, 1, v___x_3609_);
v___x_3611_ = l_Lean_MessageData_note(v___x_3610_);
v___x_3612_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3612_, 0, v_msg_3568_);
lean_ctor_set(v___x_3612_, 1, v___x_3611_);
if (v_isShared_3597_ == 0)
{
lean_ctor_set_tag(v___x_3596_, 0);
lean_ctor_set(v___x_3596_, 0, v___x_3612_);
v___x_3614_ = v___x_3596_;
goto v_reusejp_3613_;
}
else
{
lean_object* v_reuseFailAlloc_3615_; 
v_reuseFailAlloc_3615_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3615_, 0, v___x_3612_);
v___x_3614_ = v_reuseFailAlloc_3615_;
goto v_reusejp_3613_;
}
v_reusejp_3613_:
{
return v___x_3614_;
}
}
else
{
lean_object* v___x_3616_; lean_object* v___x_3617_; lean_object* v___x_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; lean_object* v___x_3621_; lean_object* v___x_3622_; lean_object* v___x_3623_; lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3627_; 
v___x_3616_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7);
v___x_3617_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3617_, 0, v___x_3616_);
lean_ctor_set(v___x_3617_, 1, v_c_3585_);
v___x_3618_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15);
v___x_3619_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3619_, 0, v___x_3617_);
lean_ctor_set(v___x_3619_, 1, v___x_3618_);
v___x_3620_ = l_Lean_MessageData_ofName(v_mod_3601_);
v___x_3621_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3621_, 0, v___x_3619_);
lean_ctor_set(v___x_3621_, 1, v___x_3620_);
v___x_3622_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17);
v___x_3623_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3623_, 0, v___x_3621_);
lean_ctor_set(v___x_3623_, 1, v___x_3622_);
v___x_3624_ = l_Lean_MessageData_note(v___x_3623_);
v___x_3625_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3625_, 0, v_msg_3568_);
lean_ctor_set(v___x_3625_, 1, v___x_3624_);
if (v_isShared_3597_ == 0)
{
lean_ctor_set_tag(v___x_3596_, 0);
lean_ctor_set(v___x_3596_, 0, v___x_3625_);
v___x_3627_ = v___x_3596_;
goto v_reusejp_3626_;
}
else
{
lean_object* v_reuseFailAlloc_3628_; 
v_reuseFailAlloc_3628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3628_, 0, v___x_3625_);
v___x_3627_ = v_reuseFailAlloc_3628_;
goto v_reusejp_3626_;
}
v_reusejp_3626_:
{
return v___x_3627_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3630_; 
lean_dec_ref(v_env_3573_);
lean_dec(v_declHint_3569_);
v___x_3630_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3630_, 0, v_msg_3568_);
return v___x_3630_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___boxed(lean_object* v_msg_3631_, lean_object* v_declHint_3632_, lean_object* v___y_3633_, lean_object* v___y_3634_){
_start:
{
lean_object* v_res_3635_; 
v_res_3635_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg(v_msg_3631_, v_declHint_3632_, v___y_3633_);
lean_dec(v___y_3633_);
return v_res_3635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28(lean_object* v_msg_3636_, lean_object* v_declHint_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_){
_start:
{
lean_object* v___x_3645_; lean_object* v_a_3646_; lean_object* v___x_3648_; uint8_t v_isShared_3649_; uint8_t v_isSharedCheck_3655_; 
v___x_3645_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg(v_msg_3636_, v_declHint_3637_, v___y_3643_);
v_a_3646_ = lean_ctor_get(v___x_3645_, 0);
v_isSharedCheck_3655_ = !lean_is_exclusive(v___x_3645_);
if (v_isSharedCheck_3655_ == 0)
{
v___x_3648_ = v___x_3645_;
v_isShared_3649_ = v_isSharedCheck_3655_;
goto v_resetjp_3647_;
}
else
{
lean_inc(v_a_3646_);
lean_dec(v___x_3645_);
v___x_3648_ = lean_box(0);
v_isShared_3649_ = v_isSharedCheck_3655_;
goto v_resetjp_3647_;
}
v_resetjp_3647_:
{
lean_object* v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3653_; 
v___x_3650_ = l_Lean_unknownIdentifierMessageTag;
v___x_3651_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3651_, 0, v___x_3650_);
lean_ctor_set(v___x_3651_, 1, v_a_3646_);
if (v_isShared_3649_ == 0)
{
lean_ctor_set(v___x_3648_, 0, v___x_3651_);
v___x_3653_ = v___x_3648_;
goto v_reusejp_3652_;
}
else
{
lean_object* v_reuseFailAlloc_3654_; 
v_reuseFailAlloc_3654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3654_, 0, v___x_3651_);
v___x_3653_ = v_reuseFailAlloc_3654_;
goto v_reusejp_3652_;
}
v_reusejp_3652_:
{
return v___x_3653_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28___boxed(lean_object* v_msg_3656_, lean_object* v_declHint_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_, lean_object* v___y_3663_, lean_object* v___y_3664_){
_start:
{
lean_object* v_res_3665_; 
v_res_3665_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28(v_msg_3656_, v_declHint_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
lean_dec(v___y_3663_);
lean_dec_ref(v___y_3662_);
lean_dec(v___y_3661_);
lean_dec_ref(v___y_3660_);
lean_dec(v___y_3659_);
lean_dec_ref(v___y_3658_);
return v_res_3665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg(lean_object* v_ref_3666_, lean_object* v_msg_3667_, lean_object* v_declHint_3668_, lean_object* v___y_3669_, lean_object* v___y_3670_, lean_object* v___y_3671_, lean_object* v___y_3672_, lean_object* v___y_3673_, lean_object* v___y_3674_){
_start:
{
lean_object* v___x_3676_; lean_object* v_a_3677_; lean_object* v___x_3678_; 
v___x_3676_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28(v_msg_3667_, v_declHint_3668_, v___y_3669_, v___y_3670_, v___y_3671_, v___y_3672_, v___y_3673_, v___y_3674_);
v_a_3677_ = lean_ctor_get(v___x_3676_, 0);
lean_inc(v_a_3677_);
lean_dec_ref(v___x_3676_);
v___x_3678_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__5___redArg(v_ref_3666_, v_a_3677_, v___y_3669_, v___y_3670_, v___y_3671_, v___y_3672_, v___y_3673_, v___y_3674_);
return v___x_3678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg___boxed(lean_object* v_ref_3679_, lean_object* v_msg_3680_, lean_object* v_declHint_3681_, lean_object* v___y_3682_, lean_object* v___y_3683_, lean_object* v___y_3684_, lean_object* v___y_3685_, lean_object* v___y_3686_, lean_object* v___y_3687_, lean_object* v___y_3688_){
_start:
{
lean_object* v_res_3689_; 
v_res_3689_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg(v_ref_3679_, v_msg_3680_, v_declHint_3681_, v___y_3682_, v___y_3683_, v___y_3684_, v___y_3685_, v___y_3686_, v___y_3687_);
lean_dec(v___y_3687_);
lean_dec_ref(v___y_3686_);
lean_dec(v___y_3685_);
lean_dec_ref(v___y_3684_);
lean_dec(v___y_3683_);
lean_dec_ref(v___y_3682_);
lean_dec(v_ref_3679_);
return v_res_3689_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1(void){
_start:
{
lean_object* v___x_3691_; lean_object* v___x_3692_; 
v___x_3691_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__0));
v___x_3692_ = l_Lean_stringToMessageData(v___x_3691_);
return v___x_3692_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3(void){
_start:
{
lean_object* v___x_3694_; lean_object* v___x_3695_; 
v___x_3694_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__2));
v___x_3695_ = l_Lean_stringToMessageData(v___x_3694_);
return v___x_3695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg(lean_object* v_ref_3696_, lean_object* v_constName_3697_, lean_object* v___y_3698_, lean_object* v___y_3699_, lean_object* v___y_3700_, lean_object* v___y_3701_, lean_object* v___y_3702_, lean_object* v___y_3703_){
_start:
{
lean_object* v___x_3705_; uint8_t v___x_3706_; lean_object* v___x_3707_; lean_object* v___x_3708_; lean_object* v___x_3709_; lean_object* v___x_3710_; lean_object* v___x_3711_; 
v___x_3705_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1);
v___x_3706_ = 0;
lean_inc(v_constName_3697_);
v___x_3707_ = l_Lean_MessageData_ofConstName(v_constName_3697_, v___x_3706_);
v___x_3708_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3708_, 0, v___x_3705_);
lean_ctor_set(v___x_3708_, 1, v___x_3707_);
v___x_3709_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3);
v___x_3710_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3710_, 0, v___x_3708_);
lean_ctor_set(v___x_3710_, 1, v___x_3709_);
v___x_3711_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg(v_ref_3696_, v___x_3710_, v_constName_3697_, v___y_3698_, v___y_3699_, v___y_3700_, v___y_3701_, v___y_3702_, v___y_3703_);
return v___x_3711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___boxed(lean_object* v_ref_3712_, lean_object* v_constName_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_, lean_object* v___y_3716_, lean_object* v___y_3717_, lean_object* v___y_3718_, lean_object* v___y_3719_, lean_object* v___y_3720_){
_start:
{
lean_object* v_res_3721_; 
v_res_3721_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg(v_ref_3712_, v_constName_3713_, v___y_3714_, v___y_3715_, v___y_3716_, v___y_3717_, v___y_3718_, v___y_3719_);
lean_dec(v___y_3719_);
lean_dec_ref(v___y_3718_);
lean_dec(v___y_3717_);
lean_dec_ref(v___y_3716_);
lean_dec(v___y_3715_);
lean_dec_ref(v___y_3714_);
lean_dec(v_ref_3712_);
return v_res_3721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg(lean_object* v_constName_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_){
_start:
{
lean_object* v_ref_3730_; lean_object* v___x_3731_; 
v_ref_3730_ = lean_ctor_get(v___y_3727_, 5);
v___x_3731_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg(v_ref_3730_, v_constName_3722_, v___y_3723_, v___y_3724_, v___y_3725_, v___y_3726_, v___y_3727_, v___y_3728_);
return v___x_3731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg___boxed(lean_object* v_constName_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_, lean_object* v___y_3737_, lean_object* v___y_3738_, lean_object* v___y_3739_){
_start:
{
lean_object* v_res_3740_; 
v_res_3740_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg(v_constName_3732_, v___y_3733_, v___y_3734_, v___y_3735_, v___y_3736_, v___y_3737_, v___y_3738_);
lean_dec(v___y_3738_);
lean_dec_ref(v___y_3737_);
lean_dec(v___y_3736_);
lean_dec_ref(v___y_3735_);
lean_dec(v___y_3734_);
lean_dec_ref(v___y_3733_);
return v_res_3740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14(lean_object* v_constName_3741_, lean_object* v___y_3742_, lean_object* v___y_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_, lean_object* v___y_3746_, lean_object* v___y_3747_){
_start:
{
lean_object* v___x_3749_; lean_object* v_env_3750_; uint8_t v___x_3751_; lean_object* v___x_3752_; 
v___x_3749_ = lean_st_ref_get(v___y_3747_);
v_env_3750_ = lean_ctor_get(v___x_3749_, 0);
lean_inc_ref(v_env_3750_);
lean_dec(v___x_3749_);
v___x_3751_ = 0;
lean_inc(v_constName_3741_);
v___x_3752_ = l_Lean_Environment_findConstVal_x3f(v_env_3750_, v_constName_3741_, v___x_3751_);
if (lean_obj_tag(v___x_3752_) == 0)
{
lean_object* v___x_3753_; 
v___x_3753_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg(v_constName_3741_, v___y_3742_, v___y_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_);
return v___x_3753_;
}
else
{
lean_object* v_val_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3761_; 
lean_dec(v_constName_3741_);
v_val_3754_ = lean_ctor_get(v___x_3752_, 0);
v_isSharedCheck_3761_ = !lean_is_exclusive(v___x_3752_);
if (v_isSharedCheck_3761_ == 0)
{
v___x_3756_ = v___x_3752_;
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_val_3754_);
lean_dec(v___x_3752_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___x_3759_; 
if (v_isShared_3757_ == 0)
{
lean_ctor_set_tag(v___x_3756_, 0);
v___x_3759_ = v___x_3756_;
goto v_reusejp_3758_;
}
else
{
lean_object* v_reuseFailAlloc_3760_; 
v_reuseFailAlloc_3760_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3760_, 0, v_val_3754_);
v___x_3759_ = v_reuseFailAlloc_3760_;
goto v_reusejp_3758_;
}
v_reusejp_3758_:
{
return v___x_3759_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14___boxed(lean_object* v_constName_3762_, lean_object* v___y_3763_, lean_object* v___y_3764_, lean_object* v___y_3765_, lean_object* v___y_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_){
_start:
{
lean_object* v_res_3770_; 
v_res_3770_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14(v_constName_3762_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
lean_dec(v___y_3768_);
lean_dec_ref(v___y_3767_);
lean_dec(v___y_3766_);
lean_dec_ref(v___y_3765_);
lean_dec(v___y_3764_);
lean_dec_ref(v___y_3763_);
return v_res_3770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7(lean_object* v_constName_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_, lean_object* v___y_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_){
_start:
{
lean_object* v___x_3779_; 
lean_inc(v_constName_3771_);
v___x_3779_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14(v_constName_3771_, v___y_3772_, v___y_3773_, v___y_3774_, v___y_3775_, v___y_3776_, v___y_3777_);
if (lean_obj_tag(v___x_3779_) == 0)
{
lean_object* v_a_3780_; lean_object* v___x_3782_; uint8_t v_isShared_3783_; uint8_t v_isSharedCheck_3791_; 
v_a_3780_ = lean_ctor_get(v___x_3779_, 0);
v_isSharedCheck_3791_ = !lean_is_exclusive(v___x_3779_);
if (v_isSharedCheck_3791_ == 0)
{
v___x_3782_ = v___x_3779_;
v_isShared_3783_ = v_isSharedCheck_3791_;
goto v_resetjp_3781_;
}
else
{
lean_inc(v_a_3780_);
lean_dec(v___x_3779_);
v___x_3782_ = lean_box(0);
v_isShared_3783_ = v_isSharedCheck_3791_;
goto v_resetjp_3781_;
}
v_resetjp_3781_:
{
lean_object* v_levelParams_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3789_; 
v_levelParams_3784_ = lean_ctor_get(v_a_3780_, 1);
lean_inc(v_levelParams_3784_);
lean_dec(v_a_3780_);
v___x_3785_ = lean_box(0);
v___x_3786_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_addRelatedDecl_spec__4(v_levelParams_3784_, v___x_3785_);
v___x_3787_ = l_Lean_mkConst(v_constName_3771_, v___x_3786_);
if (v_isShared_3783_ == 0)
{
lean_ctor_set(v___x_3782_, 0, v___x_3787_);
v___x_3789_ = v___x_3782_;
goto v_reusejp_3788_;
}
else
{
lean_object* v_reuseFailAlloc_3790_; 
v_reuseFailAlloc_3790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3790_, 0, v___x_3787_);
v___x_3789_ = v_reuseFailAlloc_3790_;
goto v_reusejp_3788_;
}
v_reusejp_3788_:
{
return v___x_3789_;
}
}
}
else
{
lean_object* v_a_3792_; lean_object* v___x_3794_; uint8_t v_isShared_3795_; uint8_t v_isSharedCheck_3799_; 
lean_dec(v_constName_3771_);
v_a_3792_ = lean_ctor_get(v___x_3779_, 0);
v_isSharedCheck_3799_ = !lean_is_exclusive(v___x_3779_);
if (v_isSharedCheck_3799_ == 0)
{
v___x_3794_ = v___x_3779_;
v_isShared_3795_ = v_isSharedCheck_3799_;
goto v_resetjp_3793_;
}
else
{
lean_inc(v_a_3792_);
lean_dec(v___x_3779_);
v___x_3794_ = lean_box(0);
v_isShared_3795_ = v_isSharedCheck_3799_;
goto v_resetjp_3793_;
}
v_resetjp_3793_:
{
lean_object* v___x_3797_; 
if (v_isShared_3795_ == 0)
{
v___x_3797_ = v___x_3794_;
goto v_reusejp_3796_;
}
else
{
lean_object* v_reuseFailAlloc_3798_; 
v_reuseFailAlloc_3798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3798_, 0, v_a_3792_);
v___x_3797_ = v_reuseFailAlloc_3798_;
goto v_reusejp_3796_;
}
v_reusejp_3796_:
{
return v___x_3797_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7___boxed(lean_object* v_constName_3800_, lean_object* v___y_3801_, lean_object* v___y_3802_, lean_object* v___y_3803_, lean_object* v___y_3804_, lean_object* v___y_3805_, lean_object* v___y_3806_, lean_object* v___y_3807_){
_start:
{
lean_object* v_res_3808_; 
v_res_3808_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7(v_constName_3800_, v___y_3801_, v___y_3802_, v___y_3803_, v___y_3804_, v___y_3805_, v___y_3806_);
lean_dec(v___y_3806_);
lean_dec_ref(v___y_3805_);
lean_dec(v___y_3804_);
lean_dec_ref(v___y_3803_);
lean_dec(v___y_3802_);
lean_dec_ref(v___y_3801_);
return v_res_3808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0(lean_object* v_attrs_3809_, lean_object* v_src_3810_, lean_object* v_tgt_3811_, uint8_t v_hoverInfo_3812_, lean_object* v_ref_3813_, lean_object* v___x_3814_, uint8_t v___x_3815_, uint8_t v___x_3816_, lean_object* v___y_3817_, lean_object* v___y_3818_, lean_object* v___y_3819_, lean_object* v___y_3820_, lean_object* v___y_3821_, lean_object* v___y_3822_){
_start:
{
lean_object* v___x_3824_; 
v___x_3824_ = lp_mathlib_Mathlib_Tactic_elabOptAttrArg(v_attrs_3809_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_, v___y_3822_);
if (lean_obj_tag(v___x_3824_) == 0)
{
lean_object* v_a_3825_; lean_object* v___x_3826_; 
v_a_3825_ = lean_ctor_get(v___x_3824_, 0);
lean_inc_n(v_a_3825_, 2);
lean_dec_ref_known(v___x_3824_, 1);
v___x_3826_ = l_Lean_Elab_Term_applyAttributes(v_src_3810_, v_a_3825_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_, v___y_3822_);
if (lean_obj_tag(v___x_3826_) == 0)
{
lean_object* v___x_3827_; 
lean_dec_ref_known(v___x_3826_, 1);
lean_inc(v_tgt_3811_);
v___x_3827_ = l_Lean_Elab_Term_applyAttributes(v_tgt_3811_, v_a_3825_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_, v___y_3822_);
if (lean_obj_tag(v___x_3827_) == 0)
{
lean_object* v___x_3829_; uint8_t v_isShared_3830_; uint8_t v_isSharedCheck_3847_; 
v_isSharedCheck_3847_ = !lean_is_exclusive(v___x_3827_);
if (v_isSharedCheck_3847_ == 0)
{
lean_object* v_unused_3848_; 
v_unused_3848_ = lean_ctor_get(v___x_3827_, 0);
lean_dec(v_unused_3848_);
v___x_3829_ = v___x_3827_;
v_isShared_3830_ = v_isSharedCheck_3847_;
goto v_resetjp_3828_;
}
else
{
lean_dec(v___x_3827_);
v___x_3829_ = lean_box(0);
v_isShared_3830_ = v_isSharedCheck_3847_;
goto v_resetjp_3828_;
}
v_resetjp_3828_:
{
if (v_hoverInfo_3812_ == 0)
{
lean_object* v___x_3831_; lean_object* v___x_3833_; 
lean_dec(v___x_3814_);
lean_dec(v_ref_3813_);
lean_dec(v_tgt_3811_);
v___x_3831_ = lean_box(0);
if (v_isShared_3830_ == 0)
{
lean_ctor_set(v___x_3829_, 0, v___x_3831_);
v___x_3833_ = v___x_3829_;
goto v_reusejp_3832_;
}
else
{
lean_object* v_reuseFailAlloc_3834_; 
v_reuseFailAlloc_3834_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3834_, 0, v___x_3831_);
v___x_3833_ = v_reuseFailAlloc_3834_;
goto v_reusejp_3832_;
}
v_reusejp_3832_:
{
return v___x_3833_;
}
}
else
{
lean_object* v___x_3835_; 
lean_del_object(v___x_3829_);
v___x_3835_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7(v_tgt_3811_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_, v___y_3822_);
if (lean_obj_tag(v___x_3835_) == 0)
{
lean_object* v_a_3836_; lean_object* v___x_3837_; lean_object* v___x_3838_; 
v_a_3836_ = lean_ctor_get(v___x_3835_, 0);
lean_inc(v_a_3836_);
lean_dec_ref_known(v___x_3835_, 1);
v___x_3837_ = lean_box(0);
v___x_3838_ = l_Lean_Elab_Term_addTermInfo_x27(v_ref_3813_, v_a_3836_, v___x_3837_, v___x_3837_, v___x_3814_, v___x_3815_, v___x_3816_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_, v___y_3822_);
return v___x_3838_;
}
else
{
lean_object* v_a_3839_; lean_object* v___x_3841_; uint8_t v_isShared_3842_; uint8_t v_isSharedCheck_3846_; 
lean_dec(v___x_3814_);
lean_dec(v_ref_3813_);
v_a_3839_ = lean_ctor_get(v___x_3835_, 0);
v_isSharedCheck_3846_ = !lean_is_exclusive(v___x_3835_);
if (v_isSharedCheck_3846_ == 0)
{
v___x_3841_ = v___x_3835_;
v_isShared_3842_ = v_isSharedCheck_3846_;
goto v_resetjp_3840_;
}
else
{
lean_inc(v_a_3839_);
lean_dec(v___x_3835_);
v___x_3841_ = lean_box(0);
v_isShared_3842_ = v_isSharedCheck_3846_;
goto v_resetjp_3840_;
}
v_resetjp_3840_:
{
lean_object* v___x_3844_; 
if (v_isShared_3842_ == 0)
{
v___x_3844_ = v___x_3841_;
goto v_reusejp_3843_;
}
else
{
lean_object* v_reuseFailAlloc_3845_; 
v_reuseFailAlloc_3845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3845_, 0, v_a_3839_);
v___x_3844_ = v_reuseFailAlloc_3845_;
goto v_reusejp_3843_;
}
v_reusejp_3843_:
{
return v___x_3844_;
}
}
}
}
}
}
else
{
lean_dec(v___x_3814_);
lean_dec(v_ref_3813_);
lean_dec(v_tgt_3811_);
return v___x_3827_;
}
}
else
{
lean_dec(v_a_3825_);
lean_dec(v___x_3814_);
lean_dec(v_ref_3813_);
lean_dec(v_tgt_3811_);
return v___x_3826_;
}
}
else
{
lean_object* v_a_3849_; lean_object* v___x_3851_; uint8_t v_isShared_3852_; uint8_t v_isSharedCheck_3856_; 
lean_dec(v___x_3814_);
lean_dec(v_ref_3813_);
lean_dec(v_tgt_3811_);
lean_dec(v_src_3810_);
v_a_3849_ = lean_ctor_get(v___x_3824_, 0);
v_isSharedCheck_3856_ = !lean_is_exclusive(v___x_3824_);
if (v_isSharedCheck_3856_ == 0)
{
v___x_3851_ = v___x_3824_;
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
else
{
lean_inc(v_a_3849_);
lean_dec(v___x_3824_);
v___x_3851_ = lean_box(0);
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
v_resetjp_3850_:
{
lean_object* v___x_3854_; 
if (v_isShared_3852_ == 0)
{
v___x_3854_ = v___x_3851_;
goto v_reusejp_3853_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v_a_3849_);
v___x_3854_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3853_;
}
v_reusejp_3853_:
{
return v___x_3854_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0___boxed(lean_object* v_attrs_3857_, lean_object* v_src_3858_, lean_object* v_tgt_3859_, lean_object* v_hoverInfo_3860_, lean_object* v_ref_3861_, lean_object* v___x_3862_, lean_object* v___x_3863_, lean_object* v___x_3864_, lean_object* v___y_3865_, lean_object* v___y_3866_, lean_object* v___y_3867_, lean_object* v___y_3868_, lean_object* v___y_3869_, lean_object* v___y_3870_, lean_object* v___y_3871_){
_start:
{
uint8_t v_hoverInfo_boxed_3872_; uint8_t v___x_24500__boxed_3873_; uint8_t v___x_24501__boxed_3874_; lean_object* v_res_3875_; 
v_hoverInfo_boxed_3872_ = lean_unbox(v_hoverInfo_3860_);
v___x_24500__boxed_3873_ = lean_unbox(v___x_3863_);
v___x_24501__boxed_3874_ = lean_unbox(v___x_3864_);
v_res_3875_ = lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0(v_attrs_3857_, v_src_3858_, v_tgt_3859_, v_hoverInfo_boxed_3872_, v_ref_3861_, v___x_3862_, v___x_24500__boxed_3873_, v___x_24501__boxed_3874_, v___y_3865_, v___y_3866_, v___y_3867_, v___y_3868_, v___y_3869_, v___y_3870_);
lean_dec(v___y_3870_);
lean_dec_ref(v___y_3869_);
lean_dec(v___y_3868_);
lean_dec_ref(v___y_3867_);
lean_dec(v___y_3866_);
lean_dec_ref(v___y_3865_);
return v_res_3875_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1(uint8_t v___x_3876_, lean_object* v_x_3877_){
_start:
{
return v___x_3876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1___boxed(lean_object* v___x_3878_, lean_object* v_x_3879_){
_start:
{
uint8_t v___x_24598__boxed_3880_; uint8_t v_res_3881_; lean_object* v_r_3882_; 
v___x_24598__boxed_3880_ = lean_unbox(v___x_3878_);
v_res_3881_ = lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__1(v___x_24598__boxed_3880_, v_x_3879_);
lean_dec(v_x_3879_);
v_r_3882_ = lean_box(v_res_3881_);
return v_r_3882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg(lean_object* v_msg_3883_, lean_object* v_declHint_3884_, lean_object* v___y_3885_){
_start:
{
lean_object* v___x_3887_; lean_object* v_env_3888_; uint8_t v___x_3889_; 
v___x_3887_ = lean_st_ref_get(v___y_3885_);
v_env_3888_ = lean_ctor_get(v___x_3887_, 0);
lean_inc_ref(v_env_3888_);
lean_dec(v___x_3887_);
v___x_3889_ = l_Lean_Name_isAnonymous(v_declHint_3884_);
if (v___x_3889_ == 0)
{
uint8_t v_isExporting_3890_; 
v_isExporting_3890_ = lean_ctor_get_uint8(v_env_3888_, sizeof(void*)*8);
if (v_isExporting_3890_ == 0)
{
lean_object* v___x_3891_; 
lean_dec_ref(v_env_3888_);
lean_dec(v_declHint_3884_);
v___x_3891_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3891_, 0, v_msg_3883_);
return v___x_3891_;
}
else
{
lean_object* v___x_3892_; uint8_t v___x_3893_; 
lean_inc_ref(v_env_3888_);
v___x_3892_ = l_Lean_Environment_setExporting(v_env_3888_, v___x_3889_);
lean_inc(v_declHint_3884_);
lean_inc_ref(v___x_3892_);
v___x_3893_ = l_Lean_Environment_contains(v___x_3892_, v_declHint_3884_, v_isExporting_3890_);
if (v___x_3893_ == 0)
{
lean_object* v___x_3894_; 
lean_dec_ref(v___x_3892_);
lean_dec_ref(v_env_3888_);
lean_dec(v_declHint_3884_);
v___x_3894_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3894_, 0, v_msg_3883_);
return v___x_3894_;
}
else
{
lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v_c_3902_; lean_object* v___x_3903_; 
v___x_3895_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__2);
v___x_3896_ = lean_unsigned_to_nat(32u);
v___x_3897_ = lean_mk_empty_array_with_capacity(v___x_3896_);
lean_dec_ref(v___x_3897_);
v___x_3898_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__5);
v___x_3899_ = l_Lean_Options_empty;
v___x_3900_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3900_, 0, v___x_3892_);
lean_ctor_set(v___x_3900_, 1, v___x_3895_);
lean_ctor_set(v___x_3900_, 2, v___x_3898_);
lean_ctor_set(v___x_3900_, 3, v___x_3899_);
lean_inc(v_declHint_3884_);
v___x_3901_ = l_Lean_MessageData_ofConstName(v_declHint_3884_, v___x_3889_);
v_c_3902_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_3902_, 0, v___x_3900_);
lean_ctor_set(v_c_3902_, 1, v___x_3901_);
v___x_3903_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_3888_, v_declHint_3884_);
if (lean_obj_tag(v___x_3903_) == 0)
{
lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; 
lean_dec_ref(v_env_3888_);
lean_dec(v_declHint_3884_);
v___x_3904_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7);
v___x_3905_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3905_, 0, v___x_3904_);
lean_ctor_set(v___x_3905_, 1, v_c_3902_);
v___x_3906_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__9);
v___x_3907_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3907_, 0, v___x_3905_);
lean_ctor_set(v___x_3907_, 1, v___x_3906_);
v___x_3908_ = l_Lean_MessageData_note(v___x_3907_);
v___x_3909_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3909_, 0, v_msg_3883_);
lean_ctor_set(v___x_3909_, 1, v___x_3908_);
v___x_3910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3910_, 0, v___x_3909_);
return v___x_3910_;
}
else
{
lean_object* v_val_3911_; lean_object* v___x_3913_; uint8_t v_isShared_3914_; uint8_t v_isSharedCheck_3946_; 
v_val_3911_ = lean_ctor_get(v___x_3903_, 0);
v_isSharedCheck_3946_ = !lean_is_exclusive(v___x_3903_);
if (v_isSharedCheck_3946_ == 0)
{
v___x_3913_ = v___x_3903_;
v_isShared_3914_ = v_isSharedCheck_3946_;
goto v_resetjp_3912_;
}
else
{
lean_inc(v_val_3911_);
lean_dec(v___x_3903_);
v___x_3913_ = lean_box(0);
v_isShared_3914_ = v_isSharedCheck_3946_;
goto v_resetjp_3912_;
}
v_resetjp_3912_:
{
lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; lean_object* v_mod_3918_; uint8_t v___x_3919_; 
v___x_3915_ = lean_box(0);
v___x_3916_ = l_Lean_Environment_header(v_env_3888_);
lean_dec_ref(v_env_3888_);
v___x_3917_ = l_Lean_EnvironmentHeader_moduleNames(v___x_3916_);
v_mod_3918_ = lean_array_get(v___x_3915_, v___x_3917_, v_val_3911_);
lean_dec(v_val_3911_);
lean_dec_ref(v___x_3917_);
v___x_3919_ = l_Lean_isPrivateName(v_declHint_3884_);
lean_dec(v_declHint_3884_);
if (v___x_3919_ == 0)
{
lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; lean_object* v___x_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3931_; 
v___x_3920_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__11);
v___x_3921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3921_, 0, v___x_3920_);
lean_ctor_set(v___x_3921_, 1, v_c_3902_);
v___x_3922_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__13);
v___x_3923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3923_, 0, v___x_3921_);
lean_ctor_set(v___x_3923_, 1, v___x_3922_);
v___x_3924_ = l_Lean_MessageData_ofName(v_mod_3918_);
v___x_3925_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3925_, 0, v___x_3923_);
lean_ctor_set(v___x_3925_, 1, v___x_3924_);
v___x_3926_ = lean_obj_once(&lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7, &lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7_once, _init_lp_mathlib_Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1___lam__1___closed__7);
v___x_3927_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3927_, 0, v___x_3925_);
lean_ctor_set(v___x_3927_, 1, v___x_3926_);
v___x_3928_ = l_Lean_MessageData_note(v___x_3927_);
v___x_3929_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3929_, 0, v_msg_3883_);
lean_ctor_set(v___x_3929_, 1, v___x_3928_);
if (v_isShared_3914_ == 0)
{
lean_ctor_set_tag(v___x_3913_, 0);
lean_ctor_set(v___x_3913_, 0, v___x_3929_);
v___x_3931_ = v___x_3913_;
goto v_reusejp_3930_;
}
else
{
lean_object* v_reuseFailAlloc_3932_; 
v_reuseFailAlloc_3932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3932_, 0, v___x_3929_);
v___x_3931_ = v_reuseFailAlloc_3932_;
goto v_reusejp_3930_;
}
v_reusejp_3930_:
{
return v___x_3931_;
}
}
else
{
lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; lean_object* v___x_3938_; lean_object* v___x_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3944_; 
v___x_3933_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__7);
v___x_3934_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3934_, 0, v___x_3933_);
lean_ctor_set(v___x_3934_, 1, v_c_3902_);
v___x_3935_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__15);
v___x_3936_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3936_, 0, v___x_3934_);
lean_ctor_set(v___x_3936_, 1, v___x_3935_);
v___x_3937_ = l_Lean_MessageData_ofName(v_mod_3918_);
v___x_3938_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3938_, 0, v___x_3936_);
lean_ctor_set(v___x_3938_, 1, v___x_3937_);
v___x_3939_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__17);
v___x_3940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3940_, 0, v___x_3938_);
lean_ctor_set(v___x_3940_, 1, v___x_3939_);
v___x_3941_ = l_Lean_MessageData_note(v___x_3940_);
v___x_3942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3942_, 0, v_msg_3883_);
lean_ctor_set(v___x_3942_, 1, v___x_3941_);
if (v_isShared_3914_ == 0)
{
lean_ctor_set_tag(v___x_3913_, 0);
lean_ctor_set(v___x_3913_, 0, v___x_3942_);
v___x_3944_ = v___x_3913_;
goto v_reusejp_3943_;
}
else
{
lean_object* v_reuseFailAlloc_3945_; 
v_reuseFailAlloc_3945_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3945_, 0, v___x_3942_);
v___x_3944_ = v_reuseFailAlloc_3945_;
goto v_reusejp_3943_;
}
v_reusejp_3943_:
{
return v___x_3944_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3947_; 
lean_dec_ref(v_env_3888_);
lean_dec(v_declHint_3884_);
v___x_3947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3947_, 0, v_msg_3883_);
return v___x_3947_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg___boxed(lean_object* v_msg_3948_, lean_object* v_declHint_3949_, lean_object* v___y_3950_, lean_object* v___y_3951_){
_start:
{
lean_object* v_res_3952_; 
v_res_3952_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg(v_msg_3948_, v_declHint_3949_, v___y_3950_);
lean_dec(v___y_3950_);
return v_res_3952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22(lean_object* v_msg_3953_, lean_object* v_declHint_3954_, lean_object* v___y_3955_, lean_object* v___y_3956_, lean_object* v___y_3957_, lean_object* v___y_3958_){
_start:
{
lean_object* v___x_3960_; lean_object* v_a_3961_; lean_object* v___x_3963_; uint8_t v_isShared_3964_; uint8_t v_isSharedCheck_3970_; 
v___x_3960_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg(v_msg_3953_, v_declHint_3954_, v___y_3958_);
v_a_3961_ = lean_ctor_get(v___x_3960_, 0);
v_isSharedCheck_3970_ = !lean_is_exclusive(v___x_3960_);
if (v_isSharedCheck_3970_ == 0)
{
v___x_3963_ = v___x_3960_;
v_isShared_3964_ = v_isSharedCheck_3970_;
goto v_resetjp_3962_;
}
else
{
lean_inc(v_a_3961_);
lean_dec(v___x_3960_);
v___x_3963_ = lean_box(0);
v_isShared_3964_ = v_isSharedCheck_3970_;
goto v_resetjp_3962_;
}
v_resetjp_3962_:
{
lean_object* v___x_3965_; lean_object* v___x_3966_; lean_object* v___x_3968_; 
v___x_3965_ = l_Lean_unknownIdentifierMessageTag;
v___x_3966_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3966_, 0, v___x_3965_);
lean_ctor_set(v___x_3966_, 1, v_a_3961_);
if (v_isShared_3964_ == 0)
{
lean_ctor_set(v___x_3963_, 0, v___x_3966_);
v___x_3968_ = v___x_3963_;
goto v_reusejp_3967_;
}
else
{
lean_object* v_reuseFailAlloc_3969_; 
v_reuseFailAlloc_3969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3969_, 0, v___x_3966_);
v___x_3968_ = v_reuseFailAlloc_3969_;
goto v_reusejp_3967_;
}
v_reusejp_3967_:
{
return v___x_3968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22___boxed(lean_object* v_msg_3971_, lean_object* v_declHint_3972_, lean_object* v___y_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_){
_start:
{
lean_object* v_res_3978_; 
v_res_3978_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22(v_msg_3971_, v_declHint_3972_, v___y_3973_, v___y_3974_, v___y_3975_, v___y_3976_);
lean_dec(v___y_3976_);
lean_dec_ref(v___y_3975_);
lean_dec(v___y_3974_);
lean_dec_ref(v___y_3973_);
return v_res_3978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(lean_object* v_msg_3979_, lean_object* v___y_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_){
_start:
{
lean_object* v_ref_3985_; lean_object* v___x_3986_; lean_object* v_a_3987_; lean_object* v___x_3989_; uint8_t v_isShared_3990_; uint8_t v_isSharedCheck_3995_; 
v_ref_3985_ = lean_ctor_get(v___y_3982_, 5);
v___x_3986_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__4_spec__15(v_msg_3979_, v___y_3980_, v___y_3981_, v___y_3982_, v___y_3983_);
v_a_3987_ = lean_ctor_get(v___x_3986_, 0);
v_isSharedCheck_3995_ = !lean_is_exclusive(v___x_3986_);
if (v_isSharedCheck_3995_ == 0)
{
v___x_3989_ = v___x_3986_;
v_isShared_3990_ = v_isSharedCheck_3995_;
goto v_resetjp_3988_;
}
else
{
lean_inc(v_a_3987_);
lean_dec(v___x_3986_);
v___x_3989_ = lean_box(0);
v_isShared_3990_ = v_isSharedCheck_3995_;
goto v_resetjp_3988_;
}
v_resetjp_3988_:
{
lean_object* v___x_3991_; lean_object* v___x_3993_; 
lean_inc(v_ref_3985_);
v___x_3991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3991_, 0, v_ref_3985_);
lean_ctor_set(v___x_3991_, 1, v_a_3987_);
if (v_isShared_3990_ == 0)
{
lean_ctor_set_tag(v___x_3989_, 1);
lean_ctor_set(v___x_3989_, 0, v___x_3991_);
v___x_3993_ = v___x_3989_;
goto v_reusejp_3992_;
}
else
{
lean_object* v_reuseFailAlloc_3994_; 
v_reuseFailAlloc_3994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3994_, 0, v___x_3991_);
v___x_3993_ = v_reuseFailAlloc_3994_;
goto v_reusejp_3992_;
}
v_reusejp_3992_:
{
return v___x_3993_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg___boxed(lean_object* v_msg_3996_, lean_object* v___y_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_){
_start:
{
lean_object* v_res_4002_; 
v_res_4002_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v_msg_3996_, v___y_3997_, v___y_3998_, v___y_3999_, v___y_4000_);
lean_dec(v___y_4000_);
lean_dec_ref(v___y_3999_);
lean_dec(v___y_3998_);
lean_dec_ref(v___y_3997_);
return v_res_4002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg(lean_object* v_ref_4003_, lean_object* v_msg_4004_, lean_object* v___y_4005_, lean_object* v___y_4006_, lean_object* v___y_4007_, lean_object* v___y_4008_){
_start:
{
lean_object* v_fileName_4010_; lean_object* v_fileMap_4011_; lean_object* v_options_4012_; lean_object* v_currRecDepth_4013_; lean_object* v_maxRecDepth_4014_; lean_object* v_ref_4015_; lean_object* v_currNamespace_4016_; lean_object* v_openDecls_4017_; lean_object* v_initHeartbeats_4018_; lean_object* v_maxHeartbeats_4019_; lean_object* v_quotContext_4020_; lean_object* v_currMacroScope_4021_; uint8_t v_diag_4022_; lean_object* v_cancelTk_x3f_4023_; uint8_t v_suppressElabErrors_4024_; lean_object* v_inheritedTraceOptions_4025_; lean_object* v_ref_4026_; lean_object* v___x_4027_; lean_object* v___x_4028_; 
v_fileName_4010_ = lean_ctor_get(v___y_4007_, 0);
v_fileMap_4011_ = lean_ctor_get(v___y_4007_, 1);
v_options_4012_ = lean_ctor_get(v___y_4007_, 2);
v_currRecDepth_4013_ = lean_ctor_get(v___y_4007_, 3);
v_maxRecDepth_4014_ = lean_ctor_get(v___y_4007_, 4);
v_ref_4015_ = lean_ctor_get(v___y_4007_, 5);
v_currNamespace_4016_ = lean_ctor_get(v___y_4007_, 6);
v_openDecls_4017_ = lean_ctor_get(v___y_4007_, 7);
v_initHeartbeats_4018_ = lean_ctor_get(v___y_4007_, 8);
v_maxHeartbeats_4019_ = lean_ctor_get(v___y_4007_, 9);
v_quotContext_4020_ = lean_ctor_get(v___y_4007_, 10);
v_currMacroScope_4021_ = lean_ctor_get(v___y_4007_, 11);
v_diag_4022_ = lean_ctor_get_uint8(v___y_4007_, sizeof(void*)*14);
v_cancelTk_x3f_4023_ = lean_ctor_get(v___y_4007_, 12);
v_suppressElabErrors_4024_ = lean_ctor_get_uint8(v___y_4007_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4025_ = lean_ctor_get(v___y_4007_, 13);
v_ref_4026_ = l_Lean_replaceRef(v_ref_4003_, v_ref_4015_);
lean_inc_ref(v_inheritedTraceOptions_4025_);
lean_inc(v_cancelTk_x3f_4023_);
lean_inc(v_currMacroScope_4021_);
lean_inc(v_quotContext_4020_);
lean_inc(v_maxHeartbeats_4019_);
lean_inc(v_initHeartbeats_4018_);
lean_inc(v_openDecls_4017_);
lean_inc(v_currNamespace_4016_);
lean_inc(v_maxRecDepth_4014_);
lean_inc(v_currRecDepth_4013_);
lean_inc_ref(v_options_4012_);
lean_inc_ref(v_fileMap_4011_);
lean_inc_ref(v_fileName_4010_);
v___x_4027_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4027_, 0, v_fileName_4010_);
lean_ctor_set(v___x_4027_, 1, v_fileMap_4011_);
lean_ctor_set(v___x_4027_, 2, v_options_4012_);
lean_ctor_set(v___x_4027_, 3, v_currRecDepth_4013_);
lean_ctor_set(v___x_4027_, 4, v_maxRecDepth_4014_);
lean_ctor_set(v___x_4027_, 5, v_ref_4026_);
lean_ctor_set(v___x_4027_, 6, v_currNamespace_4016_);
lean_ctor_set(v___x_4027_, 7, v_openDecls_4017_);
lean_ctor_set(v___x_4027_, 8, v_initHeartbeats_4018_);
lean_ctor_set(v___x_4027_, 9, v_maxHeartbeats_4019_);
lean_ctor_set(v___x_4027_, 10, v_quotContext_4020_);
lean_ctor_set(v___x_4027_, 11, v_currMacroScope_4021_);
lean_ctor_set(v___x_4027_, 12, v_cancelTk_x3f_4023_);
lean_ctor_set(v___x_4027_, 13, v_inheritedTraceOptions_4025_);
lean_ctor_set_uint8(v___x_4027_, sizeof(void*)*14, v_diag_4022_);
lean_ctor_set_uint8(v___x_4027_, sizeof(void*)*14 + 1, v_suppressElabErrors_4024_);
v___x_4028_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v_msg_4004_, v___y_4005_, v___y_4006_, v___x_4027_, v___y_4008_);
lean_dec_ref_known(v___x_4027_, 14);
return v___x_4028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg___boxed(lean_object* v_ref_4029_, lean_object* v_msg_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_, lean_object* v___y_4034_, lean_object* v___y_4035_){
_start:
{
lean_object* v_res_4036_; 
v_res_4036_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg(v_ref_4029_, v_msg_4030_, v___y_4031_, v___y_4032_, v___y_4033_, v___y_4034_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v___y_4032_);
lean_dec_ref(v___y_4031_);
lean_dec(v_ref_4029_);
return v_res_4036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg(lean_object* v_ref_4037_, lean_object* v_msg_4038_, lean_object* v_declHint_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_){
_start:
{
lean_object* v___x_4045_; lean_object* v_a_4046_; lean_object* v___x_4047_; 
v___x_4045_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22(v_msg_4038_, v_declHint_4039_, v___y_4040_, v___y_4041_, v___y_4042_, v___y_4043_);
v_a_4046_ = lean_ctor_get(v___x_4045_, 0);
lean_inc(v_a_4046_);
lean_dec_ref(v___x_4045_);
v___x_4047_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg(v_ref_4037_, v_a_4046_, v___y_4040_, v___y_4041_, v___y_4042_, v___y_4043_);
return v___x_4047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg___boxed(lean_object* v_ref_4048_, lean_object* v_msg_4049_, lean_object* v_declHint_4050_, lean_object* v___y_4051_, lean_object* v___y_4052_, lean_object* v___y_4053_, lean_object* v___y_4054_, lean_object* v___y_4055_){
_start:
{
lean_object* v_res_4056_; 
v_res_4056_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg(v_ref_4048_, v_msg_4049_, v_declHint_4050_, v___y_4051_, v___y_4052_, v___y_4053_, v___y_4054_);
lean_dec(v___y_4054_);
lean_dec_ref(v___y_4053_);
lean_dec(v___y_4052_);
lean_dec_ref(v___y_4051_);
lean_dec(v_ref_4048_);
return v_res_4056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg(lean_object* v_ref_4057_, lean_object* v_constName_4058_, lean_object* v___y_4059_, lean_object* v___y_4060_, lean_object* v___y_4061_, lean_object* v___y_4062_){
_start:
{
lean_object* v___x_4064_; uint8_t v___x_4065_; lean_object* v___x_4066_; lean_object* v___x_4067_; lean_object* v___x_4068_; lean_object* v___x_4069_; lean_object* v___x_4070_; 
v___x_4064_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__1);
v___x_4065_ = 0;
lean_inc(v_constName_4058_);
v___x_4066_ = l_Lean_MessageData_ofConstName(v_constName_4058_, v___x_4065_);
v___x_4067_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4067_, 0, v___x_4064_);
lean_ctor_set(v___x_4067_, 1, v___x_4066_);
v___x_4068_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3);
v___x_4069_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4069_, 0, v___x_4067_);
lean_ctor_set(v___x_4069_, 1, v___x_4068_);
v___x_4070_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg(v_ref_4057_, v___x_4069_, v_constName_4058_, v___y_4059_, v___y_4060_, v___y_4061_, v___y_4062_);
return v___x_4070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg___boxed(lean_object* v_ref_4071_, lean_object* v_constName_4072_, lean_object* v___y_4073_, lean_object* v___y_4074_, lean_object* v___y_4075_, lean_object* v___y_4076_, lean_object* v___y_4077_){
_start:
{
lean_object* v_res_4078_; 
v_res_4078_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg(v_ref_4071_, v_constName_4072_, v___y_4073_, v___y_4074_, v___y_4075_, v___y_4076_);
lean_dec(v___y_4076_);
lean_dec_ref(v___y_4075_);
lean_dec(v___y_4074_);
lean_dec_ref(v___y_4073_);
lean_dec(v_ref_4071_);
return v_res_4078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(lean_object* v_constName_4079_, lean_object* v___y_4080_, lean_object* v___y_4081_, lean_object* v___y_4082_, lean_object* v___y_4083_){
_start:
{
lean_object* v_ref_4085_; lean_object* v___x_4086_; 
v_ref_4085_ = lean_ctor_get(v___y_4082_, 5);
v___x_4086_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg(v_ref_4085_, v_constName_4079_, v___y_4080_, v___y_4081_, v___y_4082_, v___y_4083_);
return v___x_4086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg___boxed(lean_object* v_constName_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_, lean_object* v___y_4090_, lean_object* v___y_4091_, lean_object* v___y_4092_){
_start:
{
lean_object* v_res_4093_; 
v_res_4093_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(v_constName_4087_, v___y_4088_, v___y_4089_, v___y_4090_, v___y_4091_);
lean_dec(v___y_4091_);
lean_dec_ref(v___y_4090_);
lean_dec(v___y_4089_);
lean_dec_ref(v___y_4088_);
return v_res_4093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2(lean_object* v_constName_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_, lean_object* v___y_4097_, lean_object* v___y_4098_){
_start:
{
lean_object* v___x_4100_; lean_object* v_env_4101_; uint8_t v___x_4102_; lean_object* v___x_4103_; 
v___x_4100_ = lean_st_ref_get(v___y_4098_);
v_env_4101_ = lean_ctor_get(v___x_4100_, 0);
lean_inc_ref(v_env_4101_);
lean_dec(v___x_4100_);
v___x_4102_ = 0;
lean_inc(v_constName_4094_);
v___x_4103_ = l_Lean_Environment_find_x3f(v_env_4101_, v_constName_4094_, v___x_4102_);
if (lean_obj_tag(v___x_4103_) == 0)
{
lean_object* v___x_4104_; 
v___x_4104_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(v_constName_4094_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_);
return v___x_4104_;
}
else
{
lean_object* v_val_4105_; lean_object* v___x_4107_; uint8_t v_isShared_4108_; uint8_t v_isSharedCheck_4112_; 
lean_dec(v_constName_4094_);
v_val_4105_ = lean_ctor_get(v___x_4103_, 0);
v_isSharedCheck_4112_ = !lean_is_exclusive(v___x_4103_);
if (v_isSharedCheck_4112_ == 0)
{
v___x_4107_ = v___x_4103_;
v_isShared_4108_ = v_isSharedCheck_4112_;
goto v_resetjp_4106_;
}
else
{
lean_inc(v_val_4105_);
lean_dec(v___x_4103_);
v___x_4107_ = lean_box(0);
v_isShared_4108_ = v_isSharedCheck_4112_;
goto v_resetjp_4106_;
}
v_resetjp_4106_:
{
lean_object* v___x_4110_; 
if (v_isShared_4108_ == 0)
{
lean_ctor_set_tag(v___x_4107_, 0);
v___x_4110_ = v___x_4107_;
goto v_reusejp_4109_;
}
else
{
lean_object* v_reuseFailAlloc_4111_; 
v_reuseFailAlloc_4111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4111_, 0, v_val_4105_);
v___x_4110_ = v_reuseFailAlloc_4111_;
goto v_reusejp_4109_;
}
v_reusejp_4109_:
{
return v___x_4110_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2___boxed(lean_object* v_constName_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_){
_start:
{
lean_object* v_res_4119_; 
v_res_4119_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2(v_constName_4113_, v___y_4114_, v___y_4115_, v___y_4116_, v___y_4117_);
lean_dec(v___y_4117_);
lean_dec_ref(v___y_4116_);
lean_dec(v___y_4115_);
lean_dec_ref(v___y_4114_);
return v_res_4119_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1(void){
_start:
{
lean_object* v___x_4121_; lean_object* v___x_4122_; 
v___x_4121_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__0));
v___x_4122_ = l_Lean_stringToMessageData(v___x_4121_);
return v___x_4122_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3(void){
_start:
{
lean_object* v___x_4124_; lean_object* v___x_4125_; 
v___x_4124_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__2));
v___x_4125_ = l_Lean_stringToMessageData(v___x_4124_);
return v___x_4125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8(lean_object* v_declName_4126_, lean_object* v_docString_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_){
_start:
{
lean_object* v___y_4134_; lean_object* v___y_4135_; lean_object* v___x_4175_; lean_object* v_env_4176_; lean_object* v___x_4177_; 
v___x_4175_ = lean_st_ref_get(v___y_4131_);
v_env_4176_ = lean_ctor_get(v___x_4175_, 0);
lean_inc_ref(v_env_4176_);
lean_dec(v___x_4175_);
v___x_4177_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_4176_, v_declName_4126_);
lean_dec_ref(v_env_4176_);
if (lean_obj_tag(v___x_4177_) == 0)
{
v___y_4134_ = v___y_4129_;
v___y_4135_ = v___y_4131_;
goto v___jp_4133_;
}
else
{
uint8_t v___x_4178_; lean_object* v___x_4179_; lean_object* v___x_4180_; lean_object* v___x_4181_; lean_object* v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; 
lean_dec_ref_known(v___x_4177_, 1);
lean_dec_ref(v_docString_4127_);
v___x_4178_ = 0;
v___x_4179_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__1);
v___x_4180_ = l_Lean_MessageData_ofConstName(v_declName_4126_, v___x_4178_);
v___x_4181_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4181_, 0, v___x_4179_);
lean_ctor_set(v___x_4181_, 1, v___x_4180_);
v___x_4182_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___closed__3);
v___x_4183_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4183_, 0, v___x_4181_);
lean_ctor_set(v___x_4183_, 1, v___x_4182_);
v___x_4184_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4183_, v___y_4128_, v___y_4129_, v___y_4130_, v___y_4131_);
return v___x_4184_;
}
v___jp_4133_:
{
lean_object* v___x_4136_; lean_object* v_env_4137_; lean_object* v_nextMacroScope_4138_; lean_object* v_ngen_4139_; lean_object* v_auxDeclNGen_4140_; lean_object* v_traceState_4141_; lean_object* v_messages_4142_; lean_object* v_infoState_4143_; lean_object* v_snapshotTasks_4144_; lean_object* v___x_4146_; uint8_t v_isShared_4147_; uint8_t v_isSharedCheck_4173_; 
v___x_4136_ = lean_st_ref_take(v___y_4135_);
v_env_4137_ = lean_ctor_get(v___x_4136_, 0);
v_nextMacroScope_4138_ = lean_ctor_get(v___x_4136_, 1);
v_ngen_4139_ = lean_ctor_get(v___x_4136_, 2);
v_auxDeclNGen_4140_ = lean_ctor_get(v___x_4136_, 3);
v_traceState_4141_ = lean_ctor_get(v___x_4136_, 4);
v_messages_4142_ = lean_ctor_get(v___x_4136_, 6);
v_infoState_4143_ = lean_ctor_get(v___x_4136_, 7);
v_snapshotTasks_4144_ = lean_ctor_get(v___x_4136_, 8);
v_isSharedCheck_4173_ = !lean_is_exclusive(v___x_4136_);
if (v_isSharedCheck_4173_ == 0)
{
lean_object* v_unused_4174_; 
v_unused_4174_ = lean_ctor_get(v___x_4136_, 5);
lean_dec(v_unused_4174_);
v___x_4146_ = v___x_4136_;
v_isShared_4147_ = v_isSharedCheck_4173_;
goto v_resetjp_4145_;
}
else
{
lean_inc(v_snapshotTasks_4144_);
lean_inc(v_infoState_4143_);
lean_inc(v_messages_4142_);
lean_inc(v_traceState_4141_);
lean_inc(v_auxDeclNGen_4140_);
lean_inc(v_ngen_4139_);
lean_inc(v_nextMacroScope_4138_);
lean_inc(v_env_4137_);
lean_dec(v___x_4136_);
v___x_4146_ = lean_box(0);
v_isShared_4147_ = v_isSharedCheck_4173_;
goto v_resetjp_4145_;
}
v_resetjp_4145_:
{
lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; lean_object* v___x_4153_; 
v___x_4148_ = l_Lean_docStringExt;
v___x_4149_ = l_String_removeLeadingSpaces(v_docString_4127_);
v___x_4150_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_4148_, v_env_4137_, v_declName_4126_, v___x_4149_);
v___x_4151_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_4147_ == 0)
{
lean_ctor_set(v___x_4146_, 5, v___x_4151_);
lean_ctor_set(v___x_4146_, 0, v___x_4150_);
v___x_4153_ = v___x_4146_;
goto v_reusejp_4152_;
}
else
{
lean_object* v_reuseFailAlloc_4172_; 
v_reuseFailAlloc_4172_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4172_, 0, v___x_4150_);
lean_ctor_set(v_reuseFailAlloc_4172_, 1, v_nextMacroScope_4138_);
lean_ctor_set(v_reuseFailAlloc_4172_, 2, v_ngen_4139_);
lean_ctor_set(v_reuseFailAlloc_4172_, 3, v_auxDeclNGen_4140_);
lean_ctor_set(v_reuseFailAlloc_4172_, 4, v_traceState_4141_);
lean_ctor_set(v_reuseFailAlloc_4172_, 5, v___x_4151_);
lean_ctor_set(v_reuseFailAlloc_4172_, 6, v_messages_4142_);
lean_ctor_set(v_reuseFailAlloc_4172_, 7, v_infoState_4143_);
lean_ctor_set(v_reuseFailAlloc_4172_, 8, v_snapshotTasks_4144_);
v___x_4153_ = v_reuseFailAlloc_4172_;
goto v_reusejp_4152_;
}
v_reusejp_4152_:
{
lean_object* v___x_4154_; lean_object* v___x_4155_; lean_object* v_mctx_4156_; lean_object* v_zetaDeltaFVarIds_4157_; lean_object* v_postponed_4158_; lean_object* v_diag_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4170_; 
v___x_4154_ = lean_st_ref_set(v___y_4135_, v___x_4153_);
v___x_4155_ = lean_st_ref_take(v___y_4134_);
v_mctx_4156_ = lean_ctor_get(v___x_4155_, 0);
v_zetaDeltaFVarIds_4157_ = lean_ctor_get(v___x_4155_, 2);
v_postponed_4158_ = lean_ctor_get(v___x_4155_, 3);
v_diag_4159_ = lean_ctor_get(v___x_4155_, 4);
v_isSharedCheck_4170_ = !lean_is_exclusive(v___x_4155_);
if (v_isSharedCheck_4170_ == 0)
{
lean_object* v_unused_4171_; 
v_unused_4171_ = lean_ctor_get(v___x_4155_, 1);
lean_dec(v_unused_4171_);
v___x_4161_ = v___x_4155_;
v_isShared_4162_ = v_isSharedCheck_4170_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_diag_4159_);
lean_inc(v_postponed_4158_);
lean_inc(v_zetaDeltaFVarIds_4157_);
lean_inc(v_mctx_4156_);
lean_dec(v___x_4155_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4170_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
lean_object* v___x_4163_; lean_object* v___x_4165_; 
v___x_4163_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_4162_ == 0)
{
lean_ctor_set(v___x_4161_, 1, v___x_4163_);
v___x_4165_ = v___x_4161_;
goto v_reusejp_4164_;
}
else
{
lean_object* v_reuseFailAlloc_4169_; 
v_reuseFailAlloc_4169_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4169_, 0, v_mctx_4156_);
lean_ctor_set(v_reuseFailAlloc_4169_, 1, v___x_4163_);
lean_ctor_set(v_reuseFailAlloc_4169_, 2, v_zetaDeltaFVarIds_4157_);
lean_ctor_set(v_reuseFailAlloc_4169_, 3, v_postponed_4158_);
lean_ctor_set(v_reuseFailAlloc_4169_, 4, v_diag_4159_);
v___x_4165_ = v_reuseFailAlloc_4169_;
goto v_reusejp_4164_;
}
v_reusejp_4164_:
{
lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_4168_; 
v___x_4166_ = lean_st_ref_set(v___y_4134_, v___x_4165_);
v___x_4167_ = lean_box(0);
v___x_4168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4168_, 0, v___x_4167_);
return v___x_4168_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8___boxed(lean_object* v_declName_4185_, lean_object* v_docString_4186_, lean_object* v___y_4187_, lean_object* v___y_4188_, lean_object* v___y_4189_, lean_object* v___y_4190_, lean_object* v___y_4191_){
_start:
{
lean_object* v_res_4192_; 
v_res_4192_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8(v_declName_4185_, v_docString_4186_, v___y_4187_, v___y_4188_, v___y_4189_, v___y_4190_);
lean_dec(v___y_4190_);
lean_dec_ref(v___y_4189_);
lean_dec(v___y_4188_);
lean_dec_ref(v___y_4187_);
return v_res_4192_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1(void){
_start:
{
lean_object* v___x_4194_; lean_object* v___x_4195_; 
v___x_4194_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__0));
v___x_4195_ = l_Lean_stringToMessageData(v___x_4194_);
return v___x_4195_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3(void){
_start:
{
lean_object* v___x_4197_; lean_object* v___x_4198_; 
v___x_4197_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__2));
v___x_4198_ = l_Lean_stringToMessageData(v___x_4197_);
return v___x_4198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5(lean_object* v_addInfo_4199_, lean_object* v_declName_4200_, uint8_t v___x_4201_, lean_object* v___f_4202_, uint8_t v___x_4203_, lean_object* v_env_4204_, lean_object* v___f_4205_, lean_object* v___y_4206_, lean_object* v___y_4207_, lean_object* v___y_4208_, lean_object* v___y_4209_){
_start:
{
lean_object* v___x_4211_; 
lean_inc(v___y_4209_);
lean_inc_ref(v___y_4208_);
lean_inc(v___y_4207_);
lean_inc_ref(v___y_4206_);
lean_inc(v_declName_4200_);
v___x_4211_ = lean_apply_6(v_addInfo_4199_, v_declName_4200_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_, lean_box(0));
if (lean_obj_tag(v___x_4211_) == 0)
{
lean_object* v___x_4212_; 
lean_dec_ref_known(v___x_4211_, 1);
lean_inc(v_declName_4200_);
v___x_4212_ = l_Lean_privateToUserName_x3f(v_declName_4200_);
if (lean_obj_tag(v___x_4212_) == 0)
{
lean_object* v___x_4213_; lean_object* v___x_4214_; lean_object* v___x_4215_; lean_object* v___x_4216_; lean_object* v___x_4217_; lean_object* v___x_4218_; 
v___x_4213_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3);
v___x_4214_ = l_Lean_MessageData_ofConstName(v_declName_4200_, v___x_4201_);
v___x_4215_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4215_, 0, v___x_4213_);
lean_ctor_set(v___x_4215_, 1, v___x_4214_);
v___x_4216_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1);
v___x_4217_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4217_, 0, v___x_4215_);
lean_ctor_set(v___x_4217_, 1, v___x_4216_);
v___x_4218_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4217_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_);
lean_dec(v___y_4209_);
lean_dec_ref(v___y_4208_);
lean_dec(v___y_4207_);
lean_dec_ref(v___y_4206_);
return v___x_4218_;
}
else
{
lean_object* v_val_4219_; lean_object* v___x_4220_; lean_object* v___x_4221_; lean_object* v___x_4222_; lean_object* v___x_4223_; lean_object* v___x_4224_; lean_object* v___x_4225_; 
lean_dec(v_declName_4200_);
v_val_4219_ = lean_ctor_get(v___x_4212_, 0);
lean_inc(v_val_4219_);
lean_dec_ref_known(v___x_4212_, 1);
v___x_4220_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__3);
v___x_4221_ = l_Lean_MessageData_ofConstName(v_val_4219_, v___x_4201_);
v___x_4222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4222_, 0, v___x_4220_);
lean_ctor_set(v___x_4222_, 1, v___x_4221_);
v___x_4223_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1);
v___x_4224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4224_, 0, v___x_4222_);
lean_ctor_set(v___x_4224_, 1, v___x_4223_);
v___x_4225_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4224_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_);
lean_dec(v___y_4209_);
lean_dec_ref(v___y_4208_);
lean_dec(v___y_4207_);
lean_dec_ref(v___y_4206_);
return v___x_4225_;
}
}
else
{
lean_dec(v___y_4209_);
lean_dec_ref(v___y_4208_);
lean_dec(v___y_4207_);
lean_dec_ref(v___y_4206_);
lean_dec(v_declName_4200_);
return v___x_4211_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___boxed(lean_object* v_addInfo_4226_, lean_object* v_declName_4227_, lean_object* v___x_4228_, lean_object* v___f_4229_, lean_object* v___x_4230_, lean_object* v_env_4231_, lean_object* v___f_4232_, lean_object* v___y_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_){
_start:
{
uint8_t v___x_25144__boxed_4238_; uint8_t v___x_25146__boxed_4239_; lean_object* v_res_4240_; 
v___x_25144__boxed_4238_ = lean_unbox(v___x_4228_);
v___x_25146__boxed_4239_ = lean_unbox(v___x_4230_);
v_res_4240_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5(v_addInfo_4226_, v_declName_4227_, v___x_25144__boxed_4238_, v___f_4229_, v___x_25146__boxed_4239_, v_env_4231_, v___f_4232_, v___y_4233_, v___y_4234_, v___y_4235_, v___y_4236_);
lean_dec_ref(v___f_4232_);
lean_dec_ref(v_env_4231_);
lean_dec_ref(v___f_4229_);
return v_res_4240_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1(void){
_start:
{
lean_object* v___x_4242_; lean_object* v___x_4243_; 
v___x_4242_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__0));
v___x_4243_ = l_Lean_stringToMessageData(v___x_4242_);
return v___x_4243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1(lean_object* v_declName_4244_, lean_object* v_env_4245_, lean_object* v_addInfo_4246_, lean_object* v_____r_4247_, lean_object* v___y_4248_, lean_object* v___y_4249_, lean_object* v___y_4250_, lean_object* v___y_4251_){
_start:
{
lean_object* v___x_4253_; 
v___x_4253_ = l_Lean_privateToUserName_x3f(v_declName_4244_);
if (lean_obj_tag(v___x_4253_) == 0)
{
lean_object* v___x_4254_; lean_object* v___x_4255_; 
lean_dec_ref(v_addInfo_4246_);
lean_dec_ref(v_env_4245_);
v___x_4254_ = lean_box(0);
v___x_4255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4255_, 0, v___x_4254_);
return v___x_4255_;
}
else
{
lean_object* v_val_4256_; lean_object* v___x_4258_; uint8_t v_isShared_4259_; uint8_t v_isSharedCheck_4273_; 
v_val_4256_ = lean_ctor_get(v___x_4253_, 0);
v_isSharedCheck_4273_ = !lean_is_exclusive(v___x_4253_);
if (v_isSharedCheck_4273_ == 0)
{
v___x_4258_ = v___x_4253_;
v_isShared_4259_ = v_isSharedCheck_4273_;
goto v_resetjp_4257_;
}
else
{
lean_inc(v_val_4256_);
lean_dec(v___x_4253_);
v___x_4258_ = lean_box(0);
v_isShared_4259_ = v_isSharedCheck_4273_;
goto v_resetjp_4257_;
}
v_resetjp_4257_:
{
uint8_t v___x_4260_; uint8_t v___x_4261_; 
v___x_4260_ = 1;
lean_inc(v_val_4256_);
v___x_4261_ = l_Lean_Environment_contains(v_env_4245_, v_val_4256_, v___x_4260_);
if (v___x_4261_ == 0)
{
lean_object* v___x_4262_; lean_object* v___x_4264_; 
lean_dec(v_val_4256_);
lean_dec_ref(v_addInfo_4246_);
v___x_4262_ = lean_box(0);
if (v_isShared_4259_ == 0)
{
lean_ctor_set_tag(v___x_4258_, 0);
lean_ctor_set(v___x_4258_, 0, v___x_4262_);
v___x_4264_ = v___x_4258_;
goto v_reusejp_4263_;
}
else
{
lean_object* v_reuseFailAlloc_4265_; 
v_reuseFailAlloc_4265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4265_, 0, v___x_4262_);
v___x_4264_ = v_reuseFailAlloc_4265_;
goto v_reusejp_4263_;
}
v_reusejp_4263_:
{
return v___x_4264_;
}
}
else
{
lean_object* v___x_4266_; 
lean_del_object(v___x_4258_);
lean_inc(v___y_4251_);
lean_inc_ref(v___y_4250_);
lean_inc(v___y_4249_);
lean_inc_ref(v___y_4248_);
lean_inc(v_val_4256_);
v___x_4266_ = lean_apply_6(v_addInfo_4246_, v_val_4256_, v___y_4248_, v___y_4249_, v___y_4250_, v___y_4251_, lean_box(0));
if (lean_obj_tag(v___x_4266_) == 0)
{
lean_object* v___x_4267_; lean_object* v___x_4268_; lean_object* v___x_4269_; lean_object* v___x_4270_; lean_object* v___x_4271_; lean_object* v___x_4272_; 
lean_dec_ref_known(v___x_4266_, 1);
v___x_4267_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___closed__1);
v___x_4268_ = l_Lean_MessageData_ofConstName(v_val_4256_, v___x_4260_);
v___x_4269_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4269_, 0, v___x_4267_);
lean_ctor_set(v___x_4269_, 1, v___x_4268_);
v___x_4270_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1);
v___x_4271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4271_, 0, v___x_4269_);
lean_ctor_set(v___x_4271_, 1, v___x_4270_);
v___x_4272_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4271_, v___y_4248_, v___y_4249_, v___y_4250_, v___y_4251_);
return v___x_4272_;
}
else
{
lean_dec(v_val_4256_);
return v___x_4266_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___boxed(lean_object* v_declName_4274_, lean_object* v_env_4275_, lean_object* v_addInfo_4276_, lean_object* v_____r_4277_, lean_object* v___y_4278_, lean_object* v___y_4279_, lean_object* v___y_4280_, lean_object* v___y_4281_, lean_object* v___y_4282_){
_start:
{
lean_object* v_res_4283_; 
v_res_4283_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1(v_declName_4274_, v_env_4275_, v_addInfo_4276_, v_____r_4277_, v___y_4278_, v___y_4279_, v___y_4280_, v___y_4281_);
lean_dec(v___y_4281_);
lean_dec_ref(v___y_4280_);
lean_dec(v___y_4279_);
lean_dec_ref(v___y_4278_);
return v_res_4283_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1(void){
_start:
{
lean_object* v___x_4285_; lean_object* v___x_4286_; 
v___x_4285_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__0));
v___x_4286_ = l_Lean_stringToMessageData(v___x_4285_);
return v___x_4286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2(lean_object* v_env_4287_, lean_object* v_declName_4288_, lean_object* v___f_4289_, lean_object* v_addInfo_4290_, lean_object* v_____r_4291_, lean_object* v___y_4292_, lean_object* v___y_4293_, lean_object* v___y_4294_, lean_object* v___y_4295_){
_start:
{
lean_object* v___x_4297_; uint8_t v___x_4298_; uint8_t v___x_4299_; 
lean_inc(v_declName_4288_);
v___x_4297_ = l_Lean_mkPrivateName(v_env_4287_, v_declName_4288_);
v___x_4298_ = 1;
lean_inc(v___x_4297_);
v___x_4299_ = l_Lean_Environment_contains(v_env_4287_, v___x_4297_, v___x_4298_);
if (v___x_4299_ == 0)
{
lean_object* v___x_4300_; lean_object* v___x_4301_; 
lean_dec(v___x_4297_);
lean_dec_ref(v_addInfo_4290_);
lean_dec(v_declName_4288_);
v___x_4300_ = lean_box(0);
lean_inc(v___y_4295_);
lean_inc_ref(v___y_4294_);
lean_inc(v___y_4293_);
lean_inc_ref(v___y_4292_);
v___x_4301_ = lean_apply_6(v___f_4289_, v___x_4300_, v___y_4292_, v___y_4293_, v___y_4294_, v___y_4295_, lean_box(0));
return v___x_4301_;
}
else
{
lean_object* v___x_4302_; 
lean_dec_ref(v___f_4289_);
lean_inc(v___y_4295_);
lean_inc_ref(v___y_4294_);
lean_inc(v___y_4293_);
lean_inc_ref(v___y_4292_);
v___x_4302_ = lean_apply_6(v_addInfo_4290_, v___x_4297_, v___y_4292_, v___y_4293_, v___y_4294_, v___y_4295_, lean_box(0));
if (lean_obj_tag(v___x_4302_) == 0)
{
lean_object* v___x_4303_; lean_object* v___x_4304_; lean_object* v___x_4305_; lean_object* v___x_4306_; lean_object* v___x_4307_; lean_object* v___x_4308_; 
lean_dec_ref_known(v___x_4302_, 1);
v___x_4303_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___closed__1);
v___x_4304_ = l_Lean_MessageData_ofConstName(v_declName_4288_, v___x_4298_);
v___x_4305_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4305_, 0, v___x_4303_);
lean_ctor_set(v___x_4305_, 1, v___x_4304_);
v___x_4306_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___closed__1);
v___x_4307_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4307_, 0, v___x_4305_);
lean_ctor_set(v___x_4307_, 1, v___x_4306_);
v___x_4308_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4307_, v___y_4292_, v___y_4293_, v___y_4294_, v___y_4295_);
return v___x_4308_;
}
else
{
lean_dec(v_declName_4288_);
return v___x_4302_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___boxed(lean_object* v_env_4309_, lean_object* v_declName_4310_, lean_object* v___f_4311_, lean_object* v_addInfo_4312_, lean_object* v_____r_4313_, lean_object* v___y_4314_, lean_object* v___y_4315_, lean_object* v___y_4316_, lean_object* v___y_4317_, lean_object* v___y_4318_){
_start:
{
lean_object* v_res_4319_; 
v_res_4319_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2(v_env_4309_, v_declName_4310_, v___f_4311_, v_addInfo_4312_, v_____r_4313_, v___y_4314_, v___y_4315_, v___y_4316_, v___y_4317_);
lean_dec(v___y_4317_);
lean_dec_ref(v___y_4316_);
lean_dec(v___y_4315_);
lean_dec_ref(v___y_4314_);
return v_res_4319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4(lean_object* v___f_4320_, lean_object* v___y_4321_, lean_object* v___y_4322_, lean_object* v___y_4323_, lean_object* v___y_4324_){
_start:
{
lean_object* v___x_4326_; lean_object* v_env_4327_; lean_object* v___x_4328_; 
v___x_4326_ = lean_st_ref_get(v___y_4324_);
v_env_4327_ = lean_ctor_get(v___x_4326_, 0);
lean_inc_ref(v_env_4327_);
lean_dec(v___x_4326_);
v___x_4328_ = lean_apply_6(v___f_4320_, v_env_4327_, v___y_4321_, v___y_4322_, v___y_4323_, v___y_4324_, lean_box(0));
return v___x_4328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4___boxed(lean_object* v___f_4329_, lean_object* v___y_4330_, lean_object* v___y_4331_, lean_object* v___y_4332_, lean_object* v___y_4333_, lean_object* v___y_4334_){
_start:
{
lean_object* v_res_4335_; 
v_res_4335_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4(v___f_4329_, v___y_4330_, v___y_4331_, v___y_4332_, v___y_4333_);
return v_res_4335_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1(void){
_start:
{
lean_object* v___x_4337_; lean_object* v___x_4338_; 
v___x_4337_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__0));
v___x_4338_ = l_Lean_stringToMessageData(v___x_4337_);
return v___x_4338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3(lean_object* v___f_4339_, lean_object* v_declName_4340_, uint8_t v___x_4341_, lean_object* v_env_4342_, lean_object* v_____do__lift_4343_, lean_object* v___y_4344_, lean_object* v___y_4345_, lean_object* v___y_4346_, lean_object* v___y_4347_){
_start:
{
uint8_t v___y_4350_; lean_object* v___x_4359_; uint8_t v___x_4360_; 
lean_inc(v_declName_4340_);
v___x_4359_ = l_Lean_privateToUserName(v_declName_4340_);
lean_inc_ref(v_env_4342_);
v___x_4360_ = lean_is_reserved_name(v_env_4342_, v___x_4359_);
if (v___x_4360_ == 0)
{
lean_object* v___x_4361_; uint8_t v___x_4362_; 
lean_inc(v_declName_4340_);
v___x_4361_ = l_Lean_mkPrivateName(v_____do__lift_4343_, v_declName_4340_);
v___x_4362_ = lean_is_reserved_name(v_env_4342_, v___x_4361_);
v___y_4350_ = v___x_4362_;
goto v___jp_4349_;
}
else
{
lean_dec_ref(v_env_4342_);
v___y_4350_ = v___x_4360_;
goto v___jp_4349_;
}
v___jp_4349_:
{
if (v___y_4350_ == 0)
{
lean_object* v___x_4351_; lean_object* v___x_4352_; 
lean_dec(v_declName_4340_);
v___x_4351_ = lean_box(0);
lean_inc(v___y_4347_);
lean_inc_ref(v___y_4346_);
lean_inc(v___y_4345_);
lean_inc_ref(v___y_4344_);
v___x_4352_ = lean_apply_6(v___f_4339_, v___x_4351_, v___y_4344_, v___y_4345_, v___y_4346_, v___y_4347_, lean_box(0));
return v___x_4352_;
}
else
{
lean_object* v___x_4353_; lean_object* v___x_4354_; lean_object* v___x_4355_; lean_object* v___x_4356_; lean_object* v___x_4357_; lean_object* v___x_4358_; 
lean_dec_ref(v___f_4339_);
v___x_4353_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg___closed__3);
v___x_4354_ = l_Lean_MessageData_ofConstName(v_declName_4340_, v___x_4341_);
v___x_4355_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4355_, 0, v___x_4353_);
lean_ctor_set(v___x_4355_, 1, v___x_4354_);
v___x_4356_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___closed__1);
v___x_4357_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4357_, 0, v___x_4355_);
lean_ctor_set(v___x_4357_, 1, v___x_4356_);
v___x_4358_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_4357_, v___y_4344_, v___y_4345_, v___y_4346_, v___y_4347_);
return v___x_4358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___boxed(lean_object* v___f_4363_, lean_object* v_declName_4364_, lean_object* v___x_4365_, lean_object* v_env_4366_, lean_object* v_____do__lift_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_){
_start:
{
uint8_t v___x_25380__boxed_4373_; lean_object* v_res_4374_; 
v___x_25380__boxed_4373_ = lean_unbox(v___x_4365_);
v_res_4374_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3(v___f_4363_, v_declName_4364_, v___x_25380__boxed_4373_, v_env_4366_, v_____do__lift_4367_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_);
lean_dec(v___y_4371_);
lean_dec_ref(v___y_4370_);
lean_dec(v___y_4369_);
lean_dec_ref(v___y_4368_);
lean_dec_ref(v_____do__lift_4367_);
return v_res_4374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4(lean_object* v_constName_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_){
_start:
{
lean_object* v___x_4381_; lean_object* v_env_4382_; uint8_t v___x_4383_; lean_object* v___x_4384_; 
v___x_4381_ = lean_st_ref_get(v___y_4379_);
v_env_4382_ = lean_ctor_get(v___x_4381_, 0);
lean_inc_ref(v_env_4382_);
lean_dec(v___x_4381_);
v___x_4383_ = 0;
lean_inc(v_constName_4375_);
v___x_4384_ = l_Lean_Environment_findConstVal_x3f(v_env_4382_, v_constName_4375_, v___x_4383_);
if (lean_obj_tag(v___x_4384_) == 0)
{
lean_object* v___x_4385_; 
v___x_4385_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(v_constName_4375_, v___y_4376_, v___y_4377_, v___y_4378_, v___y_4379_);
return v___x_4385_;
}
else
{
lean_object* v_val_4386_; lean_object* v___x_4388_; uint8_t v_isShared_4389_; uint8_t v_isSharedCheck_4393_; 
lean_dec(v_constName_4375_);
v_val_4386_ = lean_ctor_get(v___x_4384_, 0);
v_isSharedCheck_4393_ = !lean_is_exclusive(v___x_4384_);
if (v_isSharedCheck_4393_ == 0)
{
v___x_4388_ = v___x_4384_;
v_isShared_4389_ = v_isSharedCheck_4393_;
goto v_resetjp_4387_;
}
else
{
lean_inc(v_val_4386_);
lean_dec(v___x_4384_);
v___x_4388_ = lean_box(0);
v_isShared_4389_ = v_isSharedCheck_4393_;
goto v_resetjp_4387_;
}
v_resetjp_4387_:
{
lean_object* v___x_4391_; 
if (v_isShared_4389_ == 0)
{
lean_ctor_set_tag(v___x_4388_, 0);
v___x_4391_ = v___x_4388_;
goto v_reusejp_4390_;
}
else
{
lean_object* v_reuseFailAlloc_4392_; 
v_reuseFailAlloc_4392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4392_, 0, v_val_4386_);
v___x_4391_ = v_reuseFailAlloc_4392_;
goto v_reusejp_4390_;
}
v_reusejp_4390_:
{
return v___x_4391_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4___boxed(lean_object* v_constName_4394_, lean_object* v___y_4395_, lean_object* v___y_4396_, lean_object* v___y_4397_, lean_object* v___y_4398_, lean_object* v___y_4399_){
_start:
{
lean_object* v_res_4400_; 
v_res_4400_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4(v_constName_4394_, v___y_4395_, v___y_4396_, v___y_4397_, v___y_4398_);
lean_dec(v___y_4398_);
lean_dec_ref(v___y_4397_);
lean_dec(v___y_4396_);
lean_dec_ref(v___y_4395_);
return v_res_4400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0(lean_object* v_constName_4401_, lean_object* v___y_4402_, lean_object* v___y_4403_, lean_object* v___y_4404_, lean_object* v___y_4405_){
_start:
{
lean_object* v___x_4407_; 
lean_inc(v_constName_4401_);
v___x_4407_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0_spec__4(v_constName_4401_, v___y_4402_, v___y_4403_, v___y_4404_, v___y_4405_);
if (lean_obj_tag(v___x_4407_) == 0)
{
lean_object* v_a_4408_; lean_object* v___x_4410_; uint8_t v_isShared_4411_; uint8_t v_isSharedCheck_4419_; 
v_a_4408_ = lean_ctor_get(v___x_4407_, 0);
v_isSharedCheck_4419_ = !lean_is_exclusive(v___x_4407_);
if (v_isSharedCheck_4419_ == 0)
{
v___x_4410_ = v___x_4407_;
v_isShared_4411_ = v_isSharedCheck_4419_;
goto v_resetjp_4409_;
}
else
{
lean_inc(v_a_4408_);
lean_dec(v___x_4407_);
v___x_4410_ = lean_box(0);
v_isShared_4411_ = v_isSharedCheck_4419_;
goto v_resetjp_4409_;
}
v_resetjp_4409_:
{
lean_object* v_levelParams_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4417_; 
v_levelParams_4412_ = lean_ctor_get(v_a_4408_, 1);
lean_inc(v_levelParams_4412_);
lean_dec(v_a_4408_);
v___x_4413_ = lean_box(0);
v___x_4414_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_addRelatedDecl_spec__4(v_levelParams_4412_, v___x_4413_);
v___x_4415_ = l_Lean_mkConst(v_constName_4401_, v___x_4414_);
if (v_isShared_4411_ == 0)
{
lean_ctor_set(v___x_4410_, 0, v___x_4415_);
v___x_4417_ = v___x_4410_;
goto v_reusejp_4416_;
}
else
{
lean_object* v_reuseFailAlloc_4418_; 
v_reuseFailAlloc_4418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4418_, 0, v___x_4415_);
v___x_4417_ = v_reuseFailAlloc_4418_;
goto v_reusejp_4416_;
}
v_reusejp_4416_:
{
return v___x_4417_;
}
}
}
else
{
lean_object* v_a_4420_; lean_object* v___x_4422_; uint8_t v_isShared_4423_; uint8_t v_isSharedCheck_4427_; 
lean_dec(v_constName_4401_);
v_a_4420_ = lean_ctor_get(v___x_4407_, 0);
v_isSharedCheck_4427_ = !lean_is_exclusive(v___x_4407_);
if (v_isSharedCheck_4427_ == 0)
{
v___x_4422_ = v___x_4407_;
v_isShared_4423_ = v_isSharedCheck_4427_;
goto v_resetjp_4421_;
}
else
{
lean_inc(v_a_4420_);
lean_dec(v___x_4407_);
v___x_4422_ = lean_box(0);
v_isShared_4423_ = v_isSharedCheck_4427_;
goto v_resetjp_4421_;
}
v_resetjp_4421_:
{
lean_object* v___x_4425_; 
if (v_isShared_4423_ == 0)
{
v___x_4425_ = v___x_4422_;
goto v_reusejp_4424_;
}
else
{
lean_object* v_reuseFailAlloc_4426_; 
v_reuseFailAlloc_4426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4426_, 0, v_a_4420_);
v___x_4425_ = v_reuseFailAlloc_4426_;
goto v_reusejp_4424_;
}
v_reusejp_4424_:
{
return v___x_4425_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0___boxed(lean_object* v_constName_4428_, lean_object* v___y_4429_, lean_object* v___y_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_){
_start:
{
lean_object* v_res_4434_; 
v_res_4434_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0(v_constName_4428_, v___y_4429_, v___y_4430_, v___y_4431_, v___y_4432_);
lean_dec(v___y_4432_);
lean_dec_ref(v___y_4431_);
lean_dec(v___y_4430_);
lean_dec_ref(v___y_4429_);
return v_res_4434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg(lean_object* v_t_4435_, lean_object* v___y_4436_){
_start:
{
lean_object* v___x_4438_; lean_object* v_infoState_4439_; uint8_t v_enabled_4440_; 
v___x_4438_ = lean_st_ref_get(v___y_4436_);
v_infoState_4439_ = lean_ctor_get(v___x_4438_, 7);
lean_inc_ref(v_infoState_4439_);
lean_dec(v___x_4438_);
v_enabled_4440_ = lean_ctor_get_uint8(v_infoState_4439_, sizeof(void*)*3);
lean_dec_ref(v_infoState_4439_);
if (v_enabled_4440_ == 0)
{
lean_object* v___x_4441_; lean_object* v___x_4442_; 
lean_dec_ref(v_t_4435_);
v___x_4441_ = lean_box(0);
v___x_4442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4442_, 0, v___x_4441_);
return v___x_4442_;
}
else
{
lean_object* v___x_4443_; lean_object* v_infoState_4444_; lean_object* v_env_4445_; lean_object* v_nextMacroScope_4446_; lean_object* v_ngen_4447_; lean_object* v_auxDeclNGen_4448_; lean_object* v_traceState_4449_; lean_object* v_cache_4450_; lean_object* v_messages_4451_; lean_object* v_snapshotTasks_4452_; lean_object* v___x_4454_; uint8_t v_isShared_4455_; uint8_t v_isSharedCheck_4474_; 
v___x_4443_ = lean_st_ref_take(v___y_4436_);
v_infoState_4444_ = lean_ctor_get(v___x_4443_, 7);
v_env_4445_ = lean_ctor_get(v___x_4443_, 0);
v_nextMacroScope_4446_ = lean_ctor_get(v___x_4443_, 1);
v_ngen_4447_ = lean_ctor_get(v___x_4443_, 2);
v_auxDeclNGen_4448_ = lean_ctor_get(v___x_4443_, 3);
v_traceState_4449_ = lean_ctor_get(v___x_4443_, 4);
v_cache_4450_ = lean_ctor_get(v___x_4443_, 5);
v_messages_4451_ = lean_ctor_get(v___x_4443_, 6);
v_snapshotTasks_4452_ = lean_ctor_get(v___x_4443_, 8);
v_isSharedCheck_4474_ = !lean_is_exclusive(v___x_4443_);
if (v_isSharedCheck_4474_ == 0)
{
v___x_4454_ = v___x_4443_;
v_isShared_4455_ = v_isSharedCheck_4474_;
goto v_resetjp_4453_;
}
else
{
lean_inc(v_snapshotTasks_4452_);
lean_inc(v_infoState_4444_);
lean_inc(v_messages_4451_);
lean_inc(v_cache_4450_);
lean_inc(v_traceState_4449_);
lean_inc(v_auxDeclNGen_4448_);
lean_inc(v_ngen_4447_);
lean_inc(v_nextMacroScope_4446_);
lean_inc(v_env_4445_);
lean_dec(v___x_4443_);
v___x_4454_ = lean_box(0);
v_isShared_4455_ = v_isSharedCheck_4474_;
goto v_resetjp_4453_;
}
v_resetjp_4453_:
{
uint8_t v_enabled_4456_; lean_object* v_assignment_4457_; lean_object* v_lazyAssignment_4458_; lean_object* v_trees_4459_; lean_object* v___x_4461_; uint8_t v_isShared_4462_; uint8_t v_isSharedCheck_4473_; 
v_enabled_4456_ = lean_ctor_get_uint8(v_infoState_4444_, sizeof(void*)*3);
v_assignment_4457_ = lean_ctor_get(v_infoState_4444_, 0);
v_lazyAssignment_4458_ = lean_ctor_get(v_infoState_4444_, 1);
v_trees_4459_ = lean_ctor_get(v_infoState_4444_, 2);
v_isSharedCheck_4473_ = !lean_is_exclusive(v_infoState_4444_);
if (v_isSharedCheck_4473_ == 0)
{
v___x_4461_ = v_infoState_4444_;
v_isShared_4462_ = v_isSharedCheck_4473_;
goto v_resetjp_4460_;
}
else
{
lean_inc(v_trees_4459_);
lean_inc(v_lazyAssignment_4458_);
lean_inc(v_assignment_4457_);
lean_dec(v_infoState_4444_);
v___x_4461_ = lean_box(0);
v_isShared_4462_ = v_isSharedCheck_4473_;
goto v_resetjp_4460_;
}
v_resetjp_4460_:
{
lean_object* v___x_4463_; lean_object* v___x_4465_; 
v___x_4463_ = l_Lean_PersistentArray_push___redArg(v_trees_4459_, v_t_4435_);
if (v_isShared_4462_ == 0)
{
lean_ctor_set(v___x_4461_, 2, v___x_4463_);
v___x_4465_ = v___x_4461_;
goto v_reusejp_4464_;
}
else
{
lean_object* v_reuseFailAlloc_4472_; 
v_reuseFailAlloc_4472_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_4472_, 0, v_assignment_4457_);
lean_ctor_set(v_reuseFailAlloc_4472_, 1, v_lazyAssignment_4458_);
lean_ctor_set(v_reuseFailAlloc_4472_, 2, v___x_4463_);
lean_ctor_set_uint8(v_reuseFailAlloc_4472_, sizeof(void*)*3, v_enabled_4456_);
v___x_4465_ = v_reuseFailAlloc_4472_;
goto v_reusejp_4464_;
}
v_reusejp_4464_:
{
lean_object* v___x_4467_; 
if (v_isShared_4455_ == 0)
{
lean_ctor_set(v___x_4454_, 7, v___x_4465_);
v___x_4467_ = v___x_4454_;
goto v_reusejp_4466_;
}
else
{
lean_object* v_reuseFailAlloc_4471_; 
v_reuseFailAlloc_4471_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4471_, 0, v_env_4445_);
lean_ctor_set(v_reuseFailAlloc_4471_, 1, v_nextMacroScope_4446_);
lean_ctor_set(v_reuseFailAlloc_4471_, 2, v_ngen_4447_);
lean_ctor_set(v_reuseFailAlloc_4471_, 3, v_auxDeclNGen_4448_);
lean_ctor_set(v_reuseFailAlloc_4471_, 4, v_traceState_4449_);
lean_ctor_set(v_reuseFailAlloc_4471_, 5, v_cache_4450_);
lean_ctor_set(v_reuseFailAlloc_4471_, 6, v_messages_4451_);
lean_ctor_set(v_reuseFailAlloc_4471_, 7, v___x_4465_);
lean_ctor_set(v_reuseFailAlloc_4471_, 8, v_snapshotTasks_4452_);
v___x_4467_ = v_reuseFailAlloc_4471_;
goto v_reusejp_4466_;
}
v_reusejp_4466_:
{
lean_object* v___x_4468_; lean_object* v___x_4469_; lean_object* v___x_4470_; 
v___x_4468_ = lean_st_ref_set(v___y_4436_, v___x_4467_);
v___x_4469_ = lean_box(0);
v___x_4470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4470_, 0, v___x_4469_);
return v___x_4470_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg___boxed(lean_object* v_t_4475_, lean_object* v___y_4476_, lean_object* v___y_4477_){
_start:
{
lean_object* v_res_4478_; 
v_res_4478_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg(v_t_4475_, v___y_4476_);
lean_dec(v___y_4476_);
return v_res_4478_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0(void){
_start:
{
lean_object* v___x_4479_; lean_object* v___x_4480_; lean_object* v___x_4481_; 
v___x_4479_ = lean_unsigned_to_nat(32u);
v___x_4480_ = lean_mk_empty_array_with_capacity(v___x_4479_);
v___x_4481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4481_, 0, v___x_4480_);
return v___x_4481_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1(void){
_start:
{
size_t v___x_4482_; lean_object* v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v___x_4486_; lean_object* v___x_4487_; 
v___x_4482_ = ((size_t)5ULL);
v___x_4483_ = lean_unsigned_to_nat(0u);
v___x_4484_ = lean_unsigned_to_nat(32u);
v___x_4485_ = lean_mk_empty_array_with_capacity(v___x_4484_);
v___x_4486_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__0);
v___x_4487_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_4487_, 0, v___x_4486_);
lean_ctor_set(v___x_4487_, 1, v___x_4485_);
lean_ctor_set(v___x_4487_, 2, v___x_4483_);
lean_ctor_set(v___x_4487_, 3, v___x_4483_);
lean_ctor_set_usize(v___x_4487_, 4, v___x_4482_);
return v___x_4487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1(lean_object* v_t_4488_, lean_object* v___y_4489_, lean_object* v___y_4490_, lean_object* v___y_4491_, lean_object* v___y_4492_){
_start:
{
lean_object* v___x_4494_; lean_object* v_infoState_4495_; uint8_t v_enabled_4496_; 
v___x_4494_ = lean_st_ref_get(v___y_4492_);
v_infoState_4495_ = lean_ctor_get(v___x_4494_, 7);
lean_inc_ref(v_infoState_4495_);
lean_dec(v___x_4494_);
v_enabled_4496_ = lean_ctor_get_uint8(v_infoState_4495_, sizeof(void*)*3);
lean_dec_ref(v_infoState_4495_);
if (v_enabled_4496_ == 0)
{
lean_object* v___x_4497_; lean_object* v___x_4498_; 
lean_dec_ref(v_t_4488_);
v___x_4497_ = lean_box(0);
v___x_4498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4498_, 0, v___x_4497_);
return v___x_4498_;
}
else
{
lean_object* v___x_4499_; lean_object* v___x_4500_; lean_object* v___x_4501_; 
v___x_4499_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___closed__1);
v___x_4500_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4500_, 0, v_t_4488_);
lean_ctor_set(v___x_4500_, 1, v___x_4499_);
v___x_4501_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg(v___x_4500_, v___y_4492_);
return v___x_4501_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1___boxed(lean_object* v_t_4502_, lean_object* v___y_4503_, lean_object* v___y_4504_, lean_object* v___y_4505_, lean_object* v___y_4506_, lean_object* v___y_4507_){
_start:
{
lean_object* v_res_4508_; 
v_res_4508_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1(v_t_4502_, v___y_4503_, v___y_4504_, v___y_4505_, v___y_4506_);
lean_dec(v___y_4506_);
lean_dec_ref(v___y_4505_);
lean_dec(v___y_4504_);
lean_dec_ref(v___y_4503_);
return v_res_4508_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0(void){
_start:
{
lean_object* v___x_4509_; 
v___x_4509_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_4509_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1(void){
_start:
{
lean_object* v___x_4510_; lean_object* v___x_4511_; 
v___x_4510_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__0);
v___x_4511_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4511_, 0, v___x_4510_);
return v___x_4511_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2(void){
_start:
{
lean_object* v___x_4512_; lean_object* v___x_4513_; lean_object* v___x_4514_; lean_object* v___x_4515_; 
v___x_4512_ = lean_box(1);
v___x_4513_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg___closed__4);
v___x_4514_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__1);
v___x_4515_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4515_, 0, v___x_4514_);
lean_ctor_set(v___x_4515_, 1, v___x_4513_);
lean_ctor_set(v___x_4515_, 2, v___x_4512_);
return v___x_4515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0(uint8_t v___x_4516_, lean_object* v_declName_4517_, lean_object* v___y_4518_, lean_object* v___y_4519_, lean_object* v___y_4520_, lean_object* v___y_4521_){
_start:
{
lean_object* v_ref_4523_; lean_object* v___x_4524_; 
v_ref_4523_ = lean_ctor_get(v___y_4520_, 5);
v___x_4524_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__0(v_declName_4517_, v___y_4518_, v___y_4519_, v___y_4520_, v___y_4521_);
if (lean_obj_tag(v___x_4524_) == 0)
{
lean_object* v_a_4525_; lean_object* v___x_4526_; lean_object* v___x_4527_; lean_object* v___x_4528_; lean_object* v___x_4529_; lean_object* v___x_4530_; lean_object* v___x_4531_; lean_object* v___x_4532_; 
v_a_4525_ = lean_ctor_get(v___x_4524_, 0);
lean_inc(v_a_4525_);
lean_dec_ref_known(v___x_4524_, 1);
v___x_4526_ = lean_box(0);
lean_inc(v_ref_4523_);
v___x_4527_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4527_, 0, v___x_4526_);
lean_ctor_set(v___x_4527_, 1, v_ref_4523_);
v___x_4528_ = lean_obj_once(&lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2, &lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2_once, _init_lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___closed__2);
v___x_4529_ = lean_box(0);
v___x_4530_ = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(v___x_4530_, 0, v___x_4527_);
lean_ctor_set(v___x_4530_, 1, v___x_4528_);
lean_ctor_set(v___x_4530_, 2, v___x_4529_);
lean_ctor_set(v___x_4530_, 3, v_a_4525_);
lean_ctor_set_uint8(v___x_4530_, sizeof(void*)*4, v___x_4516_);
lean_ctor_set_uint8(v___x_4530_, sizeof(void*)*4 + 1, v___x_4516_);
v___x_4531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4531_, 0, v___x_4530_);
v___x_4532_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1(v___x_4531_, v___y_4518_, v___y_4519_, v___y_4520_, v___y_4521_);
return v___x_4532_;
}
else
{
lean_object* v_a_4533_; lean_object* v___x_4535_; uint8_t v_isShared_4536_; uint8_t v_isSharedCheck_4540_; 
v_a_4533_ = lean_ctor_get(v___x_4524_, 0);
v_isSharedCheck_4540_ = !lean_is_exclusive(v___x_4524_);
if (v_isSharedCheck_4540_ == 0)
{
v___x_4535_ = v___x_4524_;
v_isShared_4536_ = v_isSharedCheck_4540_;
goto v_resetjp_4534_;
}
else
{
lean_inc(v_a_4533_);
lean_dec(v___x_4524_);
v___x_4535_ = lean_box(0);
v_isShared_4536_ = v_isSharedCheck_4540_;
goto v_resetjp_4534_;
}
v_resetjp_4534_:
{
lean_object* v___x_4538_; 
if (v_isShared_4536_ == 0)
{
v___x_4538_ = v___x_4535_;
goto v_reusejp_4537_;
}
else
{
lean_object* v_reuseFailAlloc_4539_; 
v_reuseFailAlloc_4539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4539_, 0, v_a_4533_);
v___x_4538_ = v_reuseFailAlloc_4539_;
goto v_reusejp_4537_;
}
v_reusejp_4537_:
{
return v___x_4538_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0___boxed(lean_object* v___x_4541_, lean_object* v_declName_4542_, lean_object* v___y_4543_, lean_object* v___y_4544_, lean_object* v___y_4545_, lean_object* v___y_4546_, lean_object* v___y_4547_){
_start:
{
uint8_t v___x_25660__boxed_4548_; lean_object* v_res_4549_; 
v___x_25660__boxed_4548_ = lean_unbox(v___x_4541_);
v_res_4549_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__0(v___x_25660__boxed_4548_, v_declName_4542_, v___y_4543_, v___y_4544_, v___y_4545_, v___y_4546_);
lean_dec(v___y_4546_);
lean_dec_ref(v___y_4545_);
lean_dec(v___y_4544_);
lean_dec_ref(v___y_4543_);
return v_res_4549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(lean_object* v_env_4550_, lean_object* v_x_4551_, lean_object* v___y_4552_, lean_object* v___y_4553_, lean_object* v___y_4554_, lean_object* v___y_4555_){
_start:
{
lean_object* v___x_4557_; lean_object* v_env_4558_; lean_object* v_a_4560_; lean_object* v___x_4570_; lean_object* v___x_4571_; 
v___x_4557_ = lean_st_ref_get(v___y_4555_);
v_env_4558_ = lean_ctor_get(v___x_4557_, 0);
lean_inc_ref(v_env_4558_);
lean_dec(v___x_4557_);
v___x_4570_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v_env_4550_, v___y_4553_, v___y_4555_);
lean_dec_ref(v___x_4570_);
lean_inc(v___y_4555_);
lean_inc_ref(v___y_4554_);
lean_inc(v___y_4553_);
lean_inc_ref(v___y_4552_);
v___x_4571_ = lean_apply_5(v_x_4551_, v___y_4552_, v___y_4553_, v___y_4554_, v___y_4555_, lean_box(0));
if (lean_obj_tag(v___x_4571_) == 0)
{
lean_object* v_a_4572_; lean_object* v___x_4573_; lean_object* v___x_4575_; uint8_t v_isShared_4576_; uint8_t v_isSharedCheck_4580_; 
v_a_4572_ = lean_ctor_get(v___x_4571_, 0);
lean_inc(v_a_4572_);
lean_dec_ref_known(v___x_4571_, 1);
v___x_4573_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v_env_4558_, v___y_4553_, v___y_4555_);
v_isSharedCheck_4580_ = !lean_is_exclusive(v___x_4573_);
if (v_isSharedCheck_4580_ == 0)
{
lean_object* v_unused_4581_; 
v_unused_4581_ = lean_ctor_get(v___x_4573_, 0);
lean_dec(v_unused_4581_);
v___x_4575_ = v___x_4573_;
v_isShared_4576_ = v_isSharedCheck_4580_;
goto v_resetjp_4574_;
}
else
{
lean_dec(v___x_4573_);
v___x_4575_ = lean_box(0);
v_isShared_4576_ = v_isSharedCheck_4580_;
goto v_resetjp_4574_;
}
v_resetjp_4574_:
{
lean_object* v___x_4578_; 
if (v_isShared_4576_ == 0)
{
lean_ctor_set(v___x_4575_, 0, v_a_4572_);
v___x_4578_ = v___x_4575_;
goto v_reusejp_4577_;
}
else
{
lean_object* v_reuseFailAlloc_4579_; 
v_reuseFailAlloc_4579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4579_, 0, v_a_4572_);
v___x_4578_ = v_reuseFailAlloc_4579_;
goto v_reusejp_4577_;
}
v_reusejp_4577_:
{
return v___x_4578_;
}
}
}
else
{
lean_object* v_a_4582_; 
v_a_4582_ = lean_ctor_get(v___x_4571_, 0);
lean_inc(v_a_4582_);
lean_dec_ref_known(v___x_4571_, 1);
v_a_4560_ = v_a_4582_;
goto v___jp_4559_;
}
v___jp_4559_:
{
lean_object* v___x_4561_; lean_object* v___x_4563_; uint8_t v_isShared_4564_; uint8_t v_isSharedCheck_4568_; 
v___x_4561_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v_env_4558_, v___y_4553_, v___y_4555_);
v_isSharedCheck_4568_ = !lean_is_exclusive(v___x_4561_);
if (v_isSharedCheck_4568_ == 0)
{
lean_object* v_unused_4569_; 
v_unused_4569_ = lean_ctor_get(v___x_4561_, 0);
lean_dec(v_unused_4569_);
v___x_4563_ = v___x_4561_;
v_isShared_4564_ = v_isSharedCheck_4568_;
goto v_resetjp_4562_;
}
else
{
lean_dec(v___x_4561_);
v___x_4563_ = lean_box(0);
v_isShared_4564_ = v_isSharedCheck_4568_;
goto v_resetjp_4562_;
}
v_resetjp_4562_:
{
lean_object* v___x_4566_; 
if (v_isShared_4564_ == 0)
{
lean_ctor_set_tag(v___x_4563_, 1);
lean_ctor_set(v___x_4563_, 0, v_a_4560_);
v___x_4566_ = v___x_4563_;
goto v_reusejp_4565_;
}
else
{
lean_object* v_reuseFailAlloc_4567_; 
v_reuseFailAlloc_4567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4567_, 0, v_a_4560_);
v___x_4566_ = v_reuseFailAlloc_4567_;
goto v_reusejp_4565_;
}
v_reusejp_4565_:
{
return v___x_4566_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg___boxed(lean_object* v_env_4583_, lean_object* v_x_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_, lean_object* v___y_4587_, lean_object* v___y_4588_, lean_object* v___y_4589_){
_start:
{
lean_object* v_res_4590_; 
v_res_4590_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(v_env_4583_, v_x_4584_, v___y_4585_, v___y_4586_, v___y_4587_, v___y_4588_);
lean_dec(v___y_4588_);
lean_dec_ref(v___y_4587_);
lean_dec(v___y_4586_);
lean_dec_ref(v___y_4585_);
return v_res_4590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0(lean_object* v_declName_4594_, lean_object* v___y_4595_, lean_object* v___y_4596_, lean_object* v___y_4597_, lean_object* v___y_4598_){
_start:
{
lean_object* v___x_4600_; lean_object* v_env_4601_; uint8_t v___x_4602_; lean_object* v_addInfo_4603_; lean_object* v_env_4604_; lean_object* v___f_4605_; lean_object* v___f_4606_; lean_object* v___x_4607_; lean_object* v___f_4608_; uint8_t v___x_4609_; uint8_t v___x_4610_; 
v___x_4600_ = lean_st_ref_get(v___y_4598_);
v_env_4601_ = lean_ctor_get(v___x_4600_, 0);
lean_inc_ref(v_env_4601_);
lean_dec(v___x_4600_);
v___x_4602_ = 0;
v_addInfo_4603_ = ((lean_object*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___closed__0));
v_env_4604_ = l_Lean_Environment_setExporting(v_env_4601_, v___x_4602_);
lean_inc_ref_n(v_env_4604_, 4);
lean_inc_n(v_declName_4594_, 4);
v___f_4605_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__1___boxed), 9, 3);
lean_closure_set(v___f_4605_, 0, v_declName_4594_);
lean_closure_set(v___f_4605_, 1, v_env_4604_);
lean_closure_set(v___f_4605_, 2, v_addInfo_4603_);
v___f_4606_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__2___boxed), 10, 4);
lean_closure_set(v___f_4606_, 0, v_env_4604_);
lean_closure_set(v___f_4606_, 1, v_declName_4594_);
lean_closure_set(v___f_4606_, 2, v___f_4605_);
lean_closure_set(v___f_4606_, 3, v_addInfo_4603_);
v___x_4607_ = lean_box(v___x_4602_);
lean_inc_ref(v___f_4606_);
v___f_4608_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__3___boxed), 10, 4);
lean_closure_set(v___f_4608_, 0, v___f_4606_);
lean_closure_set(v___f_4608_, 1, v_declName_4594_);
lean_closure_set(v___f_4608_, 2, v___x_4607_);
lean_closure_set(v___f_4608_, 3, v_env_4604_);
v___x_4609_ = 1;
v___x_4610_ = l_Lean_Environment_contains(v_env_4604_, v_declName_4594_, v___x_4609_);
if (v___x_4610_ == 0)
{
lean_object* v___f_4611_; lean_object* v___x_4612_; 
lean_dec_ref(v___f_4606_);
lean_dec(v_declName_4594_);
v___f_4611_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__4___boxed), 6, 1);
lean_closure_set(v___f_4611_, 0, v___f_4608_);
v___x_4612_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(v_env_4604_, v___f_4611_, v___y_4595_, v___y_4596_, v___y_4597_, v___y_4598_);
return v___x_4612_;
}
else
{
lean_object* v___x_4613_; lean_object* v___x_4614_; lean_object* v___f_4615_; lean_object* v___x_4616_; 
v___x_4613_ = lean_box(v___x_4609_);
v___x_4614_ = lean_box(v___x_4602_);
lean_inc_ref(v_env_4604_);
v___f_4615_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___lam__5___boxed), 12, 7);
lean_closure_set(v___f_4615_, 0, v_addInfo_4603_);
lean_closure_set(v___f_4615_, 1, v_declName_4594_);
lean_closure_set(v___f_4615_, 2, v___x_4613_);
lean_closure_set(v___f_4615_, 3, v___f_4606_);
lean_closure_set(v___f_4615_, 4, v___x_4614_);
lean_closure_set(v___f_4615_, 5, v_env_4604_);
lean_closure_set(v___f_4615_, 6, v___f_4608_);
v___x_4616_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(v_env_4604_, v___f_4615_, v___y_4595_, v___y_4596_, v___y_4597_, v___y_4598_);
return v___x_4616_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0___boxed(lean_object* v_declName_4617_, lean_object* v___y_4618_, lean_object* v___y_4619_, lean_object* v___y_4620_, lean_object* v___y_4621_, lean_object* v___y_4622_){
_start:
{
lean_object* v_res_4623_; 
v_res_4623_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0(v_declName_4617_, v___y_4618_, v___y_4619_, v___y_4620_, v___y_4621_);
lean_dec(v___y_4621_);
lean_dec_ref(v___y_4620_);
lean_dec(v___y_4619_);
lean_dec_ref(v___y_4618_);
return v_res_4623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(lean_object* v_stx_4624_, lean_object* v___y_4625_){
_start:
{
uint8_t v___x_4627_; lean_object* v___x_4628_; 
v___x_4627_ = 0;
v___x_4628_ = l_Lean_Syntax_getRange_x3f(v_stx_4624_, v___x_4627_);
if (lean_obj_tag(v___x_4628_) == 1)
{
lean_object* v_val_4629_; lean_object* v___x_4631_; uint8_t v_isShared_4632_; uint8_t v_isSharedCheck_4641_; 
v_val_4629_ = lean_ctor_get(v___x_4628_, 0);
v_isSharedCheck_4641_ = !lean_is_exclusive(v___x_4628_);
if (v_isSharedCheck_4641_ == 0)
{
v___x_4631_ = v___x_4628_;
v_isShared_4632_ = v_isSharedCheck_4641_;
goto v_resetjp_4630_;
}
else
{
lean_inc(v_val_4629_);
lean_dec(v___x_4628_);
v___x_4631_ = lean_box(0);
v_isShared_4632_ = v_isSharedCheck_4641_;
goto v_resetjp_4630_;
}
v_resetjp_4630_:
{
lean_object* v_fileMap_4633_; lean_object* v_start_4634_; lean_object* v_stop_4635_; lean_object* v___x_4636_; lean_object* v___x_4638_; 
v_fileMap_4633_ = lean_ctor_get(v___y_4625_, 1);
v_start_4634_ = lean_ctor_get(v_val_4629_, 0);
lean_inc(v_start_4634_);
v_stop_4635_ = lean_ctor_get(v_val_4629_, 1);
lean_inc(v_stop_4635_);
lean_dec(v_val_4629_);
lean_inc_ref(v_fileMap_4633_);
v___x_4636_ = l_Lean_DeclarationRange_ofStringPositions(v_fileMap_4633_, v_start_4634_, v_stop_4635_);
lean_dec(v_stop_4635_);
lean_dec(v_start_4634_);
if (v_isShared_4632_ == 0)
{
lean_ctor_set(v___x_4631_, 0, v___x_4636_);
v___x_4638_ = v___x_4631_;
goto v_reusejp_4637_;
}
else
{
lean_object* v_reuseFailAlloc_4640_; 
v_reuseFailAlloc_4640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4640_, 0, v___x_4636_);
v___x_4638_ = v_reuseFailAlloc_4640_;
goto v_reusejp_4637_;
}
v_reusejp_4637_:
{
lean_object* v___x_4639_; 
v___x_4639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4639_, 0, v___x_4638_);
return v___x_4639_;
}
}
}
else
{
lean_object* v___x_4642_; lean_object* v___x_4643_; 
lean_dec(v___x_4628_);
v___x_4642_ = lean_box(0);
v___x_4643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4643_, 0, v___x_4642_);
return v___x_4643_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg___boxed(lean_object* v_stx_4644_, lean_object* v___y_4645_, lean_object* v___y_4646_){
_start:
{
lean_object* v_res_4647_; 
v_res_4647_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(v_stx_4644_, v___y_4645_);
lean_dec_ref(v___y_4645_);
lean_dec(v_stx_4644_);
return v_res_4647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg(lean_object* v_declName_4648_, lean_object* v_declRanges_4649_, lean_object* v___y_4650_, lean_object* v___y_4651_){
_start:
{
uint8_t v___x_4653_; 
v___x_4653_ = l_Lean_Name_isAnonymous(v_declName_4648_);
if (v___x_4653_ == 0)
{
lean_object* v___x_4654_; lean_object* v_env_4655_; lean_object* v_nextMacroScope_4656_; lean_object* v_ngen_4657_; lean_object* v_auxDeclNGen_4658_; lean_object* v_traceState_4659_; lean_object* v_messages_4660_; lean_object* v_infoState_4661_; lean_object* v_snapshotTasks_4662_; lean_object* v___x_4664_; uint8_t v_isShared_4665_; uint8_t v_isSharedCheck_4690_; 
v___x_4654_ = lean_st_ref_take(v___y_4651_);
v_env_4655_ = lean_ctor_get(v___x_4654_, 0);
v_nextMacroScope_4656_ = lean_ctor_get(v___x_4654_, 1);
v_ngen_4657_ = lean_ctor_get(v___x_4654_, 2);
v_auxDeclNGen_4658_ = lean_ctor_get(v___x_4654_, 3);
v_traceState_4659_ = lean_ctor_get(v___x_4654_, 4);
v_messages_4660_ = lean_ctor_get(v___x_4654_, 6);
v_infoState_4661_ = lean_ctor_get(v___x_4654_, 7);
v_snapshotTasks_4662_ = lean_ctor_get(v___x_4654_, 8);
v_isSharedCheck_4690_ = !lean_is_exclusive(v___x_4654_);
if (v_isSharedCheck_4690_ == 0)
{
lean_object* v_unused_4691_; 
v_unused_4691_ = lean_ctor_get(v___x_4654_, 5);
lean_dec(v_unused_4691_);
v___x_4664_ = v___x_4654_;
v_isShared_4665_ = v_isSharedCheck_4690_;
goto v_resetjp_4663_;
}
else
{
lean_inc(v_snapshotTasks_4662_);
lean_inc(v_infoState_4661_);
lean_inc(v_messages_4660_);
lean_inc(v_traceState_4659_);
lean_inc(v_auxDeclNGen_4658_);
lean_inc(v_ngen_4657_);
lean_inc(v_nextMacroScope_4656_);
lean_inc(v_env_4655_);
lean_dec(v___x_4654_);
v___x_4664_ = lean_box(0);
v_isShared_4665_ = v_isSharedCheck_4690_;
goto v_resetjp_4663_;
}
v_resetjp_4663_:
{
lean_object* v___x_4666_; lean_object* v___x_4667_; lean_object* v___x_4668_; lean_object* v___x_4670_; 
v___x_4666_ = l_Lean_declRangeExt;
v___x_4667_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_4666_, v_env_4655_, v_declName_4648_, v_declRanges_4649_);
v___x_4668_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_4665_ == 0)
{
lean_ctor_set(v___x_4664_, 5, v___x_4668_);
lean_ctor_set(v___x_4664_, 0, v___x_4667_);
v___x_4670_ = v___x_4664_;
goto v_reusejp_4669_;
}
else
{
lean_object* v_reuseFailAlloc_4689_; 
v_reuseFailAlloc_4689_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4689_, 0, v___x_4667_);
lean_ctor_set(v_reuseFailAlloc_4689_, 1, v_nextMacroScope_4656_);
lean_ctor_set(v_reuseFailAlloc_4689_, 2, v_ngen_4657_);
lean_ctor_set(v_reuseFailAlloc_4689_, 3, v_auxDeclNGen_4658_);
lean_ctor_set(v_reuseFailAlloc_4689_, 4, v_traceState_4659_);
lean_ctor_set(v_reuseFailAlloc_4689_, 5, v___x_4668_);
lean_ctor_set(v_reuseFailAlloc_4689_, 6, v_messages_4660_);
lean_ctor_set(v_reuseFailAlloc_4689_, 7, v_infoState_4661_);
lean_ctor_set(v_reuseFailAlloc_4689_, 8, v_snapshotTasks_4662_);
v___x_4670_ = v_reuseFailAlloc_4689_;
goto v_reusejp_4669_;
}
v_reusejp_4669_:
{
lean_object* v___x_4671_; lean_object* v___x_4672_; lean_object* v_mctx_4673_; lean_object* v_zetaDeltaFVarIds_4674_; lean_object* v_postponed_4675_; lean_object* v_diag_4676_; lean_object* v___x_4678_; uint8_t v_isShared_4679_; uint8_t v_isSharedCheck_4687_; 
v___x_4671_ = lean_st_ref_set(v___y_4651_, v___x_4670_);
v___x_4672_ = lean_st_ref_take(v___y_4650_);
v_mctx_4673_ = lean_ctor_get(v___x_4672_, 0);
v_zetaDeltaFVarIds_4674_ = lean_ctor_get(v___x_4672_, 2);
v_postponed_4675_ = lean_ctor_get(v___x_4672_, 3);
v_diag_4676_ = lean_ctor_get(v___x_4672_, 4);
v_isSharedCheck_4687_ = !lean_is_exclusive(v___x_4672_);
if (v_isSharedCheck_4687_ == 0)
{
lean_object* v_unused_4688_; 
v_unused_4688_ = lean_ctor_get(v___x_4672_, 1);
lean_dec(v_unused_4688_);
v___x_4678_ = v___x_4672_;
v_isShared_4679_ = v_isSharedCheck_4687_;
goto v_resetjp_4677_;
}
else
{
lean_inc(v_diag_4676_);
lean_inc(v_postponed_4675_);
lean_inc(v_zetaDeltaFVarIds_4674_);
lean_inc(v_mctx_4673_);
lean_dec(v___x_4672_);
v___x_4678_ = lean_box(0);
v_isShared_4679_ = v_isSharedCheck_4687_;
goto v_resetjp_4677_;
}
v_resetjp_4677_:
{
lean_object* v___x_4680_; lean_object* v___x_4682_; 
v___x_4680_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_4679_ == 0)
{
lean_ctor_set(v___x_4678_, 1, v___x_4680_);
v___x_4682_ = v___x_4678_;
goto v_reusejp_4681_;
}
else
{
lean_object* v_reuseFailAlloc_4686_; 
v_reuseFailAlloc_4686_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4686_, 0, v_mctx_4673_);
lean_ctor_set(v_reuseFailAlloc_4686_, 1, v___x_4680_);
lean_ctor_set(v_reuseFailAlloc_4686_, 2, v_zetaDeltaFVarIds_4674_);
lean_ctor_set(v_reuseFailAlloc_4686_, 3, v_postponed_4675_);
lean_ctor_set(v_reuseFailAlloc_4686_, 4, v_diag_4676_);
v___x_4682_ = v_reuseFailAlloc_4686_;
goto v_reusejp_4681_;
}
v_reusejp_4681_:
{
lean_object* v___x_4683_; lean_object* v___x_4684_; lean_object* v___x_4685_; 
v___x_4683_ = lean_st_ref_set(v___y_4650_, v___x_4682_);
v___x_4684_ = lean_box(0);
v___x_4685_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4685_, 0, v___x_4684_);
return v___x_4685_;
}
}
}
}
}
else
{
lean_object* v___x_4692_; lean_object* v___x_4693_; 
lean_dec_ref(v_declRanges_4649_);
lean_dec(v_declName_4648_);
v___x_4692_ = lean_box(0);
v___x_4693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4693_, 0, v___x_4692_);
return v___x_4693_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg___boxed(lean_object* v_declName_4694_, lean_object* v_declRanges_4695_, lean_object* v___y_4696_, lean_object* v___y_4697_, lean_object* v___y_4698_){
_start:
{
lean_object* v_res_4699_; 
v_res_4699_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg(v_declName_4694_, v_declRanges_4695_, v___y_4696_, v___y_4697_);
lean_dec(v___y_4697_);
lean_dec(v___y_4696_);
return v_res_4699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1(lean_object* v_declName_4700_, lean_object* v_rangeStx_4701_, lean_object* v_selectionRangeStx_4702_, lean_object* v___y_4703_, lean_object* v___y_4704_, lean_object* v___y_4705_, lean_object* v___y_4706_){
_start:
{
lean_object* v___x_4708_; lean_object* v_a_4709_; lean_object* v___x_4711_; uint8_t v_isShared_4712_; uint8_t v_isSharedCheck_4725_; 
v___x_4708_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(v_rangeStx_4701_, v___y_4705_);
v_a_4709_ = lean_ctor_get(v___x_4708_, 0);
v_isSharedCheck_4725_ = !lean_is_exclusive(v___x_4708_);
if (v_isSharedCheck_4725_ == 0)
{
v___x_4711_ = v___x_4708_;
v_isShared_4712_ = v_isSharedCheck_4725_;
goto v_resetjp_4710_;
}
else
{
lean_inc(v_a_4709_);
lean_dec(v___x_4708_);
v___x_4711_ = lean_box(0);
v_isShared_4712_ = v_isSharedCheck_4725_;
goto v_resetjp_4710_;
}
v_resetjp_4710_:
{
if (lean_obj_tag(v_a_4709_) == 1)
{
lean_object* v_val_4713_; lean_object* v___x_4714_; lean_object* v_a_4715_; lean_object* v_a_4717_; 
lean_del_object(v___x_4711_);
v_val_4713_ = lean_ctor_get(v_a_4709_, 0);
lean_inc(v_val_4713_);
lean_dec_ref_known(v_a_4709_, 1);
v___x_4714_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(v_selectionRangeStx_4702_, v___y_4705_);
v_a_4715_ = lean_ctor_get(v___x_4714_, 0);
lean_inc(v_a_4715_);
lean_dec_ref(v___x_4714_);
if (lean_obj_tag(v_a_4715_) == 0)
{
lean_inc(v_val_4713_);
v_a_4717_ = v_val_4713_;
goto v___jp_4716_;
}
else
{
lean_object* v_val_4720_; 
v_val_4720_ = lean_ctor_get(v_a_4715_, 0);
lean_inc(v_val_4720_);
lean_dec_ref_known(v_a_4715_, 1);
v_a_4717_ = v_val_4720_;
goto v___jp_4716_;
}
v___jp_4716_:
{
lean_object* v___x_4718_; lean_object* v___x_4719_; 
v___x_4718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4718_, 0, v_val_4713_);
lean_ctor_set(v___x_4718_, 1, v_a_4717_);
v___x_4719_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg(v_declName_4700_, v___x_4718_, v___y_4704_, v___y_4706_);
return v___x_4719_;
}
}
else
{
lean_object* v___x_4721_; lean_object* v___x_4723_; 
lean_dec(v_a_4709_);
lean_dec(v_declName_4700_);
v___x_4721_ = lean_box(0);
if (v_isShared_4712_ == 0)
{
lean_ctor_set(v___x_4711_, 0, v___x_4721_);
v___x_4723_ = v___x_4711_;
goto v_reusejp_4722_;
}
else
{
lean_object* v_reuseFailAlloc_4724_; 
v_reuseFailAlloc_4724_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4724_, 0, v___x_4721_);
v___x_4723_ = v_reuseFailAlloc_4724_;
goto v_reusejp_4722_;
}
v_reusejp_4722_:
{
return v___x_4723_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1___boxed(lean_object* v_declName_4726_, lean_object* v_rangeStx_4727_, lean_object* v_selectionRangeStx_4728_, lean_object* v___y_4729_, lean_object* v___y_4730_, lean_object* v___y_4731_, lean_object* v___y_4732_, lean_object* v___y_4733_){
_start:
{
lean_object* v_res_4734_; 
v_res_4734_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1(v_declName_4726_, v_rangeStx_4727_, v_selectionRangeStx_4728_, v___y_4729_, v___y_4730_, v___y_4731_, v___y_4732_);
lean_dec(v___y_4732_);
lean_dec_ref(v___y_4731_);
lean_dec(v___y_4730_);
lean_dec_ref(v___y_4729_);
lean_dec(v_selectionRangeStx_4728_);
lean_dec(v_rangeStx_4727_);
return v_res_4734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg(lean_object* v_x_4735_, uint8_t v_isExporting_4736_, lean_object* v___y_4737_, lean_object* v___y_4738_, lean_object* v___y_4739_, lean_object* v___y_4740_){
_start:
{
lean_object* v___x_4742_; lean_object* v_env_4743_; uint8_t v_isExporting_4744_; lean_object* v___x_4810_; uint8_t v_isModule_4811_; 
v___x_4742_ = lean_st_ref_get(v___y_4740_);
v_env_4743_ = lean_ctor_get(v___x_4742_, 0);
lean_inc_ref(v_env_4743_);
lean_dec(v___x_4742_);
v_isExporting_4744_ = lean_ctor_get_uint8(v_env_4743_, sizeof(void*)*8);
v___x_4810_ = l_Lean_Environment_header(v_env_4743_);
lean_dec_ref(v_env_4743_);
v_isModule_4811_ = lean_ctor_get_uint8(v___x_4810_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_4810_);
if (v_isModule_4811_ == 0)
{
lean_object* v___x_4812_; 
lean_inc(v___y_4740_);
lean_inc_ref(v___y_4739_);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
v___x_4812_ = lean_apply_5(v_x_4735_, v___y_4737_, v___y_4738_, v___y_4739_, v___y_4740_, lean_box(0));
return v___x_4812_;
}
else
{
if (v_isExporting_4744_ == 0)
{
if (v_isExporting_4736_ == 0)
{
lean_object* v___x_4813_; 
lean_inc(v___y_4740_);
lean_inc_ref(v___y_4739_);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
v___x_4813_ = lean_apply_5(v_x_4735_, v___y_4737_, v___y_4738_, v___y_4739_, v___y_4740_, lean_box(0));
return v___x_4813_;
}
else
{
goto v___jp_4745_;
}
}
else
{
if (v_isExporting_4736_ == 0)
{
goto v___jp_4745_;
}
else
{
lean_object* v___x_4814_; 
lean_inc(v___y_4740_);
lean_inc_ref(v___y_4739_);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
v___x_4814_ = lean_apply_5(v_x_4735_, v___y_4737_, v___y_4738_, v___y_4739_, v___y_4740_, lean_box(0));
return v___x_4814_;
}
}
}
v___jp_4745_:
{
lean_object* v___x_4746_; lean_object* v_env_4747_; lean_object* v_nextMacroScope_4748_; lean_object* v_ngen_4749_; lean_object* v_auxDeclNGen_4750_; lean_object* v_traceState_4751_; lean_object* v_messages_4752_; lean_object* v_infoState_4753_; lean_object* v_snapshotTasks_4754_; lean_object* v___x_4756_; uint8_t v_isShared_4757_; uint8_t v_isSharedCheck_4808_; 
v___x_4746_ = lean_st_ref_take(v___y_4740_);
v_env_4747_ = lean_ctor_get(v___x_4746_, 0);
v_nextMacroScope_4748_ = lean_ctor_get(v___x_4746_, 1);
v_ngen_4749_ = lean_ctor_get(v___x_4746_, 2);
v_auxDeclNGen_4750_ = lean_ctor_get(v___x_4746_, 3);
v_traceState_4751_ = lean_ctor_get(v___x_4746_, 4);
v_messages_4752_ = lean_ctor_get(v___x_4746_, 6);
v_infoState_4753_ = lean_ctor_get(v___x_4746_, 7);
v_snapshotTasks_4754_ = lean_ctor_get(v___x_4746_, 8);
v_isSharedCheck_4808_ = !lean_is_exclusive(v___x_4746_);
if (v_isSharedCheck_4808_ == 0)
{
lean_object* v_unused_4809_; 
v_unused_4809_ = lean_ctor_get(v___x_4746_, 5);
lean_dec(v_unused_4809_);
v___x_4756_ = v___x_4746_;
v_isShared_4757_ = v_isSharedCheck_4808_;
goto v_resetjp_4755_;
}
else
{
lean_inc(v_snapshotTasks_4754_);
lean_inc(v_infoState_4753_);
lean_inc(v_messages_4752_);
lean_inc(v_traceState_4751_);
lean_inc(v_auxDeclNGen_4750_);
lean_inc(v_ngen_4749_);
lean_inc(v_nextMacroScope_4748_);
lean_inc(v_env_4747_);
lean_dec(v___x_4746_);
v___x_4756_ = lean_box(0);
v_isShared_4757_ = v_isSharedCheck_4808_;
goto v_resetjp_4755_;
}
v_resetjp_4755_:
{
lean_object* v___x_4758_; lean_object* v___x_4759_; lean_object* v___x_4761_; 
v___x_4758_ = l_Lean_Environment_setExporting(v_env_4747_, v_isExporting_4736_);
v___x_4759_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__2);
if (v_isShared_4757_ == 0)
{
lean_ctor_set(v___x_4756_, 5, v___x_4759_);
lean_ctor_set(v___x_4756_, 0, v___x_4758_);
v___x_4761_ = v___x_4756_;
goto v_reusejp_4760_;
}
else
{
lean_object* v_reuseFailAlloc_4807_; 
v_reuseFailAlloc_4807_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4807_, 0, v___x_4758_);
lean_ctor_set(v_reuseFailAlloc_4807_, 1, v_nextMacroScope_4748_);
lean_ctor_set(v_reuseFailAlloc_4807_, 2, v_ngen_4749_);
lean_ctor_set(v_reuseFailAlloc_4807_, 3, v_auxDeclNGen_4750_);
lean_ctor_set(v_reuseFailAlloc_4807_, 4, v_traceState_4751_);
lean_ctor_set(v_reuseFailAlloc_4807_, 5, v___x_4759_);
lean_ctor_set(v_reuseFailAlloc_4807_, 6, v_messages_4752_);
lean_ctor_set(v_reuseFailAlloc_4807_, 7, v_infoState_4753_);
lean_ctor_set(v_reuseFailAlloc_4807_, 8, v_snapshotTasks_4754_);
v___x_4761_ = v_reuseFailAlloc_4807_;
goto v_reusejp_4760_;
}
v_reusejp_4760_:
{
lean_object* v___x_4762_; lean_object* v___x_4763_; lean_object* v_mctx_4764_; lean_object* v_zetaDeltaFVarIds_4765_; lean_object* v_postponed_4766_; lean_object* v_diag_4767_; lean_object* v___x_4769_; uint8_t v_isShared_4770_; uint8_t v_isSharedCheck_4805_; 
v___x_4762_ = lean_st_ref_set(v___y_4740_, v___x_4761_);
v___x_4763_ = lean_st_ref_take(v___y_4738_);
v_mctx_4764_ = lean_ctor_get(v___x_4763_, 0);
v_zetaDeltaFVarIds_4765_ = lean_ctor_get(v___x_4763_, 2);
v_postponed_4766_ = lean_ctor_get(v___x_4763_, 3);
v_diag_4767_ = lean_ctor_get(v___x_4763_, 4);
v_isSharedCheck_4805_ = !lean_is_exclusive(v___x_4763_);
if (v_isSharedCheck_4805_ == 0)
{
lean_object* v_unused_4806_; 
v_unused_4806_ = lean_ctor_get(v___x_4763_, 1);
lean_dec(v_unused_4806_);
v___x_4769_ = v___x_4763_;
v_isShared_4770_ = v_isSharedCheck_4805_;
goto v_resetjp_4768_;
}
else
{
lean_inc(v_diag_4767_);
lean_inc(v_postponed_4766_);
lean_inc(v_zetaDeltaFVarIds_4765_);
lean_inc(v_mctx_4764_);
lean_dec(v___x_4763_);
v___x_4769_ = lean_box(0);
v_isShared_4770_ = v_isSharedCheck_4805_;
goto v_resetjp_4768_;
}
v_resetjp_4768_:
{
lean_object* v___x_4771_; lean_object* v___x_4773_; 
v___x_4771_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___closed__3);
if (v_isShared_4770_ == 0)
{
lean_ctor_set(v___x_4769_, 1, v___x_4771_);
v___x_4773_ = v___x_4769_;
goto v_reusejp_4772_;
}
else
{
lean_object* v_reuseFailAlloc_4804_; 
v_reuseFailAlloc_4804_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4804_, 0, v_mctx_4764_);
lean_ctor_set(v_reuseFailAlloc_4804_, 1, v___x_4771_);
lean_ctor_set(v_reuseFailAlloc_4804_, 2, v_zetaDeltaFVarIds_4765_);
lean_ctor_set(v_reuseFailAlloc_4804_, 3, v_postponed_4766_);
lean_ctor_set(v_reuseFailAlloc_4804_, 4, v_diag_4767_);
v___x_4773_ = v_reuseFailAlloc_4804_;
goto v_reusejp_4772_;
}
v_reusejp_4772_:
{
lean_object* v___x_4774_; lean_object* v_r_4775_; 
v___x_4774_ = lean_st_ref_set(v___y_4738_, v___x_4773_);
lean_inc(v___y_4740_);
lean_inc_ref(v___y_4739_);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
v_r_4775_ = lean_apply_5(v_x_4735_, v___y_4737_, v___y_4738_, v___y_4739_, v___y_4740_, lean_box(0));
if (lean_obj_tag(v_r_4775_) == 0)
{
lean_object* v_a_4776_; lean_object* v___x_4778_; uint8_t v_isShared_4779_; uint8_t v_isSharedCheck_4792_; 
v_a_4776_ = lean_ctor_get(v_r_4775_, 0);
v_isSharedCheck_4792_ = !lean_is_exclusive(v_r_4775_);
if (v_isSharedCheck_4792_ == 0)
{
v___x_4778_ = v_r_4775_;
v_isShared_4779_ = v_isSharedCheck_4792_;
goto v_resetjp_4777_;
}
else
{
lean_inc(v_a_4776_);
lean_dec(v_r_4775_);
v___x_4778_ = lean_box(0);
v_isShared_4779_ = v_isSharedCheck_4792_;
goto v_resetjp_4777_;
}
v_resetjp_4777_:
{
lean_object* v___x_4781_; 
lean_inc(v_a_4776_);
if (v_isShared_4779_ == 0)
{
lean_ctor_set_tag(v___x_4778_, 1);
v___x_4781_ = v___x_4778_;
goto v_reusejp_4780_;
}
else
{
lean_object* v_reuseFailAlloc_4791_; 
v_reuseFailAlloc_4791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4791_, 0, v_a_4776_);
v___x_4781_ = v_reuseFailAlloc_4791_;
goto v_reusejp_4780_;
}
v_reusejp_4780_:
{
lean_object* v___x_4782_; lean_object* v___x_4784_; uint8_t v_isShared_4785_; uint8_t v_isSharedCheck_4789_; 
v___x_4782_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(v___y_4740_, v_isExporting_4744_, v___x_4759_, v___y_4738_, v___x_4771_, v___x_4781_);
lean_dec_ref(v___x_4781_);
v_isSharedCheck_4789_ = !lean_is_exclusive(v___x_4782_);
if (v_isSharedCheck_4789_ == 0)
{
lean_object* v_unused_4790_; 
v_unused_4790_ = lean_ctor_get(v___x_4782_, 0);
lean_dec(v_unused_4790_);
v___x_4784_ = v___x_4782_;
v_isShared_4785_ = v_isSharedCheck_4789_;
goto v_resetjp_4783_;
}
else
{
lean_dec(v___x_4782_);
v___x_4784_ = lean_box(0);
v_isShared_4785_ = v_isSharedCheck_4789_;
goto v_resetjp_4783_;
}
v_resetjp_4783_:
{
lean_object* v___x_4787_; 
if (v_isShared_4785_ == 0)
{
lean_ctor_set(v___x_4784_, 0, v_a_4776_);
v___x_4787_ = v___x_4784_;
goto v_reusejp_4786_;
}
else
{
lean_object* v_reuseFailAlloc_4788_; 
v_reuseFailAlloc_4788_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4788_, 0, v_a_4776_);
v___x_4787_ = v_reuseFailAlloc_4788_;
goto v_reusejp_4786_;
}
v_reusejp_4786_:
{
return v___x_4787_;
}
}
}
}
}
else
{
lean_object* v_a_4793_; lean_object* v___x_4794_; lean_object* v___x_4795_; lean_object* v___x_4797_; uint8_t v_isShared_4798_; uint8_t v_isSharedCheck_4802_; 
v_a_4793_ = lean_ctor_get(v_r_4775_, 0);
lean_inc(v_a_4793_);
lean_dec_ref_known(v_r_4775_, 1);
v___x_4794_ = lean_box(0);
v___x_4795_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Lean_Elab_elabAttr___at___00Lean_Elab_elabAttrs___at___00Mathlib_Tactic_elabOptAttrArg_spec__1_spec__1_spec__6_spec__19___redArg___lam__0(v___y_4740_, v_isExporting_4744_, v___x_4759_, v___y_4738_, v___x_4771_, v___x_4794_);
v_isSharedCheck_4802_ = !lean_is_exclusive(v___x_4795_);
if (v_isSharedCheck_4802_ == 0)
{
lean_object* v_unused_4803_; 
v_unused_4803_ = lean_ctor_get(v___x_4795_, 0);
lean_dec(v_unused_4803_);
v___x_4797_ = v___x_4795_;
v_isShared_4798_ = v_isSharedCheck_4802_;
goto v_resetjp_4796_;
}
else
{
lean_dec(v___x_4795_);
v___x_4797_ = lean_box(0);
v_isShared_4798_ = v_isSharedCheck_4802_;
goto v_resetjp_4796_;
}
v_resetjp_4796_:
{
lean_object* v___x_4800_; 
if (v_isShared_4798_ == 0)
{
lean_ctor_set_tag(v___x_4797_, 1);
lean_ctor_set(v___x_4797_, 0, v_a_4793_);
v___x_4800_ = v___x_4797_;
goto v_reusejp_4799_;
}
else
{
lean_object* v_reuseFailAlloc_4801_; 
v_reuseFailAlloc_4801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4801_, 0, v_a_4793_);
v___x_4800_ = v_reuseFailAlloc_4801_;
goto v_reusejp_4799_;
}
v_reusejp_4799_:
{
return v___x_4800_;
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg___boxed(lean_object* v_x_4815_, lean_object* v_isExporting_4816_, lean_object* v___y_4817_, lean_object* v___y_4818_, lean_object* v___y_4819_, lean_object* v___y_4820_, lean_object* v___y_4821_){
_start:
{
uint8_t v_isExporting_boxed_4822_; lean_object* v_res_4823_; 
v_isExporting_boxed_4822_ = lean_unbox(v_isExporting_4816_);
v_res_4823_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg(v_x_4815_, v_isExporting_boxed_4822_, v___y_4817_, v___y_4818_, v___y_4819_, v___y_4820_);
lean_dec(v___y_4820_);
lean_dec_ref(v___y_4819_);
lean_dec(v___y_4818_);
lean_dec_ref(v___y_4817_);
return v_res_4823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg(lean_object* v_x_4824_, uint8_t v_when_4825_, lean_object* v___y_4826_, lean_object* v___y_4827_, lean_object* v___y_4828_, lean_object* v___y_4829_){
_start:
{
if (v_when_4825_ == 0)
{
lean_object* v___x_4831_; 
lean_inc(v___y_4829_);
lean_inc_ref(v___y_4828_);
lean_inc(v___y_4827_);
lean_inc_ref(v___y_4826_);
v___x_4831_ = lean_apply_5(v_x_4824_, v___y_4826_, v___y_4827_, v___y_4828_, v___y_4829_, lean_box(0));
return v___x_4831_;
}
else
{
uint8_t v___x_4832_; lean_object* v___x_4833_; 
v___x_4832_ = 0;
v___x_4833_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg(v_x_4824_, v___x_4832_, v___y_4826_, v___y_4827_, v___y_4828_, v___y_4829_);
return v___x_4833_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg___boxed(lean_object* v_x_4834_, lean_object* v_when_4835_, lean_object* v___y_4836_, lean_object* v___y_4837_, lean_object* v___y_4838_, lean_object* v___y_4839_, lean_object* v___y_4840_){
_start:
{
uint8_t v_when_boxed_4841_; lean_object* v_res_4842_; 
v_when_boxed_4841_ = lean_unbox(v_when_4835_);
v_res_4842_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg(v_x_4834_, v_when_boxed_4841_, v___y_4836_, v___y_4837_, v___y_4838_, v___y_4839_);
lean_dec(v___y_4839_);
lean_dec_ref(v___y_4838_);
lean_dec(v___y_4837_);
lean_dec_ref(v___y_4836_);
return v_res_4842_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4(void){
_start:
{
lean_object* v___x_4850_; lean_object* v___x_4851_; 
v___x_4850_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__3));
v___x_4851_ = l_Lean_stringToMessageData(v___x_4850_);
return v___x_4851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl(lean_object* v_src_4852_, lean_object* v_tgt_4853_, lean_object* v_ref_4854_, lean_object* v_attrs_4855_, lean_object* v_construct_4856_, lean_object* v_docstringPrefix_x3f_4857_, uint8_t v_hoverInfo_4858_, lean_object* v_a_4859_, lean_object* v_a_4860_, lean_object* v_a_4861_, lean_object* v_a_4862_){
_start:
{
lean_object* v___x_4864_; 
lean_inc(v_tgt_4853_);
v___x_4864_ = lp_mathlib_Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0(v_tgt_4853_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
if (lean_obj_tag(v___x_4864_) == 0)
{
lean_object* v_ref_4865_; lean_object* v___x_4866_; 
lean_dec_ref_known(v___x_4864_, 1);
v_ref_4865_ = lean_ctor_get(v_a_4861_, 5);
lean_inc(v_tgt_4853_);
v___x_4866_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1(v_tgt_4853_, v_ref_4865_, v_ref_4854_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
if (lean_obj_tag(v___x_4866_) == 0)
{
lean_object* v___x_4868_; uint8_t v_isShared_4869_; uint8_t v_isSharedCheck_5042_; 
v_isSharedCheck_5042_ = !lean_is_exclusive(v___x_4866_);
if (v_isSharedCheck_5042_ == 0)
{
lean_object* v_unused_5043_; 
v_unused_5043_ = lean_ctor_get(v___x_4866_, 0);
lean_dec(v_unused_5043_);
v___x_4868_ = v___x_4866_;
v_isShared_4869_ = v_isSharedCheck_5042_;
goto v_resetjp_4867_;
}
else
{
lean_dec(v___x_4866_);
v___x_4868_ = lean_box(0);
v_isShared_4869_ = v_isSharedCheck_5042_;
goto v_resetjp_4867_;
}
v_resetjp_4867_:
{
lean_object* v___x_4870_; uint8_t v___x_4871_; lean_object* v___y_4873_; lean_object* v___y_4874_; uint8_t v___y_4875_; lean_object* v___y_4876_; lean_object* v___y_4877_; lean_object* v___y_4878_; lean_object* v___y_4879_; lean_object* v___y_4880_; lean_object* v___y_4907_; uint8_t v___y_4908_; lean_object* v___y_4909_; lean_object* v___y_4910_; lean_object* v_doc_4911_; lean_object* v___y_4912_; lean_object* v___y_4913_; lean_object* v___y_4914_; lean_object* v___y_4915_; uint8_t v___y_4918_; lean_object* v___y_4919_; uint8_t v___y_4920_; lean_object* v___y_4921_; lean_object* v___y_4922_; lean_object* v___y_4923_; lean_object* v___y_4924_; lean_object* v___y_4925_; lean_object* v___x_4960_; 
lean_inc(v_src_4852_);
v___x_4870_ = lean_alloc_closure((void*)(lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2___boxed), 6, 1);
lean_closure_set(v___x_4870_, 0, v_src_4852_);
v___x_4871_ = 1;
v___x_4960_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg(v___x_4870_, v___x_4871_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
if (lean_obj_tag(v___x_4960_) == 0)
{
lean_object* v_a_4961_; lean_object* v___x_4962_; lean_object* v___x_4963_; lean_object* v___x_4964_; lean_object* v___x_4965_; lean_object* v___x_4966_; 
v_a_4961_ = lean_ctor_get(v___x_4960_, 0);
lean_inc(v_a_4961_);
lean_dec_ref_known(v___x_4960_, 1);
v___x_4962_ = l_Lean_ConstantInfo_levelParams(v_a_4961_);
lean_dec(v_a_4961_);
v___x_4963_ = lean_box(0);
lean_inc(v___x_4962_);
v___x_4964_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_addRelatedDecl_spec__4(v___x_4962_, v___x_4963_);
lean_inc(v_src_4852_);
v___x_4965_ = l_Lean_Expr_const___override(v_src_4852_, v___x_4964_);
lean_inc(v_a_4862_);
lean_inc_ref(v_a_4861_);
lean_inc(v_a_4860_);
lean_inc_ref(v_a_4859_);
v___x_4966_ = lean_apply_7(v_construct_4856_, v___x_4965_, v___x_4962_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_, lean_box(0));
if (lean_obj_tag(v___x_4966_) == 0)
{
lean_object* v_a_4967_; lean_object* v_fst_4968_; lean_object* v_snd_4969_; lean_object* v___x_4971_; uint8_t v_isShared_4972_; uint8_t v_isSharedCheck_5025_; 
v_a_4967_ = lean_ctor_get(v___x_4966_, 0);
lean_inc(v_a_4967_);
lean_dec_ref_known(v___x_4966_, 1);
v_fst_4968_ = lean_ctor_get(v_a_4967_, 0);
v_snd_4969_ = lean_ctor_get(v_a_4967_, 1);
v_isSharedCheck_5025_ = !lean_is_exclusive(v_a_4967_);
if (v_isSharedCheck_5025_ == 0)
{
v___x_4971_ = v_a_4967_;
v_isShared_4972_ = v_isSharedCheck_5025_;
goto v_resetjp_4970_;
}
else
{
lean_inc(v_snd_4969_);
lean_inc(v_fst_4968_);
lean_dec(v_a_4967_);
v___x_4971_ = lean_box(0);
v_isShared_4972_ = v_isSharedCheck_5025_;
goto v_resetjp_4970_;
}
v_resetjp_4970_:
{
lean_object* v___x_4973_; lean_object* v_a_4974_; lean_object* v___x_4975_; 
v___x_4973_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(v_fst_4968_, v_a_4860_);
v_a_4974_ = lean_ctor_get(v___x_4973_, 0);
lean_inc_n(v_a_4974_, 2);
lean_dec_ref(v___x_4973_);
lean_inc(v_a_4862_);
lean_inc_ref(v_a_4861_);
lean_inc(v_a_4860_);
lean_inc_ref(v_a_4859_);
v___x_4975_ = lean_infer_type(v_a_4974_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
if (lean_obj_tag(v___x_4975_) == 0)
{
lean_object* v_a_4976_; lean_object* v___x_4977_; lean_object* v_a_4978_; lean_object* v___y_4980_; lean_object* v___y_4981_; lean_object* v___y_4982_; lean_object* v___y_4983_; lean_object* v___x_5002_; 
v_a_4976_ = lean_ctor_get(v___x_4975_, 0);
lean_inc(v_a_4976_);
lean_dec_ref_known(v___x_4975_, 1);
v___x_4977_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_addRelatedDecl_spec__5___redArg(v_a_4976_, v_a_4860_);
v_a_4978_ = lean_ctor_get(v___x_4977_, 0);
lean_inc_n(v_a_4978_, 2);
lean_dec_ref(v___x_4977_);
v___x_5002_ = l_Lean_Meta_isProp(v_a_4978_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
if (lean_obj_tag(v___x_5002_) == 0)
{
lean_object* v_a_5003_; uint8_t v___x_5004_; 
v_a_5003_ = lean_ctor_get(v___x_5002_, 0);
lean_inc(v_a_5003_);
lean_dec_ref_known(v___x_5002_, 1);
v___x_5004_ = lean_unbox(v_a_5003_);
lean_dec(v_a_5003_);
if (v___x_5004_ == 0)
{
lean_object* v___x_5005_; lean_object* v___x_5006_; lean_object* v___x_5007_; lean_object* v___x_5008_; 
lean_dec(v_a_4974_);
lean_del_object(v___x_4971_);
lean_dec(v_snd_4969_);
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v___x_5005_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4, &lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__4);
v___x_5006_ = l_Lean_MessageData_ofExpr(v_a_4978_);
v___x_5007_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5007_, 0, v___x_5005_);
lean_ctor_set(v___x_5007_, 1, v___x_5006_);
v___x_5008_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v___x_5007_, v_a_4859_, v_a_4860_, v_a_4861_, v_a_4862_);
return v___x_5008_;
}
else
{
v___y_4980_ = v_a_4859_;
v___y_4981_ = v_a_4860_;
v___y_4982_ = v_a_4861_;
v___y_4983_ = v_a_4862_;
goto v___jp_4979_;
}
}
else
{
lean_object* v_a_5009_; lean_object* v___x_5011_; uint8_t v_isShared_5012_; uint8_t v_isSharedCheck_5016_; 
lean_dec(v_a_4978_);
lean_dec(v_a_4974_);
lean_del_object(v___x_4971_);
lean_dec(v_snd_4969_);
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v_a_5009_ = lean_ctor_get(v___x_5002_, 0);
v_isSharedCheck_5016_ = !lean_is_exclusive(v___x_5002_);
if (v_isSharedCheck_5016_ == 0)
{
v___x_5011_ = v___x_5002_;
v_isShared_5012_ = v_isSharedCheck_5016_;
goto v_resetjp_5010_;
}
else
{
lean_inc(v_a_5009_);
lean_dec(v___x_5002_);
v___x_5011_ = lean_box(0);
v_isShared_5012_ = v_isSharedCheck_5016_;
goto v_resetjp_5010_;
}
v_resetjp_5010_:
{
lean_object* v___x_5014_; 
if (v_isShared_5012_ == 0)
{
v___x_5014_ = v___x_5011_;
goto v_reusejp_5013_;
}
else
{
lean_object* v_reuseFailAlloc_5015_; 
v_reuseFailAlloc_5015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5015_, 0, v_a_5009_);
v___x_5014_ = v_reuseFailAlloc_5015_;
goto v_reusejp_5013_;
}
v_reusejp_5013_:
{
return v___x_5014_;
}
}
}
v___jp_4979_:
{
lean_object* v___x_4984_; 
lean_inc(v_a_4978_);
lean_inc(v_tgt_4853_);
lean_inc(v_ref_4854_);
v___x_4984_ = lp_mathlib_Mathlib_Tactic_warnIfImplicitIllTyped(v_ref_4854_, v_tgt_4853_, v_a_4978_, v___y_4980_, v___y_4981_, v___y_4982_, v___y_4983_);
if (lean_obj_tag(v___x_4984_) == 0)
{
lean_object* v___x_4985_; lean_object* v___x_4987_; 
lean_dec_ref_known(v___x_4984_, 1);
lean_inc_n(v_tgt_4853_, 2);
v___x_4985_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4985_, 0, v_tgt_4853_);
lean_ctor_set(v___x_4985_, 1, v_snd_4969_);
lean_ctor_set(v___x_4985_, 2, v_a_4978_);
if (v_isShared_4972_ == 0)
{
lean_ctor_set_tag(v___x_4971_, 1);
lean_ctor_set(v___x_4971_, 1, v___x_4963_);
lean_ctor_set(v___x_4971_, 0, v_tgt_4853_);
v___x_4987_ = v___x_4971_;
goto v_reusejp_4986_;
}
else
{
lean_object* v_reuseFailAlloc_5001_; 
v_reuseFailAlloc_5001_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5001_, 0, v_tgt_4853_);
lean_ctor_set(v_reuseFailAlloc_5001_, 1, v___x_4963_);
v___x_4987_ = v_reuseFailAlloc_5001_;
goto v_reusejp_4986_;
}
v_reusejp_4986_:
{
lean_object* v___x_4988_; lean_object* v___x_4989_; lean_object* v_a_4990_; uint8_t v___x_4991_; lean_object* v___x_4992_; 
v___x_4988_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4988_, 0, v___x_4985_);
lean_ctor_set(v___x_4988_, 1, v_a_4974_);
lean_ctor_set(v___x_4988_, 2, v___x_4987_);
v___x_4989_ = lp_mathlib_Lean_mkThmOrUnsafeDef___at___00Mathlib_Tactic_addRelatedDecl_spec__6___redArg(v___x_4988_, v___y_4983_);
v_a_4990_ = lean_ctor_get(v___x_4989_, 0);
lean_inc(v_a_4990_);
lean_dec_ref(v___x_4989_);
v___x_4991_ = 0;
v___x_4992_ = l_Lean_addDecl(v_a_4990_, v___x_4991_, v___y_4982_, v___y_4983_);
if (lean_obj_tag(v___x_4992_) == 0)
{
lean_object* v___x_4993_; lean_object* v_env_4994_; lean_object* v___f_4995_; uint8_t v___x_4996_; 
lean_dec_ref_known(v___x_4992_, 1);
v___x_4993_ = lean_st_ref_get(v___y_4983_);
v_env_4994_ = lean_ctor_get(v___x_4993_, 0);
lean_inc_ref(v_env_4994_);
lean_dec(v___x_4993_);
v___f_4995_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__2));
lean_inc(v_src_4852_);
v___x_4996_ = l_Lean_isProtected(v_env_4994_, v_src_4852_);
if (v___x_4996_ == 0)
{
v___y_4918_ = v___x_4991_;
v___y_4919_ = v___f_4995_;
v___y_4920_ = v___x_4991_;
v___y_4921_ = v___x_4963_;
v___y_4922_ = v___y_4980_;
v___y_4923_ = v___y_4981_;
v___y_4924_ = v___y_4982_;
v___y_4925_ = v___y_4983_;
goto v___jp_4917_;
}
else
{
lean_object* v___x_4997_; lean_object* v_env_4998_; lean_object* v___x_4999_; lean_object* v___x_5000_; 
v___x_4997_ = lean_st_ref_get(v___y_4983_);
v_env_4998_ = lean_ctor_get(v___x_4997_, 0);
lean_inc_ref(v_env_4998_);
lean_dec(v___x_4997_);
lean_inc(v_tgt_4853_);
v___x_4999_ = l_Lean_addProtected(v_env_4998_, v_tgt_4853_);
v___x_5000_ = lp_mathlib_Lean_setEnv___at___00Mathlib_Tactic_addRelatedDecl_spec__9___redArg(v___x_4999_, v___y_4981_, v___y_4983_);
lean_dec_ref(v___x_5000_);
v___y_4918_ = v___x_4991_;
v___y_4919_ = v___f_4995_;
v___y_4920_ = v___x_4991_;
v___y_4921_ = v___x_4963_;
v___y_4922_ = v___y_4980_;
v___y_4923_ = v___y_4981_;
v___y_4924_ = v___y_4982_;
v___y_4925_ = v___y_4983_;
goto v___jp_4917_;
}
}
else
{
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
return v___x_4992_;
}
}
}
else
{
lean_dec(v_a_4978_);
lean_dec(v_a_4974_);
lean_del_object(v___x_4971_);
lean_dec(v_snd_4969_);
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
return v___x_4984_;
}
}
}
else
{
lean_object* v_a_5017_; lean_object* v___x_5019_; uint8_t v_isShared_5020_; uint8_t v_isSharedCheck_5024_; 
lean_dec(v_a_4974_);
lean_del_object(v___x_4971_);
lean_dec(v_snd_4969_);
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v_a_5017_ = lean_ctor_get(v___x_4975_, 0);
v_isSharedCheck_5024_ = !lean_is_exclusive(v___x_4975_);
if (v_isSharedCheck_5024_ == 0)
{
v___x_5019_ = v___x_4975_;
v_isShared_5020_ = v_isSharedCheck_5024_;
goto v_resetjp_5018_;
}
else
{
lean_inc(v_a_5017_);
lean_dec(v___x_4975_);
v___x_5019_ = lean_box(0);
v_isShared_5020_ = v_isSharedCheck_5024_;
goto v_resetjp_5018_;
}
v_resetjp_5018_:
{
lean_object* v___x_5022_; 
if (v_isShared_5020_ == 0)
{
v___x_5022_ = v___x_5019_;
goto v_reusejp_5021_;
}
else
{
lean_object* v_reuseFailAlloc_5023_; 
v_reuseFailAlloc_5023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5023_, 0, v_a_5017_);
v___x_5022_ = v_reuseFailAlloc_5023_;
goto v_reusejp_5021_;
}
v_reusejp_5021_:
{
return v___x_5022_;
}
}
}
}
}
else
{
lean_object* v_a_5026_; lean_object* v___x_5028_; uint8_t v_isShared_5029_; uint8_t v_isSharedCheck_5033_; 
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v_a_5026_ = lean_ctor_get(v___x_4966_, 0);
v_isSharedCheck_5033_ = !lean_is_exclusive(v___x_4966_);
if (v_isSharedCheck_5033_ == 0)
{
v___x_5028_ = v___x_4966_;
v_isShared_5029_ = v_isSharedCheck_5033_;
goto v_resetjp_5027_;
}
else
{
lean_inc(v_a_5026_);
lean_dec(v___x_4966_);
v___x_5028_ = lean_box(0);
v_isShared_5029_ = v_isSharedCheck_5033_;
goto v_resetjp_5027_;
}
v_resetjp_5027_:
{
lean_object* v___x_5031_; 
if (v_isShared_5029_ == 0)
{
v___x_5031_ = v___x_5028_;
goto v_reusejp_5030_;
}
else
{
lean_object* v_reuseFailAlloc_5032_; 
v_reuseFailAlloc_5032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5032_, 0, v_a_5026_);
v___x_5031_ = v_reuseFailAlloc_5032_;
goto v_reusejp_5030_;
}
v_reusejp_5030_:
{
return v___x_5031_;
}
}
}
}
else
{
lean_object* v_a_5034_; lean_object* v___x_5036_; uint8_t v_isShared_5037_; uint8_t v_isSharedCheck_5041_; 
lean_del_object(v___x_4868_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec_ref(v_construct_4856_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v_a_5034_ = lean_ctor_get(v___x_4960_, 0);
v_isSharedCheck_5041_ = !lean_is_exclusive(v___x_4960_);
if (v_isSharedCheck_5041_ == 0)
{
v___x_5036_ = v___x_4960_;
v_isShared_5037_ = v_isSharedCheck_5041_;
goto v_resetjp_5035_;
}
else
{
lean_inc(v_a_5034_);
lean_dec(v___x_4960_);
v___x_5036_ = lean_box(0);
v_isShared_5037_ = v_isSharedCheck_5041_;
goto v_resetjp_5035_;
}
v_resetjp_5035_:
{
lean_object* v___x_5039_; 
if (v_isShared_5037_ == 0)
{
v___x_5039_ = v___x_5036_;
goto v_reusejp_5038_;
}
else
{
lean_object* v_reuseFailAlloc_5040_; 
v_reuseFailAlloc_5040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5040_, 0, v_a_5034_);
v___x_5039_ = v_reuseFailAlloc_5040_;
goto v_reusejp_5038_;
}
v_reusejp_5038_:
{
return v___x_5039_;
}
}
}
v___jp_4872_:
{
lean_object* v___x_4881_; 
v___x_4881_ = l_Lean_inferDefEqAttr(v_tgt_4853_, v___y_4877_, v___y_4878_, v___y_4879_, v___y_4880_);
if (lean_obj_tag(v___x_4881_) == 0)
{
lean_object* v___x_4882_; lean_object* v___x_4883_; lean_object* v___x_4884_; lean_object* v___x_4885_; lean_object* v___x_4886_; lean_object* v___x_4887_; lean_object* v___x_4888_; 
lean_dec_ref_known(v___x_4881_, 1);
v___x_4882_ = lean_box(0);
v___x_4883_ = lean_box(0);
v___x_4884_ = lean_box(1);
v___x_4885_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__0));
lean_inc_ref(v___y_4873_);
v___x_4886_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_4886_, 0, v___x_4882_);
lean_ctor_set(v___x_4886_, 1, v___x_4883_);
lean_ctor_set(v___x_4886_, 2, v___x_4882_);
lean_ctor_set(v___x_4886_, 3, v___y_4873_);
lean_ctor_set(v___x_4886_, 4, v___x_4884_);
lean_ctor_set(v___x_4886_, 5, v___x_4884_);
lean_ctor_set(v___x_4886_, 6, v___x_4882_);
lean_ctor_set(v___x_4886_, 7, v___x_4885_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8, v___x_4871_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 1, v___x_4871_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 2, v___x_4871_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 3, v___x_4871_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 4, v___y_4875_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 5, v___y_4875_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 6, v___y_4875_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 7, v___y_4875_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 8, v___x_4871_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 9, v___y_4875_);
lean_ctor_set_uint8(v___x_4886_, sizeof(void*)*8 + 10, v___x_4871_);
v___x_4887_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_4887_, 0, v___y_4874_);
lean_ctor_set(v___x_4887_, 1, v___x_4884_);
lean_ctor_set(v___x_4887_, 2, v___x_4883_);
lean_ctor_set(v___x_4887_, 3, v___x_4883_);
lean_ctor_set(v___x_4887_, 4, v___x_4883_);
lean_ctor_set(v___x_4887_, 5, v___x_4884_);
lean_ctor_set(v___x_4887_, 6, v___x_4883_);
v___x_4888_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___y_4876_, v___x_4886_, v___x_4887_, v___y_4877_, v___y_4878_, v___y_4879_, v___y_4880_);
if (lean_obj_tag(v___x_4888_) == 0)
{
lean_object* v_a_4889_; lean_object* v___x_4891_; uint8_t v_isShared_4892_; uint8_t v_isSharedCheck_4897_; 
v_a_4889_ = lean_ctor_get(v___x_4888_, 0);
v_isSharedCheck_4897_ = !lean_is_exclusive(v___x_4888_);
if (v_isSharedCheck_4897_ == 0)
{
v___x_4891_ = v___x_4888_;
v_isShared_4892_ = v_isSharedCheck_4897_;
goto v_resetjp_4890_;
}
else
{
lean_inc(v_a_4889_);
lean_dec(v___x_4888_);
v___x_4891_ = lean_box(0);
v_isShared_4892_ = v_isSharedCheck_4897_;
goto v_resetjp_4890_;
}
v_resetjp_4890_:
{
lean_object* v_fst_4893_; lean_object* v___x_4895_; 
v_fst_4893_ = lean_ctor_get(v_a_4889_, 0);
lean_inc(v_fst_4893_);
lean_dec(v_a_4889_);
if (v_isShared_4892_ == 0)
{
lean_ctor_set(v___x_4891_, 0, v_fst_4893_);
v___x_4895_ = v___x_4891_;
goto v_reusejp_4894_;
}
else
{
lean_object* v_reuseFailAlloc_4896_; 
v_reuseFailAlloc_4896_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4896_, 0, v_fst_4893_);
v___x_4895_ = v_reuseFailAlloc_4896_;
goto v_reusejp_4894_;
}
v_reusejp_4894_:
{
return v___x_4895_;
}
}
}
else
{
lean_object* v_a_4898_; lean_object* v___x_4900_; uint8_t v_isShared_4901_; uint8_t v_isSharedCheck_4905_; 
v_a_4898_ = lean_ctor_get(v___x_4888_, 0);
v_isSharedCheck_4905_ = !lean_is_exclusive(v___x_4888_);
if (v_isSharedCheck_4905_ == 0)
{
v___x_4900_ = v___x_4888_;
v_isShared_4901_ = v_isSharedCheck_4905_;
goto v_resetjp_4899_;
}
else
{
lean_inc(v_a_4898_);
lean_dec(v___x_4888_);
v___x_4900_ = lean_box(0);
v_isShared_4901_ = v_isSharedCheck_4905_;
goto v_resetjp_4899_;
}
v_resetjp_4899_:
{
lean_object* v___x_4903_; 
if (v_isShared_4901_ == 0)
{
v___x_4903_ = v___x_4900_;
goto v_reusejp_4902_;
}
else
{
lean_object* v_reuseFailAlloc_4904_; 
v_reuseFailAlloc_4904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4904_, 0, v_a_4898_);
v___x_4903_ = v_reuseFailAlloc_4904_;
goto v_reusejp_4902_;
}
v_reusejp_4902_:
{
return v___x_4903_;
}
}
}
}
else
{
lean_dec_ref(v___y_4876_);
lean_dec(v___y_4874_);
return v___x_4881_;
}
}
v___jp_4906_:
{
lean_object* v___x_4916_; 
lean_inc(v_tgt_4853_);
v___x_4916_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8(v_tgt_4853_, v_doc_4911_, v___y_4912_, v___y_4913_, v___y_4914_, v___y_4915_);
if (lean_obj_tag(v___x_4916_) == 0)
{
lean_dec_ref_known(v___x_4916_, 1);
v___y_4873_ = v___y_4907_;
v___y_4874_ = v___y_4909_;
v___y_4875_ = v___y_4908_;
v___y_4876_ = v___y_4910_;
v___y_4877_ = v___y_4912_;
v___y_4878_ = v___y_4913_;
v___y_4879_ = v___y_4914_;
v___y_4880_ = v___y_4915_;
goto v___jp_4872_;
}
else
{
lean_dec_ref(v___y_4910_);
lean_dec(v___y_4909_);
lean_dec(v_tgt_4853_);
return v___x_4916_;
}
}
v___jp_4917_:
{
lean_object* v___x_4926_; lean_object* v_env_4927_; lean_object* v___x_4928_; lean_object* v___x_4929_; lean_object* v___x_4930_; lean_object* v___x_4931_; 
v___x_4926_ = lean_st_ref_get(v___y_4925_);
v_env_4927_ = lean_ctor_get(v___x_4926_, 0);
lean_inc_ref(v_env_4927_);
lean_dec(v___x_4926_);
v___x_4928_ = l_Lean_Options_empty;
v___x_4929_ = lean_box(0);
v___x_4930_ = lean_box(0);
lean_inc(v_src_4852_);
v___x_4931_ = l_Lean_findDocString_x3f(v_env_4927_, v_src_4852_, v___x_4871_, v___x_4928_, v___x_4929_, v___x_4930_);
if (lean_obj_tag(v___x_4931_) == 0)
{
lean_object* v_a_4932_; lean_object* v___x_4933_; lean_object* v___x_4934_; lean_object* v___x_4935_; lean_object* v___f_4936_; 
lean_del_object(v___x_4868_);
v_a_4932_ = lean_ctor_get(v___x_4931_, 0);
lean_inc(v_a_4932_);
lean_dec_ref_known(v___x_4931_, 1);
v___x_4933_ = lean_box(v_hoverInfo_4858_);
v___x_4934_ = lean_box(v___x_4871_);
v___x_4935_ = lean_box(v___y_4918_);
lean_inc(v_tgt_4853_);
v___f_4936_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_addRelatedDecl___lam__0___boxed), 15, 8);
lean_closure_set(v___f_4936_, 0, v_attrs_4855_);
lean_closure_set(v___f_4936_, 1, v_src_4852_);
lean_closure_set(v___f_4936_, 2, v_tgt_4853_);
lean_closure_set(v___f_4936_, 3, v___x_4933_);
lean_closure_set(v___f_4936_, 4, v_ref_4854_);
lean_closure_set(v___f_4936_, 5, v___x_4929_);
lean_closure_set(v___f_4936_, 6, v___x_4934_);
lean_closure_set(v___f_4936_, 7, v___x_4935_);
if (lean_obj_tag(v_docstringPrefix_x3f_4857_) == 0)
{
if (lean_obj_tag(v_a_4932_) == 0)
{
v___y_4873_ = v___y_4919_;
v___y_4874_ = v___y_4921_;
v___y_4875_ = v___y_4920_;
v___y_4876_ = v___f_4936_;
v___y_4877_ = v___y_4922_;
v___y_4878_ = v___y_4923_;
v___y_4879_ = v___y_4924_;
v___y_4880_ = v___y_4925_;
goto v___jp_4872_;
}
else
{
lean_object* v_val_4937_; 
v_val_4937_ = lean_ctor_get(v_a_4932_, 0);
lean_inc(v_val_4937_);
lean_dec_ref_known(v_a_4932_, 1);
v___y_4907_ = v___y_4919_;
v___y_4908_ = v___y_4920_;
v___y_4909_ = v___y_4921_;
v___y_4910_ = v___f_4936_;
v_doc_4911_ = v_val_4937_;
v___y_4912_ = v___y_4922_;
v___y_4913_ = v___y_4923_;
v___y_4914_ = v___y_4924_;
v___y_4915_ = v___y_4925_;
goto v___jp_4906_;
}
}
else
{
if (lean_obj_tag(v_a_4932_) == 0)
{
lean_object* v_val_4938_; 
v_val_4938_ = lean_ctor_get(v_docstringPrefix_x3f_4857_, 0);
lean_inc(v_val_4938_);
lean_dec_ref_known(v_docstringPrefix_x3f_4857_, 1);
v___y_4907_ = v___y_4919_;
v___y_4908_ = v___y_4920_;
v___y_4909_ = v___y_4921_;
v___y_4910_ = v___f_4936_;
v_doc_4911_ = v_val_4938_;
v___y_4912_ = v___y_4922_;
v___y_4913_ = v___y_4923_;
v___y_4914_ = v___y_4924_;
v___y_4915_ = v___y_4925_;
goto v___jp_4906_;
}
else
{
lean_object* v_val_4939_; lean_object* v_val_4940_; lean_object* v___x_4941_; lean_object* v___x_4942_; lean_object* v___x_4943_; lean_object* v___x_4944_; 
v_val_4939_ = lean_ctor_get(v_docstringPrefix_x3f_4857_, 0);
lean_inc(v_val_4939_);
lean_dec_ref_known(v_docstringPrefix_x3f_4857_, 1);
v_val_4940_ = lean_ctor_get(v_a_4932_, 0);
lean_inc(v_val_4940_);
lean_dec_ref_known(v_a_4932_, 1);
v___x_4941_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_addRelatedDecl___closed__1));
v___x_4942_ = lean_string_append(v_val_4939_, v___x_4941_);
v___x_4943_ = lean_string_append(v___x_4942_, v_val_4940_);
lean_dec(v_val_4940_);
lean_inc(v_tgt_4853_);
v___x_4944_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_addRelatedDecl_spec__8(v_tgt_4853_, v___x_4943_, v___y_4922_, v___y_4923_, v___y_4924_, v___y_4925_);
if (lean_obj_tag(v___x_4944_) == 0)
{
lean_dec_ref_known(v___x_4944_, 1);
v___y_4873_ = v___y_4919_;
v___y_4874_ = v___y_4921_;
v___y_4875_ = v___y_4920_;
v___y_4876_ = v___f_4936_;
v___y_4877_ = v___y_4922_;
v___y_4878_ = v___y_4923_;
v___y_4879_ = v___y_4924_;
v___y_4880_ = v___y_4925_;
goto v___jp_4872_;
}
else
{
lean_dec_ref(v___f_4936_);
lean_dec(v___y_4921_);
lean_dec(v_tgt_4853_);
return v___x_4944_;
}
}
}
}
else
{
lean_object* v_a_4945_; lean_object* v___x_4947_; uint8_t v_isShared_4948_; uint8_t v_isSharedCheck_4959_; 
lean_dec(v___y_4921_);
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
v_a_4945_ = lean_ctor_get(v___x_4931_, 0);
v_isSharedCheck_4959_ = !lean_is_exclusive(v___x_4931_);
if (v_isSharedCheck_4959_ == 0)
{
v___x_4947_ = v___x_4931_;
v_isShared_4948_ = v_isSharedCheck_4959_;
goto v_resetjp_4946_;
}
else
{
lean_inc(v_a_4945_);
lean_dec(v___x_4931_);
v___x_4947_ = lean_box(0);
v_isShared_4948_ = v_isSharedCheck_4959_;
goto v_resetjp_4946_;
}
v_resetjp_4946_:
{
lean_object* v_ref_4949_; lean_object* v___x_4950_; lean_object* v___x_4952_; 
v_ref_4949_ = lean_ctor_get(v___y_4924_, 5);
v___x_4950_ = lean_io_error_to_string(v_a_4945_);
if (v_isShared_4869_ == 0)
{
lean_ctor_set_tag(v___x_4868_, 3);
lean_ctor_set(v___x_4868_, 0, v___x_4950_);
v___x_4952_ = v___x_4868_;
goto v_reusejp_4951_;
}
else
{
lean_object* v_reuseFailAlloc_4958_; 
v_reuseFailAlloc_4958_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4958_, 0, v___x_4950_);
v___x_4952_ = v_reuseFailAlloc_4958_;
goto v_reusejp_4951_;
}
v_reusejp_4951_:
{
lean_object* v___x_4953_; lean_object* v___x_4954_; lean_object* v___x_4956_; 
v___x_4953_ = l_Lean_MessageData_ofFormat(v___x_4952_);
lean_inc(v_ref_4949_);
v___x_4954_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4954_, 0, v_ref_4949_);
lean_ctor_set(v___x_4954_, 1, v___x_4953_);
if (v_isShared_4948_ == 0)
{
lean_ctor_set(v___x_4947_, 0, v___x_4954_);
v___x_4956_ = v___x_4947_;
goto v_reusejp_4955_;
}
else
{
lean_object* v_reuseFailAlloc_4957_; 
v_reuseFailAlloc_4957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4957_, 0, v___x_4954_);
v___x_4956_ = v_reuseFailAlloc_4957_;
goto v_reusejp_4955_;
}
v_reusejp_4955_:
{
return v___x_4956_;
}
}
}
}
}
}
}
else
{
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec_ref(v_construct_4856_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
return v___x_4866_;
}
}
else
{
lean_dec(v_docstringPrefix_x3f_4857_);
lean_dec_ref(v_construct_4856_);
lean_dec(v_attrs_4855_);
lean_dec(v_ref_4854_);
lean_dec(v_tgt_4853_);
lean_dec(v_src_4852_);
return v___x_4864_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_addRelatedDecl___boxed(lean_object* v_src_5044_, lean_object* v_tgt_5045_, lean_object* v_ref_5046_, lean_object* v_attrs_5047_, lean_object* v_construct_5048_, lean_object* v_docstringPrefix_x3f_5049_, lean_object* v_hoverInfo_5050_, lean_object* v_a_5051_, lean_object* v_a_5052_, lean_object* v_a_5053_, lean_object* v_a_5054_, lean_object* v_a_5055_){
_start:
{
uint8_t v_hoverInfo_boxed_5056_; lean_object* v_res_5057_; 
v_hoverInfo_boxed_5056_ = lean_unbox(v_hoverInfo_5050_);
v_res_5057_ = lp_mathlib_Mathlib_Tactic_addRelatedDecl(v_src_5044_, v_tgt_5045_, v_ref_5046_, v_attrs_5047_, v_construct_5048_, v_docstringPrefix_x3f_5049_, v_hoverInfo_boxed_5056_, v_a_5051_, v_a_5052_, v_a_5053_, v_a_5054_);
lean_dec(v_a_5054_);
lean_dec_ref(v_a_5053_);
lean_dec(v_a_5052_);
lean_dec_ref(v_a_5051_);
return v_res_5057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4(lean_object* v_stx_5058_, lean_object* v___y_5059_, lean_object* v___y_5060_, lean_object* v___y_5061_, lean_object* v___y_5062_){
_start:
{
lean_object* v___x_5064_; 
v___x_5064_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___redArg(v_stx_5058_, v___y_5061_);
return v___x_5064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4___boxed(lean_object* v_stx_5065_, lean_object* v___y_5066_, lean_object* v___y_5067_, lean_object* v___y_5068_, lean_object* v___y_5069_, lean_object* v___y_5070_){
_start:
{
lean_object* v_res_5071_; 
v_res_5071_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__4(v_stx_5065_, v___y_5066_, v___y_5067_, v___y_5068_, v___y_5069_);
lean_dec(v___y_5069_);
lean_dec_ref(v___y_5068_);
lean_dec(v___y_5067_);
lean_dec_ref(v___y_5066_);
lean_dec(v_stx_5065_);
return v_res_5071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5(lean_object* v_declName_5072_, lean_object* v_declRanges_5073_, lean_object* v___y_5074_, lean_object* v___y_5075_, lean_object* v___y_5076_, lean_object* v___y_5077_){
_start:
{
lean_object* v___x_5079_; 
v___x_5079_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___redArg(v_declName_5072_, v_declRanges_5073_, v___y_5075_, v___y_5077_);
return v___x_5079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5___boxed(lean_object* v_declName_5080_, lean_object* v_declRanges_5081_, lean_object* v___y_5082_, lean_object* v___y_5083_, lean_object* v___y_5084_, lean_object* v___y_5085_, lean_object* v___y_5086_){
_start:
{
lean_object* v_res_5087_; 
v_res_5087_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_addRelatedDecl_spec__1_spec__5(v_declName_5080_, v_declRanges_5081_, v___y_5082_, v___y_5083_, v___y_5084_, v___y_5085_);
lean_dec(v___y_5085_);
lean_dec_ref(v___y_5084_);
lean_dec(v___y_5083_);
lean_dec_ref(v___y_5082_);
return v_res_5087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9(lean_object* v_00_u03b1_5088_, lean_object* v_x_5089_, uint8_t v_isExporting_5090_, lean_object* v___y_5091_, lean_object* v___y_5092_, lean_object* v___y_5093_, lean_object* v___y_5094_){
_start:
{
lean_object* v___x_5096_; 
v___x_5096_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___redArg(v_x_5089_, v_isExporting_5090_, v___y_5091_, v___y_5092_, v___y_5093_, v___y_5094_);
return v___x_5096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9___boxed(lean_object* v_00_u03b1_5097_, lean_object* v_x_5098_, lean_object* v_isExporting_5099_, lean_object* v___y_5100_, lean_object* v___y_5101_, lean_object* v___y_5102_, lean_object* v___y_5103_, lean_object* v___y_5104_){
_start:
{
uint8_t v_isExporting_boxed_5105_; lean_object* v_res_5106_; 
v_isExporting_boxed_5105_ = lean_unbox(v_isExporting_5099_);
v_res_5106_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3_spec__9(v_00_u03b1_5097_, v_x_5098_, v_isExporting_boxed_5105_, v___y_5100_, v___y_5101_, v___y_5102_, v___y_5103_);
lean_dec(v___y_5103_);
lean_dec_ref(v___y_5102_);
lean_dec(v___y_5101_);
lean_dec_ref(v___y_5100_);
return v_res_5106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3(lean_object* v_00_u03b1_5107_, lean_object* v_x_5108_, uint8_t v_when_5109_, lean_object* v___y_5110_, lean_object* v___y_5111_, lean_object* v___y_5112_, lean_object* v___y_5113_){
_start:
{
lean_object* v___x_5115_; 
v___x_5115_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___redArg(v_x_5108_, v_when_5109_, v___y_5110_, v___y_5111_, v___y_5112_, v___y_5113_);
return v___x_5115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3___boxed(lean_object* v_00_u03b1_5116_, lean_object* v_x_5117_, lean_object* v_when_5118_, lean_object* v___y_5119_, lean_object* v___y_5120_, lean_object* v___y_5121_, lean_object* v___y_5122_, lean_object* v___y_5123_){
_start:
{
uint8_t v_when_boxed_5124_; lean_object* v_res_5125_; 
v_when_boxed_5124_ = lean_unbox(v_when_5118_);
v_res_5125_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Tactic_addRelatedDecl_spec__3(v_00_u03b1_5116_, v_x_5117_, v_when_boxed_5124_, v___y_5119_, v___y_5120_, v___y_5121_, v___y_5122_);
lean_dec(v___y_5122_);
lean_dec_ref(v___y_5121_);
lean_dec(v___y_5120_);
lean_dec_ref(v___y_5119_);
return v_res_5125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10(lean_object* v_00_u03b1_5126_, lean_object* v_msg_5127_, lean_object* v___y_5128_, lean_object* v___y_5129_, lean_object* v___y_5130_, lean_object* v___y_5131_){
_start:
{
lean_object* v___x_5133_; 
v___x_5133_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___redArg(v_msg_5127_, v___y_5128_, v___y_5129_, v___y_5130_, v___y_5131_);
return v___x_5133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10___boxed(lean_object* v_00_u03b1_5134_, lean_object* v_msg_5135_, lean_object* v___y_5136_, lean_object* v___y_5137_, lean_object* v___y_5138_, lean_object* v___y_5139_, lean_object* v___y_5140_){
_start:
{
lean_object* v_res_5141_; 
v_res_5141_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_addRelatedDecl_spec__10(v_00_u03b1_5134_, v_msg_5135_, v___y_5136_, v___y_5137_, v___y_5138_, v___y_5139_);
lean_dec(v___y_5139_);
lean_dec_ref(v___y_5138_);
lean_dec(v___y_5137_);
lean_dec_ref(v___y_5136_);
return v_res_5141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6(lean_object* v_t_5142_, lean_object* v___y_5143_, lean_object* v___y_5144_, lean_object* v___y_5145_, lean_object* v___y_5146_){
_start:
{
lean_object* v___x_5148_; 
v___x_5148_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___redArg(v_t_5142_, v___y_5146_);
return v___x_5148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6___boxed(lean_object* v_t_5149_, lean_object* v___y_5150_, lean_object* v___y_5151_, lean_object* v___y_5152_, lean_object* v___y_5153_, lean_object* v___y_5154_){
_start:
{
lean_object* v_res_5155_; 
v_res_5155_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__1_spec__6(v_t_5149_, v___y_5150_, v___y_5151_, v___y_5152_, v___y_5153_);
lean_dec(v___y_5153_);
lean_dec_ref(v___y_5152_);
lean_dec(v___y_5151_);
lean_dec_ref(v___y_5150_);
return v_res_5155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2(lean_object* v_00_u03b1_5156_, lean_object* v_env_5157_, lean_object* v_x_5158_, lean_object* v___y_5159_, lean_object* v___y_5160_, lean_object* v___y_5161_, lean_object* v___y_5162_){
_start:
{
lean_object* v___x_5164_; 
v___x_5164_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___redArg(v_env_5157_, v_x_5158_, v___y_5159_, v___y_5160_, v___y_5161_, v___y_5162_);
return v___x_5164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2___boxed(lean_object* v_00_u03b1_5165_, lean_object* v_env_5166_, lean_object* v_x_5167_, lean_object* v___y_5168_, lean_object* v___y_5169_, lean_object* v___y_5170_, lean_object* v___y_5171_, lean_object* v___y_5172_){
_start:
{
lean_object* v_res_5173_; 
v_res_5173_ = lp_mathlib_Lean_withEnv___at___00Lean_Elab_checkNotAlreadyDeclared___at___00Mathlib_Tactic_addRelatedDecl_spec__0_spec__2(v_00_u03b1_5165_, v_env_5166_, v_x_5167_, v___y_5168_, v___y_5169_, v___y_5170_, v___y_5171_);
lean_dec(v___y_5171_);
lean_dec_ref(v___y_5170_);
lean_dec(v___y_5169_);
lean_dec_ref(v___y_5168_);
return v_res_5173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7(lean_object* v_00_u03b1_5174_, lean_object* v_constName_5175_, lean_object* v___y_5176_, lean_object* v___y_5177_, lean_object* v___y_5178_, lean_object* v___y_5179_){
_start:
{
lean_object* v___x_5181_; 
v___x_5181_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___redArg(v_constName_5175_, v___y_5176_, v___y_5177_, v___y_5178_, v___y_5179_);
return v___x_5181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7___boxed(lean_object* v_00_u03b1_5182_, lean_object* v_constName_5183_, lean_object* v___y_5184_, lean_object* v___y_5185_, lean_object* v___y_5186_, lean_object* v___y_5187_, lean_object* v___y_5188_){
_start:
{
lean_object* v_res_5189_; 
v_res_5189_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7(v_00_u03b1_5182_, v_constName_5183_, v___y_5184_, v___y_5185_, v___y_5186_, v___y_5187_);
lean_dec(v___y_5187_);
lean_dec_ref(v___y_5186_);
lean_dec(v___y_5185_);
lean_dec_ref(v___y_5184_);
return v_res_5189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13(lean_object* v_00_u03b1_5190_, lean_object* v_ref_5191_, lean_object* v_constName_5192_, lean_object* v___y_5193_, lean_object* v___y_5194_, lean_object* v___y_5195_, lean_object* v___y_5196_){
_start:
{
lean_object* v___x_5198_; 
v___x_5198_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___redArg(v_ref_5191_, v_constName_5192_, v___y_5193_, v___y_5194_, v___y_5195_, v___y_5196_);
return v___x_5198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13___boxed(lean_object* v_00_u03b1_5199_, lean_object* v_ref_5200_, lean_object* v_constName_5201_, lean_object* v___y_5202_, lean_object* v___y_5203_, lean_object* v___y_5204_, lean_object* v___y_5205_, lean_object* v___y_5206_){
_start:
{
lean_object* v_res_5207_; 
v_res_5207_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13(v_00_u03b1_5199_, v_ref_5200_, v_constName_5201_, v___y_5202_, v___y_5203_, v___y_5204_, v___y_5205_);
lean_dec(v___y_5205_);
lean_dec_ref(v___y_5204_);
lean_dec(v___y_5203_);
lean_dec_ref(v___y_5202_);
lean_dec(v_ref_5200_);
return v_res_5207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19(lean_object* v_00_u03b1_5208_, lean_object* v_constName_5209_, lean_object* v___y_5210_, lean_object* v___y_5211_, lean_object* v___y_5212_, lean_object* v___y_5213_, lean_object* v___y_5214_, lean_object* v___y_5215_){
_start:
{
lean_object* v___x_5217_; 
v___x_5217_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___redArg(v_constName_5209_, v___y_5210_, v___y_5211_, v___y_5212_, v___y_5213_, v___y_5214_, v___y_5215_);
return v___x_5217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19___boxed(lean_object* v_00_u03b1_5218_, lean_object* v_constName_5219_, lean_object* v___y_5220_, lean_object* v___y_5221_, lean_object* v___y_5222_, lean_object* v___y_5223_, lean_object* v___y_5224_, lean_object* v___y_5225_, lean_object* v___y_5226_){
_start:
{
lean_object* v_res_5227_; 
v_res_5227_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19(v_00_u03b1_5218_, v_constName_5219_, v___y_5220_, v___y_5221_, v___y_5222_, v___y_5223_, v___y_5224_, v___y_5225_);
lean_dec(v___y_5225_);
lean_dec_ref(v___y_5224_);
lean_dec(v___y_5223_);
lean_dec_ref(v___y_5222_);
lean_dec(v___y_5221_);
lean_dec_ref(v___y_5220_);
return v_res_5227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20(lean_object* v_00_u03b1_5228_, lean_object* v_ref_5229_, lean_object* v_msg_5230_, lean_object* v_declHint_5231_, lean_object* v___y_5232_, lean_object* v___y_5233_, lean_object* v___y_5234_, lean_object* v___y_5235_){
_start:
{
lean_object* v___x_5237_; 
v___x_5237_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___redArg(v_ref_5229_, v_msg_5230_, v_declHint_5231_, v___y_5232_, v___y_5233_, v___y_5234_, v___y_5235_);
return v___x_5237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20___boxed(lean_object* v_00_u03b1_5238_, lean_object* v_ref_5239_, lean_object* v_msg_5240_, lean_object* v_declHint_5241_, lean_object* v___y_5242_, lean_object* v___y_5243_, lean_object* v___y_5244_, lean_object* v___y_5245_, lean_object* v___y_5246_){
_start:
{
lean_object* v_res_5247_; 
v_res_5247_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20(v_00_u03b1_5238_, v_ref_5239_, v_msg_5240_, v_declHint_5241_, v___y_5242_, v___y_5243_, v___y_5244_, v___y_5245_);
lean_dec(v___y_5245_);
lean_dec_ref(v___y_5244_);
lean_dec(v___y_5243_);
lean_dec_ref(v___y_5242_);
lean_dec(v_ref_5239_);
return v_res_5247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23(lean_object* v_00_u03b1_5248_, lean_object* v_ref_5249_, lean_object* v_constName_5250_, lean_object* v___y_5251_, lean_object* v___y_5252_, lean_object* v___y_5253_, lean_object* v___y_5254_, lean_object* v___y_5255_, lean_object* v___y_5256_){
_start:
{
lean_object* v___x_5258_; 
v___x_5258_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___redArg(v_ref_5249_, v_constName_5250_, v___y_5251_, v___y_5252_, v___y_5253_, v___y_5254_, v___y_5255_, v___y_5256_);
return v___x_5258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23___boxed(lean_object* v_00_u03b1_5259_, lean_object* v_ref_5260_, lean_object* v_constName_5261_, lean_object* v___y_5262_, lean_object* v___y_5263_, lean_object* v___y_5264_, lean_object* v___y_5265_, lean_object* v___y_5266_, lean_object* v___y_5267_, lean_object* v___y_5268_){
_start:
{
lean_object* v_res_5269_; 
v_res_5269_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23(v_00_u03b1_5259_, v_ref_5260_, v_constName_5261_, v___y_5262_, v___y_5263_, v___y_5264_, v___y_5265_, v___y_5266_, v___y_5267_);
lean_dec(v___y_5267_);
lean_dec_ref(v___y_5266_);
lean_dec(v___y_5265_);
lean_dec_ref(v___y_5264_);
lean_dec(v___y_5263_);
lean_dec_ref(v___y_5262_);
lean_dec(v_ref_5260_);
return v_res_5269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24(lean_object* v_msg_5270_, lean_object* v_declHint_5271_, lean_object* v___y_5272_, lean_object* v___y_5273_, lean_object* v___y_5274_, lean_object* v___y_5275_){
_start:
{
lean_object* v___x_5277_; 
v___x_5277_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___redArg(v_msg_5270_, v_declHint_5271_, v___y_5275_);
return v___x_5277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24___boxed(lean_object* v_msg_5278_, lean_object* v_declHint_5279_, lean_object* v___y_5280_, lean_object* v___y_5281_, lean_object* v___y_5282_, lean_object* v___y_5283_, lean_object* v___y_5284_){
_start:
{
lean_object* v_res_5285_; 
v_res_5285_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__22_spec__24(v_msg_5278_, v_declHint_5279_, v___y_5280_, v___y_5281_, v___y_5282_, v___y_5283_);
lean_dec(v___y_5283_);
lean_dec_ref(v___y_5282_);
lean_dec(v___y_5281_);
lean_dec_ref(v___y_5280_);
return v_res_5285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23(lean_object* v_00_u03b1_5286_, lean_object* v_ref_5287_, lean_object* v_msg_5288_, lean_object* v___y_5289_, lean_object* v___y_5290_, lean_object* v___y_5291_, lean_object* v___y_5292_){
_start:
{
lean_object* v___x_5294_; 
v___x_5294_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___redArg(v_ref_5287_, v_msg_5288_, v___y_5289_, v___y_5290_, v___y_5291_, v___y_5292_);
return v___x_5294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23___boxed(lean_object* v_00_u03b1_5295_, lean_object* v_ref_5296_, lean_object* v_msg_5297_, lean_object* v___y_5298_, lean_object* v___y_5299_, lean_object* v___y_5300_, lean_object* v___y_5301_, lean_object* v___y_5302_){
_start:
{
lean_object* v_res_5303_; 
v_res_5303_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_addRelatedDecl_spec__2_spec__7_spec__13_spec__20_spec__23(v_00_u03b1_5295_, v_ref_5296_, v_msg_5297_, v___y_5298_, v___y_5299_, v___y_5300_, v___y_5301_);
lean_dec(v___y_5301_);
lean_dec_ref(v___y_5300_);
lean_dec(v___y_5299_);
lean_dec_ref(v___y_5298_);
lean_dec(v_ref_5296_);
return v_res_5303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26(lean_object* v_00_u03b1_5304_, lean_object* v_ref_5305_, lean_object* v_msg_5306_, lean_object* v_declHint_5307_, lean_object* v___y_5308_, lean_object* v___y_5309_, lean_object* v___y_5310_, lean_object* v___y_5311_, lean_object* v___y_5312_, lean_object* v___y_5313_){
_start:
{
lean_object* v___x_5315_; 
v___x_5315_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___redArg(v_ref_5305_, v_msg_5306_, v_declHint_5307_, v___y_5308_, v___y_5309_, v___y_5310_, v___y_5311_, v___y_5312_, v___y_5313_);
return v___x_5315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26___boxed(lean_object* v_00_u03b1_5316_, lean_object* v_ref_5317_, lean_object* v_msg_5318_, lean_object* v_declHint_5319_, lean_object* v___y_5320_, lean_object* v___y_5321_, lean_object* v___y_5322_, lean_object* v___y_5323_, lean_object* v___y_5324_, lean_object* v___y_5325_, lean_object* v___y_5326_){
_start:
{
lean_object* v_res_5327_; 
v_res_5327_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26(v_00_u03b1_5316_, v_ref_5317_, v_msg_5318_, v_declHint_5319_, v___y_5320_, v___y_5321_, v___y_5322_, v___y_5323_, v___y_5324_, v___y_5325_);
lean_dec(v___y_5325_);
lean_dec_ref(v___y_5324_);
lean_dec(v___y_5323_);
lean_dec_ref(v___y_5322_);
lean_dec(v___y_5321_);
lean_dec_ref(v___y_5320_);
lean_dec(v_ref_5317_);
return v_res_5327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29(lean_object* v_msg_5328_, lean_object* v_declHint_5329_, lean_object* v___y_5330_, lean_object* v___y_5331_, lean_object* v___y_5332_, lean_object* v___y_5333_, lean_object* v___y_5334_, lean_object* v___y_5335_){
_start:
{
lean_object* v___x_5337_; 
v___x_5337_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___redArg(v_msg_5328_, v_declHint_5329_, v___y_5335_);
return v___x_5337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29___boxed(lean_object* v_msg_5338_, lean_object* v_declHint_5339_, lean_object* v___y_5340_, lean_object* v___y_5341_, lean_object* v___y_5342_, lean_object* v___y_5343_, lean_object* v___y_5344_, lean_object* v___y_5345_, lean_object* v___y_5346_){
_start:
{
lean_object* v_res_5347_; 
v_res_5347_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_addRelatedDecl_spec__7_spec__14_spec__19_spec__23_spec__26_spec__28_spec__29(v_msg_5338_, v_declHint_5339_, v___y_5340_, v___y_5341_, v___y_5342_, v___y_5343_, v___y_5344_, v___y_5345_);
lean_dec(v___y_5345_);
lean_dec_ref(v___y_5344_);
lean_dec(v___y_5343_);
lean_dec_ref(v___y_5342_);
lean_dec(v___y_5341_);
lean_dec_ref(v___y_5340_);
return v_res_5347_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_AddRelatedDecl(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_AddRelatedDecl(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_AddRelatedDecl(uint8_t builtin) {
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
res = initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AddRelatedDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_AddRelatedDecl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_AddRelatedDecl(builtin);
}
#ifdef __cplusplus
}
#endif
