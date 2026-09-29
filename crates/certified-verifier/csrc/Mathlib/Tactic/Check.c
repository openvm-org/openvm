// Lean compiler output
// Module: Mathlib.Tactic.Check
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Basic public meta import Lean.PrettyPrinter public meta import Lean.Elab.SyntheticMVars public meta import Lean.PrettyPrinter.Delaborator.Builtins
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
lean_object* l_Lean_stringToMessageData(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
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
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delabForallParamsWithSignature___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppExprWithInfos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormatWithInfosM(lean_object*);
lean_object* l_Lean_MessageData_signature(lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_check(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_levelMVarToParam___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isSyntheticSorry(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstWithInfos(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withDeclName___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_withMainContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_runTermElabM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_unlockAsync(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "explicitBinder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 119, 193, 23, 170, 93, 183, 238)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "forall"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(195, 142, 115, 15, 55, 103, 31, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(22, 101, 130, 251, 183, 19, 113, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delabForallParamsWithSignature___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__3 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__4 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_check"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(124, 206, 113, 80, 4, 121, 2, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__3_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "command#check'_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 179, 166, 61, 112, 81, 237, 121)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "#check' "};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_command_x23check_x27__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tactic#check__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 68, 177, 137, 53, 47, 72, 23)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#check "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tactic#check'__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(7, 107, 121, 227, 240, 16, 114, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tactic_x23check_x27____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check_x27______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check_x27______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(size_t v_sz_1_, size_t v_i_2_, lean_object* v_bs_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = lean_usize_dec_lt(v_i_2_, v_sz_1_);
if (v___x_4_ == 0)
{
return v_bs_3_;
}
else
{
lean_object* v_v_5_; lean_object* v___x_6_; lean_object* v_bs_x27_7_; size_t v___x_8_; size_t v___x_9_; lean_object* v___x_10_; 
v_v_5_ = lean_array_uget(v_bs_3_, v_i_2_);
v___x_6_ = lean_unsigned_to_nat(0u);
v_bs_x27_7_ = lean_array_uset(v_bs_3_, v_i_2_, v___x_6_);
v___x_8_ = ((size_t)1ULL);
v___x_9_ = lean_usize_add(v_i_2_, v___x_8_);
v___x_10_ = lean_array_uset(v_bs_x27_7_, v_i_2_, v_v_5_);
v_i_2_ = v___x_9_;
v_bs_3_ = v___x_10_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1___boxed(lean_object* v_sz_12_, lean_object* v_i_13_, lean_object* v_bs_14_){
_start:
{
size_t v_sz_boxed_15_; size_t v_i_boxed_16_; lean_object* v_res_17_; 
v_sz_boxed_15_ = lean_unbox_usize(v_sz_12_);
lean_dec(v_sz_12_);
v_i_boxed_16_ = lean_unbox_usize(v_i_13_);
lean_dec(v_i_13_);
v_res_17_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(v_sz_boxed_15_, v_i_boxed_16_, v_bs_14_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(size_t v_sz_18_, size_t v_i_19_, lean_object* v_bs_20_){
_start:
{
uint8_t v___x_21_; 
v___x_21_ = lean_usize_dec_lt(v_i_19_, v_sz_18_);
if (v___x_21_ == 0)
{
return v_bs_20_;
}
else
{
lean_object* v_v_22_; lean_object* v___x_23_; lean_object* v_bs_x27_24_; size_t v___x_25_; size_t v___x_26_; lean_object* v___x_27_; 
v_v_22_ = lean_array_uget(v_bs_20_, v_i_19_);
v___x_23_ = lean_unsigned_to_nat(0u);
v_bs_x27_24_ = lean_array_uset(v_bs_20_, v_i_19_, v___x_23_);
v___x_25_ = ((size_t)1ULL);
v___x_26_ = lean_usize_add(v_i_19_, v___x_25_);
v___x_27_ = lean_array_uset(v_bs_x27_24_, v_i_19_, v_v_22_);
v_i_19_ = v___x_26_;
v_bs_20_ = v___x_27_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0___boxed(lean_object* v_sz_29_, lean_object* v_i_30_, lean_object* v_bs_31_){
_start:
{
size_t v_sz_boxed_32_; size_t v_i_boxed_33_; lean_object* v_res_34_; 
v_sz_boxed_32_ = lean_unbox_usize(v_sz_29_);
lean_dec(v_sz_29_);
v_i_boxed_33_ = lean_unbox_usize(v_i_30_);
lean_dec(v_i_30_);
v_res_34_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(v_sz_boxed_32_, v_i_boxed_33_, v_bs_31_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2(lean_object* v_as_44_, size_t v_i_45_, size_t v_stop_46_, lean_object* v_b_47_){
_start:
{
lean_object* v___y_49_; uint8_t v___x_53_; 
v___x_53_ = lean_usize_dec_eq(v_i_45_, v_stop_46_);
if (v___x_53_ == 0)
{
lean_object* v___x_54_; lean_object* v___x_55_; uint8_t v___x_56_; 
v___x_54_ = lean_array_uget_borrowed(v_as_44_, v_i_45_);
v___x_55_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4));
lean_inc(v___x_54_);
v___x_56_ = l_Lean_Syntax_isOfKind(v___x_54_, v___x_55_);
if (v___x_56_ == 0)
{
v___y_49_ = v_b_47_;
goto v___jp_48_;
}
else
{
lean_object* v___x_57_; 
lean_inc(v___x_54_);
v___x_57_ = lean_array_push(v_b_47_, v___x_54_);
v___y_49_ = v___x_57_;
goto v___jp_48_;
}
}
else
{
return v_b_47_;
}
v___jp_48_:
{
size_t v___x_50_; size_t v___x_51_; 
v___x_50_ = ((size_t)1ULL);
v___x_51_ = lean_usize_add(v_i_45_, v___x_50_);
v_i_45_ = v___x_51_;
v_b_47_ = v___y_49_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___boxed(lean_object* v_as_58_, lean_object* v_i_59_, lean_object* v_stop_60_, lean_object* v_b_61_){
_start:
{
size_t v_i_boxed_62_; size_t v_stop_boxed_63_; lean_object* v_res_64_; 
v_i_boxed_62_ = lean_unbox_usize(v_i_59_);
lean_dec(v_i_59_);
v_stop_boxed_63_ = lean_unbox_usize(v_stop_60_);
lean_dec(v_stop_60_);
v_res_64_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2(v_as_58_, v_i_boxed_62_, v_stop_boxed_63_, v_b_61_);
lean_dec_ref(v_as_58_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3(lean_object* v_as_65_, size_t v_i_66_, size_t v_stop_67_, lean_object* v_b_68_){
_start:
{
lean_object* v___y_70_; uint8_t v___x_74_; 
v___x_74_ = lean_usize_dec_eq(v_i_66_, v_stop_67_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_75_ = lean_array_uget_borrowed(v_as_65_, v_i_66_);
v___x_76_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__4));
lean_inc(v___x_75_);
v___x_77_ = l_Lean_Syntax_isOfKind(v___x_75_, v___x_76_);
if (v___x_77_ == 0)
{
v___y_70_ = v_b_68_;
goto v___jp_69_;
}
else
{
lean_object* v___x_78_; 
lean_inc(v___x_75_);
v___x_78_ = lean_array_push(v_b_68_, v___x_75_);
v___y_70_ = v___x_78_;
goto v___jp_69_;
}
}
else
{
return v_b_68_;
}
v___jp_69_:
{
size_t v___x_71_; size_t v___x_72_; 
v___x_71_ = ((size_t)1ULL);
v___x_72_ = lean_usize_add(v_i_66_, v___x_71_);
v_i_66_ = v___x_72_;
v_b_68_ = v___y_70_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3___boxed(lean_object* v_as_79_, lean_object* v_i_80_, lean_object* v_stop_81_, lean_object* v_b_82_){
_start:
{
size_t v_i_boxed_83_; size_t v_stop_boxed_84_; lean_object* v_res_85_; 
v_i_boxed_83_ = lean_unbox_usize(v_i_80_);
lean_dec(v_i_80_);
v_stop_boxed_84_ = lean_unbox_usize(v_stop_81_);
lean_dec(v_stop_81_);
v_res_85_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3(v_as_79_, v_i_boxed_83_, v_stop_boxed_84_, v_b_82_);
lean_dec_ref(v_as_79_);
return v_res_85_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = l_Array_mkArray0(lean_box(0));
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0(lean_object* v_binders_112_, lean_object* v_type_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
lean_object* v___y_122_; lean_object* v___y_123_; lean_object* v___y_124_; lean_object* v___y_125_; lean_object* v___y_126_; lean_object* v___y_127_; lean_object* v___x_151_; lean_object* v___y_153_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; 
v___x_151_ = lean_unsigned_to_nat(0u);
v___x_213_ = lean_array_get_size(v_binders_112_);
v___x_214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__11));
v___x_215_ = lean_nat_dec_lt(v___x_151_, v___x_213_);
if (v___x_215_ == 0)
{
v___y_153_ = v___x_214_;
goto v___jp_152_;
}
else
{
uint8_t v___x_216_; 
v___x_216_ = lean_nat_dec_le(v___x_213_, v___x_213_);
if (v___x_216_ == 0)
{
if (v___x_215_ == 0)
{
v___y_153_ = v___x_214_;
goto v___jp_152_;
}
else
{
size_t v___x_217_; size_t v___x_218_; lean_object* v___x_219_; 
v___x_217_ = ((size_t)0ULL);
v___x_218_ = lean_usize_of_nat(v___x_213_);
v___x_219_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3(v_binders_112_, v___x_217_, v___x_218_, v___x_214_);
v___y_153_ = v___x_219_;
goto v___jp_152_;
}
}
else
{
size_t v___x_220_; size_t v___x_221_; lean_object* v___x_222_; 
v___x_220_ = ((size_t)0ULL);
v___x_221_ = lean_usize_of_nat(v___x_213_);
v___x_222_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__3(v_binders_112_, v___x_220_, v___x_221_, v___x_214_);
v___y_153_ = v___x_222_;
goto v___jp_152_;
}
}
v___jp_121_:
{
lean_object* v_ref_128_; uint8_t v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; size_t v_sz_136_; size_t v___x_137_; lean_object* v___x_138_; size_t v_sz_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v_ref_128_ = lean_ctor_get(v___y_118_, 5);
v___x_129_ = 0;
v___x_130_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__0));
v___x_132_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__1));
lean_inc_ref_n(v___y_126_, 2);
lean_inc_ref_n(v___y_122_, 2);
v___x_133_ = l_Lean_Name_mkStr4(v___y_122_, v___y_126_, v___x_131_, v___x_132_);
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3));
v___x_135_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4);
v_sz_136_ = lean_array_size(v___y_124_);
v___x_137_ = ((size_t)0ULL);
v___x_138_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(v_sz_136_, v___x_137_, v___y_124_);
v_sz_139_ = lean_array_size(v___x_138_);
v___x_140_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(v_sz_139_, v___x_137_, v___x_138_);
v___x_141_ = l_Array_append___redArg(v___x_135_, v___x_140_);
lean_dec_ref(v___x_140_);
v___x_142_ = l_Array_append___redArg(v___x_141_, v___y_127_);
lean_dec_ref(v___y_127_);
lean_inc_n(v___x_130_, 3);
v___x_143_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_143_, 0, v___x_130_);
lean_ctor_set(v___x_143_, 1, v___x_134_);
lean_ctor_set(v___x_143_, 2, v___x_142_);
v___x_144_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__5));
lean_inc_ref(v___y_125_);
v___x_145_ = l_Lean_Name_mkStr4(v___y_122_, v___y_126_, v___y_125_, v___x_144_);
v___x_146_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6));
v___x_147_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_130_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = l_Lean_Syntax_node2(v___x_130_, v___x_145_, v___x_147_, v___y_123_);
v___x_149_ = l_Lean_Syntax_node2(v___x_130_, v___x_133_, v___x_143_, v___x_148_);
v___x_150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
return v___x_150_;
}
v___jp_152_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_154_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__0));
v___x_155_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__1));
v___x_156_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2___closed__2));
v___x_157_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__8));
lean_inc(v_type_113_);
v___x_158_ = l_Lean_Syntax_isOfKind(v_type_113_, v___x_157_);
if (v___x_158_ == 0)
{
lean_object* v_ref_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; size_t v_sz_164_; size_t v___x_165_; lean_object* v___x_166_; size_t v_sz_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_ref_159_ = lean_ctor_get(v___y_118_, 5);
v___x_160_ = l_Lean_SourceInfo_fromRef(v_ref_159_, v___x_158_);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9));
v___x_162_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3));
v___x_163_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4);
v_sz_164_ = lean_array_size(v___y_153_);
v___x_165_ = ((size_t)0ULL);
v___x_166_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(v_sz_164_, v___x_165_, v___y_153_);
v_sz_167_ = lean_array_size(v___x_166_);
v___x_168_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(v_sz_167_, v___x_165_, v___x_166_);
v___x_169_ = l_Array_append___redArg(v___x_163_, v___x_168_);
lean_dec_ref(v___x_168_);
lean_inc_n(v___x_160_, 3);
v___x_170_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_170_, 0, v___x_160_);
lean_ctor_set(v___x_170_, 1, v___x_162_);
lean_ctor_set(v___x_170_, 2, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10));
v___x_172_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6));
v___x_173_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_160_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
v___x_174_ = l_Lean_Syntax_node2(v___x_160_, v___x_171_, v___x_173_, v_type_113_);
v___x_175_ = l_Lean_Syntax_node2(v___x_160_, v___x_161_, v___x_170_, v___x_174_);
v___x_176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_177_ = lean_unsigned_to_nat(2u);
v___x_178_ = l_Lean_Syntax_getArg(v_type_113_, v___x_177_);
v___x_179_ = l_Lean_Syntax_matchesNull(v___x_178_, v___x_151_);
if (v___x_179_ == 0)
{
lean_object* v_ref_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; size_t v_sz_185_; size_t v___x_186_; lean_object* v___x_187_; size_t v_sz_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v_ref_180_ = lean_ctor_get(v___y_118_, 5);
v___x_181_ = l_Lean_SourceInfo_fromRef(v_ref_180_, v___x_179_);
v___x_182_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__9));
v___x_183_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__3));
v___x_184_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__4);
v_sz_185_ = lean_array_size(v___y_153_);
v___x_186_ = ((size_t)0ULL);
v___x_187_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__0(v_sz_185_, v___x_186_, v___y_153_);
v_sz_188_ = lean_array_size(v___x_187_);
v___x_189_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__1(v_sz_188_, v___x_186_, v___x_187_);
v___x_190_ = l_Array_append___redArg(v___x_184_, v___x_189_);
lean_dec_ref(v___x_189_);
lean_inc_n(v___x_181_, 3);
v___x_191_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_191_, 0, v___x_181_);
lean_ctor_set(v___x_191_, 1, v___x_183_);
lean_ctor_set(v___x_191_, 2, v___x_190_);
v___x_192_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__10));
v___x_193_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__6));
v___x_194_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_181_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = l_Lean_Syntax_node2(v___x_181_, v___x_192_, v___x_194_, v_type_113_);
v___x_196_ = l_Lean_Syntax_node2(v___x_181_, v___x_182_, v___x_191_, v___x_195_);
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v_binders_x27_202_; lean_object* v___x_203_; lean_object* v___x_204_; uint8_t v___x_205_; 
v___x_198_ = lean_unsigned_to_nat(1u);
v___x_199_ = l_Lean_Syntax_getArg(v_type_113_, v___x_198_);
v___x_200_ = lean_unsigned_to_nat(4u);
v___x_201_ = l_Lean_Syntax_getArg(v_type_113_, v___x_200_);
lean_dec(v_type_113_);
v_binders_x27_202_ = l_Lean_Syntax_getArgs(v___x_199_);
lean_dec(v___x_199_);
v___x_203_ = lean_array_get_size(v_binders_x27_202_);
v___x_204_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___closed__11));
v___x_205_ = lean_nat_dec_lt(v___x_151_, v___x_203_);
if (v___x_205_ == 0)
{
lean_dec_ref(v_binders_x27_202_);
v___y_122_ = v___x_154_;
v___y_123_ = v___x_201_;
v___y_124_ = v___y_153_;
v___y_125_ = v___x_156_;
v___y_126_ = v___x_155_;
v___y_127_ = v___x_204_;
goto v___jp_121_;
}
else
{
uint8_t v___x_206_; 
v___x_206_ = lean_nat_dec_le(v___x_203_, v___x_203_);
if (v___x_206_ == 0)
{
if (v___x_205_ == 0)
{
lean_dec_ref(v_binders_x27_202_);
v___y_122_ = v___x_154_;
v___y_123_ = v___x_201_;
v___y_124_ = v___y_153_;
v___y_125_ = v___x_156_;
v___y_126_ = v___x_155_;
v___y_127_ = v___x_204_;
goto v___jp_121_;
}
else
{
size_t v___x_207_; size_t v___x_208_; lean_object* v___x_209_; 
v___x_207_ = ((size_t)0ULL);
v___x_208_ = lean_usize_of_nat(v___x_203_);
v___x_209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2(v_binders_x27_202_, v___x_207_, v___x_208_, v___x_204_);
lean_dec_ref(v_binders_x27_202_);
v___y_122_ = v___x_154_;
v___y_123_ = v___x_201_;
v___y_124_ = v___y_153_;
v___y_125_ = v___x_156_;
v___y_126_ = v___x_155_;
v___y_127_ = v___x_209_;
goto v___jp_121_;
}
}
else
{
size_t v___x_210_; size_t v___x_211_; lean_object* v___x_212_; 
v___x_210_ = ((size_t)0ULL);
v___x_211_ = lean_usize_of_nat(v___x_203_);
v___x_212_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit_spec__2(v_binders_x27_202_, v___x_210_, v___x_211_, v___x_204_);
lean_dec_ref(v_binders_x27_202_);
v___y_122_ = v___x_154_;
v___y_123_ = v___x_201_;
v___y_124_ = v___y_153_;
v___y_125_ = v___x_156_;
v___y_126_ = v___x_155_;
v___y_127_ = v___x_212_;
goto v___jp_121_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0___boxed(lean_object* v_binders_223_, lean_object* v_type_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___lam__0(v_binders_223_, v_type_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
lean_dec(v___y_230_);
lean_dec_ref(v___y_229_);
lean_dec(v___y_228_);
lean_dec_ref(v___y_227_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec_ref(v_binders_223_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit(lean_object* v_type_236_){
_start:
{
lean_object* v_delab_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_delab_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit___closed__1));
v___x_238_ = lean_box(1);
v___x_239_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_ppExprWithInfos___boxed), 8, 3);
lean_closure_set(v___x_239_, 0, v_type_236_);
lean_closure_set(v___x_239_, 1, v___x_238_);
lean_closure_set(v___x_239_, 2, v_delab_237_);
v___x_240_ = l_Lean_MessageData_ofFormatWithInfosM(v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg(lean_object* v_e_241_, lean_object* v___y_242_){
_start:
{
uint8_t v___x_244_; 
v___x_244_ = l_Lean_Expr_hasMVar(v_e_241_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; 
v___x_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_245_, 0, v_e_241_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v_mctx_247_; lean_object* v___x_248_; lean_object* v_fst_249_; lean_object* v_snd_250_; lean_object* v___x_251_; lean_object* v_cache_252_; lean_object* v_zetaDeltaFVarIds_253_; lean_object* v_postponed_254_; lean_object* v_diag_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_264_; 
v___x_246_ = lean_st_ref_get(v___y_242_);
v_mctx_247_ = lean_ctor_get(v___x_246_, 0);
lean_inc_ref(v_mctx_247_);
lean_dec(v___x_246_);
v___x_248_ = l_Lean_instantiateMVarsCore(v_mctx_247_, v_e_241_);
v_fst_249_ = lean_ctor_get(v___x_248_, 0);
lean_inc(v_fst_249_);
v_snd_250_ = lean_ctor_get(v___x_248_, 1);
lean_inc(v_snd_250_);
lean_dec_ref(v___x_248_);
v___x_251_ = lean_st_ref_take(v___y_242_);
v_cache_252_ = lean_ctor_get(v___x_251_, 1);
v_zetaDeltaFVarIds_253_ = lean_ctor_get(v___x_251_, 2);
v_postponed_254_ = lean_ctor_get(v___x_251_, 3);
v_diag_255_ = lean_ctor_get(v___x_251_, 4);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_264_ == 0)
{
lean_object* v_unused_265_; 
v_unused_265_ = lean_ctor_get(v___x_251_, 0);
lean_dec(v_unused_265_);
v___x_257_ = v___x_251_;
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_diag_255_);
lean_inc(v_postponed_254_);
lean_inc(v_zetaDeltaFVarIds_253_);
lean_inc(v_cache_252_);
lean_dec(v___x_251_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_264_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_260_; 
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v_snd_250_);
v___x_260_ = v___x_257_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_snd_250_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v_cache_252_);
lean_ctor_set(v_reuseFailAlloc_263_, 2, v_zetaDeltaFVarIds_253_);
lean_ctor_set(v_reuseFailAlloc_263_, 3, v_postponed_254_);
lean_ctor_set(v_reuseFailAlloc_263_, 4, v_diag_255_);
v___x_260_ = v_reuseFailAlloc_263_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = lean_st_ref_set(v___y_242_, v___x_260_);
v___x_262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_262_, 0, v_fst_249_);
return v___x_262_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg___boxed(lean_object* v_e_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg(v_e_266_, v___y_267_);
lean_dec(v___y_267_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0(lean_object* v_e_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg(v_e_270_, v___y_274_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___boxed(lean_object* v_e_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0(v_e_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
return v_res_287_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0(lean_object* v_x_288_){
_start:
{
uint8_t v___x_289_; 
v___x_289_ = 0;
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0___boxed(lean_object* v_x_290_){
_start:
{
uint8_t v_res_291_; lean_object* v_r_292_; 
v_res_291_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__0(v_x_290_);
lean_dec(v_x_290_);
v_r_292_ = lean_box(v_res_291_);
return v_r_292_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3(lean_object* v_opts_293_, lean_object* v_opt_294_){
_start:
{
lean_object* v_name_295_; lean_object* v_defValue_296_; lean_object* v_map_297_; lean_object* v___x_298_; 
v_name_295_ = lean_ctor_get(v_opt_294_, 0);
v_defValue_296_ = lean_ctor_get(v_opt_294_, 1);
v_map_297_ = lean_ctor_get(v_opts_293_, 0);
v___x_298_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_297_, v_name_295_);
if (lean_obj_tag(v___x_298_) == 0)
{
uint8_t v___x_299_; 
v___x_299_ = lean_unbox(v_defValue_296_);
return v___x_299_;
}
else
{
lean_object* v_val_300_; 
v_val_300_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_val_300_);
lean_dec_ref_known(v___x_298_, 1);
if (lean_obj_tag(v_val_300_) == 1)
{
uint8_t v_v_301_; 
v_v_301_ = lean_ctor_get_uint8(v_val_300_, 0);
lean_dec_ref_known(v_val_300_, 0);
return v_v_301_;
}
else
{
uint8_t v___x_302_; 
lean_dec(v_val_300_);
v___x_302_ = lean_unbox(v_defValue_296_);
return v___x_302_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3___boxed(lean_object* v_opts_303_, lean_object* v_opt_304_){
_start:
{
uint8_t v_res_305_; lean_object* v_r_306_; 
v_res_305_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3(v_opts_303_, v_opt_304_);
lean_dec_ref(v_opt_304_);
lean_dec_ref(v_opts_303_);
v_r_306_ = lean_box(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2(lean_object* v_msgData_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v___x_313_; lean_object* v_env_314_; lean_object* v___x_315_; lean_object* v_mctx_316_; lean_object* v_lctx_317_; lean_object* v_options_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_313_ = lean_st_ref_get(v___y_311_);
v_env_314_ = lean_ctor_get(v___x_313_, 0);
lean_inc_ref(v_env_314_);
lean_dec(v___x_313_);
v___x_315_ = lean_st_ref_get(v___y_309_);
v_mctx_316_ = lean_ctor_get(v___x_315_, 0);
lean_inc_ref(v_mctx_316_);
lean_dec(v___x_315_);
v_lctx_317_ = lean_ctor_get(v___y_308_, 2);
v_options_318_ = lean_ctor_get(v___y_310_, 2);
lean_inc_ref(v_options_318_);
lean_inc_ref(v_lctx_317_);
v___x_319_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_319_, 0, v_env_314_);
lean_ctor_set(v___x_319_, 1, v_mctx_316_);
lean_ctor_set(v___x_319_, 2, v_lctx_317_);
lean_ctor_set(v___x_319_, 3, v_options_318_);
v___x_320_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
lean_ctor_set(v___x_320_, 1, v_msgData_307_);
v___x_321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_321_, 0, v___x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2(v_msgData_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
return v_res_328_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0(uint8_t v___y_337_, uint8_t v_suppressElabErrors_338_, lean_object* v_x_339_){
_start:
{
if (lean_obj_tag(v_x_339_) == 1)
{
lean_object* v_pre_340_; 
v_pre_340_ = lean_ctor_get(v_x_339_, 0);
switch(lean_obj_tag(v_pre_340_))
{
case 1:
{
lean_object* v_pre_341_; 
v_pre_341_ = lean_ctor_get(v_pre_340_, 0);
switch(lean_obj_tag(v_pre_341_))
{
case 0:
{
lean_object* v_str_342_; lean_object* v_str_343_; lean_object* v___x_344_; uint8_t v___x_345_; 
v_str_342_ = lean_ctor_get(v_x_339_, 1);
v_str_343_ = lean_ctor_get(v_pre_340_, 1);
v___x_344_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__0));
v___x_345_ = lean_string_dec_eq(v_str_343_, v___x_344_);
if (v___x_345_ == 0)
{
lean_object* v___x_346_; uint8_t v___x_347_; 
v___x_346_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__1));
v___x_347_ = lean_string_dec_eq(v_str_343_, v___x_346_);
if (v___x_347_ == 0)
{
return v___y_337_;
}
else
{
lean_object* v___x_348_; uint8_t v___x_349_; 
v___x_348_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__2));
v___x_349_ = lean_string_dec_eq(v_str_342_, v___x_348_);
if (v___x_349_ == 0)
{
return v___y_337_;
}
else
{
return v_suppressElabErrors_338_;
}
}
}
else
{
lean_object* v___x_350_; uint8_t v___x_351_; 
v___x_350_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__3));
v___x_351_ = lean_string_dec_eq(v_str_342_, v___x_350_);
if (v___x_351_ == 0)
{
return v___y_337_;
}
else
{
return v_suppressElabErrors_338_;
}
}
}
case 1:
{
lean_object* v_pre_352_; 
v_pre_352_ = lean_ctor_get(v_pre_341_, 0);
if (lean_obj_tag(v_pre_352_) == 0)
{
lean_object* v_str_353_; lean_object* v_str_354_; lean_object* v_str_355_; lean_object* v___x_356_; uint8_t v___x_357_; 
v_str_353_ = lean_ctor_get(v_x_339_, 1);
v_str_354_ = lean_ctor_get(v_pre_340_, 1);
v_str_355_ = lean_ctor_get(v_pre_341_, 1);
v___x_356_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__4));
v___x_357_ = lean_string_dec_eq(v_str_355_, v___x_356_);
if (v___x_357_ == 0)
{
return v___y_337_;
}
else
{
lean_object* v___x_358_; uint8_t v___x_359_; 
v___x_358_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__5));
v___x_359_ = lean_string_dec_eq(v_str_354_, v___x_358_);
if (v___x_359_ == 0)
{
return v___y_337_;
}
else
{
lean_object* v___x_360_; uint8_t v___x_361_; 
v___x_360_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__6));
v___x_361_ = lean_string_dec_eq(v_str_353_, v___x_360_);
if (v___x_361_ == 0)
{
return v___y_337_;
}
else
{
return v_suppressElabErrors_338_;
}
}
}
}
else
{
return v___y_337_;
}
}
default: 
{
return v___y_337_;
}
}
}
case 0:
{
lean_object* v_str_362_; lean_object* v___x_363_; uint8_t v___x_364_; 
v_str_362_ = lean_ctor_get(v_x_339_, 1);
v___x_363_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___closed__7));
v___x_364_ = lean_string_dec_eq(v_str_362_, v___x_363_);
if (v___x_364_ == 0)
{
return v___y_337_;
}
else
{
return v_suppressElabErrors_338_;
}
}
default: 
{
return v___y_337_;
}
}
}
else
{
return v___y_337_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v___y_365_, lean_object* v_suppressElabErrors_366_, lean_object* v_x_367_){
_start:
{
uint8_t v___y_16209__boxed_368_; uint8_t v_suppressElabErrors_boxed_369_; uint8_t v_res_370_; lean_object* v_r_371_; 
v___y_16209__boxed_368_ = lean_unbox(v___y_365_);
v_suppressElabErrors_boxed_369_ = lean_unbox(v_suppressElabErrors_366_);
v_res_370_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0(v___y_16209__boxed_368_, v_suppressElabErrors_boxed_369_, v_x_367_);
lean_dec(v_x_367_);
v_r_371_ = lean_box(v_res_370_);
return v_r_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg(lean_object* v_ref_373_, lean_object* v_msgData_374_, uint8_t v_severity_375_, uint8_t v_isSilent_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_){
_start:
{
uint8_t v___y_383_; lean_object* v___y_384_; uint8_t v___y_385_; lean_object* v___y_386_; lean_object* v___y_387_; lean_object* v___y_388_; lean_object* v___y_389_; lean_object* v___y_390_; lean_object* v___y_391_; lean_object* v___y_419_; uint8_t v___y_420_; uint8_t v___y_421_; lean_object* v___y_422_; uint8_t v___y_423_; lean_object* v___y_424_; lean_object* v___y_425_; lean_object* v___y_426_; lean_object* v___y_444_; uint8_t v___y_445_; uint8_t v___y_446_; lean_object* v___y_447_; lean_object* v___y_448_; uint8_t v___y_449_; lean_object* v___y_450_; lean_object* v___y_451_; lean_object* v___y_455_; uint8_t v___y_456_; uint8_t v___y_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; uint8_t v___y_461_; uint8_t v___x_466_; uint8_t v___y_468_; lean_object* v___y_469_; lean_object* v___y_470_; lean_object* v___y_471_; lean_object* v___y_472_; uint8_t v___y_473_; uint8_t v___y_474_; uint8_t v___y_476_; uint8_t v___x_491_; 
v___x_466_ = 2;
v___x_491_ = l_Lean_instBEqMessageSeverity_beq(v_severity_375_, v___x_466_);
if (v___x_491_ == 0)
{
v___y_476_ = v___x_491_;
goto v___jp_475_;
}
else
{
uint8_t v___x_492_; 
lean_inc_ref(v_msgData_374_);
v___x_492_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_374_);
v___y_476_ = v___x_492_;
goto v___jp_475_;
}
v___jp_382_:
{
lean_object* v___x_392_; lean_object* v_currNamespace_393_; lean_object* v_openDecls_394_; lean_object* v_env_395_; lean_object* v_nextMacroScope_396_; lean_object* v_ngen_397_; lean_object* v_auxDeclNGen_398_; lean_object* v_traceState_399_; lean_object* v_cache_400_; lean_object* v_messages_401_; lean_object* v_infoState_402_; lean_object* v_snapshotTasks_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_417_; 
v___x_392_ = lean_st_ref_take(v___y_391_);
v_currNamespace_393_ = lean_ctor_get(v___y_390_, 6);
v_openDecls_394_ = lean_ctor_get(v___y_390_, 7);
v_env_395_ = lean_ctor_get(v___x_392_, 0);
v_nextMacroScope_396_ = lean_ctor_get(v___x_392_, 1);
v_ngen_397_ = lean_ctor_get(v___x_392_, 2);
v_auxDeclNGen_398_ = lean_ctor_get(v___x_392_, 3);
v_traceState_399_ = lean_ctor_get(v___x_392_, 4);
v_cache_400_ = lean_ctor_get(v___x_392_, 5);
v_messages_401_ = lean_ctor_get(v___x_392_, 6);
v_infoState_402_ = lean_ctor_get(v___x_392_, 7);
v_snapshotTasks_403_ = lean_ctor_get(v___x_392_, 8);
v_isSharedCheck_417_ = !lean_is_exclusive(v___x_392_);
if (v_isSharedCheck_417_ == 0)
{
v___x_405_ = v___x_392_;
v_isShared_406_ = v_isSharedCheck_417_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_snapshotTasks_403_);
lean_inc(v_infoState_402_);
lean_inc(v_messages_401_);
lean_inc(v_cache_400_);
lean_inc(v_traceState_399_);
lean_inc(v_auxDeclNGen_398_);
lean_inc(v_ngen_397_);
lean_inc(v_nextMacroScope_396_);
lean_inc(v_env_395_);
lean_dec(v___x_392_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_417_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_412_; 
lean_inc(v_openDecls_394_);
lean_inc(v_currNamespace_393_);
v___x_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_407_, 0, v_currNamespace_393_);
lean_ctor_set(v___x_407_, 1, v_openDecls_394_);
v___x_408_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_407_);
lean_ctor_set(v___x_408_, 1, v___y_384_);
lean_inc_ref(v___y_388_);
lean_inc_ref(v___y_389_);
v___x_409_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_409_, 0, v___y_389_);
lean_ctor_set(v___x_409_, 1, v___y_387_);
lean_ctor_set(v___x_409_, 2, v___y_386_);
lean_ctor_set(v___x_409_, 3, v___y_388_);
lean_ctor_set(v___x_409_, 4, v___x_408_);
lean_ctor_set_uint8(v___x_409_, sizeof(void*)*5, v___y_383_);
lean_ctor_set_uint8(v___x_409_, sizeof(void*)*5 + 1, v___y_385_);
lean_ctor_set_uint8(v___x_409_, sizeof(void*)*5 + 2, v_isSilent_376_);
v___x_410_ = l_Lean_MessageLog_add(v___x_409_, v_messages_401_);
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 6, v___x_410_);
v___x_412_ = v___x_405_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_env_395_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v_nextMacroScope_396_);
lean_ctor_set(v_reuseFailAlloc_416_, 2, v_ngen_397_);
lean_ctor_set(v_reuseFailAlloc_416_, 3, v_auxDeclNGen_398_);
lean_ctor_set(v_reuseFailAlloc_416_, 4, v_traceState_399_);
lean_ctor_set(v_reuseFailAlloc_416_, 5, v_cache_400_);
lean_ctor_set(v_reuseFailAlloc_416_, 6, v___x_410_);
lean_ctor_set(v_reuseFailAlloc_416_, 7, v_infoState_402_);
lean_ctor_set(v_reuseFailAlloc_416_, 8, v_snapshotTasks_403_);
v___x_412_ = v_reuseFailAlloc_416_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_413_ = lean_st_ref_set(v___y_391_, v___x_412_);
v___x_414_ = lean_box(0);
v___x_415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
return v___x_415_;
}
}
}
v___jp_418_:
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_442_; 
v___x_427_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_374_);
v___x_428_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2(v___x_427_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
v_a_429_ = lean_ctor_get(v___x_428_, 0);
v_isSharedCheck_442_ = !lean_is_exclusive(v___x_428_);
if (v_isSharedCheck_442_ == 0)
{
v___x_431_ = v___x_428_;
v_isShared_432_ = v_isSharedCheck_442_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_428_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_442_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
lean_inc_ref_n(v___y_422_, 2);
v___x_433_ = l_Lean_FileMap_toPosition(v___y_422_, v___y_425_);
lean_dec(v___y_425_);
v___x_434_ = l_Lean_FileMap_toPosition(v___y_422_, v___y_426_);
lean_dec(v___y_426_);
v___x_435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_435_, 0, v___x_434_);
v___x_436_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___closed__0));
if (v___y_421_ == 0)
{
lean_del_object(v___x_431_);
lean_dec_ref(v___y_419_);
v___y_383_ = v___y_420_;
v___y_384_ = v_a_429_;
v___y_385_ = v___y_423_;
v___y_386_ = v___x_435_;
v___y_387_ = v___x_433_;
v___y_388_ = v___x_436_;
v___y_389_ = v___y_424_;
v___y_390_ = v___y_379_;
v___y_391_ = v___y_380_;
goto v___jp_382_;
}
else
{
uint8_t v___x_437_; 
lean_inc(v_a_429_);
v___x_437_ = l_Lean_MessageData_hasTag(v___y_419_, v_a_429_);
if (v___x_437_ == 0)
{
lean_object* v___x_438_; lean_object* v___x_440_; 
lean_dec_ref_known(v___x_435_, 1);
lean_dec_ref(v___x_433_);
lean_dec(v_a_429_);
v___x_438_ = lean_box(0);
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 0, v___x_438_);
v___x_440_ = v___x_431_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v___x_438_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
else
{
lean_del_object(v___x_431_);
v___y_383_ = v___y_420_;
v___y_384_ = v_a_429_;
v___y_385_ = v___y_423_;
v___y_386_ = v___x_435_;
v___y_387_ = v___x_433_;
v___y_388_ = v___x_436_;
v___y_389_ = v___y_424_;
v___y_390_ = v___y_379_;
v___y_391_ = v___y_380_;
goto v___jp_382_;
}
}
}
}
v___jp_443_:
{
lean_object* v___x_452_; 
v___x_452_ = l_Lean_Syntax_getTailPos_x3f(v___y_448_, v___y_445_);
lean_dec(v___y_448_);
if (lean_obj_tag(v___x_452_) == 0)
{
lean_inc(v___y_451_);
v___y_419_ = v___y_444_;
v___y_420_ = v___y_445_;
v___y_421_ = v___y_446_;
v___y_422_ = v___y_447_;
v___y_423_ = v___y_449_;
v___y_424_ = v___y_450_;
v___y_425_ = v___y_451_;
v___y_426_ = v___y_451_;
goto v___jp_418_;
}
else
{
lean_object* v_val_453_; 
v_val_453_ = lean_ctor_get(v___x_452_, 0);
lean_inc(v_val_453_);
lean_dec_ref_known(v___x_452_, 1);
v___y_419_ = v___y_444_;
v___y_420_ = v___y_445_;
v___y_421_ = v___y_446_;
v___y_422_ = v___y_447_;
v___y_423_ = v___y_449_;
v___y_424_ = v___y_450_;
v___y_425_ = v___y_451_;
v___y_426_ = v_val_453_;
goto v___jp_418_;
}
}
v___jp_454_:
{
lean_object* v_ref_462_; lean_object* v___x_463_; 
v_ref_462_ = l_Lean_replaceRef(v_ref_373_, v___y_459_);
v___x_463_ = l_Lean_Syntax_getPos_x3f(v_ref_462_, v___y_456_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_object* v___x_464_; 
v___x_464_ = lean_unsigned_to_nat(0u);
v___y_444_ = v___y_455_;
v___y_445_ = v___y_456_;
v___y_446_ = v___y_457_;
v___y_447_ = v___y_458_;
v___y_448_ = v_ref_462_;
v___y_449_ = v___y_461_;
v___y_450_ = v___y_460_;
v___y_451_ = v___x_464_;
goto v___jp_443_;
}
else
{
lean_object* v_val_465_; 
v_val_465_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_val_465_);
lean_dec_ref_known(v___x_463_, 1);
v___y_444_ = v___y_455_;
v___y_445_ = v___y_456_;
v___y_446_ = v___y_457_;
v___y_447_ = v___y_458_;
v___y_448_ = v_ref_462_;
v___y_449_ = v___y_461_;
v___y_450_ = v___y_460_;
v___y_451_ = v_val_465_;
goto v___jp_443_;
}
}
v___jp_467_:
{
if (v___y_474_ == 0)
{
v___y_455_ = v___y_470_;
v___y_456_ = v___y_473_;
v___y_457_ = v___y_468_;
v___y_458_ = v___y_469_;
v___y_459_ = v___y_471_;
v___y_460_ = v___y_472_;
v___y_461_ = v_severity_375_;
goto v___jp_454_;
}
else
{
v___y_455_ = v___y_470_;
v___y_456_ = v___y_473_;
v___y_457_ = v___y_468_;
v___y_458_ = v___y_469_;
v___y_459_ = v___y_471_;
v___y_460_ = v___y_472_;
v___y_461_ = v___x_466_;
goto v___jp_454_;
}
}
v___jp_475_:
{
if (v___y_476_ == 0)
{
lean_object* v_fileName_477_; lean_object* v_fileMap_478_; lean_object* v_options_479_; lean_object* v_ref_480_; uint8_t v_suppressElabErrors_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___f_484_; uint8_t v___x_485_; uint8_t v___x_486_; 
v_fileName_477_ = lean_ctor_get(v___y_379_, 0);
v_fileMap_478_ = lean_ctor_get(v___y_379_, 1);
v_options_479_ = lean_ctor_get(v___y_379_, 2);
v_ref_480_ = lean_ctor_get(v___y_379_, 5);
v_suppressElabErrors_481_ = lean_ctor_get_uint8(v___y_379_, sizeof(void*)*14 + 1);
v___x_482_ = lean_box(v___y_476_);
v___x_483_ = lean_box(v_suppressElabErrors_481_);
v___f_484_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_484_, 0, v___x_482_);
lean_closure_set(v___f_484_, 1, v___x_483_);
v___x_485_ = 1;
v___x_486_ = l_Lean_instBEqMessageSeverity_beq(v_severity_375_, v___x_485_);
if (v___x_486_ == 0)
{
v___y_468_ = v_suppressElabErrors_481_;
v___y_469_ = v_fileMap_478_;
v___y_470_ = v___f_484_;
v___y_471_ = v_ref_480_;
v___y_472_ = v_fileName_477_;
v___y_473_ = v___y_476_;
v___y_474_ = v___x_486_;
goto v___jp_467_;
}
else
{
lean_object* v___x_487_; uint8_t v___x_488_; 
v___x_487_ = l_Lean_warningAsError;
v___x_488_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3(v_options_479_, v___x_487_);
v___y_468_ = v_suppressElabErrors_481_;
v___y_469_ = v_fileMap_478_;
v___y_470_ = v___f_484_;
v___y_471_ = v_ref_480_;
v___y_472_ = v_fileName_477_;
v___y_473_ = v___y_476_;
v___y_474_ = v___x_488_;
goto v___jp_467_;
}
}
else
{
lean_object* v___x_489_; lean_object* v___x_490_; 
lean_dec_ref(v_msgData_374_);
v___x_489_ = lean_box(0);
v___x_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
return v___x_490_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg___boxed(lean_object* v_ref_493_, lean_object* v_msgData_494_, lean_object* v_severity_495_, lean_object* v_isSilent_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_){
_start:
{
uint8_t v_severity_boxed_502_; uint8_t v_isSilent_boxed_503_; lean_object* v_res_504_; 
v_severity_boxed_502_ = lean_unbox(v_severity_495_);
v_isSilent_boxed_503_ = lean_unbox(v_isSilent_496_);
v_res_504_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg(v_ref_493_, v_msgData_494_, v_severity_boxed_502_, v_isSilent_boxed_503_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
lean_dec(v___y_500_);
lean_dec_ref(v___y_499_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec(v_ref_493_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1(lean_object* v_ref_505_, lean_object* v_msgData_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
uint8_t v___x_514_; uint8_t v___x_515_; lean_object* v___x_516_; 
v___x_514_ = 0;
v___x_515_ = 0;
v___x_516_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg(v_ref_505_, v_msgData_506_, v___x_514_, v___x_515_, v___y_509_, v___y_510_, v___y_511_, v___y_512_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1___boxed(lean_object* v_ref_517_, lean_object* v_msgData_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1(v_ref_517_, v_msgData_518_, v___y_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_);
lean_dec(v___y_524_);
lean_dec_ref(v___y_523_);
lean_dec(v___y_522_);
lean_dec_ref(v___y_521_);
lean_dec(v___y_520_);
lean_dec_ref(v___y_519_);
lean_dec(v_ref_517_);
return v_res_526_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1(void){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__0));
v___x_529_ = l_Lean_stringToMessageData(v___x_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1(lean_object* v_term_530_, lean_object* v_tk_531_, lean_object* v___f_532_, lean_object* v_____r_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_){
_start:
{
lean_object* v___x_541_; uint8_t v___x_542_; lean_object* v___x_543_; 
v___x_541_ = lean_box(0);
v___x_542_ = 1;
v___x_543_ = l_Lean_Elab_Term_elabTerm(v_term_530_, v___x_541_, v___x_542_, v___x_542_, v___y_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
if (lean_obj_tag(v___x_543_) == 0)
{
lean_object* v_a_544_; lean_object* v___x_545_; 
v_a_544_ = lean_ctor_get(v___x_543_, 0);
lean_inc(v_a_544_);
lean_dec_ref_known(v___x_543_, 1);
v___x_545_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_542_, v___y_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
if (lean_obj_tag(v___x_545_) == 0)
{
lean_object* v_fileName_546_; lean_object* v_fileMap_547_; lean_object* v_options_548_; lean_object* v_currRecDepth_549_; lean_object* v_maxRecDepth_550_; lean_object* v_ref_551_; lean_object* v_currNamespace_552_; lean_object* v_openDecls_553_; lean_object* v_initHeartbeats_554_; lean_object* v_maxHeartbeats_555_; lean_object* v_quotContext_556_; lean_object* v_currMacroScope_557_; uint8_t v_diag_558_; lean_object* v_cancelTk_x3f_559_; uint8_t v_suppressElabErrors_560_; lean_object* v_inheritedTraceOptions_561_; uint8_t v___x_562_; lean_object* v_ref_563_; lean_object* v___x_564_; lean_object* v___x_565_; 
lean_dec_ref_known(v___x_545_, 1);
v_fileName_546_ = lean_ctor_get(v___y_538_, 0);
v_fileMap_547_ = lean_ctor_get(v___y_538_, 1);
v_options_548_ = lean_ctor_get(v___y_538_, 2);
v_currRecDepth_549_ = lean_ctor_get(v___y_538_, 3);
v_maxRecDepth_550_ = lean_ctor_get(v___y_538_, 4);
v_ref_551_ = lean_ctor_get(v___y_538_, 5);
v_currNamespace_552_ = lean_ctor_get(v___y_538_, 6);
v_openDecls_553_ = lean_ctor_get(v___y_538_, 7);
v_initHeartbeats_554_ = lean_ctor_get(v___y_538_, 8);
v_maxHeartbeats_555_ = lean_ctor_get(v___y_538_, 9);
v_quotContext_556_ = lean_ctor_get(v___y_538_, 10);
v_currMacroScope_557_ = lean_ctor_get(v___y_538_, 11);
v_diag_558_ = lean_ctor_get_uint8(v___y_538_, sizeof(void*)*14);
v_cancelTk_x3f_559_ = lean_ctor_get(v___y_538_, 12);
v_suppressElabErrors_560_ = lean_ctor_get_uint8(v___y_538_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_561_ = lean_ctor_get(v___y_538_, 13);
v___x_562_ = 0;
v_ref_563_ = l_Lean_replaceRef(v_tk_531_, v_ref_551_);
lean_inc_ref(v_inheritedTraceOptions_561_);
lean_inc(v_cancelTk_x3f_559_);
lean_inc(v_currMacroScope_557_);
lean_inc(v_quotContext_556_);
lean_inc(v_maxHeartbeats_555_);
lean_inc(v_initHeartbeats_554_);
lean_inc(v_openDecls_553_);
lean_inc(v_currNamespace_552_);
lean_inc(v_maxRecDepth_550_);
lean_inc(v_currRecDepth_549_);
lean_inc_ref(v_options_548_);
lean_inc_ref(v_fileMap_547_);
lean_inc_ref(v_fileName_546_);
v___x_564_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_564_, 0, v_fileName_546_);
lean_ctor_set(v___x_564_, 1, v_fileMap_547_);
lean_ctor_set(v___x_564_, 2, v_options_548_);
lean_ctor_set(v___x_564_, 3, v_currRecDepth_549_);
lean_ctor_set(v___x_564_, 4, v_maxRecDepth_550_);
lean_ctor_set(v___x_564_, 5, v_ref_563_);
lean_ctor_set(v___x_564_, 6, v_currNamespace_552_);
lean_ctor_set(v___x_564_, 7, v_openDecls_553_);
lean_ctor_set(v___x_564_, 8, v_initHeartbeats_554_);
lean_ctor_set(v___x_564_, 9, v_maxHeartbeats_555_);
lean_ctor_set(v___x_564_, 10, v_quotContext_556_);
lean_ctor_set(v___x_564_, 11, v_currMacroScope_557_);
lean_ctor_set(v___x_564_, 12, v_cancelTk_x3f_559_);
lean_ctor_set(v___x_564_, 13, v_inheritedTraceOptions_561_);
lean_ctor_set_uint8(v___x_564_, sizeof(void*)*14, v_diag_558_);
lean_ctor_set_uint8(v___x_564_, sizeof(void*)*14 + 1, v_suppressElabErrors_560_);
lean_inc(v_a_544_);
v___x_565_ = l_Lean_Meta_check(v_a_544_, v___x_562_, v___y_536_, v___y_537_, v___x_564_, v___y_539_);
lean_dec_ref_known(v___x_564_, 14);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v___x_566_; lean_object* v_a_567_; lean_object* v___x_568_; 
lean_dec_ref_known(v___x_565_, 1);
v___x_566_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__0___redArg(v_a_544_, v___y_537_);
v_a_567_ = lean_ctor_get(v___x_566_, 0);
lean_inc(v_a_567_);
lean_dec_ref(v___x_566_);
v___x_568_ = l_Lean_Elab_Term_levelMVarToParam___redArg(v_a_567_, v___f_532_, v___y_535_, v___y_537_);
if (lean_obj_tag(v___x_568_) == 0)
{
lean_object* v_a_569_; lean_object* v___x_571_; uint8_t v_isShared_572_; uint8_t v_isSharedCheck_594_; 
v_a_569_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_594_ == 0)
{
v___x_571_ = v___x_568_;
v_isShared_572_ = v_isSharedCheck_594_;
goto v_resetjp_570_;
}
else
{
lean_inc(v_a_569_);
lean_dec(v___x_568_);
v___x_571_ = lean_box(0);
v_isShared_572_ = v_isSharedCheck_594_;
goto v_resetjp_570_;
}
v_resetjp_570_:
{
uint8_t v___x_573_; 
v___x_573_ = l_Lean_Expr_isSyntheticSorry(v_a_569_);
if (v___x_573_ == 0)
{
lean_object* v___x_574_; 
lean_del_object(v___x_571_);
lean_inc(v___y_539_);
lean_inc_ref(v___y_538_);
lean_inc(v___y_537_);
lean_inc_ref(v___y_536_);
lean_inc(v_a_569_);
v___x_574_ = lean_infer_type(v_a_569_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
if (lean_obj_tag(v___x_574_) == 0)
{
lean_object* v_a_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v_a_575_ = lean_ctor_get(v___x_574_, 0);
lean_inc(v_a_575_);
lean_dec_ref_known(v___x_574_, 1);
v___x_576_ = l_Lean_MessageData_ofExpr(v_a_569_);
v___x_577_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___closed__1);
v___x_578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_576_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = l_Lean_MessageData_ofExpr(v_a_575_);
v___x_580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_580_, 0, v___x_578_);
lean_ctor_set(v___x_580_, 1, v___x_579_);
v___x_581_ = lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1(v_tk_531_, v___x_580_, v___y_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
return v___x_581_;
}
else
{
lean_object* v_a_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_589_; 
lean_dec(v_a_569_);
v_a_582_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_589_ == 0)
{
v___x_584_ = v___x_574_;
v_isShared_585_ = v_isSharedCheck_589_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_a_582_);
lean_dec(v___x_574_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_589_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v___x_587_; 
if (v_isShared_585_ == 0)
{
v___x_587_ = v___x_584_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v_a_582_);
v___x_587_ = v_reuseFailAlloc_588_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
return v___x_587_;
}
}
}
}
else
{
lean_object* v___x_590_; lean_object* v___x_592_; 
lean_dec(v_a_569_);
v___x_590_ = lean_box(0);
if (v_isShared_572_ == 0)
{
lean_ctor_set(v___x_571_, 0, v___x_590_);
v___x_592_ = v___x_571_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v___x_590_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
}
else
{
lean_object* v_a_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_602_; 
v_a_595_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_602_ == 0)
{
v___x_597_ = v___x_568_;
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_a_595_);
lean_dec(v___x_568_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_600_; 
if (v_isShared_598_ == 0)
{
v___x_600_ = v___x_597_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_a_595_);
v___x_600_ = v_reuseFailAlloc_601_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
return v___x_600_;
}
}
}
}
else
{
lean_dec(v_a_544_);
lean_dec_ref(v___f_532_);
return v___x_565_;
}
}
else
{
lean_dec(v_a_544_);
lean_dec_ref(v___f_532_);
return v___x_545_;
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_dec_ref(v___f_532_);
v_a_603_ = lean_ctor_get(v___x_543_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_543_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_543_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_543_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___boxed(lean_object* v_term_611_, lean_object* v_tk_612_, lean_object* v___f_613_, lean_object* v_____r_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1(v_term_611_, v_tk_612_, v___f_613_, v_____r_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_, v___y_620_);
lean_dec(v___y_620_);
lean_dec_ref(v___y_619_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
lean_dec(v___y_616_);
lean_dec_ref(v___y_615_);
lean_dec(v_tk_612_);
return v_res_622_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0(void){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_623_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1(void){
_start:
{
lean_object* v___x_624_; lean_object* v___x_625_; 
v___x_624_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__0);
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v___x_624_);
return v___x_625_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2(void){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_626_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1);
v___x_627_ = lean_unsigned_to_nat(0u);
v___x_628_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
lean_ctor_set(v___x_628_, 2, v___x_627_);
lean_ctor_set(v___x_628_, 3, v___x_627_);
lean_ctor_set(v___x_628_, 4, v___x_626_);
lean_ctor_set(v___x_628_, 5, v___x_626_);
lean_ctor_set(v___x_628_, 6, v___x_626_);
lean_ctor_set(v___x_628_, 7, v___x_626_);
lean_ctor_set(v___x_628_, 8, v___x_626_);
lean_ctor_set(v___x_628_, 9, v___x_626_);
return v___x_628_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = lean_unsigned_to_nat(32u);
v___x_630_ = lean_mk_empty_array_with_capacity(v___x_629_);
v___x_631_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_631_, 0, v___x_630_);
return v___x_631_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4(void){
_start:
{
size_t v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_632_ = ((size_t)5ULL);
v___x_633_ = lean_unsigned_to_nat(0u);
v___x_634_ = lean_unsigned_to_nat(32u);
v___x_635_ = lean_mk_empty_array_with_capacity(v___x_634_);
v___x_636_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__3);
v___x_637_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_637_, 0, v___x_636_);
lean_ctor_set(v___x_637_, 1, v___x_635_);
lean_ctor_set(v___x_637_, 2, v___x_633_);
lean_ctor_set(v___x_637_, 3, v___x_633_);
lean_ctor_set_usize(v___x_637_, 4, v___x_632_);
return v___x_637_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_638_ = lean_box(1);
v___x_639_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4);
v___x_640_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__1);
v___x_641_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_641_, 0, v___x_640_);
lean_ctor_set(v___x_641_, 1, v___x_639_);
lean_ctor_set(v___x_641_, 2, v___x_638_);
return v___x_641_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7(void){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_643_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__6));
v___x_644_ = l_Lean_stringToMessageData(v___x_643_);
return v___x_644_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9(void){
_start:
{
lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_646_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__8));
v___x_647_ = l_Lean_stringToMessageData(v___x_646_);
return v___x_647_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_649_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__10));
v___x_650_ = l_Lean_stringToMessageData(v___x_649_);
return v___x_650_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_652_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__12));
v___x_653_ = l_Lean_stringToMessageData(v___x_652_);
return v___x_653_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_655_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__14));
v___x_656_ = l_Lean_stringToMessageData(v___x_655_);
return v___x_656_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17(void){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_658_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__16));
v___x_659_ = l_Lean_stringToMessageData(v___x_658_);
return v___x_659_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19(void){
_start:
{
lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_661_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__18));
v___x_662_ = l_Lean_stringToMessageData(v___x_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg(lean_object* v_msg_663_, lean_object* v_declHint_664_, lean_object* v___y_665_){
_start:
{
lean_object* v___x_667_; lean_object* v_env_668_; uint8_t v___x_669_; 
v___x_667_ = lean_st_ref_get(v___y_665_);
v_env_668_ = lean_ctor_get(v___x_667_, 0);
lean_inc_ref(v_env_668_);
lean_dec(v___x_667_);
v___x_669_ = l_Lean_Name_isAnonymous(v_declHint_664_);
if (v___x_669_ == 0)
{
uint8_t v_isExporting_670_; 
v_isExporting_670_ = lean_ctor_get_uint8(v_env_668_, sizeof(void*)*8);
if (v_isExporting_670_ == 0)
{
lean_object* v___x_671_; 
lean_dec_ref(v_env_668_);
lean_dec(v_declHint_664_);
v___x_671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_671_, 0, v_msg_663_);
return v___x_671_;
}
else
{
lean_object* v___x_672_; uint8_t v___x_673_; 
lean_inc_ref(v_env_668_);
v___x_672_ = l_Lean_Environment_setExporting(v_env_668_, v___x_669_);
lean_inc(v_declHint_664_);
lean_inc_ref(v___x_672_);
v___x_673_ = l_Lean_Environment_contains(v___x_672_, v_declHint_664_, v_isExporting_670_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; 
lean_dec_ref(v___x_672_);
lean_dec_ref(v_env_668_);
lean_dec(v_declHint_664_);
v___x_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_674_, 0, v_msg_663_);
return v___x_674_;
}
else
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v_c_680_; lean_object* v___x_681_; 
v___x_675_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__2);
v___x_676_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__5);
v___x_677_ = l_Lean_Options_empty;
v___x_678_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_678_, 0, v___x_672_);
lean_ctor_set(v___x_678_, 1, v___x_675_);
lean_ctor_set(v___x_678_, 2, v___x_676_);
lean_ctor_set(v___x_678_, 3, v___x_677_);
lean_inc(v_declHint_664_);
v___x_679_ = l_Lean_MessageData_ofConstName(v_declHint_664_, v___x_669_);
v_c_680_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_680_, 0, v___x_678_);
lean_ctor_set(v_c_680_, 1, v___x_679_);
v___x_681_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_668_, v_declHint_664_);
if (lean_obj_tag(v___x_681_) == 0)
{
lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; 
lean_dec_ref(v_env_668_);
lean_dec(v_declHint_664_);
v___x_682_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7);
v___x_683_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_683_, 0, v___x_682_);
lean_ctor_set(v___x_683_, 1, v_c_680_);
v___x_684_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__9);
v___x_685_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_683_);
lean_ctor_set(v___x_685_, 1, v___x_684_);
v___x_686_ = l_Lean_MessageData_note(v___x_685_);
v___x_687_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_687_, 0, v_msg_663_);
lean_ctor_set(v___x_687_, 1, v___x_686_);
v___x_688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_688_, 0, v___x_687_);
return v___x_688_;
}
else
{
lean_object* v_val_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_724_; 
v_val_689_ = lean_ctor_get(v___x_681_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_681_);
if (v_isSharedCheck_724_ == 0)
{
v___x_691_ = v___x_681_;
v_isShared_692_ = v_isSharedCheck_724_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_val_689_);
lean_dec(v___x_681_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_724_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v_mod_696_; uint8_t v___x_697_; 
v___x_693_ = lean_box(0);
v___x_694_ = l_Lean_Environment_header(v_env_668_);
lean_dec_ref(v_env_668_);
v___x_695_ = l_Lean_EnvironmentHeader_moduleNames(v___x_694_);
v_mod_696_ = lean_array_get(v___x_693_, v___x_695_, v_val_689_);
lean_dec(v_val_689_);
lean_dec_ref(v___x_695_);
v___x_697_ = l_Lean_isPrivateName(v_declHint_664_);
lean_dec(v_declHint_664_);
if (v___x_697_ == 0)
{
lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_709_; 
v___x_698_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__11);
v___x_699_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_698_);
lean_ctor_set(v___x_699_, 1, v_c_680_);
v___x_700_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__13);
v___x_701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_701_, 0, v___x_699_);
lean_ctor_set(v___x_701_, 1, v___x_700_);
v___x_702_ = l_Lean_MessageData_ofName(v_mod_696_);
v___x_703_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_703_, 0, v___x_701_);
lean_ctor_set(v___x_703_, 1, v___x_702_);
v___x_704_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__15);
v___x_705_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_703_);
lean_ctor_set(v___x_705_, 1, v___x_704_);
v___x_706_ = l_Lean_MessageData_note(v___x_705_);
v___x_707_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_707_, 0, v_msg_663_);
lean_ctor_set(v___x_707_, 1, v___x_706_);
if (v_isShared_692_ == 0)
{
lean_ctor_set_tag(v___x_691_, 0);
lean_ctor_set(v___x_691_, 0, v___x_707_);
v___x_709_ = v___x_691_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v___x_707_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
else
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_722_; 
v___x_711_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__7);
v___x_712_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_712_, 0, v___x_711_);
lean_ctor_set(v___x_712_, 1, v_c_680_);
v___x_713_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__17);
v___x_714_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_714_, 0, v___x_712_);
lean_ctor_set(v___x_714_, 1, v___x_713_);
v___x_715_ = l_Lean_MessageData_ofName(v_mod_696_);
v___x_716_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_716_, 0, v___x_714_);
lean_ctor_set(v___x_716_, 1, v___x_715_);
v___x_717_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__19);
v___x_718_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_716_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = l_Lean_MessageData_note(v___x_718_);
v___x_720_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_720_, 0, v_msg_663_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
if (v_isShared_692_ == 0)
{
lean_ctor_set_tag(v___x_691_, 0);
lean_ctor_set(v___x_691_, 0, v___x_720_);
v___x_722_ = v___x_691_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v___x_720_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_725_; 
lean_dec_ref(v_env_668_);
lean_dec(v_declHint_664_);
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v_msg_663_);
return v___x_725_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___boxed(lean_object* v_msg_726_, lean_object* v_declHint_727_, lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg(v_msg_726_, v_declHint_727_, v___y_728_);
lean_dec(v___y_728_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12(lean_object* v_msg_731_, lean_object* v_declHint_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v___x_740_; lean_object* v_a_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_750_; 
v___x_740_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg(v_msg_731_, v_declHint_732_, v___y_738_);
v_a_741_ = lean_ctor_get(v___x_740_, 0);
v_isSharedCheck_750_ = !lean_is_exclusive(v___x_740_);
if (v_isSharedCheck_750_ == 0)
{
v___x_743_ = v___x_740_;
v_isShared_744_ = v_isSharedCheck_750_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_a_741_);
lean_dec(v___x_740_);
v___x_743_ = lean_box(0);
v_isShared_744_ = v_isSharedCheck_750_;
goto v_resetjp_742_;
}
v_resetjp_742_:
{
lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_748_; 
v___x_745_ = l_Lean_unknownIdentifierMessageTag;
v___x_746_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_746_, 0, v___x_745_);
lean_ctor_set(v___x_746_, 1, v_a_741_);
if (v_isShared_744_ == 0)
{
lean_ctor_set(v___x_743_, 0, v___x_746_);
v___x_748_ = v___x_743_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_749_; 
v_reuseFailAlloc_749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_749_, 0, v___x_746_);
v___x_748_ = v_reuseFailAlloc_749_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
return v___x_748_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12___boxed(lean_object* v_msg_751_, lean_object* v_declHint_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12(v_msg_751_, v_declHint_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
return v_res_760_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0(void){
_start:
{
lean_object* v___x_761_; lean_object* v___x_762_; 
v___x_761_ = lean_box(1);
v___x_762_ = l_Lean_MessageData_ofFormat(v___x_761_);
return v___x_762_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3(void){
_start:
{
lean_object* v___x_766_; lean_object* v___x_767_; 
v___x_766_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__2));
v___x_767_ = l_Lean_MessageData_ofFormat(v___x_766_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17(lean_object* v_x_768_, lean_object* v_x_769_){
_start:
{
if (lean_obj_tag(v_x_769_) == 0)
{
return v_x_768_;
}
else
{
lean_object* v_head_770_; lean_object* v_tail_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_793_; 
v_head_770_ = lean_ctor_get(v_x_769_, 0);
v_tail_771_ = lean_ctor_get(v_x_769_, 1);
v_isSharedCheck_793_ = !lean_is_exclusive(v_x_769_);
if (v_isSharedCheck_793_ == 0)
{
v___x_773_ = v_x_769_;
v_isShared_774_ = v_isSharedCheck_793_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_tail_771_);
lean_inc(v_head_770_);
lean_dec(v_x_769_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_793_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v_before_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_791_; 
v_before_775_ = lean_ctor_get(v_head_770_, 0);
v_isSharedCheck_791_ = !lean_is_exclusive(v_head_770_);
if (v_isSharedCheck_791_ == 0)
{
lean_object* v_unused_792_; 
v_unused_792_ = lean_ctor_get(v_head_770_, 1);
lean_dec(v_unused_792_);
v___x_777_ = v_head_770_;
v_isShared_778_ = v_isSharedCheck_791_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_before_775_);
lean_dec(v_head_770_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_791_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v___x_779_; lean_object* v___x_781_; 
v___x_779_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0);
if (v_isShared_778_ == 0)
{
lean_ctor_set_tag(v___x_777_, 7);
lean_ctor_set(v___x_777_, 1, v___x_779_);
lean_ctor_set(v___x_777_, 0, v_x_768_);
v___x_781_ = v___x_777_;
goto v_reusejp_780_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v_x_768_);
lean_ctor_set(v_reuseFailAlloc_790_, 1, v___x_779_);
v___x_781_ = v_reuseFailAlloc_790_;
goto v_reusejp_780_;
}
v_reusejp_780_:
{
lean_object* v___x_782_; lean_object* v___x_784_; 
v___x_782_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__3);
if (v_isShared_774_ == 0)
{
lean_ctor_set_tag(v___x_773_, 7);
lean_ctor_set(v___x_773_, 1, v___x_782_);
lean_ctor_set(v___x_773_, 0, v___x_781_);
v___x_784_ = v___x_773_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v___x_781_);
lean_ctor_set(v_reuseFailAlloc_789_, 1, v___x_782_);
v___x_784_ = v_reuseFailAlloc_789_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; 
v___x_785_ = l_Lean_MessageData_ofSyntax(v_before_775_);
v___x_786_ = l_Lean_indentD(v___x_785_);
v___x_787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_787_, 0, v___x_784_);
lean_ctor_set(v___x_787_, 1, v___x_786_);
v_x_768_ = v___x_787_;
v_x_769_ = v_tail_771_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2(void){
_start:
{
lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_797_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__1));
v___x_798_ = l_Lean_MessageData_ofFormat(v___x_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg(lean_object* v_msgData_799_, lean_object* v_macroStack_800_, lean_object* v___y_801_){
_start:
{
lean_object* v_options_803_; lean_object* v___x_804_; uint8_t v___x_805_; 
v_options_803_ = lean_ctor_get(v___y_801_, 2);
v___x_804_ = l_Lean_Elab_pp_macroStack;
v___x_805_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__3(v_options_803_, v___x_804_);
if (v___x_805_ == 0)
{
lean_object* v___x_806_; 
lean_dec(v_macroStack_800_);
v___x_806_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_806_, 0, v_msgData_799_);
return v___x_806_;
}
else
{
if (lean_obj_tag(v_macroStack_800_) == 0)
{
lean_object* v___x_807_; 
v___x_807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_807_, 0, v_msgData_799_);
return v___x_807_;
}
else
{
lean_object* v_head_808_; lean_object* v_after_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_824_; 
v_head_808_ = lean_ctor_get(v_macroStack_800_, 0);
lean_inc(v_head_808_);
v_after_809_ = lean_ctor_get(v_head_808_, 1);
v_isSharedCheck_824_ = !lean_is_exclusive(v_head_808_);
if (v_isSharedCheck_824_ == 0)
{
lean_object* v_unused_825_; 
v_unused_825_ = lean_ctor_get(v_head_808_, 0);
lean_dec(v_unused_825_);
v___x_811_ = v_head_808_;
v_isShared_812_ = v_isSharedCheck_824_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_after_809_);
lean_dec(v_head_808_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_824_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_813_; lean_object* v___x_815_; 
v___x_813_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17___closed__0);
if (v_isShared_812_ == 0)
{
lean_ctor_set_tag(v___x_811_, 7);
lean_ctor_set(v___x_811_, 1, v___x_813_);
lean_ctor_set(v___x_811_, 0, v_msgData_799_);
v___x_815_ = v___x_811_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_msgData_799_);
lean_ctor_set(v_reuseFailAlloc_823_, 1, v___x_813_);
v___x_815_ = v_reuseFailAlloc_823_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v_msgData_820_; lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_816_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___closed__2);
v___x_817_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_817_, 0, v___x_815_);
lean_ctor_set(v___x_817_, 1, v___x_816_);
v___x_818_ = l_Lean_MessageData_ofSyntax(v_after_809_);
v___x_819_ = l_Lean_indentD(v___x_818_);
v_msgData_820_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_820_, 0, v___x_817_);
lean_ctor_set(v_msgData_820_, 1, v___x_819_);
v___x_821_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16_spec__17(v_msgData_820_, v_macroStack_800_);
v___x_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_822_, 0, v___x_821_);
return v___x_822_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg___boxed(lean_object* v_msgData_826_, lean_object* v_macroStack_827_, lean_object* v___y_828_, lean_object* v___y_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg(v_msgData_826_, v_macroStack_827_, v___y_828_);
lean_dec_ref(v___y_828_);
return v_res_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg(lean_object* v_msg_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v_ref_839_; lean_object* v___x_840_; lean_object* v_a_841_; lean_object* v_macroStack_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_853_; 
v_ref_839_ = lean_ctor_get(v___y_836_, 5);
v___x_840_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1_spec__2(v_msg_831_, v___y_834_, v___y_835_, v___y_836_, v___y_837_);
v_a_841_ = lean_ctor_get(v___x_840_, 0);
lean_inc(v_a_841_);
lean_dec_ref(v___x_840_);
v_macroStack_842_ = lean_ctor_get(v___y_832_, 1);
v___x_843_ = l_Lean_Elab_getBetterRef(v_ref_839_, v_macroStack_842_);
lean_inc(v_macroStack_842_);
v___x_844_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg(v_a_841_, v_macroStack_842_, v___y_836_);
v_a_845_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_853_ == 0)
{
v___x_847_ = v___x_844_;
v_isShared_848_ = v_isSharedCheck_853_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_844_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_853_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
lean_object* v___x_849_; lean_object* v___x_851_; 
v___x_849_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_849_, 0, v___x_843_);
lean_ctor_set(v___x_849_, 1, v_a_845_);
if (v_isShared_848_ == 0)
{
lean_ctor_set_tag(v___x_847_, 1);
lean_ctor_set(v___x_847_, 0, v___x_849_);
v___x_851_ = v___x_847_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v___x_849_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg___boxed(lean_object* v_msg_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_){
_start:
{
lean_object* v_res_862_; 
v_res_862_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg(v_msg_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_, v___y_860_);
lean_dec(v___y_860_);
lean_dec_ref(v___y_859_);
lean_dec(v___y_858_);
lean_dec_ref(v___y_857_);
lean_dec(v___y_856_);
lean_dec_ref(v___y_855_);
return v_res_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg(lean_object* v_ref_863_, lean_object* v_msg_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_){
_start:
{
lean_object* v_fileName_872_; lean_object* v_fileMap_873_; lean_object* v_options_874_; lean_object* v_currRecDepth_875_; lean_object* v_maxRecDepth_876_; lean_object* v_ref_877_; lean_object* v_currNamespace_878_; lean_object* v_openDecls_879_; lean_object* v_initHeartbeats_880_; lean_object* v_maxHeartbeats_881_; lean_object* v_quotContext_882_; lean_object* v_currMacroScope_883_; uint8_t v_diag_884_; lean_object* v_cancelTk_x3f_885_; uint8_t v_suppressElabErrors_886_; lean_object* v_inheritedTraceOptions_887_; lean_object* v_ref_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v_fileName_872_ = lean_ctor_get(v___y_869_, 0);
v_fileMap_873_ = lean_ctor_get(v___y_869_, 1);
v_options_874_ = lean_ctor_get(v___y_869_, 2);
v_currRecDepth_875_ = lean_ctor_get(v___y_869_, 3);
v_maxRecDepth_876_ = lean_ctor_get(v___y_869_, 4);
v_ref_877_ = lean_ctor_get(v___y_869_, 5);
v_currNamespace_878_ = lean_ctor_get(v___y_869_, 6);
v_openDecls_879_ = lean_ctor_get(v___y_869_, 7);
v_initHeartbeats_880_ = lean_ctor_get(v___y_869_, 8);
v_maxHeartbeats_881_ = lean_ctor_get(v___y_869_, 9);
v_quotContext_882_ = lean_ctor_get(v___y_869_, 10);
v_currMacroScope_883_ = lean_ctor_get(v___y_869_, 11);
v_diag_884_ = lean_ctor_get_uint8(v___y_869_, sizeof(void*)*14);
v_cancelTk_x3f_885_ = lean_ctor_get(v___y_869_, 12);
v_suppressElabErrors_886_ = lean_ctor_get_uint8(v___y_869_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_887_ = lean_ctor_get(v___y_869_, 13);
v_ref_888_ = l_Lean_replaceRef(v_ref_863_, v_ref_877_);
lean_inc_ref(v_inheritedTraceOptions_887_);
lean_inc(v_cancelTk_x3f_885_);
lean_inc(v_currMacroScope_883_);
lean_inc(v_quotContext_882_);
lean_inc(v_maxHeartbeats_881_);
lean_inc(v_initHeartbeats_880_);
lean_inc(v_openDecls_879_);
lean_inc(v_currNamespace_878_);
lean_inc(v_maxRecDepth_876_);
lean_inc(v_currRecDepth_875_);
lean_inc_ref(v_options_874_);
lean_inc_ref(v_fileMap_873_);
lean_inc_ref(v_fileName_872_);
v___x_889_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_889_, 0, v_fileName_872_);
lean_ctor_set(v___x_889_, 1, v_fileMap_873_);
lean_ctor_set(v___x_889_, 2, v_options_874_);
lean_ctor_set(v___x_889_, 3, v_currRecDepth_875_);
lean_ctor_set(v___x_889_, 4, v_maxRecDepth_876_);
lean_ctor_set(v___x_889_, 5, v_ref_888_);
lean_ctor_set(v___x_889_, 6, v_currNamespace_878_);
lean_ctor_set(v___x_889_, 7, v_openDecls_879_);
lean_ctor_set(v___x_889_, 8, v_initHeartbeats_880_);
lean_ctor_set(v___x_889_, 9, v_maxHeartbeats_881_);
lean_ctor_set(v___x_889_, 10, v_quotContext_882_);
lean_ctor_set(v___x_889_, 11, v_currMacroScope_883_);
lean_ctor_set(v___x_889_, 12, v_cancelTk_x3f_885_);
lean_ctor_set(v___x_889_, 13, v_inheritedTraceOptions_887_);
lean_ctor_set_uint8(v___x_889_, sizeof(void*)*14, v_diag_884_);
lean_ctor_set_uint8(v___x_889_, sizeof(void*)*14 + 1, v_suppressElabErrors_886_);
v___x_890_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg(v_msg_864_, v___y_865_, v___y_866_, v___y_867_, v___y_868_, v___x_889_, v___y_870_);
lean_dec_ref_known(v___x_889_, 14);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg___boxed(lean_object* v_ref_891_, lean_object* v_msg_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg(v_ref_891_, v_msg_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
lean_dec(v___y_894_);
lean_dec_ref(v___y_893_);
lean_dec(v_ref_891_);
return v_res_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg(lean_object* v_ref_901_, lean_object* v_msg_902_, lean_object* v_declHint_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v___x_911_; lean_object* v_a_912_; lean_object* v___x_913_; 
v___x_911_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12(v_msg_902_, v_declHint_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_);
v_a_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_a_912_);
lean_dec_ref(v___x_911_);
v___x_913_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg(v_ref_901_, v_a_912_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg___boxed(lean_object* v_ref_914_, lean_object* v_msg_915_, lean_object* v_declHint_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
lean_object* v_res_924_; 
v_res_924_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg(v_ref_914_, v_msg_915_, v_declHint_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_);
lean_dec(v___y_922_);
lean_dec_ref(v___y_921_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
lean_dec(v_ref_914_);
return v_res_924_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1(void){
_start:
{
lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_926_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__0));
v___x_927_ = l_Lean_stringToMessageData(v___x_926_);
return v___x_927_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3(void){
_start:
{
lean_object* v___x_929_; lean_object* v___x_930_; 
v___x_929_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__2));
v___x_930_ = l_Lean_stringToMessageData(v___x_929_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg(lean_object* v_ref_931_, lean_object* v_constName_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_){
_start:
{
lean_object* v___x_940_; uint8_t v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; 
v___x_940_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__1);
v___x_941_ = 0;
lean_inc(v_constName_932_);
v___x_942_ = l_Lean_MessageData_ofConstName(v_constName_932_, v___x_941_);
v___x_943_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_943_, 0, v___x_940_);
lean_ctor_set(v___x_943_, 1, v___x_942_);
v___x_944_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___closed__3);
v___x_945_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_945_, 0, v___x_943_);
lean_ctor_set(v___x_945_, 1, v___x_944_);
v___x_946_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg(v_ref_931_, v___x_945_, v_constName_932_, v___y_933_, v___y_934_, v___y_935_, v___y_936_, v___y_937_, v___y_938_);
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg___boxed(lean_object* v_ref_947_, lean_object* v_constName_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_){
_start:
{
lean_object* v_res_956_; 
v_res_956_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg(v_ref_947_, v_constName_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_);
lean_dec(v___y_954_);
lean_dec_ref(v___y_953_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
lean_dec(v_ref_947_);
return v_res_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg(lean_object* v_constName_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v_ref_965_; lean_object* v___x_966_; 
v_ref_965_ = lean_ctor_get(v___y_962_, 5);
v___x_966_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg(v_ref_965_, v_constName_957_, v___y_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_, v___y_963_);
return v___x_966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg___boxed(lean_object* v_constName_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
lean_object* v_res_975_; 
v_res_975_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg(v_constName_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3(lean_object* v_constName_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
lean_object* v___x_984_; lean_object* v_env_985_; uint8_t v___x_986_; lean_object* v___x_987_; 
v___x_984_ = lean_st_ref_get(v___y_982_);
v_env_985_ = lean_ctor_get(v___x_984_, 0);
lean_inc_ref(v_env_985_);
lean_dec(v___x_984_);
v___x_986_ = 0;
lean_inc(v_constName_976_);
v___x_987_ = l_Lean_Environment_find_x3f(v_env_985_, v_constName_976_, v___x_986_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v___x_988_; 
v___x_988_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg(v_constName_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
return v___x_988_;
}
else
{
lean_object* v_val_989_; lean_object* v___x_991_; uint8_t v_isShared_992_; uint8_t v_isSharedCheck_996_; 
lean_dec(v_constName_976_);
v_val_989_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_996_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_996_ == 0)
{
v___x_991_ = v___x_987_;
v_isShared_992_ = v_isSharedCheck_996_;
goto v_resetjp_990_;
}
else
{
lean_inc(v_val_989_);
lean_dec(v___x_987_);
v___x_991_ = lean_box(0);
v_isShared_992_ = v_isSharedCheck_996_;
goto v_resetjp_990_;
}
v_resetjp_990_:
{
lean_object* v___x_994_; 
if (v_isShared_992_ == 0)
{
lean_ctor_set_tag(v___x_991_, 0);
v___x_994_ = v___x_991_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v_val_989_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3___boxed(lean_object* v_constName_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_){
_start:
{
lean_object* v_res_1005_; 
v_res_1005_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3(v_constName_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_, v___y_1003_);
lean_dec(v___y_1003_);
lean_dec_ref(v___y_1002_);
lean_dec(v___y_1001_);
lean_dec_ref(v___y_1000_);
lean_dec(v___y_999_);
lean_dec_ref(v___y_998_);
return v_res_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg(lean_object* v_t_1006_, lean_object* v___y_1007_){
_start:
{
lean_object* v___x_1009_; lean_object* v_infoState_1010_; uint8_t v_enabled_1011_; 
v___x_1009_ = lean_st_ref_get(v___y_1007_);
v_infoState_1010_ = lean_ctor_get(v___x_1009_, 7);
lean_inc_ref(v_infoState_1010_);
lean_dec(v___x_1009_);
v_enabled_1011_ = lean_ctor_get_uint8(v_infoState_1010_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1010_);
if (v_enabled_1011_ == 0)
{
lean_object* v___x_1012_; lean_object* v___x_1013_; 
lean_dec_ref(v_t_1006_);
v___x_1012_ = lean_box(0);
v___x_1013_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1013_, 0, v___x_1012_);
return v___x_1013_;
}
else
{
lean_object* v___x_1014_; lean_object* v_infoState_1015_; lean_object* v_env_1016_; lean_object* v_nextMacroScope_1017_; lean_object* v_ngen_1018_; lean_object* v_auxDeclNGen_1019_; lean_object* v_traceState_1020_; lean_object* v_cache_1021_; lean_object* v_messages_1022_; lean_object* v_snapshotTasks_1023_; lean_object* v___x_1025_; uint8_t v_isShared_1026_; uint8_t v_isSharedCheck_1045_; 
v___x_1014_ = lean_st_ref_take(v___y_1007_);
v_infoState_1015_ = lean_ctor_get(v___x_1014_, 7);
v_env_1016_ = lean_ctor_get(v___x_1014_, 0);
v_nextMacroScope_1017_ = lean_ctor_get(v___x_1014_, 1);
v_ngen_1018_ = lean_ctor_get(v___x_1014_, 2);
v_auxDeclNGen_1019_ = lean_ctor_get(v___x_1014_, 3);
v_traceState_1020_ = lean_ctor_get(v___x_1014_, 4);
v_cache_1021_ = lean_ctor_get(v___x_1014_, 5);
v_messages_1022_ = lean_ctor_get(v___x_1014_, 6);
v_snapshotTasks_1023_ = lean_ctor_get(v___x_1014_, 8);
v_isSharedCheck_1045_ = !lean_is_exclusive(v___x_1014_);
if (v_isSharedCheck_1045_ == 0)
{
v___x_1025_ = v___x_1014_;
v_isShared_1026_ = v_isSharedCheck_1045_;
goto v_resetjp_1024_;
}
else
{
lean_inc(v_snapshotTasks_1023_);
lean_inc(v_infoState_1015_);
lean_inc(v_messages_1022_);
lean_inc(v_cache_1021_);
lean_inc(v_traceState_1020_);
lean_inc(v_auxDeclNGen_1019_);
lean_inc(v_ngen_1018_);
lean_inc(v_nextMacroScope_1017_);
lean_inc(v_env_1016_);
lean_dec(v___x_1014_);
v___x_1025_ = lean_box(0);
v_isShared_1026_ = v_isSharedCheck_1045_;
goto v_resetjp_1024_;
}
v_resetjp_1024_:
{
uint8_t v_enabled_1027_; lean_object* v_assignment_1028_; lean_object* v_lazyAssignment_1029_; lean_object* v_trees_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1044_; 
v_enabled_1027_ = lean_ctor_get_uint8(v_infoState_1015_, sizeof(void*)*3);
v_assignment_1028_ = lean_ctor_get(v_infoState_1015_, 0);
v_lazyAssignment_1029_ = lean_ctor_get(v_infoState_1015_, 1);
v_trees_1030_ = lean_ctor_get(v_infoState_1015_, 2);
v_isSharedCheck_1044_ = !lean_is_exclusive(v_infoState_1015_);
if (v_isSharedCheck_1044_ == 0)
{
v___x_1032_ = v_infoState_1015_;
v_isShared_1033_ = v_isSharedCheck_1044_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_trees_1030_);
lean_inc(v_lazyAssignment_1029_);
lean_inc(v_assignment_1028_);
lean_dec(v_infoState_1015_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1044_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1034_; lean_object* v___x_1036_; 
v___x_1034_ = l_Lean_PersistentArray_push___redArg(v_trees_1030_, v_t_1006_);
if (v_isShared_1033_ == 0)
{
lean_ctor_set(v___x_1032_, 2, v___x_1034_);
v___x_1036_ = v___x_1032_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v_assignment_1028_);
lean_ctor_set(v_reuseFailAlloc_1043_, 1, v_lazyAssignment_1029_);
lean_ctor_set(v_reuseFailAlloc_1043_, 2, v___x_1034_);
lean_ctor_set_uint8(v_reuseFailAlloc_1043_, sizeof(void*)*3, v_enabled_1027_);
v___x_1036_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
lean_object* v___x_1038_; 
if (v_isShared_1026_ == 0)
{
lean_ctor_set(v___x_1025_, 7, v___x_1036_);
v___x_1038_ = v___x_1025_;
goto v_reusejp_1037_;
}
else
{
lean_object* v_reuseFailAlloc_1042_; 
v_reuseFailAlloc_1042_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1042_, 0, v_env_1016_);
lean_ctor_set(v_reuseFailAlloc_1042_, 1, v_nextMacroScope_1017_);
lean_ctor_set(v_reuseFailAlloc_1042_, 2, v_ngen_1018_);
lean_ctor_set(v_reuseFailAlloc_1042_, 3, v_auxDeclNGen_1019_);
lean_ctor_set(v_reuseFailAlloc_1042_, 4, v_traceState_1020_);
lean_ctor_set(v_reuseFailAlloc_1042_, 5, v_cache_1021_);
lean_ctor_set(v_reuseFailAlloc_1042_, 6, v_messages_1022_);
lean_ctor_set(v_reuseFailAlloc_1042_, 7, v___x_1036_);
lean_ctor_set(v_reuseFailAlloc_1042_, 8, v_snapshotTasks_1023_);
v___x_1038_ = v_reuseFailAlloc_1042_;
goto v_reusejp_1037_;
}
v_reusejp_1037_:
{
lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1039_ = lean_st_ref_set(v___y_1007_, v___x_1038_);
v___x_1040_ = lean_box(0);
v___x_1041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
return v___x_1041_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_t_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_){
_start:
{
lean_object* v_res_1049_; 
v_res_1049_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg(v_t_1046_, v___y_1047_);
lean_dec(v___y_1047_);
return v_res_1049_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1050_ = lean_unsigned_to_nat(32u);
v___x_1051_ = lean_mk_empty_array_with_capacity(v___x_1050_);
v___x_1052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1051_);
return v___x_1052_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1(void){
_start:
{
size_t v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1053_ = ((size_t)5ULL);
v___x_1054_ = lean_unsigned_to_nat(0u);
v___x_1055_ = lean_unsigned_to_nat(32u);
v___x_1056_ = lean_mk_empty_array_with_capacity(v___x_1055_);
v___x_1057_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__0);
v___x_1058_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
lean_ctor_set(v___x_1058_, 1, v___x_1056_);
lean_ctor_set(v___x_1058_, 2, v___x_1054_);
lean_ctor_set(v___x_1058_, 3, v___x_1054_);
lean_ctor_set_usize(v___x_1058_, 4, v___x_1053_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3(lean_object* v_t_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v___x_1067_; lean_object* v_infoState_1068_; uint8_t v_enabled_1069_; 
v___x_1067_ = lean_st_ref_get(v___y_1065_);
v_infoState_1068_ = lean_ctor_get(v___x_1067_, 7);
lean_inc_ref(v_infoState_1068_);
lean_dec(v___x_1067_);
v_enabled_1069_ = lean_ctor_get_uint8(v_infoState_1068_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1068_);
if (v_enabled_1069_ == 0)
{
lean_object* v___x_1070_; lean_object* v___x_1071_; 
lean_dec_ref(v_t_1059_);
v___x_1070_ = lean_box(0);
v___x_1071_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1071_, 0, v___x_1070_);
return v___x_1071_;
}
else
{
lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___x_1072_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___closed__1);
v___x_1073_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1073_, 0, v_t_1059_);
lean_ctor_set(v___x_1073_, 1, v___x_1072_);
v___x_1074_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg(v___x_1073_, v___y_1065_);
return v___x_1074_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3___boxed(lean_object* v_t_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3(v_t_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
return v_res_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2(lean_object* v_info_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v___x_1092_; lean_object* v___x_1093_; 
v___x_1092_ = lean_alloc_ctor(8, 1, 0);
lean_ctor_set(v___x_1092_, 0, v_info_1084_);
v___x_1093_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3(v___x_1092_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2___boxed(lean_object* v_info_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_){
_start:
{
lean_object* v_res_1102_; 
v_res_1102_ = lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2(v_info_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_);
lean_dec(v___y_1100_);
lean_dec_ref(v___y_1099_);
lean_dec(v___y_1098_);
lean_dec_ref(v___y_1097_);
lean_dec(v___y_1096_);
lean_dec_ref(v___y_1095_);
return v_res_1102_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_1103_; 
v___x_1103_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1103_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__0);
v___x_1105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1104_);
return v___x_1105_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
v___x_1106_ = lean_box(1);
v___x_1107_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg___closed__4);
v___x_1108_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__1);
v___x_1109_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1109_, 0, v___x_1108_);
lean_ctor_set(v___x_1109_, 1, v___x_1107_);
lean_ctor_set(v___x_1109_, 2, v___x_1106_);
return v___x_1109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg(lean_object* v_tk_1115_, lean_object* v_id_1116_, lean_object* v___x_1117_, uint8_t v_showImplicit_1118_, lean_object* v_as_x27_1119_, lean_object* v_b_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_){
_start:
{
if (lean_obj_tag(v_as_x27_1119_) == 0)
{
lean_object* v___x_1128_; 
lean_dec(v___x_1117_);
v___x_1128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1128_, 0, v_b_1120_);
return v___x_1128_;
}
else
{
lean_object* v_head_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; uint8_t v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v_____do__lift_1137_; lean_object* v___y_1138_; lean_object* v___y_1139_; lean_object* v___y_1140_; lean_object* v___y_1141_; lean_object* v___y_1142_; lean_object* v___y_1143_; 
lean_dec_ref(v_b_1120_);
v_head_1129_ = lean_ctor_get(v_as_x27_1119_, 0);
v___x_1130_ = lean_box(0);
v___x_1131_ = l_Lean_TSyntax_getId(v_id_1116_);
v___x_1132_ = 0;
v___x_1133_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__2);
v___x_1134_ = lean_alloc_ctor(1, 4, 1);
lean_ctor_set(v___x_1134_, 0, v___x_1117_);
lean_ctor_set(v___x_1134_, 1, v___x_1131_);
lean_ctor_set(v___x_1134_, 2, v___x_1133_);
lean_ctor_set(v___x_1134_, 3, v___x_1130_);
lean_ctor_set_uint8(v___x_1134_, sizeof(void*)*4, v___x_1132_);
v___x_1135_ = lp_mathlib_Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2(v___x_1134_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_);
lean_dec_ref(v___x_1135_);
if (v_showImplicit_1118_ == 0)
{
lean_object* v___x_1162_; 
lean_inc(v_head_1129_);
v___x_1162_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3(v_head_1129_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_);
if (lean_obj_tag(v___x_1162_) == 0)
{
lean_object* v_a_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; 
v_a_1163_ = lean_ctor_get(v___x_1162_, 0);
lean_inc(v_a_1163_);
lean_dec_ref_known(v___x_1162_, 1);
lean_inc(v_head_1129_);
v___x_1164_ = l_Lean_MessageData_ofConstName(v_head_1129_, v___x_1132_);
v___x_1165_ = l_Lean_ConstantInfo_type(v_a_1163_);
lean_dec(v_a_1163_);
v___x_1166_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_delabSignatureWithoutImplicit(v___x_1165_);
v___x_1167_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1164_);
lean_ctor_set(v___x_1167_, 1, v___x_1166_);
v_____do__lift_1137_ = v___x_1167_;
v___y_1138_ = v___y_1121_;
v___y_1139_ = v___y_1122_;
v___y_1140_ = v___y_1123_;
v___y_1141_ = v___y_1124_;
v___y_1142_ = v___y_1125_;
v___y_1143_ = v___y_1126_;
goto v___jp_1136_;
}
else
{
lean_object* v_a_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1175_; 
v_a_1168_ = lean_ctor_get(v___x_1162_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1162_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1170_ = v___x_1162_;
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_a_1168_);
lean_dec(v___x_1162_);
v___x_1170_ = lean_box(0);
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
v_resetjp_1169_:
{
lean_object* v___x_1173_; 
if (v_isShared_1171_ == 0)
{
v___x_1173_ = v___x_1170_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v_a_1168_);
v___x_1173_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
return v___x_1173_;
}
}
}
}
else
{
lean_object* v___x_1176_; 
lean_inc(v_head_1129_);
v___x_1176_ = l_Lean_MessageData_signature(v_head_1129_);
v_____do__lift_1137_ = v___x_1176_;
v___y_1138_ = v___y_1121_;
v___y_1139_ = v___y_1122_;
v___y_1140_ = v___y_1123_;
v___y_1141_ = v___y_1124_;
v___y_1142_ = v___y_1125_;
v___y_1143_ = v___y_1126_;
goto v___jp_1136_;
}
v___jp_1136_:
{
lean_object* v___x_1144_; 
v___x_1144_ = lp_mathlib_Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1(v_tk_1115_, v_____do__lift_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
if (lean_obj_tag(v___x_1144_) == 0)
{
lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1152_; 
v_isSharedCheck_1152_ = !lean_is_exclusive(v___x_1144_);
if (v_isSharedCheck_1152_ == 0)
{
lean_object* v_unused_1153_; 
v_unused_1153_ = lean_ctor_get(v___x_1144_, 0);
lean_dec(v_unused_1153_);
v___x_1146_ = v___x_1144_;
v_isShared_1147_ = v_isSharedCheck_1152_;
goto v_resetjp_1145_;
}
else
{
lean_dec(v___x_1144_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1152_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v___x_1148_; lean_object* v___x_1150_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___closed__4));
if (v_isShared_1147_ == 0)
{
lean_ctor_set(v___x_1146_, 0, v___x_1148_);
v___x_1150_ = v___x_1146_;
goto v_reusejp_1149_;
}
else
{
lean_object* v_reuseFailAlloc_1151_; 
v_reuseFailAlloc_1151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1151_, 0, v___x_1148_);
v___x_1150_ = v_reuseFailAlloc_1151_;
goto v_reusejp_1149_;
}
v_reusejp_1149_:
{
return v___x_1150_;
}
}
}
else
{
lean_object* v_a_1154_; lean_object* v___x_1156_; uint8_t v_isShared_1157_; uint8_t v_isSharedCheck_1161_; 
v_a_1154_ = lean_ctor_get(v___x_1144_, 0);
v_isSharedCheck_1161_ = !lean_is_exclusive(v___x_1144_);
if (v_isSharedCheck_1161_ == 0)
{
v___x_1156_ = v___x_1144_;
v_isShared_1157_ = v_isSharedCheck_1161_;
goto v_resetjp_1155_;
}
else
{
lean_inc(v_a_1154_);
lean_dec(v___x_1144_);
v___x_1156_ = lean_box(0);
v_isShared_1157_ = v_isSharedCheck_1161_;
goto v_resetjp_1155_;
}
v_resetjp_1155_:
{
lean_object* v___x_1159_; 
if (v_isShared_1157_ == 0)
{
v___x_1159_ = v___x_1156_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1160_; 
v_reuseFailAlloc_1160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1160_, 0, v_a_1154_);
v___x_1159_ = v_reuseFailAlloc_1160_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
return v___x_1159_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg___boxed(lean_object* v_tk_1177_, lean_object* v_id_1178_, lean_object* v___x_1179_, lean_object* v_showImplicit_1180_, lean_object* v_as_x27_1181_, lean_object* v_b_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
uint8_t v_showImplicit_boxed_1190_; lean_object* v_res_1191_; 
v_showImplicit_boxed_1190_ = lean_unbox(v_showImplicit_1180_);
v_res_1191_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg(v_tk_1177_, v_id_1178_, v___x_1179_, v_showImplicit_boxed_1190_, v_as_x27_1181_, v_b_1182_, v___y_1183_, v___y_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
lean_dec(v___y_1186_);
lean_dec_ref(v___y_1185_);
lean_dec(v___y_1184_);
lean_dec_ref(v___y_1183_);
lean_dec(v_as_x27_1181_);
lean_dec(v_id_1178_);
lean_dec(v_tk_1177_);
return v_res_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2(uint8_t v___x_1195_, lean_object* v___f_1196_, lean_object* v_term_1197_, lean_object* v_tk_1198_, uint8_t v_showImplicit_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_){
_start:
{
lean_object* v___y_1208_; uint8_t v___y_1209_; lean_object* v_a_1214_; 
if (v___x_1195_ == 0)
{
lean_object* v___x_1217_; lean_object* v___x_1218_; 
lean_dec(v_term_1197_);
v___x_1217_ = lean_box(0);
lean_inc(v___y_1205_);
lean_inc_ref(v___y_1204_);
lean_inc(v___y_1203_);
lean_inc_ref(v___y_1202_);
lean_inc(v___y_1201_);
lean_inc_ref(v___y_1200_);
v___x_1218_ = lean_apply_8(v___f_1196_, v___x_1217_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, lean_box(0));
return v___x_1218_;
}
else
{
lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1219_ = lean_box(0);
lean_inc(v_term_1197_);
v___x_1220_ = l_Lean_Elab_realizeGlobalConstWithInfos(v_term_1197_, v___x_1219_, v___y_1204_, v___y_1205_);
if (lean_obj_tag(v___x_1220_) == 0)
{
lean_object* v_a_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; 
v_a_1221_ = lean_ctor_get(v___x_1220_, 0);
lean_inc(v_a_1221_);
lean_dec_ref_known(v___x_1220_, 1);
v___x_1222_ = lean_box(0);
v___x_1223_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___closed__0));
lean_inc(v_term_1197_);
v___x_1224_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg(v_tk_1198_, v_term_1197_, v_term_1197_, v_showImplicit_1199_, v_a_1221_, v___x_1223_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v_a_1221_);
lean_dec(v_term_1197_);
if (lean_obj_tag(v___x_1224_) == 0)
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1235_; 
v_a_1225_ = lean_ctor_get(v___x_1224_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1224_);
if (v_isSharedCheck_1235_ == 0)
{
v___x_1227_ = v___x_1224_;
v_isShared_1228_ = v_isSharedCheck_1235_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1224_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1235_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v_fst_1229_; 
v_fst_1229_ = lean_ctor_get(v_a_1225_, 0);
lean_inc(v_fst_1229_);
lean_dec(v_a_1225_);
if (lean_obj_tag(v_fst_1229_) == 0)
{
lean_object* v___x_1230_; 
lean_del_object(v___x_1227_);
lean_inc(v___y_1205_);
lean_inc_ref(v___y_1204_);
lean_inc(v___y_1203_);
lean_inc_ref(v___y_1202_);
lean_inc(v___y_1201_);
lean_inc_ref(v___y_1200_);
v___x_1230_ = lean_apply_8(v___f_1196_, v___x_1222_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, lean_box(0));
return v___x_1230_;
}
else
{
lean_object* v_val_1231_; lean_object* v___x_1233_; 
lean_dec_ref(v___f_1196_);
v_val_1231_ = lean_ctor_get(v_fst_1229_, 0);
lean_inc(v_val_1231_);
lean_dec_ref_known(v_fst_1229_, 1);
if (v_isShared_1228_ == 0)
{
lean_ctor_set(v___x_1227_, 0, v_val_1231_);
v___x_1233_ = v___x_1227_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v_val_1231_);
v___x_1233_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1232_;
}
v_reusejp_1232_:
{
return v___x_1233_;
}
}
}
}
else
{
lean_object* v_a_1236_; 
v_a_1236_ = lean_ctor_get(v___x_1224_, 0);
lean_inc(v_a_1236_);
lean_dec_ref_known(v___x_1224_, 1);
v_a_1214_ = v_a_1236_;
goto v___jp_1213_;
}
}
else
{
lean_object* v_a_1237_; 
lean_dec(v_term_1197_);
v_a_1237_ = lean_ctor_get(v___x_1220_, 0);
lean_inc(v_a_1237_);
lean_dec_ref_known(v___x_1220_, 1);
v_a_1214_ = v_a_1237_;
goto v___jp_1213_;
}
}
v___jp_1207_:
{
if (v___y_1209_ == 0)
{
lean_object* v___x_1210_; lean_object* v___x_1211_; 
lean_dec_ref(v___y_1208_);
v___x_1210_ = lean_box(0);
lean_inc(v___y_1205_);
lean_inc_ref(v___y_1204_);
lean_inc(v___y_1203_);
lean_inc_ref(v___y_1202_);
lean_inc(v___y_1201_);
lean_inc_ref(v___y_1200_);
v___x_1211_ = lean_apply_8(v___f_1196_, v___x_1210_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, lean_box(0));
return v___x_1211_;
}
else
{
lean_object* v___x_1212_; 
lean_dec_ref(v___f_1196_);
v___x_1212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1212_, 0, v___y_1208_);
return v___x_1212_;
}
}
v___jp_1213_:
{
uint8_t v___x_1215_; 
v___x_1215_ = l_Lean_Exception_isInterrupt(v_a_1214_);
if (v___x_1215_ == 0)
{
uint8_t v___x_1216_; 
lean_inc_ref(v_a_1214_);
v___x_1216_ = l_Lean_Exception_isRuntime(v_a_1214_);
v___y_1208_ = v_a_1214_;
v___y_1209_ = v___x_1216_;
goto v___jp_1207_;
}
else
{
v___y_1208_ = v_a_1214_;
v___y_1209_ = v___x_1215_;
goto v___jp_1207_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___boxed(lean_object* v___x_1238_, lean_object* v___f_1239_, lean_object* v_term_1240_, lean_object* v_tk_1241_, lean_object* v_showImplicit_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_){
_start:
{
uint8_t v___x_17634__boxed_1250_; uint8_t v_showImplicit_boxed_1251_; lean_object* v_res_1252_; 
v___x_17634__boxed_1250_ = lean_unbox(v___x_1238_);
v_showImplicit_boxed_1251_ = lean_unbox(v_showImplicit_1242_);
v_res_1252_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2(v___x_17634__boxed_1250_, v___f_1239_, v_term_1240_, v_tk_1241_, v_showImplicit_boxed_1251_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_);
lean_dec(v___y_1248_);
lean_dec_ref(v___y_1247_);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec(v_tk_1241_);
return v_res_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux(lean_object* v_tk_1260_, lean_object* v_term_1261_, uint8_t v_showImplicit_1262_, lean_object* v_a_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_, lean_object* v_a_1268_){
_start:
{
lean_object* v___f_1270_; lean_object* v___f_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; uint8_t v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___y_1277_; lean_object* v___x_1278_; 
v___f_1270_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__0));
lean_inc(v_tk_1260_);
lean_inc_n(v_term_1261_, 2);
v___f_1271_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__1___boxed), 11, 3);
lean_closure_set(v___f_1271_, 0, v_term_1261_);
lean_closure_set(v___f_1271_, 1, v_tk_1260_);
lean_closure_set(v___f_1271_, 2, v___f_1270_);
v___x_1272_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__2));
v___x_1273_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___closed__4));
v___x_1274_ = l_Lean_Syntax_isOfKind(v_term_1261_, v___x_1273_);
v___x_1275_ = lean_box(v___x_1274_);
v___x_1276_ = lean_box(v_showImplicit_1262_);
v___y_1277_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___lam__2___boxed), 12, 5);
lean_closure_set(v___y_1277_, 0, v___x_1275_);
lean_closure_set(v___y_1277_, 1, v___f_1271_);
lean_closure_set(v___y_1277_, 2, v_term_1261_);
lean_closure_set(v___y_1277_, 3, v_tk_1260_);
lean_closure_set(v___y_1277_, 4, v___x_1276_);
v___x_1278_ = l_Lean_Elab_Term_withDeclName___redArg(v___x_1272_, v___y_1277_, v_a_1263_, v_a_1264_, v_a_1265_, v_a_1266_, v_a_1267_, v_a_1268_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux___boxed(lean_object* v_tk_1279_, lean_object* v_term_1280_, lean_object* v_showImplicit_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_, lean_object* v_a_1285_, lean_object* v_a_1286_, lean_object* v_a_1287_, lean_object* v_a_1288_){
_start:
{
uint8_t v_showImplicit_boxed_1289_; lean_object* v_res_1290_; 
v_showImplicit_boxed_1289_ = lean_unbox(v_showImplicit_1281_);
v_res_1290_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux(v_tk_1279_, v_term_1280_, v_showImplicit_boxed_1289_, v_a_1282_, v_a_1283_, v_a_1284_, v_a_1285_, v_a_1286_, v_a_1287_);
lean_dec(v_a_1287_);
lean_dec_ref(v_a_1286_);
lean_dec(v_a_1285_);
lean_dec_ref(v_a_1284_);
lean_dec(v_a_1283_);
lean_dec_ref(v_a_1282_);
return v_res_1290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4(lean_object* v_tk_1291_, lean_object* v_id_1292_, lean_object* v___x_1293_, uint8_t v_showImplicit_1294_, lean_object* v_as_1295_, lean_object* v_as_x27_1296_, lean_object* v_b_1297_, lean_object* v_a_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___redArg(v_tk_1291_, v_id_1292_, v___x_1293_, v_showImplicit_1294_, v_as_x27_1296_, v_b_1297_, v___y_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4___boxed(lean_object* v_tk_1307_, lean_object* v_id_1308_, lean_object* v___x_1309_, lean_object* v_showImplicit_1310_, lean_object* v_as_1311_, lean_object* v_as_x27_1312_, lean_object* v_b_1313_, lean_object* v_a_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
uint8_t v_showImplicit_boxed_1322_; lean_object* v_res_1323_; 
v_showImplicit_boxed_1322_ = lean_unbox(v_showImplicit_1310_);
v_res_1323_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__4(v_tk_1307_, v_id_1308_, v___x_1309_, v_showImplicit_boxed_1322_, v_as_1311_, v_as_x27_1312_, v_b_1313_, v_a_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
lean_dec(v_as_x27_1312_);
lean_dec(v_as_1311_);
lean_dec(v_id_1308_);
lean_dec(v_tk_1307_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1(lean_object* v_ref_1324_, lean_object* v_msgData_1325_, uint8_t v_severity_1326_, uint8_t v_isSilent_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
lean_object* v___x_1335_; 
v___x_1335_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___redArg(v_ref_1324_, v_msgData_1325_, v_severity_1326_, v_isSilent_1327_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_);
return v___x_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1___boxed(lean_object* v_ref_1336_, lean_object* v_msgData_1337_, lean_object* v_severity_1338_, lean_object* v_isSilent_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_){
_start:
{
uint8_t v_severity_boxed_1347_; uint8_t v_isSilent_boxed_1348_; lean_object* v_res_1349_; 
v_severity_boxed_1347_ = lean_unbox(v_severity_1338_);
v_isSilent_boxed_1348_ = lean_unbox(v_isSilent_1339_);
v_res_1349_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__1_spec__1(v_ref_1336_, v_msgData_1337_, v_severity_boxed_1347_, v_isSilent_boxed_1348_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_, v___y_1344_, v___y_1345_);
lean_dec(v___y_1345_);
lean_dec_ref(v___y_1344_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec(v_ref_1336_);
return v_res_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6(lean_object* v_t_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_){
_start:
{
lean_object* v___x_1358_; 
v___x_1358_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___redArg(v_t_1350_, v___y_1356_);
return v___x_1358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6___boxed(lean_object* v_t_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_){
_start:
{
lean_object* v_res_1367_; 
v_res_1367_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__2_spec__3_spec__6(v_t_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_);
lean_dec(v___y_1365_);
lean_dec_ref(v___y_1364_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
lean_dec(v___y_1361_);
lean_dec_ref(v___y_1360_);
return v_res_1367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5(lean_object* v_00_u03b1_1368_, lean_object* v_constName_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_){
_start:
{
lean_object* v___x_1377_; 
v___x_1377_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___redArg(v_constName_1369_, v___y_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_);
return v___x_1377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5___boxed(lean_object* v_00_u03b1_1378_, lean_object* v_constName_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_){
_start:
{
lean_object* v_res_1387_; 
v_res_1387_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5(v_00_u03b1_1378_, v_constName_1379_, v___y_1380_, v___y_1381_, v___y_1382_, v___y_1383_, v___y_1384_, v___y_1385_);
lean_dec(v___y_1385_);
lean_dec_ref(v___y_1384_);
lean_dec(v___y_1383_);
lean_dec_ref(v___y_1382_);
lean_dec(v___y_1381_);
lean_dec_ref(v___y_1380_);
return v_res_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9(lean_object* v_00_u03b1_1388_, lean_object* v_ref_1389_, lean_object* v_constName_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_){
_start:
{
lean_object* v___x_1398_; 
v___x_1398_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___redArg(v_ref_1389_, v_constName_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_, v___y_1396_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9___boxed(lean_object* v_00_u03b1_1399_, lean_object* v_ref_1400_, lean_object* v_constName_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_){
_start:
{
lean_object* v_res_1409_; 
v_res_1409_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9(v_00_u03b1_1399_, v_ref_1400_, v_constName_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
lean_dec(v___y_1407_);
lean_dec_ref(v___y_1406_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec(v_ref_1400_);
return v_res_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11(lean_object* v_00_u03b1_1410_, lean_object* v_ref_1411_, lean_object* v_msg_1412_, lean_object* v_declHint_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_){
_start:
{
lean_object* v___x_1421_; 
v___x_1421_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___redArg(v_ref_1411_, v_msg_1412_, v_declHint_1413_, v___y_1414_, v___y_1415_, v___y_1416_, v___y_1417_, v___y_1418_, v___y_1419_);
return v___x_1421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11___boxed(lean_object* v_00_u03b1_1422_, lean_object* v_ref_1423_, lean_object* v_msg_1424_, lean_object* v_declHint_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_){
_start:
{
lean_object* v_res_1433_; 
v_res_1433_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11(v_00_u03b1_1422_, v_ref_1423_, v_msg_1424_, v_declHint_1425_, v___y_1426_, v___y_1427_, v___y_1428_, v___y_1429_, v___y_1430_, v___y_1431_);
lean_dec(v___y_1431_);
lean_dec_ref(v___y_1430_);
lean_dec(v___y_1429_);
lean_dec_ref(v___y_1428_);
lean_dec(v___y_1427_);
lean_dec_ref(v___y_1426_);
lean_dec(v_ref_1423_);
return v_res_1433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13(lean_object* v_msg_1434_, lean_object* v_declHint_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_){
_start:
{
lean_object* v___x_1443_; 
v___x_1443_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___redArg(v_msg_1434_, v_declHint_1435_, v___y_1441_);
return v___x_1443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13___boxed(lean_object* v_msg_1444_, lean_object* v_declHint_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_){
_start:
{
lean_object* v_res_1453_; 
v_res_1453_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__12_spec__13(v_msg_1444_, v_declHint_1445_, v___y_1446_, v___y_1447_, v___y_1448_, v___y_1449_, v___y_1450_, v___y_1451_);
lean_dec(v___y_1451_);
lean_dec_ref(v___y_1450_);
lean_dec(v___y_1449_);
lean_dec_ref(v___y_1448_);
lean_dec(v___y_1447_);
lean_dec_ref(v___y_1446_);
return v_res_1453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13(lean_object* v_00_u03b1_1454_, lean_object* v_ref_1455_, lean_object* v_msg_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_){
_start:
{
lean_object* v___x_1464_; 
v___x_1464_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___redArg(v_ref_1455_, v_msg_1456_, v___y_1457_, v___y_1458_, v___y_1459_, v___y_1460_, v___y_1461_, v___y_1462_);
return v___x_1464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13___boxed(lean_object* v_00_u03b1_1465_, lean_object* v_ref_1466_, lean_object* v_msg_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
lean_object* v_res_1475_; 
v_res_1475_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13(v_00_u03b1_1465_, v_ref_1466_, v_msg_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
lean_dec(v___y_1471_);
lean_dec_ref(v___y_1470_);
lean_dec(v___y_1469_);
lean_dec_ref(v___y_1468_);
lean_dec(v_ref_1466_);
return v_res_1475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15(lean_object* v_00_u03b1_1476_, lean_object* v_msg_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_){
_start:
{
lean_object* v___x_1485_; 
v___x_1485_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___redArg(v_msg_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_, v___y_1482_, v___y_1483_);
return v___x_1485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15___boxed(lean_object* v_00_u03b1_1486_, lean_object* v_msg_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_){
_start:
{
lean_object* v_res_1495_; 
v_res_1495_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15(v_00_u03b1_1486_, v_msg_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_);
lean_dec(v___y_1493_);
lean_dec_ref(v___y_1492_);
lean_dec(v___y_1491_);
lean_dec_ref(v___y_1490_);
lean_dec(v___y_1489_);
lean_dec_ref(v___y_1488_);
return v_res_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16(lean_object* v_msgData_1496_, lean_object* v_macroStack_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_){
_start:
{
lean_object* v___x_1505_; 
v___x_1505_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___redArg(v_msgData_1496_, v_macroStack_1497_, v___y_1502_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16___boxed(lean_object* v_msgData_1506_, lean_object* v_macroStack_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_){
_start:
{
lean_object* v_res_1515_; 
v_res_1515_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux_spec__3_spec__5_spec__9_spec__11_spec__13_spec__15_spec__16(v_msgData_1506_, v_macroStack_1507_, v___y_1508_, v___y_1509_, v___y_1510_, v___y_1511_, v___y_1512_, v___y_1513_);
lean_dec(v___y_1513_);
lean_dec_ref(v___y_1512_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec(v___y_1509_);
lean_dec_ref(v___y_1508_);
return v_res_1515_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; 
v___x_1543_ = lean_box(0);
v___x_1544_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1545_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1545_, 0, v___x_1544_);
lean_ctor_set(v___x_1545_, 1, v___x_1543_);
return v___x_1545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1547_; lean_object* v___x_1548_; 
v___x_1547_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0);
v___x_1548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1548_, 0, v___x_1547_);
return v___x_1548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___boxed(lean_object* v___y_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg();
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0(lean_object* v_00_u03b1_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_){
_start:
{
lean_object* v___x_1555_; 
v___x_1555_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg();
return v___x_1555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___boxed(lean_object* v_00_u03b1_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_){
_start:
{
lean_object* v_res_1560_; 
v_res_1560_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0(v_00_u03b1_1556_, v___y_1557_, v___y_1558_);
lean_dec(v___y_1558_);
lean_dec_ref(v___y_1557_);
return v_res_1560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0(lean_object* v_tk_1561_, lean_object* v_t_1562_, lean_object* v_x_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_){
_start:
{
uint8_t v___x_1571_; lean_object* v___x_1572_; 
v___x_1571_ = 0;
v___x_1572_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux(v_tk_1561_, v_t_1562_, v___x_1571_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_);
return v___x_1572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0___boxed(lean_object* v_tk_1573_, lean_object* v_t_1574_, lean_object* v_x_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_){
_start:
{
lean_object* v_res_1583_; 
v_res_1583_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0(v_tk_1573_, v_t_1574_, v_x_1575_, v___y_1576_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_);
lean_dec(v___y_1581_);
lean_dec_ref(v___y_1580_);
lean_dec(v___y_1579_);
lean_dec_ref(v___y_1578_);
lean_dec(v___y_1577_);
lean_dec_ref(v___y_1576_);
lean_dec_ref(v_x_1575_);
return v_res_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(lean_object* v_env_1584_, lean_object* v___y_1585_){
_start:
{
lean_object* v___x_1587_; lean_object* v_messages_1588_; lean_object* v_scopes_1589_; lean_object* v_usedQuotCtxts_1590_; lean_object* v_nextMacroScope_1591_; lean_object* v_maxRecDepth_1592_; lean_object* v_ngen_1593_; lean_object* v_auxDeclNGen_1594_; lean_object* v_infoState_1595_; lean_object* v_traceState_1596_; lean_object* v_snapshotTasks_1597_; lean_object* v_prevLinterStates_1598_; lean_object* v___x_1600_; uint8_t v_isShared_1601_; uint8_t v_isSharedCheck_1608_; 
v___x_1587_ = lean_st_ref_take(v___y_1585_);
v_messages_1588_ = lean_ctor_get(v___x_1587_, 1);
v_scopes_1589_ = lean_ctor_get(v___x_1587_, 2);
v_usedQuotCtxts_1590_ = lean_ctor_get(v___x_1587_, 3);
v_nextMacroScope_1591_ = lean_ctor_get(v___x_1587_, 4);
v_maxRecDepth_1592_ = lean_ctor_get(v___x_1587_, 5);
v_ngen_1593_ = lean_ctor_get(v___x_1587_, 6);
v_auxDeclNGen_1594_ = lean_ctor_get(v___x_1587_, 7);
v_infoState_1595_ = lean_ctor_get(v___x_1587_, 8);
v_traceState_1596_ = lean_ctor_get(v___x_1587_, 9);
v_snapshotTasks_1597_ = lean_ctor_get(v___x_1587_, 10);
v_prevLinterStates_1598_ = lean_ctor_get(v___x_1587_, 11);
v_isSharedCheck_1608_ = !lean_is_exclusive(v___x_1587_);
if (v_isSharedCheck_1608_ == 0)
{
lean_object* v_unused_1609_; 
v_unused_1609_ = lean_ctor_get(v___x_1587_, 0);
lean_dec(v_unused_1609_);
v___x_1600_ = v___x_1587_;
v_isShared_1601_ = v_isSharedCheck_1608_;
goto v_resetjp_1599_;
}
else
{
lean_inc(v_prevLinterStates_1598_);
lean_inc(v_snapshotTasks_1597_);
lean_inc(v_traceState_1596_);
lean_inc(v_infoState_1595_);
lean_inc(v_auxDeclNGen_1594_);
lean_inc(v_ngen_1593_);
lean_inc(v_maxRecDepth_1592_);
lean_inc(v_nextMacroScope_1591_);
lean_inc(v_usedQuotCtxts_1590_);
lean_inc(v_scopes_1589_);
lean_inc(v_messages_1588_);
lean_dec(v___x_1587_);
v___x_1600_ = lean_box(0);
v_isShared_1601_ = v_isSharedCheck_1608_;
goto v_resetjp_1599_;
}
v_resetjp_1599_:
{
lean_object* v___x_1603_; 
if (v_isShared_1601_ == 0)
{
lean_ctor_set(v___x_1600_, 0, v_env_1584_);
v___x_1603_ = v___x_1600_;
goto v_reusejp_1602_;
}
else
{
lean_object* v_reuseFailAlloc_1607_; 
v_reuseFailAlloc_1607_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1607_, 0, v_env_1584_);
lean_ctor_set(v_reuseFailAlloc_1607_, 1, v_messages_1588_);
lean_ctor_set(v_reuseFailAlloc_1607_, 2, v_scopes_1589_);
lean_ctor_set(v_reuseFailAlloc_1607_, 3, v_usedQuotCtxts_1590_);
lean_ctor_set(v_reuseFailAlloc_1607_, 4, v_nextMacroScope_1591_);
lean_ctor_set(v_reuseFailAlloc_1607_, 5, v_maxRecDepth_1592_);
lean_ctor_set(v_reuseFailAlloc_1607_, 6, v_ngen_1593_);
lean_ctor_set(v_reuseFailAlloc_1607_, 7, v_auxDeclNGen_1594_);
lean_ctor_set(v_reuseFailAlloc_1607_, 8, v_infoState_1595_);
lean_ctor_set(v_reuseFailAlloc_1607_, 9, v_traceState_1596_);
lean_ctor_set(v_reuseFailAlloc_1607_, 10, v_snapshotTasks_1597_);
lean_ctor_set(v_reuseFailAlloc_1607_, 11, v_prevLinterStates_1598_);
v___x_1603_ = v_reuseFailAlloc_1607_;
goto v_reusejp_1602_;
}
v_reusejp_1602_:
{
lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; 
v___x_1604_ = lean_st_ref_set(v___y_1585_, v___x_1603_);
v___x_1605_ = lean_box(0);
v___x_1606_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1606_, 0, v___x_1605_);
return v___x_1606_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg___boxed(lean_object* v_env_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
lean_object* v_res_1613_; 
v_res_1613_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(v_env_1610_, v___y_1611_);
lean_dec(v___y_1611_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg(lean_object* v_env_1614_, lean_object* v_x_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_){
_start:
{
lean_object* v___x_1619_; lean_object* v_env_1620_; lean_object* v_a_1622_; lean_object* v___x_1632_; lean_object* v___x_1633_; 
v___x_1619_ = lean_st_ref_get(v___y_1617_);
v_env_1620_ = lean_ctor_get(v___x_1619_, 0);
lean_inc_ref(v_env_1620_);
lean_dec(v___x_1619_);
v___x_1632_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(v_env_1614_, v___y_1617_);
lean_dec_ref(v___x_1632_);
lean_inc(v___y_1617_);
lean_inc_ref(v___y_1616_);
v___x_1633_ = lean_apply_3(v_x_1615_, v___y_1616_, v___y_1617_, lean_box(0));
if (lean_obj_tag(v___x_1633_) == 0)
{
lean_object* v_a_1634_; lean_object* v___x_1635_; lean_object* v___x_1637_; uint8_t v_isShared_1638_; uint8_t v_isSharedCheck_1642_; 
v_a_1634_ = lean_ctor_get(v___x_1633_, 0);
lean_inc(v_a_1634_);
lean_dec_ref_known(v___x_1633_, 1);
v___x_1635_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(v_env_1620_, v___y_1617_);
v_isSharedCheck_1642_ = !lean_is_exclusive(v___x_1635_);
if (v_isSharedCheck_1642_ == 0)
{
lean_object* v_unused_1643_; 
v_unused_1643_ = lean_ctor_get(v___x_1635_, 0);
lean_dec(v_unused_1643_);
v___x_1637_ = v___x_1635_;
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
else
{
lean_dec(v___x_1635_);
v___x_1637_ = lean_box(0);
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
v_resetjp_1636_:
{
lean_object* v___x_1640_; 
if (v_isShared_1638_ == 0)
{
lean_ctor_set(v___x_1637_, 0, v_a_1634_);
v___x_1640_ = v___x_1637_;
goto v_reusejp_1639_;
}
else
{
lean_object* v_reuseFailAlloc_1641_; 
v_reuseFailAlloc_1641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1641_, 0, v_a_1634_);
v___x_1640_ = v_reuseFailAlloc_1641_;
goto v_reusejp_1639_;
}
v_reusejp_1639_:
{
return v___x_1640_;
}
}
}
else
{
lean_object* v_a_1644_; 
v_a_1644_ = lean_ctor_get(v___x_1633_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1633_, 1);
v_a_1622_ = v_a_1644_;
goto v___jp_1621_;
}
v___jp_1621_:
{
lean_object* v___x_1623_; lean_object* v___x_1625_; uint8_t v_isShared_1626_; uint8_t v_isSharedCheck_1630_; 
v___x_1623_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(v_env_1620_, v___y_1617_);
v_isSharedCheck_1630_ = !lean_is_exclusive(v___x_1623_);
if (v_isSharedCheck_1630_ == 0)
{
lean_object* v_unused_1631_; 
v_unused_1631_ = lean_ctor_get(v___x_1623_, 0);
lean_dec(v_unused_1631_);
v___x_1625_ = v___x_1623_;
v_isShared_1626_ = v_isSharedCheck_1630_;
goto v_resetjp_1624_;
}
else
{
lean_dec(v___x_1623_);
v___x_1625_ = lean_box(0);
v_isShared_1626_ = v_isSharedCheck_1630_;
goto v_resetjp_1624_;
}
v_resetjp_1624_:
{
lean_object* v___x_1628_; 
if (v_isShared_1626_ == 0)
{
lean_ctor_set_tag(v___x_1625_, 1);
lean_ctor_set(v___x_1625_, 0, v_a_1622_);
v___x_1628_ = v___x_1625_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v_a_1622_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg___boxed(lean_object* v_env_1645_, lean_object* v_x_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_){
_start:
{
lean_object* v_res_1650_; 
v_res_1650_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg(v_env_1645_, v_x_1646_, v___y_1647_, v___y_1648_);
lean_dec(v___y_1648_);
lean_dec_ref(v___y_1647_);
return v_res_1650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1(lean_object* v_x_1651_, lean_object* v_a_1652_, lean_object* v_a_1653_){
_start:
{
lean_object* v___x_1655_; uint8_t v___x_1656_; 
v___x_1655_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_command_x23check_x27___00__closed__2));
lean_inc(v_x_1651_);
v___x_1656_ = l_Lean_Syntax_isOfKind(v_x_1651_, v___x_1655_);
if (v___x_1656_ == 0)
{
lean_object* v___x_1657_; 
lean_dec(v_x_1651_);
v___x_1657_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg();
return v___x_1657_;
}
else
{
lean_object* v___x_1658_; lean_object* v_env_1659_; lean_object* v___x_1660_; lean_object* v_tk_1661_; lean_object* v___x_1662_; lean_object* v_t_1663_; lean_object* v___f_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; 
v___x_1658_ = lean_st_ref_get(v_a_1653_);
v_env_1659_ = lean_ctor_get(v___x_1658_, 0);
lean_inc_ref(v_env_1659_);
lean_dec(v___x_1658_);
v___x_1660_ = lean_unsigned_to_nat(0u);
v_tk_1661_ = l_Lean_Syntax_getArg(v_x_1651_, v___x_1660_);
v___x_1662_ = lean_unsigned_to_nat(1u);
v_t_1663_ = l_Lean_Syntax_getArg(v_x_1651_, v___x_1662_);
lean_dec(v_x_1651_);
v___f_1664_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___lam__0___boxed), 10, 2);
lean_closure_set(v___f_1664_, 0, v_tk_1661_);
lean_closure_set(v___f_1664_, 1, v_t_1663_);
v___x_1665_ = lean_alloc_closure((void*)(l_Lean_Elab_Command_runTermElabM___boxed), 5, 2);
lean_closure_set(v___x_1665_, 0, lean_box(0));
lean_closure_set(v___x_1665_, 1, v___f_1664_);
v___x_1666_ = l_Lean_Environment_unlockAsync(v_env_1659_);
v___x_1667_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg(v___x_1666_, v___x_1665_, v_a_1652_, v_a_1653_);
return v___x_1667_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1___boxed(lean_object* v_x_1668_, lean_object* v_a_1669_, lean_object* v_a_1670_, lean_object* v_a_1671_){
_start:
{
lean_object* v_res_1672_; 
v_res_1672_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1(v_x_1668_, v_a_1669_, v_a_1670_);
lean_dec(v_a_1670_);
lean_dec_ref(v_a_1669_);
return v_res_1672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1(lean_object* v_env_1673_, lean_object* v___y_1674_, lean_object* v___y_1675_){
_start:
{
lean_object* v___x_1677_; 
v___x_1677_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___redArg(v_env_1673_, v___y_1675_);
return v___x_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1___boxed(lean_object* v_env_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_){
_start:
{
lean_object* v_res_1682_; 
v_res_1682_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1_spec__1(v_env_1678_, v___y_1679_, v___y_1680_);
lean_dec(v___y_1680_);
lean_dec_ref(v___y_1679_);
return v_res_1682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1(lean_object* v_00_u03b1_1683_, lean_object* v_env_1684_, lean_object* v_x_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_){
_start:
{
lean_object* v___x_1689_; 
v___x_1689_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___redArg(v_env_1684_, v_x_1685_, v___y_1686_, v___y_1687_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1___boxed(lean_object* v_00_u03b1_1690_, lean_object* v_env_1691_, lean_object* v_x_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_){
_start:
{
lean_object* v_res_1696_; 
v_res_1696_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__1(v_00_u03b1_1690_, v_env_1691_, v_x_1692_, v___y_1693_, v___y_1694_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
return v_res_1696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg(){
_start:
{
lean_object* v___x_1731_; lean_object* v___x_1732_; 
v___x_1731_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__command_x23check_x27____1_spec__0___redArg___closed__0);
v___x_1732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1732_, 0, v___x_1731_);
return v___x_1732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg___boxed(lean_object* v___y_1733_){
_start:
{
lean_object* v_res_1734_; 
v_res_1734_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg();
return v_res_1734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0(lean_object* v_00_u03b1_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_){
_start:
{
lean_object* v___x_1745_; 
v___x_1745_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg();
return v___x_1745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___boxed(lean_object* v_00_u03b1_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_){
_start:
{
lean_object* v_res_1756_; 
v_res_1756_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0(v_00_u03b1_1746_, v___y_1747_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_);
lean_dec(v___y_1754_);
lean_dec_ref(v___y_1753_);
lean_dec(v___y_1752_);
lean_dec_ref(v___y_1751_);
lean_dec(v___y_1750_);
lean_dec_ref(v___y_1749_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
return v_res_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0(lean_object* v_x_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_){
_start:
{
lean_object* v___x_1767_; 
lean_inc(v___y_1759_);
lean_inc_ref(v___y_1758_);
v___x_1767_ = lean_apply_9(v_x_1757_, v___y_1758_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_, v___y_1765_, lean_box(0));
return v___x_1767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0___boxed(lean_object* v_x_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_){
_start:
{
lean_object* v_res_1778_; 
v_res_1778_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0(v_x_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_);
lean_dec(v___y_1770_);
lean_dec_ref(v___y_1769_);
return v_res_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(lean_object* v_x_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v___f_1789_; lean_object* v___x_1790_; 
lean_inc(v___y_1781_);
lean_inc_ref(v___y_1780_);
v___f_1789_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1789_, 0, v_x_1779_);
lean_closure_set(v___f_1789_, 1, v___y_1780_);
lean_closure_set(v___f_1789_, 2, v___y_1781_);
v___x_1790_ = l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_box(0), v___f_1789_, v___y_1782_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_, v___y_1787_);
if (lean_obj_tag(v___x_1790_) == 0)
{
return v___x_1790_;
}
else
{
lean_object* v_a_1791_; lean_object* v___x_1793_; uint8_t v_isShared_1794_; uint8_t v_isSharedCheck_1798_; 
v_a_1791_ = lean_ctor_get(v___x_1790_, 0);
v_isSharedCheck_1798_ = !lean_is_exclusive(v___x_1790_);
if (v_isSharedCheck_1798_ == 0)
{
v___x_1793_ = v___x_1790_;
v_isShared_1794_ = v_isSharedCheck_1798_;
goto v_resetjp_1792_;
}
else
{
lean_inc(v_a_1791_);
lean_dec(v___x_1790_);
v___x_1793_ = lean_box(0);
v_isShared_1794_ = v_isSharedCheck_1798_;
goto v_resetjp_1792_;
}
v_resetjp_1792_:
{
lean_object* v___x_1796_; 
if (v_isShared_1794_ == 0)
{
v___x_1796_ = v___x_1793_;
goto v_reusejp_1795_;
}
else
{
lean_object* v_reuseFailAlloc_1797_; 
v_reuseFailAlloc_1797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1797_, 0, v_a_1791_);
v___x_1796_ = v_reuseFailAlloc_1797_;
goto v_reusejp_1795_;
}
v_reusejp_1795_:
{
return v___x_1796_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg___boxed(lean_object* v_x_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_){
_start:
{
lean_object* v_res_1809_; 
v_res_1809_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(v_x_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_, v___y_1807_);
lean_dec(v___y_1807_);
lean_dec_ref(v___y_1806_);
lean_dec(v___y_1805_);
lean_dec_ref(v___y_1804_);
lean_dec(v___y_1803_);
lean_dec_ref(v___y_1802_);
lean_dec(v___y_1801_);
lean_dec_ref(v___y_1800_);
return v_res_1809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1(lean_object* v_00_u03b1_1810_, lean_object* v_x_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_){
_start:
{
lean_object* v___x_1821_; 
v___x_1821_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(v_x_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
return v___x_1821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___boxed(lean_object* v_00_u03b1_1822_, lean_object* v_x_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_, lean_object* v___y_1832_){
_start:
{
lean_object* v_res_1833_; 
v_res_1833_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1(v_00_u03b1_1822_, v_x_1823_, v___y_1824_, v___y_1825_, v___y_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_, v___y_1831_);
lean_dec(v___y_1831_);
lean_dec_ref(v___y_1830_);
lean_dec(v___y_1829_);
lean_dec_ref(v___y_1828_);
lean_dec(v___y_1827_);
lean_dec_ref(v___y_1826_);
lean_dec(v___y_1825_);
lean_dec_ref(v___y_1824_);
return v_res_1833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0(lean_object* v_tk_1834_, lean_object* v_term_1835_, uint8_t v___x_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_){
_start:
{
lean_object* v___x_1846_; 
v___x_1846_ = lp_mathlib___private_Mathlib_Tactic_Check_0__Mathlib_Tactic_checkCoreAux(v_tk_1834_, v_term_1835_, v___x_1836_, v___y_1839_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
return v___x_1846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0___boxed(lean_object* v_tk_1847_, lean_object* v_term_1848_, lean_object* v___x_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
uint8_t v___x_1580__boxed_1859_; lean_object* v_res_1860_; 
v___x_1580__boxed_1859_ = lean_unbox(v___x_1849_);
v_res_1860_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0(v_tk_1847_, v_term_1848_, v___x_1580__boxed_1859_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_, v___y_1857_);
lean_dec(v___y_1857_);
lean_dec_ref(v___y_1856_);
lean_dec(v___y_1855_);
lean_dec_ref(v___y_1854_);
lean_dec(v___y_1853_);
lean_dec_ref(v___y_1852_);
lean_dec(v___y_1851_);
lean_dec_ref(v___y_1850_);
return v_res_1860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1(lean_object* v_x_1861_, lean_object* v_a_1862_, lean_object* v_a_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_, lean_object* v_a_1868_, lean_object* v_a_1869_){
_start:
{
lean_object* v___x_1871_; uint8_t v___x_1872_; 
v___x_1871_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tactic_x23check_____00__closed__1));
lean_inc(v_x_1861_);
v___x_1872_ = l_Lean_Syntax_isOfKind(v_x_1861_, v___x_1871_);
if (v___x_1872_ == 0)
{
lean_object* v___x_1873_; 
lean_dec(v_x_1861_);
v___x_1873_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg();
return v___x_1873_;
}
else
{
lean_object* v___x_1874_; lean_object* v_tk_1875_; lean_object* v___x_1876_; lean_object* v_term_1877_; lean_object* v___x_1878_; lean_object* v___f_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1874_ = lean_unsigned_to_nat(0u);
v_tk_1875_ = l_Lean_Syntax_getArg(v_x_1861_, v___x_1874_);
v___x_1876_ = lean_unsigned_to_nat(2u);
v_term_1877_ = l_Lean_Syntax_getArg(v_x_1861_, v___x_1876_);
lean_dec(v_x_1861_);
v___x_1878_ = lean_box(v___x_1872_);
v___f_1879_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_1879_, 0, v_tk_1875_);
lean_closure_set(v___f_1879_, 1, v_term_1877_);
lean_closure_set(v___f_1879_, 2, v___x_1878_);
v___x_1880_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withMainContext___boxed), 11, 2);
lean_closure_set(v___x_1880_, 0, lean_box(0));
lean_closure_set(v___x_1880_, 1, v___f_1879_);
v___x_1881_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(v___x_1880_, v_a_1862_, v_a_1863_, v_a_1864_, v_a_1865_, v_a_1866_, v_a_1867_, v_a_1868_, v_a_1869_);
return v___x_1881_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___boxed(lean_object* v_x_1882_, lean_object* v_a_1883_, lean_object* v_a_1884_, lean_object* v_a_1885_, lean_object* v_a_1886_, lean_object* v_a_1887_, lean_object* v_a_1888_, lean_object* v_a_1889_, lean_object* v_a_1890_, lean_object* v_a_1891_){
_start:
{
lean_object* v_res_1892_; 
v_res_1892_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1(v_x_1882_, v_a_1883_, v_a_1884_, v_a_1885_, v_a_1886_, v_a_1887_, v_a_1888_, v_a_1889_, v_a_1890_);
lean_dec(v_a_1890_);
lean_dec_ref(v_a_1889_);
lean_dec(v_a_1888_);
lean_dec_ref(v_a_1887_);
lean_dec(v_a_1886_);
lean_dec_ref(v_a_1885_);
lean_dec(v_a_1884_);
lean_dec_ref(v_a_1883_);
return v_res_1892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check_x27______1(lean_object* v_x_1914_, lean_object* v_a_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_, lean_object* v_a_1921_, lean_object* v_a_1922_){
_start:
{
lean_object* v___x_1924_; uint8_t v___x_1925_; 
v___x_1924_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tactic_x23check_x27_____00__closed__1));
lean_inc(v_x_1914_);
v___x_1925_ = l_Lean_Syntax_isOfKind(v_x_1914_, v___x_1924_);
if (v___x_1925_ == 0)
{
lean_object* v___x_1926_; 
lean_dec(v_x_1914_);
v___x_1926_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__0___redArg();
return v___x_1926_;
}
else
{
lean_object* v___x_1927_; lean_object* v_tk_1928_; lean_object* v___x_1929_; lean_object* v_term_1930_; uint8_t v___x_1931_; lean_object* v___x_1932_; lean_object* v___f_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; 
v___x_1927_ = lean_unsigned_to_nat(0u);
v_tk_1928_ = l_Lean_Syntax_getArg(v_x_1914_, v___x_1927_);
v___x_1929_ = lean_unsigned_to_nat(2u);
v_term_1930_ = l_Lean_Syntax_getArg(v_x_1914_, v___x_1929_);
lean_dec(v_x_1914_);
v___x_1931_ = 0;
v___x_1932_ = lean_box(v___x_1931_);
v___f_1933_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_1933_, 0, v_tk_1928_);
lean_closure_set(v___f_1933_, 1, v_term_1930_);
lean_closure_set(v___f_1933_, 2, v___x_1932_);
v___x_1934_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withMainContext___boxed), 11, 2);
lean_closure_set(v___x_1934_, 0, lean_box(0));
lean_closure_set(v___x_1934_, 1, v___f_1933_);
v___x_1935_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check______1_spec__1___redArg(v___x_1934_, v_a_1915_, v_a_1916_, v_a_1917_, v_a_1918_, v_a_1919_, v_a_1920_, v_a_1921_, v_a_1922_);
return v___x_1935_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check_x27______1___boxed(lean_object* v_x_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_, lean_object* v_a_1939_, lean_object* v_a_1940_, lean_object* v_a_1941_, lean_object* v_a_1942_, lean_object* v_a_1943_, lean_object* v_a_1944_, lean_object* v_a_1945_){
_start:
{
lean_object* v_res_1946_; 
v_res_1946_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Check______elabRules__Mathlib__Tactic__tactic_x23check_x27______1(v_x_1936_, v_a_1937_, v_a_1938_, v_a_1939_, v_a_1940_, v_a_1941_, v_a_1942_, v_a_1943_, v_a_1944_);
lean_dec(v_a_1944_);
lean_dec_ref(v_a_1943_);
lean_dec(v_a_1942_);
lean_dec_ref(v_a_1941_);
lean_dec(v_a_1940_);
lean_dec_ref(v_a_1939_);
lean_dec(v_a_1938_);
lean_dec_ref(v_a_1937_);
return v_res_1946_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter(uint8_t builtin);
lean_object* initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Check(builtin);
}
#ifdef __cplusplus
}
#endif
