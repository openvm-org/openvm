// Lean compiler output
// Module: Mathlib.Tactic.FBinop
// Imports: public import Init public meta import Init public meta import Lean.Elab.App public meta import Lean.Elab.BuiltinNotation public import Mathlib.Tactic.ToExpr
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
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedParamInfo_default;
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_ParamInfo_isInstImplicit(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Elab_Term_resolveId_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_expandMacroImpl_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_expandMacroImpl_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
lean_object* l_Lean_mkPrivateName(lean_object*, lean_object*);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_privateToUserName(lean_object*);
lean_object* l_Lean_ResolveName_resolveNamespace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Environment_header(lean_object*);
extern lean_object* l_Lean_instInhabitedEffectiveImport_default;
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instHashableExtraModUse_hash___boxed(lean_object*);
lean_object* l_Lean_instBEqExtraModUse_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l___private_Lean_ExtraModUses_0__Lean_extraModUses;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableExtraModUse_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_instBEqExtraModUse_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
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
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
uint8_t l_Lean_isMarkedMeta(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withPushMacroExpansionStack___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getHygieneInfo(lean_object*);
uint8_t l_Lean_Elab_Term_hasCDot(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_Elab_Term_mkCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEqGuarded(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Expr_letBody_x21(lean_object*);
lean_object* l_Lean_Expr_letValue_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Meta_isType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_Term_elabAppArgs(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l___private_Lean_ToExpr_0__Lean_Name_toExprAux(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_coerceSimple_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withTermInfoContext_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withPushMacroExpansionStack___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_ensureHasType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__0_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__0_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__0_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__1_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "fbinop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__1_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__1_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__0_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__1_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(158, 127, 135, 18, 105, 41, 252, 176)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__3_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__3_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__3_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__4_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__3_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__4_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__4_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__5_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__5_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__5_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__6_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__4_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__5_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__6_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__6_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__7_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__7_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__7_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__8_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__6_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__7_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__8_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__8_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__9_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "FBinop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__9_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__9_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__10_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__8_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__9_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(167, 229, 200, 182, 158, 23, 24, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__10_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__10_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__11_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__10_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(26, 155, 127, 134, 101, 60, 20, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__11_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__11_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "FBinopElab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__13_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__11_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 119, 65, 188, 106, 198, 10, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__13_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__13_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__14_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__14_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__14_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__15_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__13_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__14_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(228, 157, 231, 30, 60, 7, 55, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__15_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__15_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__16_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__16_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__16_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__17_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__15_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__16_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(237, 149, 186, 252, 250, 3, 22, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__17_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__17_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__18_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__17_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__5_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(232, 194, 155, 148, 130, 195, 222, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__18_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__18_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__19_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__18_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__7_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(149, 81, 6, 231, 76, 66, 55, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__19_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__19_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__20_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__19_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__9_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(117, 61, 129, 242, 117, 6, 139, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__20_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__20_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__21_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__20_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1378038792) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(175, 7, 67, 143, 183, 227, 35, 40)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__21_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__21_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__22_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__22_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__22_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__23_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__21_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__22_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(164, 92, 15, 103, 177, 218, 80, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__23_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__23_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__24_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__24_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__24_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__25_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__23_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__24_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 156, 182, 74, 54, 93, 125, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__25_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__25_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__26_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__25_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(97, 172, 238, 212, 11, 163, 91, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__26_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__26_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "prodSyntax"};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__0 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__0_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(152, 220, 174, 132, 195, 251, 161, 76)}};
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__1_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(139, 24, 209, 97, 181, 86, 62, 237)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__1 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__1_value;
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__2 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__2_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__3 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value;
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fbinop% "};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__4 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__4_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__4_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__5 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__5_value;
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__6 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__6_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__6_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__7 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__7_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__7_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__8 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__8_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__5_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__8_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__9 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__9_value;
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__10 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__10_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__11 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__11_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__11_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__12 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__12_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__9_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__12_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__13 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__13_value;
static const lean_string_object lp_mathlib_FBinopElab_prodSyntax___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__14 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__14_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__14_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__15 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__15_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__15_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__16 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__16_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__13_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__16_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__17 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__17_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__17_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__12_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__18 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__18_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__3_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__18_value),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__16_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__19 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__19_value;
static const lean_ctor_object lp_mathlib_FBinopElab_prodSyntax___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__19_value)}};
static const lean_object* lp_mathlib_FBinopElab_prodSyntax___closed__20 = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib_FBinopElab_prodSyntax = (const lean_object*)&lp_mathlib_FBinopElab_prodSyntax___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_term_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_term_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_binop_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_binop_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_macroExpansion_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_macroExpansion_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqExtraModUse_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__0_value;
static const lean_closure_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableExtraModUse_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extraModUses"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__7 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__7_value),LEAN_SCALAR_PTR_LITERAL(27, 95, 70, 98, 97, 66, 56, 109)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__8 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__8_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " extra mod use "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__9 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " of "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__11 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__14 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__14_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "recording "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__17 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__19 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__19_value;
static lean_once_cell_t lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "regular"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__21 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__21_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__22 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__22_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__23 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__23_value;
static const lean_string_object lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__24 = (const lean_object*)&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2;
static const lean_array_object lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__3 = (const lean_object*)&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 158, .m_capacity = 158, .m_length = 157, .m_data = "maximum recursion depth has been reached\nuse `set_option maxRecDepth <num>` to increase limit\nuse `set_option diagnostics true` to get diagnostic information"};
static const lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__3_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__7_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__0 = (const lean_object*)&lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__0_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__0_value)}};
static const lean_object* lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__1 = (const lean_object*)&lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_FBinopElab_instInhabitedSRec_default = (const lean_object*)&lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_FBinopElab_instInhabitedSRec = (const lean_object*)&lp_mathlib_FBinopElab_instInhabitedSRec_default___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SRec"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__0 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__1 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__1_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(152, 220, 174, 132, 195, 251, 161, 76)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 158, 96, 57, 68, 41, 30, 114)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 118, 93, 209, 139, 78, 201, 233)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Expr"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__4 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__4_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7_value;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "toArray"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__8 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__8_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__8_value),LEAN_SCALAR_PTR_LITERAL(225, 54, 189, 64, 249, 49, 198, 116)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__12 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__12_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__12_value),LEAN_SCALAR_PTR_LITERAL(90, 150, 134, 113, 145, 38, 173, 251)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15;
static const lean_string_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__16 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__16_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__16_value),LEAN_SCALAR_PTR_LITERAL(98, 170, 59, 223, 79, 132, 139, 119)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_FBinopElab_instToExprSRec___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FBinopElab_instToExprSRec_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec___closed__0 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec___closed__0_value;
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__12_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(152, 220, 174, 132, 195, 251, 161, 76)}};
static const lean_ctor_object lp_mathlib_FBinopElab_instToExprSRec___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec___closed__1_value_aux_0),((lean_object*)&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 158, 96, 57, 68, 41, 30, 114)}};
static const lean_object* lp_mathlib_FBinopElab_instToExprSRec___closed__1 = (const lean_object*)&lp_mathlib_FBinopElab_instToExprSRec___closed__1_value;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec___closed__2;
static lean_once_cell_t lp_mathlib_FBinopElab_instToExprSRec___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FBinopElab_instToExprSRec___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_instToExprSRec;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "v"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 108, 188, 174, 117, 112, 110, 72)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "fromType = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = ", toType = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "uncomparable types: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "added coercion: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " =\?= "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "applying "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "defeq hint "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "mvar apply failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "visiting "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr_spec__0(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "result: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "hasUncomparable: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = ", maxType: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Option"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__6_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(149, 114, 34, 228, 75, 195, 143, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "some"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__6_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__11_value),LEAN_SCALAR_PTR_LITERAL(89, 148, 40, 55, 221, 242, 231, 67)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_elabBinOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_elabBinOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_61_; uint8_t v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_62_ = 0;
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__26_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_64_ = l_Lean_registerTraceClass(v___x_61_, v___x_62_, v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2____boxed(lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_();
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorIdx(lean_object* v_x_118_){
_start:
{
switch(lean_obj_tag(v_x_118_))
{
case 0:
{
lean_object* v___x_119_; 
v___x_119_ = lean_unsigned_to_nat(0u);
return v___x_119_;
}
case 1:
{
lean_object* v___x_120_; 
v___x_120_ = lean_unsigned_to_nat(1u);
return v___x_120_;
}
default: 
{
lean_object* v___x_121_; 
v___x_121_ = lean_unsigned_to_nat(2u);
return v___x_121_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorIdx___boxed(lean_object* v_x_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorIdx(v_x_122_);
lean_dec_ref(v_x_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(lean_object* v_t_124_, lean_object* v_k_125_){
_start:
{
switch(lean_obj_tag(v_t_124_))
{
case 0:
{
lean_object* v_ref_126_; lean_object* v_infoTrees_127_; lean_object* v_val_128_; lean_object* v___x_129_; 
v_ref_126_ = lean_ctor_get(v_t_124_, 0);
lean_inc(v_ref_126_);
v_infoTrees_127_ = lean_ctor_get(v_t_124_, 1);
lean_inc_ref(v_infoTrees_127_);
v_val_128_ = lean_ctor_get(v_t_124_, 2);
lean_inc_ref(v_val_128_);
lean_dec_ref_known(v_t_124_, 3);
v___x_129_ = lean_apply_3(v_k_125_, v_ref_126_, v_infoTrees_127_, v_val_128_);
return v___x_129_;
}
case 1:
{
lean_object* v_ref_130_; lean_object* v_f_131_; lean_object* v_lhs_132_; lean_object* v_rhs_133_; lean_object* v___x_134_; 
v_ref_130_ = lean_ctor_get(v_t_124_, 0);
lean_inc(v_ref_130_);
v_f_131_ = lean_ctor_get(v_t_124_, 1);
lean_inc_ref(v_f_131_);
v_lhs_132_ = lean_ctor_get(v_t_124_, 2);
lean_inc_ref(v_lhs_132_);
v_rhs_133_ = lean_ctor_get(v_t_124_, 3);
lean_inc_ref(v_rhs_133_);
lean_dec_ref_known(v_t_124_, 4);
v___x_134_ = lean_apply_4(v_k_125_, v_ref_130_, v_f_131_, v_lhs_132_, v_rhs_133_);
return v___x_134_;
}
default: 
{
lean_object* v_macroName_135_; lean_object* v_stx_136_; lean_object* v_stx_x27_137_; lean_object* v_nested_138_; lean_object* v___x_139_; 
v_macroName_135_ = lean_ctor_get(v_t_124_, 0);
lean_inc(v_macroName_135_);
v_stx_136_ = lean_ctor_get(v_t_124_, 1);
lean_inc(v_stx_136_);
v_stx_x27_137_ = lean_ctor_get(v_t_124_, 2);
lean_inc(v_stx_x27_137_);
v_nested_138_ = lean_ctor_get(v_t_124_, 3);
lean_inc_ref(v_nested_138_);
lean_dec_ref_known(v_t_124_, 4);
v___x_139_ = lean_apply_4(v_k_125_, v_macroName_135_, v_stx_136_, v_stx_x27_137_, v_nested_138_);
return v___x_139_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim(lean_object* v_motive_140_, lean_object* v_ctorIdx_141_, lean_object* v_t_142_, lean_object* v_h_143_, lean_object* v_k_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_142_, v_k_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___boxed(lean_object* v_motive_146_, lean_object* v_ctorIdx_147_, lean_object* v_t_148_, lean_object* v_h_149_, lean_object* v_k_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim(v_motive_146_, v_ctorIdx_147_, v_t_148_, v_h_149_, v_k_150_);
lean_dec(v_ctorIdx_147_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_term_elim___redArg(lean_object* v_t_152_, lean_object* v_term_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_152_, v_term_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_term_elim(lean_object* v_motive_155_, lean_object* v_t_156_, lean_object* v_h_157_, lean_object* v_term_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_156_, v_term_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_binop_elim___redArg(lean_object* v_t_160_, lean_object* v_binop_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_160_, v_binop_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_binop_elim(lean_object* v_motive_163_, lean_object* v_t_164_, lean_object* v_h_165_, lean_object* v_binop_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_164_, v_binop_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_macroExpansion_elim___redArg(lean_object* v_t_168_, lean_object* v_macroExpansion_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_168_, v_macroExpansion_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_macroExpansion_elim(lean_object* v_motive_171_, lean_object* v_t_172_, lean_object* v_h_173_, lean_object* v_macroExpansion_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_Tree_ctorElim___redArg(v_t_172_, v_macroExpansion_174_);
return v___x_175_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_176_ = lean_unsigned_to_nat(32u);
v___x_177_ = lean_mk_empty_array_with_capacity(v___x_176_);
v___x_178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_179_ = ((size_t)5ULL);
v___x_180_ = lean_unsigned_to_nat(0u);
v___x_181_ = lean_unsigned_to_nat(32u);
v___x_182_ = lean_mk_empty_array_with_capacity(v___x_181_);
v___x_183_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__0);
v___x_184_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_182_);
lean_ctor_set(v___x_184_, 2, v___x_180_);
lean_ctor_set(v___x_184_, 3, v___x_180_);
lean_ctor_set_usize(v___x_184_, 4, v___x_179_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(lean_object* v___y_185_){
_start:
{
lean_object* v___x_187_; lean_object* v_infoState_188_; lean_object* v_trees_189_; lean_object* v___x_190_; lean_object* v_infoState_191_; lean_object* v_env_192_; lean_object* v_nextMacroScope_193_; lean_object* v_ngen_194_; lean_object* v_auxDeclNGen_195_; lean_object* v_traceState_196_; lean_object* v_cache_197_; lean_object* v_messages_198_; lean_object* v_snapshotTasks_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_220_; 
v___x_187_ = lean_st_ref_get(v___y_185_);
v_infoState_188_ = lean_ctor_get(v___x_187_, 7);
lean_inc_ref(v_infoState_188_);
lean_dec(v___x_187_);
v_trees_189_ = lean_ctor_get(v_infoState_188_, 2);
lean_inc_ref(v_trees_189_);
lean_dec_ref(v_infoState_188_);
v___x_190_ = lean_st_ref_take(v___y_185_);
v_infoState_191_ = lean_ctor_get(v___x_190_, 7);
v_env_192_ = lean_ctor_get(v___x_190_, 0);
v_nextMacroScope_193_ = lean_ctor_get(v___x_190_, 1);
v_ngen_194_ = lean_ctor_get(v___x_190_, 2);
v_auxDeclNGen_195_ = lean_ctor_get(v___x_190_, 3);
v_traceState_196_ = lean_ctor_get(v___x_190_, 4);
v_cache_197_ = lean_ctor_get(v___x_190_, 5);
v_messages_198_ = lean_ctor_get(v___x_190_, 6);
v_snapshotTasks_199_ = lean_ctor_get(v___x_190_, 8);
v_isSharedCheck_220_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_220_ == 0)
{
v___x_201_ = v___x_190_;
v_isShared_202_ = v_isSharedCheck_220_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_snapshotTasks_199_);
lean_inc(v_infoState_191_);
lean_inc(v_messages_198_);
lean_inc(v_cache_197_);
lean_inc(v_traceState_196_);
lean_inc(v_auxDeclNGen_195_);
lean_inc(v_ngen_194_);
lean_inc(v_nextMacroScope_193_);
lean_inc(v_env_192_);
lean_dec(v___x_190_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_220_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
uint8_t v_enabled_203_; lean_object* v_assignment_204_; lean_object* v_lazyAssignment_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_218_; 
v_enabled_203_ = lean_ctor_get_uint8(v_infoState_191_, sizeof(void*)*3);
v_assignment_204_ = lean_ctor_get(v_infoState_191_, 0);
v_lazyAssignment_205_ = lean_ctor_get(v_infoState_191_, 1);
v_isSharedCheck_218_ = !lean_is_exclusive(v_infoState_191_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v_infoState_191_, 2);
lean_dec(v_unused_219_);
v___x_207_ = v_infoState_191_;
v_isShared_208_ = v_isSharedCheck_218_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_lazyAssignment_205_);
lean_inc(v_assignment_204_);
lean_dec(v_infoState_191_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_218_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_209_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___closed__1);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 2, v___x_209_);
v___x_211_ = v___x_207_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_assignment_204_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v_lazyAssignment_205_);
lean_ctor_set(v_reuseFailAlloc_217_, 2, v___x_209_);
lean_ctor_set_uint8(v_reuseFailAlloc_217_, sizeof(void*)*3, v_enabled_203_);
v___x_211_ = v_reuseFailAlloc_217_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_213_; 
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 7, v___x_211_);
v___x_213_ = v___x_201_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_env_192_);
lean_ctor_set(v_reuseFailAlloc_216_, 1, v_nextMacroScope_193_);
lean_ctor_set(v_reuseFailAlloc_216_, 2, v_ngen_194_);
lean_ctor_set(v_reuseFailAlloc_216_, 3, v_auxDeclNGen_195_);
lean_ctor_set(v_reuseFailAlloc_216_, 4, v_traceState_196_);
lean_ctor_set(v_reuseFailAlloc_216_, 5, v_cache_197_);
lean_ctor_set(v_reuseFailAlloc_216_, 6, v_messages_198_);
lean_ctor_set(v_reuseFailAlloc_216_, 7, v___x_211_);
lean_ctor_set(v_reuseFailAlloc_216_, 8, v_snapshotTasks_199_);
v___x_213_ = v_reuseFailAlloc_216_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_214_ = lean_st_ref_set(v___y_185_, v___x_213_);
v___x_215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_215_, 0, v_trees_189_);
return v___x_215_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg___boxed(lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(v___y_221_);
lean_dec(v___y_221_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0(lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(v___y_229_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___boxed(lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0(v___y_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
lean_dec(v___y_233_);
lean_dec_ref(v___y_232_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(lean_object* v_s_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; uint8_t v___x_249_; lean_object* v___x_250_; 
v___x_248_ = lean_box(0);
v___x_249_ = 1;
lean_inc(v_s_240_);
v___x_250_ = l_Lean_Elab_Term_elabTerm(v_s_240_, v___x_248_, v___x_249_, v___x_249_, v_a_241_, v_a_242_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
if (lean_obj_tag(v___x_250_) == 0)
{
lean_object* v_a_251_; lean_object* v___x_252_; lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_261_; 
v_a_251_ = lean_ctor_get(v___x_250_, 0);
lean_inc(v_a_251_);
lean_dec_ref_known(v___x_250_, 1);
v___x_252_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(v_a_246_);
v_a_253_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_261_ == 0)
{
v___x_255_ = v___x_252_;
v_isShared_256_ = v_isSharedCheck_261_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_252_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_261_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v___x_259_; 
v___x_257_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_257_, 0, v_s_240_);
lean_ctor_set(v___x_257_, 1, v_a_253_);
lean_ctor_set(v___x_257_, 2, v_a_251_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v___x_257_);
v___x_259_ = v___x_255_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v___x_257_);
v___x_259_ = v_reuseFailAlloc_260_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
return v___x_259_;
}
}
}
else
{
lean_object* v_a_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_269_; 
lean_dec(v_s_240_);
v_a_262_ = lean_ctor_get(v___x_250_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_269_ == 0)
{
v___x_264_ = v___x_250_;
v_isShared_265_ = v_isSharedCheck_269_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_a_262_);
lean_dec(v___x_250_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_269_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___x_267_; 
if (v_isShared_265_ == 0)
{
v___x_267_ = v___x_264_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v_a_262_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf___boxed(lean_object* v_s_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(v_s_270_, v_a_271_, v_a_272_, v_a_273_, v_a_274_, v_a_275_, v_a_276_);
lean_dec(v_a_276_);
lean_dec_ref(v_a_275_);
lean_dec(v_a_274_);
lean_dec_ref(v_a_273_);
lean_dec(v_a_272_);
lean_dec_ref(v_a_271_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(lean_object* v_x_279_, lean_object* v___y_280_){
_start:
{
if (lean_obj_tag(v_x_279_) == 0)
{
lean_object* v_a_281_; lean_object* v___x_282_; 
v_a_281_ = lean_ctor_get(v_x_279_, 0);
lean_inc(v_a_281_);
v___x_282_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_282_, 0, v_a_281_);
lean_ctor_set(v___x_282_, 1, v___y_280_);
return v___x_282_;
}
else
{
lean_object* v_a_283_; lean_object* v___x_284_; 
v_a_283_ = lean_ctor_get(v_x_279_, 0);
lean_inc(v_a_283_);
v___x_284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_284_, 0, v_a_283_);
lean_ctor_set(v___x_284_, 1, v___y_280_);
return v___x_284_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg___boxed(lean_object* v_x_285_, lean_object* v___y_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(v_x_285_, v___y_286_);
lean_dec_ref(v_x_285_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1(lean_object* v_00_u03b1_288_, lean_object* v_x_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(v_x_289_, v___y_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___boxed(lean_object* v_00_u03b1_293_, lean_object* v_x_294_, lean_object* v___y_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1(v_00_u03b1_293_, v_x_294_, v___y_295_, v___y_296_);
lean_dec_ref(v___y_295_);
lean_dec_ref(v_x_294_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4(lean_object* v_env_298_, lean_object* v_options_299_, lean_object* v_currNamespace_300_, lean_object* v_openDecls_301_, lean_object* v_n_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_305_ = l_Lean_ResolveName_resolveGlobalName(v_env_298_, v_options_299_, v_currNamespace_300_, v_openDecls_301_, v_n_302_);
v___x_306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
lean_ctor_set(v___x_306_, 1, v___y_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4___boxed(lean_object* v_env_307_, lean_object* v_options_308_, lean_object* v_currNamespace_309_, lean_object* v_openDecls_310_, lean_object* v_n_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4(v_env_307_, v_options_308_, v_currNamespace_309_, v_openDecls_310_, v_n_311_, v___y_312_, v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec_ref(v_options_308_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2(lean_object* v_env_315_, lean_object* v_currNamespace_316_, lean_object* v_openDecls_317_, lean_object* v_n_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_321_ = l_Lean_ResolveName_resolveNamespace(v_env_315_, v_currNamespace_316_, v_openDecls_317_, v_n_318_);
v___x_322_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_321_);
lean_ctor_set(v___x_322_, 1, v___y_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2___boxed(lean_object* v_env_323_, lean_object* v_currNamespace_324_, lean_object* v_openDecls_325_, lean_object* v_n_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2(v_env_323_, v_currNamespace_324_, v_openDecls_325_, v_n_326_, v___y_327_, v___y_328_);
lean_dec_ref(v___y_327_);
return v_res_329_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_330_ = lean_box(0);
v___x_331_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_332_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
lean_ctor_set(v___x_332_, 1, v___x_330_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg(){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___closed__0);
v___x_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg___boxed(lean_object* v___y_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg();
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg(lean_object* v_a_338_, lean_object* v_x_339_){
_start:
{
if (lean_obj_tag(v_x_339_) == 0)
{
lean_object* v___x_340_; 
v___x_340_ = lean_box(0);
return v___x_340_;
}
else
{
lean_object* v_key_341_; lean_object* v_value_342_; lean_object* v_tail_343_; uint8_t v___x_344_; 
v_key_341_ = lean_ctor_get(v_x_339_, 0);
v_value_342_ = lean_ctor_get(v_x_339_, 1);
v_tail_343_ = lean_ctor_get(v_x_339_, 2);
v___x_344_ = lean_name_eq(v_key_341_, v_a_338_);
if (v___x_344_ == 0)
{
v_x_339_ = v_tail_343_;
goto _start;
}
else
{
lean_object* v___x_346_; 
lean_inc(v_value_342_);
v___x_346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_346_, 0, v_value_342_);
return v___x_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg___boxed(lean_object* v_a_347_, lean_object* v_x_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg(v_a_347_, v_x_348_);
lean_dec(v_x_348_);
lean_dec(v_a_347_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg(lean_object* v_m_350_, lean_object* v_a_351_){
_start:
{
lean_object* v_buckets_352_; lean_object* v___x_353_; uint64_t v___y_355_; 
v_buckets_352_ = lean_ctor_get(v_m_350_, 1);
v___x_353_ = lean_array_get_size(v_buckets_352_);
if (lean_obj_tag(v_a_351_) == 0)
{
uint64_t v___x_369_; 
v___x_369_ = 1723ULL;
v___y_355_ = v___x_369_;
goto v___jp_354_;
}
else
{
uint64_t v_hash_370_; 
v_hash_370_ = lean_ctor_get_uint64(v_a_351_, sizeof(void*)*2);
v___y_355_ = v_hash_370_;
goto v___jp_354_;
}
v___jp_354_:
{
uint64_t v___x_356_; uint64_t v___x_357_; uint64_t v_fold_358_; uint64_t v___x_359_; uint64_t v___x_360_; uint64_t v___x_361_; size_t v___x_362_; size_t v___x_363_; size_t v___x_364_; size_t v___x_365_; size_t v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_356_ = 32ULL;
v___x_357_ = lean_uint64_shift_right(v___y_355_, v___x_356_);
v_fold_358_ = lean_uint64_xor(v___y_355_, v___x_357_);
v___x_359_ = 16ULL;
v___x_360_ = lean_uint64_shift_right(v_fold_358_, v___x_359_);
v___x_361_ = lean_uint64_xor(v_fold_358_, v___x_360_);
v___x_362_ = lean_uint64_to_usize(v___x_361_);
v___x_363_ = lean_usize_of_nat(v___x_353_);
v___x_364_ = ((size_t)1ULL);
v___x_365_ = lean_usize_sub(v___x_363_, v___x_364_);
v___x_366_ = lean_usize_land(v___x_362_, v___x_365_);
v___x_367_ = lean_array_uget_borrowed(v_buckets_352_, v___x_366_);
v___x_368_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg(v_a_351_, v___x_367_);
return v___x_368_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg___boxed(lean_object* v_m_371_, lean_object* v_a_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg(v_m_371_, v_a_372_);
lean_dec(v_a_372_);
lean_dec_ref(v_m_371_);
return v_res_373_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg(lean_object* v_keys_374_, lean_object* v_i_375_, lean_object* v_k_376_){
_start:
{
lean_object* v___x_377_; uint8_t v___x_378_; 
v___x_377_ = lean_array_get_size(v_keys_374_);
v___x_378_ = lean_nat_dec_lt(v_i_375_, v___x_377_);
if (v___x_378_ == 0)
{
lean_dec(v_i_375_);
return v___x_378_;
}
else
{
lean_object* v_k_x27_379_; uint8_t v___x_380_; 
v_k_x27_379_ = lean_array_fget_borrowed(v_keys_374_, v_i_375_);
v___x_380_ = l_Lean_instBEqExtraModUse_beq(v_k_376_, v_k_x27_379_);
if (v___x_380_ == 0)
{
lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_381_ = lean_unsigned_to_nat(1u);
v___x_382_ = lean_nat_add(v_i_375_, v___x_381_);
lean_dec(v_i_375_);
v_i_375_ = v___x_382_;
goto _start;
}
else
{
lean_dec(v_i_375_);
return v___x_380_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg___boxed(lean_object* v_keys_384_, lean_object* v_i_385_, lean_object* v_k_386_){
_start:
{
uint8_t v_res_387_; lean_object* v_r_388_; 
v_res_387_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg(v_keys_384_, v_i_385_, v_k_386_);
lean_dec_ref(v_k_386_);
lean_dec_ref(v_keys_384_);
v_r_388_ = lean_box(v_res_387_);
return v_r_388_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg(lean_object* v_x_389_, size_t v_x_390_, lean_object* v_x_391_){
_start:
{
if (lean_obj_tag(v_x_389_) == 0)
{
lean_object* v_es_392_; lean_object* v___x_393_; size_t v___x_394_; size_t v___x_395_; lean_object* v_j_396_; lean_object* v___x_397_; 
v_es_392_ = lean_ctor_get(v_x_389_, 0);
v___x_393_ = lean_box(2);
v___x_394_ = ((size_t)31ULL);
v___x_395_ = lean_usize_land(v_x_390_, v___x_394_);
v_j_396_ = lean_usize_to_nat(v___x_395_);
v___x_397_ = lean_array_get_borrowed(v___x_393_, v_es_392_, v_j_396_);
lean_dec(v_j_396_);
switch(lean_obj_tag(v___x_397_))
{
case 0:
{
lean_object* v_key_398_; uint8_t v___x_399_; 
v_key_398_ = lean_ctor_get(v___x_397_, 0);
v___x_399_ = l_Lean_instBEqExtraModUse_beq(v_x_391_, v_key_398_);
return v___x_399_;
}
case 1:
{
lean_object* v_node_400_; size_t v___x_401_; size_t v___x_402_; 
v_node_400_ = lean_ctor_get(v___x_397_, 0);
v___x_401_ = ((size_t)5ULL);
v___x_402_ = lean_usize_shift_right(v_x_390_, v___x_401_);
v_x_389_ = v_node_400_;
v_x_390_ = v___x_402_;
goto _start;
}
default: 
{
uint8_t v___x_404_; 
v___x_404_ = 0;
return v___x_404_;
}
}
}
else
{
lean_object* v_ks_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v_ks_405_ = lean_ctor_get(v_x_389_, 0);
v___x_406_ = lean_unsigned_to_nat(0u);
v___x_407_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg(v_ks_405_, v___x_406_, v_x_391_);
return v___x_407_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg___boxed(lean_object* v_x_408_, lean_object* v_x_409_, lean_object* v_x_410_){
_start:
{
size_t v_x_26092__boxed_411_; uint8_t v_res_412_; lean_object* v_r_413_; 
v_x_26092__boxed_411_ = lean_unbox_usize(v_x_409_);
lean_dec(v_x_409_);
v_res_412_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg(v_x_408_, v_x_26092__boxed_411_, v_x_410_);
lean_dec_ref(v_x_410_);
lean_dec_ref(v_x_408_);
v_r_413_ = lean_box(v_res_412_);
return v_r_413_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg(lean_object* v_x_414_, lean_object* v_x_415_){
_start:
{
uint64_t v___x_416_; size_t v___x_417_; uint8_t v___x_418_; 
v___x_416_ = l_Lean_instHashableExtraModUse_hash(v_x_415_);
v___x_417_ = lean_uint64_to_usize(v___x_416_);
v___x_418_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg(v_x_414_, v___x_417_, v_x_415_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg___boxed(lean_object* v_x_419_, lean_object* v_x_420_){
_start:
{
uint8_t v_res_421_; lean_object* v_r_422_; 
v_res_421_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg(v_x_419_, v_x_420_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_x_419_);
v_r_422_ = lean_box(v_res_421_);
return v_r_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(lean_object* v_msgData_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v___x_429_; lean_object* v_env_430_; lean_object* v___x_431_; lean_object* v_mctx_432_; lean_object* v_lctx_433_; lean_object* v_options_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_429_ = lean_st_ref_get(v___y_427_);
v_env_430_ = lean_ctor_get(v___x_429_, 0);
lean_inc_ref(v_env_430_);
lean_dec(v___x_429_);
v___x_431_ = lean_st_ref_get(v___y_425_);
v_mctx_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc_ref(v_mctx_432_);
lean_dec(v___x_431_);
v_lctx_433_ = lean_ctor_get(v___y_424_, 2);
v_options_434_ = lean_ctor_get(v___y_426_, 2);
lean_inc_ref(v_options_434_);
lean_inc_ref(v_lctx_433_);
v___x_435_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_435_, 0, v_env_430_);
lean_ctor_set(v___x_435_, 1, v_mctx_432_);
lean_ctor_set(v___x_435_, 2, v_lctx_433_);
lean_ctor_set(v___x_435_, 3, v_options_434_);
v___x_436_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_435_);
lean_ctor_set(v___x_436_, 1, v_msgData_423_);
v___x_437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_437_, 0, v___x_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3___boxed(lean_object* v_msgData_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(v_msgData_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
return v_res_444_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_445_; double v___x_446_; 
v___x_445_ = lean_unsigned_to_nat(0u);
v___x_446_ = lean_float_of_nat(v___x_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(lean_object* v_cls_450_, lean_object* v_msg_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_){
_start:
{
lean_object* v_ref_457_; lean_object* v___x_458_; lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_503_; 
v_ref_457_ = lean_ctor_get(v___y_454_, 5);
v___x_458_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(v_msg_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_);
v_a_459_ = lean_ctor_get(v___x_458_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_458_);
if (v_isSharedCheck_503_ == 0)
{
v___x_461_ = v___x_458_;
v_isShared_462_ = v_isSharedCheck_503_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_458_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_503_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_463_; lean_object* v_traceState_464_; lean_object* v_env_465_; lean_object* v_nextMacroScope_466_; lean_object* v_ngen_467_; lean_object* v_auxDeclNGen_468_; lean_object* v_cache_469_; lean_object* v_messages_470_; lean_object* v_infoState_471_; lean_object* v_snapshotTasks_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_502_; 
v___x_463_ = lean_st_ref_take(v___y_455_);
v_traceState_464_ = lean_ctor_get(v___x_463_, 4);
v_env_465_ = lean_ctor_get(v___x_463_, 0);
v_nextMacroScope_466_ = lean_ctor_get(v___x_463_, 1);
v_ngen_467_ = lean_ctor_get(v___x_463_, 2);
v_auxDeclNGen_468_ = lean_ctor_get(v___x_463_, 3);
v_cache_469_ = lean_ctor_get(v___x_463_, 5);
v_messages_470_ = lean_ctor_get(v___x_463_, 6);
v_infoState_471_ = lean_ctor_get(v___x_463_, 7);
v_snapshotTasks_472_ = lean_ctor_get(v___x_463_, 8);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_502_ == 0)
{
v___x_474_ = v___x_463_;
v_isShared_475_ = v_isSharedCheck_502_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_snapshotTasks_472_);
lean_inc(v_infoState_471_);
lean_inc(v_messages_470_);
lean_inc(v_cache_469_);
lean_inc(v_traceState_464_);
lean_inc(v_auxDeclNGen_468_);
lean_inc(v_ngen_467_);
lean_inc(v_nextMacroScope_466_);
lean_inc(v_env_465_);
lean_dec(v___x_463_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_502_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
uint64_t v_tid_476_; lean_object* v_traces_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_501_; 
v_tid_476_ = lean_ctor_get_uint64(v_traceState_464_, sizeof(void*)*1);
v_traces_477_ = lean_ctor_get(v_traceState_464_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v_traceState_464_);
if (v_isSharedCheck_501_ == 0)
{
v___x_479_ = v_traceState_464_;
v_isShared_480_ = v_isSharedCheck_501_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_traces_477_);
lean_dec(v_traceState_464_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_501_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; double v___x_482_; uint8_t v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_491_; 
v___x_481_ = lean_box(0);
v___x_482_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0);
v___x_483_ = 0;
v___x_484_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1));
v___x_485_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_485_, 0, v_cls_450_);
lean_ctor_set(v___x_485_, 1, v___x_481_);
lean_ctor_set(v___x_485_, 2, v___x_484_);
lean_ctor_set_float(v___x_485_, sizeof(void*)*3, v___x_482_);
lean_ctor_set_float(v___x_485_, sizeof(void*)*3 + 8, v___x_482_);
lean_ctor_set_uint8(v___x_485_, sizeof(void*)*3 + 16, v___x_483_);
v___x_486_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__2));
v___x_487_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_487_, 0, v___x_485_);
lean_ctor_set(v___x_487_, 1, v_a_459_);
lean_ctor_set(v___x_487_, 2, v___x_486_);
lean_inc(v_ref_457_);
v___x_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_488_, 0, v_ref_457_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = l_Lean_PersistentArray_push___redArg(v_traces_477_, v___x_488_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 0, v___x_489_);
v___x_491_ = v___x_479_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_489_);
lean_ctor_set_uint64(v_reuseFailAlloc_500_, sizeof(void*)*1, v_tid_476_);
v___x_491_ = v_reuseFailAlloc_500_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
lean_object* v___x_493_; 
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 4, v___x_491_);
v___x_493_ = v___x_474_;
goto v_reusejp_492_;
}
else
{
lean_object* v_reuseFailAlloc_499_; 
v_reuseFailAlloc_499_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_499_, 0, v_env_465_);
lean_ctor_set(v_reuseFailAlloc_499_, 1, v_nextMacroScope_466_);
lean_ctor_set(v_reuseFailAlloc_499_, 2, v_ngen_467_);
lean_ctor_set(v_reuseFailAlloc_499_, 3, v_auxDeclNGen_468_);
lean_ctor_set(v_reuseFailAlloc_499_, 4, v___x_491_);
lean_ctor_set(v_reuseFailAlloc_499_, 5, v_cache_469_);
lean_ctor_set(v_reuseFailAlloc_499_, 6, v_messages_470_);
lean_ctor_set(v_reuseFailAlloc_499_, 7, v_infoState_471_);
lean_ctor_set(v_reuseFailAlloc_499_, 8, v_snapshotTasks_472_);
v___x_493_ = v_reuseFailAlloc_499_;
goto v_reusejp_492_;
}
v_reusejp_492_:
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_497_; 
v___x_494_ = lean_st_ref_set(v___y_455_, v___x_493_);
v___x_495_ = lean_box(0);
if (v_isShared_462_ == 0)
{
lean_ctor_set(v___x_461_, 0, v___x_495_);
v___x_497_ = v___x_461_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_495_);
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___boxed(lean_object* v_cls_504_, lean_object* v_msg_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v_cls_504_, v_msg_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
lean_dec(v___y_507_);
lean_dec_ref(v___y_506_);
return v_res_511_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2(void){
_start:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; 
v___x_514_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__1));
v___x_515_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__0));
v___x_516_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___x_515_, v___x_514_);
return v___x_516_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3(void){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_517_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4(void){
_start:
{
lean_object* v___x_518_; lean_object* v___x_519_; 
v___x_518_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__3);
v___x_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_519_, 0, v___x_518_);
return v___x_519_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5(void){
_start:
{
lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_520_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4);
v___x_521_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v___x_520_);
return v___x_521_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6(void){
_start:
{
lean_object* v___x_522_; lean_object* v___x_523_; 
v___x_522_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__4);
v___x_523_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_523_, 0, v___x_522_);
lean_ctor_set(v___x_523_, 1, v___x_522_);
lean_ctor_set(v___x_523_, 2, v___x_522_);
lean_ctor_set(v___x_523_, 3, v___x_522_);
lean_ctor_set(v___x_523_, 4, v___x_522_);
lean_ctor_set(v___x_523_, 5, v___x_522_);
return v___x_523_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10(void){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__9));
v___x_529_ = l_Lean_stringToMessageData(v___x_528_);
return v___x_529_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12(void){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_531_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__11));
v___x_532_ = l_Lean_stringToMessageData(v___x_531_);
return v___x_532_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13(void){
_start:
{
lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_533_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1));
v___x_534_ = l_Lean_stringToMessageData(v___x_533_);
return v___x_534_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16(void){
_start:
{
lean_object* v_cls_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v_cls_538_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__8));
v___x_539_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15));
v___x_540_ = l_Lean_Name_append(v___x_539_, v_cls_538_);
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; 
v___x_542_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__17));
v___x_543_ = l_Lean_stringToMessageData(v___x_542_);
return v___x_543_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_545_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__19));
v___x_546_ = l_Lean_stringToMessageData(v___x_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5(lean_object* v_mod_551_, uint8_t v_isMeta_552_, lean_object* v_hint_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
lean_object* v___x_561_; lean_object* v_env_562_; uint8_t v_isExporting_563_; lean_object* v___x_564_; lean_object* v_env_565_; lean_object* v___x_566_; lean_object* v_entry_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___y_572_; lean_object* v___y_573_; lean_object* v___x_613_; uint8_t v___x_614_; 
v___x_561_ = lean_st_ref_get(v___y_559_);
v_env_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc_ref(v_env_562_);
lean_dec(v___x_561_);
v_isExporting_563_ = lean_ctor_get_uint8(v_env_562_, sizeof(void*)*8);
lean_dec_ref(v_env_562_);
v___x_564_ = lean_st_ref_get(v___y_559_);
v_env_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc_ref(v_env_565_);
lean_dec(v___x_564_);
v___x_566_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__2);
lean_inc(v_mod_551_);
v_entry_567_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v_entry_567_, 0, v_mod_551_);
lean_ctor_set_uint8(v_entry_567_, sizeof(void*)*1, v_isExporting_563_);
lean_ctor_set_uint8(v_entry_567_, sizeof(void*)*1 + 1, v_isMeta_552_);
v___x_568_ = l___private_Lean_ExtraModUses_0__Lean_extraModUses;
v___x_569_ = lean_box(1);
v___x_570_ = lean_box(0);
v___x_613_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_566_, v___x_568_, v_env_565_, v___x_569_, v___x_570_);
v___x_614_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg(v___x_613_, v_entry_567_);
lean_dec(v___x_613_);
if (v___x_614_ == 0)
{
lean_object* v_options_615_; uint8_t v_hasTrace_616_; 
v_options_615_ = lean_ctor_get(v___y_558_, 2);
v_hasTrace_616_ = lean_ctor_get_uint8(v_options_615_, sizeof(void*)*1);
if (v_hasTrace_616_ == 0)
{
lean_dec(v_hint_553_);
lean_dec(v_mod_551_);
v___y_572_ = v___y_557_;
v___y_573_ = v___y_559_;
goto v___jp_571_;
}
else
{
lean_object* v_inheritedTraceOptions_617_; lean_object* v_cls_618_; lean_object* v___y_620_; lean_object* v___y_621_; lean_object* v___y_625_; lean_object* v___y_626_; lean_object* v___x_638_; uint8_t v___x_639_; 
v_inheritedTraceOptions_617_ = lean_ctor_get(v___y_558_, 13);
v_cls_618_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__8));
v___x_638_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__16);
v___x_639_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_617_, v_options_615_, v___x_638_);
if (v___x_639_ == 0)
{
lean_dec(v_hint_553_);
lean_dec(v_mod_551_);
v___y_572_ = v___y_557_;
v___y_573_ = v___y_559_;
goto v___jp_571_;
}
else
{
lean_object* v___x_640_; lean_object* v___y_642_; 
v___x_640_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__18);
if (v_isExporting_563_ == 0)
{
lean_object* v___x_649_; 
v___x_649_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__23));
v___y_642_ = v___x_649_;
goto v___jp_641_;
}
else
{
lean_object* v___x_650_; 
v___x_650_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__24));
v___y_642_ = v___x_650_;
goto v___jp_641_;
}
v___jp_641_:
{
lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
lean_inc_ref(v___y_642_);
v___x_643_ = l_Lean_stringToMessageData(v___y_642_);
v___x_644_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_644_, 0, v___x_640_);
lean_ctor_set(v___x_644_, 1, v___x_643_);
v___x_645_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20);
v___x_646_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_646_, 0, v___x_644_);
lean_ctor_set(v___x_646_, 1, v___x_645_);
if (v_isMeta_552_ == 0)
{
lean_object* v___x_647_; 
v___x_647_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__21));
v___y_625_ = v___x_646_;
v___y_626_ = v___x_647_;
goto v___jp_624_;
}
else
{
lean_object* v___x_648_; 
v___x_648_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__22));
v___y_625_ = v___x_646_;
v___y_626_ = v___x_648_;
goto v___jp_624_;
}
}
}
v___jp_619_:
{
lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_622_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_622_, 0, v___y_620_);
lean_ctor_set(v___x_622_, 1, v___y_621_);
v___x_623_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v_cls_618_, v___x_622_, v___y_556_, v___y_557_, v___y_558_, v___y_559_);
if (lean_obj_tag(v___x_623_) == 0)
{
lean_dec_ref_known(v___x_623_, 1);
v___y_572_ = v___y_557_;
v___y_573_ = v___y_559_;
goto v___jp_571_;
}
else
{
lean_dec_ref_known(v_entry_567_, 1);
return v___x_623_;
}
}
v___jp_624_:
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; uint8_t v___x_633_; 
lean_inc_ref(v___y_626_);
v___x_627_ = l_Lean_stringToMessageData(v___y_626_);
v___x_628_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_628_, 0, v___y_625_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
v___x_629_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__10);
v___x_630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_628_);
lean_ctor_set(v___x_630_, 1, v___x_629_);
v___x_631_ = l_Lean_MessageData_ofName(v_mod_551_);
v___x_632_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_632_, 0, v___x_630_);
lean_ctor_set(v___x_632_, 1, v___x_631_);
v___x_633_ = l_Lean_Name_isAnonymous(v_hint_553_);
if (v___x_633_ == 0)
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; 
v___x_634_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__12);
v___x_635_ = l_Lean_MessageData_ofName(v_hint_553_);
v___x_636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_634_);
lean_ctor_set(v___x_636_, 1, v___x_635_);
v___y_620_ = v___x_632_;
v___y_621_ = v___x_636_;
goto v___jp_619_;
}
else
{
lean_object* v___x_637_; 
lean_dec(v_hint_553_);
v___x_637_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__13);
v___y_620_ = v___x_632_;
v___y_621_ = v___x_637_;
goto v___jp_619_;
}
}
}
}
else
{
lean_object* v___x_651_; lean_object* v___x_652_; 
lean_dec_ref_known(v_entry_567_, 1);
lean_dec(v_hint_553_);
lean_dec(v_mod_551_);
v___x_651_ = lean_box(0);
v___x_652_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
return v___x_652_;
}
v___jp_571_:
{
lean_object* v___x_574_; lean_object* v_toEnvExtension_575_; lean_object* v_env_576_; lean_object* v_nextMacroScope_577_; lean_object* v_ngen_578_; lean_object* v_auxDeclNGen_579_; lean_object* v_traceState_580_; lean_object* v_messages_581_; lean_object* v_infoState_582_; lean_object* v_snapshotTasks_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_611_; 
v___x_574_ = lean_st_ref_take(v___y_573_);
v_toEnvExtension_575_ = lean_ctor_get(v___x_568_, 0);
v_env_576_ = lean_ctor_get(v___x_574_, 0);
v_nextMacroScope_577_ = lean_ctor_get(v___x_574_, 1);
v_ngen_578_ = lean_ctor_get(v___x_574_, 2);
v_auxDeclNGen_579_ = lean_ctor_get(v___x_574_, 3);
v_traceState_580_ = lean_ctor_get(v___x_574_, 4);
v_messages_581_ = lean_ctor_get(v___x_574_, 6);
v_infoState_582_ = lean_ctor_get(v___x_574_, 7);
v_snapshotTasks_583_ = lean_ctor_get(v___x_574_, 8);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_611_ == 0)
{
lean_object* v_unused_612_; 
v_unused_612_ = lean_ctor_get(v___x_574_, 5);
lean_dec(v_unused_612_);
v___x_585_ = v___x_574_;
v_isShared_586_ = v_isSharedCheck_611_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_snapshotTasks_583_);
lean_inc(v_infoState_582_);
lean_inc(v_messages_581_);
lean_inc(v_traceState_580_);
lean_inc(v_auxDeclNGen_579_);
lean_inc(v_ngen_578_);
lean_inc(v_nextMacroScope_577_);
lean_inc(v_env_576_);
lean_dec(v___x_574_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_611_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v_asyncMode_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_591_; 
v_asyncMode_587_ = lean_ctor_get(v_toEnvExtension_575_, 2);
v___x_588_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_568_, v_env_576_, v_entry_567_, v_asyncMode_587_, v___x_570_);
v___x_589_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__5);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 5, v___x_589_);
lean_ctor_set(v___x_585_, 0, v___x_588_);
v___x_591_ = v___x_585_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_588_);
lean_ctor_set(v_reuseFailAlloc_610_, 1, v_nextMacroScope_577_);
lean_ctor_set(v_reuseFailAlloc_610_, 2, v_ngen_578_);
lean_ctor_set(v_reuseFailAlloc_610_, 3, v_auxDeclNGen_579_);
lean_ctor_set(v_reuseFailAlloc_610_, 4, v_traceState_580_);
lean_ctor_set(v_reuseFailAlloc_610_, 5, v___x_589_);
lean_ctor_set(v_reuseFailAlloc_610_, 6, v_messages_581_);
lean_ctor_set(v_reuseFailAlloc_610_, 7, v_infoState_582_);
lean_ctor_set(v_reuseFailAlloc_610_, 8, v_snapshotTasks_583_);
v___x_591_ = v_reuseFailAlloc_610_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v_mctx_594_; lean_object* v_zetaDeltaFVarIds_595_; lean_object* v_postponed_596_; lean_object* v_diag_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_608_; 
v___x_592_ = lean_st_ref_set(v___y_573_, v___x_591_);
v___x_593_ = lean_st_ref_take(v___y_572_);
v_mctx_594_ = lean_ctor_get(v___x_593_, 0);
v_zetaDeltaFVarIds_595_ = lean_ctor_get(v___x_593_, 2);
v_postponed_596_ = lean_ctor_get(v___x_593_, 3);
v_diag_597_ = lean_ctor_get(v___x_593_, 4);
v_isSharedCheck_608_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_608_ == 0)
{
lean_object* v_unused_609_; 
v_unused_609_ = lean_ctor_get(v___x_593_, 1);
lean_dec(v_unused_609_);
v___x_599_ = v___x_593_;
v_isShared_600_ = v_isSharedCheck_608_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_diag_597_);
lean_inc(v_postponed_596_);
lean_inc(v_zetaDeltaFVarIds_595_);
lean_inc(v_mctx_594_);
lean_dec(v___x_593_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_608_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_601_; lean_object* v___x_603_; 
v___x_601_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__6);
if (v_isShared_600_ == 0)
{
lean_ctor_set(v___x_599_, 1, v___x_601_);
v___x_603_ = v___x_599_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_mctx_594_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v___x_601_);
lean_ctor_set(v_reuseFailAlloc_607_, 2, v_zetaDeltaFVarIds_595_);
lean_ctor_set(v_reuseFailAlloc_607_, 3, v_postponed_596_);
lean_ctor_set(v_reuseFailAlloc_607_, 4, v_diag_597_);
v___x_603_ = v_reuseFailAlloc_607_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; 
v___x_604_ = lean_st_ref_set(v___y_572_, v___x_603_);
v___x_605_ = lean_box(0);
v___x_606_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_606_, 0, v___x_605_);
return v___x_606_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___boxed(lean_object* v_mod_653_, lean_object* v_isMeta_654_, lean_object* v_hint_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
uint8_t v_isMeta_boxed_663_; lean_object* v_res_664_; 
v_isMeta_boxed_663_ = lean_unbox(v_isMeta_654_);
v_res_664_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5(v_mod_653_, v_isMeta_boxed_663_, v_hint_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6(lean_object* v___x_665_, lean_object* v_declName_666_, lean_object* v_as_667_, size_t v_sz_668_, size_t v_i_669_, lean_object* v_b_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_){
_start:
{
uint8_t v___x_678_; 
v___x_678_ = lean_usize_dec_lt(v_i_669_, v_sz_668_);
if (v___x_678_ == 0)
{
lean_object* v___x_679_; 
lean_dec(v_declName_666_);
v___x_679_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_679_, 0, v_b_670_);
return v___x_679_;
}
else
{
lean_object* v___x_680_; lean_object* v_modules_681_; lean_object* v___x_682_; lean_object* v_a_683_; lean_object* v___x_684_; lean_object* v_toImport_685_; lean_object* v_module_686_; uint8_t v___x_687_; lean_object* v___x_688_; 
v___x_680_ = l_Lean_Environment_header(v___x_665_);
v_modules_681_ = lean_ctor_get(v___x_680_, 3);
lean_inc_ref(v_modules_681_);
lean_dec_ref(v___x_680_);
v___x_682_ = l_Lean_instInhabitedEffectiveImport_default;
v_a_683_ = lean_array_uget_borrowed(v_as_667_, v_i_669_);
v___x_684_ = lean_array_get(v___x_682_, v_modules_681_, v_a_683_);
lean_dec_ref(v_modules_681_);
v_toImport_685_ = lean_ctor_get(v___x_684_, 0);
lean_inc_ref(v_toImport_685_);
lean_dec(v___x_684_);
v_module_686_ = lean_ctor_get(v_toImport_685_, 0);
lean_inc(v_module_686_);
lean_dec_ref(v_toImport_685_);
v___x_687_ = 0;
lean_inc(v_declName_666_);
v___x_688_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5(v_module_686_, v___x_687_, v_declName_666_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_);
if (lean_obj_tag(v___x_688_) == 0)
{
lean_object* v___x_689_; size_t v___x_690_; size_t v___x_691_; 
lean_dec_ref_known(v___x_688_, 1);
v___x_689_ = lean_box(0);
v___x_690_ = ((size_t)1ULL);
v___x_691_ = lean_usize_add(v_i_669_, v___x_690_);
v_i_669_ = v___x_691_;
v_b_670_ = v___x_689_;
goto _start;
}
else
{
lean_dec(v_declName_666_);
return v___x_688_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6___boxed(lean_object* v___x_693_, lean_object* v_declName_694_, lean_object* v_as_695_, lean_object* v_sz_696_, lean_object* v_i_697_, lean_object* v_b_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
size_t v_sz_boxed_706_; size_t v_i_boxed_707_; lean_object* v_res_708_; 
v_sz_boxed_706_ = lean_unbox_usize(v_sz_696_);
lean_dec(v_sz_696_);
v_i_boxed_707_ = lean_unbox_usize(v_i_697_);
lean_dec(v_i_697_);
v_res_708_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6(v___x_693_, v_declName_694_, v_as_695_, v_sz_boxed_706_, v_i_boxed_707_, v_b_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
lean_dec_ref(v_as_695_);
lean_dec_ref(v___x_693_);
return v_res_708_;
}
}
static lean_object* _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; 
v___x_711_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__1));
v___x_712_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__0));
v___x_713_ = l_Std_HashMap_instInhabited(lean_box(0), lean_box(0), v___x_712_, v___x_711_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1(lean_object* v_declName_716_, uint8_t v_isMeta_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_){
_start:
{
lean_object* v___x_725_; lean_object* v_env_729_; lean_object* v___y_731_; lean_object* v___x_744_; 
v___x_725_ = lean_st_ref_get(v___y_723_);
v_env_729_ = lean_ctor_get(v___x_725_, 0);
lean_inc_ref(v_env_729_);
lean_dec(v___x_725_);
v___x_744_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_729_, v_declName_716_);
if (lean_obj_tag(v___x_744_) == 0)
{
lean_dec_ref(v_env_729_);
lean_dec(v_declName_716_);
goto v___jp_726_;
}
else
{
lean_object* v_val_745_; lean_object* v___x_746_; lean_object* v_modules_747_; lean_object* v___x_748_; uint8_t v___x_749_; 
v_val_745_ = lean_ctor_get(v___x_744_, 0);
lean_inc(v_val_745_);
lean_dec_ref_known(v___x_744_, 1);
v___x_746_ = l_Lean_Environment_header(v_env_729_);
v_modules_747_ = lean_ctor_get(v___x_746_, 3);
lean_inc_ref(v_modules_747_);
lean_dec_ref(v___x_746_);
v___x_748_ = lean_array_get_size(v_modules_747_);
v___x_749_ = lean_nat_dec_lt(v_val_745_, v___x_748_);
if (v___x_749_ == 0)
{
lean_dec_ref(v_modules_747_);
lean_dec(v_val_745_);
lean_dec_ref(v_env_729_);
lean_dec(v_declName_716_);
goto v___jp_726_;
}
else
{
lean_object* v___x_750_; lean_object* v_env_751_; lean_object* v___x_752_; lean_object* v___x_753_; uint8_t v___y_755_; 
v___x_750_ = lean_st_ref_get(v___y_723_);
v_env_751_ = lean_ctor_get(v___x_750_, 0);
lean_inc_ref(v_env_751_);
lean_dec(v___x_750_);
v___x_752_ = lean_obj_once(&lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2, &lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2_once, _init_lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__2);
v___x_753_ = lean_array_fget(v_modules_747_, v_val_745_);
lean_dec(v_val_745_);
lean_dec_ref(v_modules_747_);
if (v_isMeta_717_ == 0)
{
lean_dec_ref(v_env_751_);
v___y_755_ = v_isMeta_717_;
goto v___jp_754_;
}
else
{
uint8_t v___x_766_; 
lean_inc(v_declName_716_);
v___x_766_ = l_Lean_isMarkedMeta(v_env_751_, v_declName_716_);
if (v___x_766_ == 0)
{
v___y_755_ = v_isMeta_717_;
goto v___jp_754_;
}
else
{
uint8_t v___x_767_; 
v___x_767_ = 0;
v___y_755_ = v___x_767_;
goto v___jp_754_;
}
}
v___jp_754_:
{
lean_object* v_toImport_756_; lean_object* v_module_757_; lean_object* v___x_758_; 
v_toImport_756_ = lean_ctor_get(v___x_753_, 0);
lean_inc_ref(v_toImport_756_);
lean_dec(v___x_753_);
v_module_757_ = lean_ctor_get(v_toImport_756_, 0);
lean_inc(v_module_757_);
lean_dec_ref(v_toImport_756_);
lean_inc(v_declName_716_);
v___x_758_ = lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5(v_module_757_, v___y_755_, v_declName_716_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; 
lean_dec_ref_known(v___x_758_, 1);
v___x_759_ = l_Lean_indirectModUseExt;
v___x_760_ = lean_box(1);
v___x_761_ = lean_box(0);
lean_inc_ref(v_env_729_);
v___x_762_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_752_, v___x_759_, v_env_729_, v___x_760_, v___x_761_);
v___x_763_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg(v___x_762_, v_declName_716_);
lean_dec(v___x_762_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v___x_764_; 
v___x_764_ = ((lean_object*)(lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___closed__3));
v___y_731_ = v___x_764_;
goto v___jp_730_;
}
else
{
lean_object* v_val_765_; 
v_val_765_ = lean_ctor_get(v___x_763_, 0);
lean_inc(v_val_765_);
lean_dec_ref_known(v___x_763_, 1);
v___y_731_ = v_val_765_;
goto v___jp_730_;
}
}
else
{
lean_dec_ref(v_env_729_);
lean_dec(v_declName_716_);
return v___x_758_;
}
}
}
}
v___jp_726_:
{
lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_727_ = lean_box(0);
v___x_728_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
return v___x_728_;
}
v___jp_730_:
{
lean_object* v___x_732_; size_t v_sz_733_; size_t v___x_734_; lean_object* v___x_735_; 
v___x_732_ = lean_box(0);
v_sz_733_ = lean_array_size(v___y_731_);
v___x_734_ = ((size_t)0ULL);
v___x_735_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__6(v_env_729_, v_declName_716_, v___y_731_, v_sz_733_, v___x_734_, v___x_732_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
lean_dec_ref(v___y_731_);
lean_dec_ref(v_env_729_);
if (lean_obj_tag(v___x_735_) == 0)
{
lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_742_; 
v_isSharedCheck_742_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_742_ == 0)
{
lean_object* v_unused_743_; 
v_unused_743_ = lean_ctor_get(v___x_735_, 0);
lean_dec(v_unused_743_);
v___x_737_ = v___x_735_;
v_isShared_738_ = v_isSharedCheck_742_;
goto v_resetjp_736_;
}
else
{
lean_dec(v___x_735_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_742_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
lean_object* v___x_740_; 
if (v_isShared_738_ == 0)
{
lean_ctor_set(v___x_737_, 0, v___x_732_);
v___x_740_ = v___x_737_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v___x_732_);
v___x_740_ = v_reuseFailAlloc_741_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
return v___x_740_;
}
}
}
else
{
return v___x_735_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1___boxed(lean_object* v_declName_768_, lean_object* v_isMeta_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_){
_start:
{
uint8_t v_isMeta_boxed_777_; lean_object* v_res_778_; 
v_isMeta_boxed_777_ = lean_unbox(v_isMeta_769_);
v_res_778_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1(v_declName_768_, v_isMeta_boxed_777_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg(lean_object* v_as_x27_779_, lean_object* v_b_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
if (lean_obj_tag(v_as_x27_779_) == 0)
{
lean_object* v___x_788_; 
v___x_788_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_788_, 0, v_b_780_);
return v___x_788_;
}
else
{
lean_object* v_head_789_; lean_object* v_tail_790_; uint8_t v___x_791_; lean_object* v___x_792_; 
v_head_789_ = lean_ctor_get(v_as_x27_779_, 0);
v_tail_790_ = lean_ctor_get(v_as_x27_779_, 1);
v___x_791_ = 1;
lean_inc(v_head_789_);
v___x_792_ = lp_mathlib_Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1(v_head_789_, v___x_791_, v___y_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_, v___y_786_);
if (lean_obj_tag(v___x_792_) == 0)
{
lean_object* v___x_793_; 
lean_dec_ref_known(v___x_792_, 1);
v___x_793_ = lean_box(0);
v_as_x27_779_ = v_tail_790_;
v_b_780_ = v___x_793_;
goto _start;
}
else
{
return v___x_792_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg___boxed(lean_object* v_as_x27_795_, lean_object* v_b_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_){
_start:
{
lean_object* v_res_804_; 
v_res_804_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg(v_as_x27_795_, v_b_796_, v___y_797_, v___y_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
lean_dec(v___y_802_);
lean_dec_ref(v___y_801_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
lean_dec(v___y_798_);
lean_dec_ref(v___y_797_);
lean_dec(v_as_x27_795_);
return v_res_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3(lean_object* v_currNamespace_805_, lean_object* v___y_806_, lean_object* v___y_807_){
_start:
{
lean_object* v___x_808_; 
v___x_808_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_808_, 0, v_currNamespace_805_);
lean_ctor_set(v___x_808_, 1, v___y_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3___boxed(lean_object* v_currNamespace_809_, lean_object* v___y_810_, lean_object* v___y_811_){
_start:
{
lean_object* v_res_812_; 
v_res_812_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3(v_currNamespace_809_, v___y_810_, v___y_811_);
lean_dec_ref(v___y_810_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1(lean_object* v_env_813_, lean_object* v_declName_814_, lean_object* v___y_815_, lean_object* v___y_816_){
_start:
{
uint8_t v___x_817_; lean_object* v_env_818_; lean_object* v___x_819_; uint8_t v___x_820_; uint8_t v___x_821_; 
v___x_817_ = 0;
v_env_818_ = l_Lean_Environment_setExporting(v_env_813_, v___x_817_);
lean_inc(v_declName_814_);
v___x_819_ = l_Lean_mkPrivateName(v_env_818_, v_declName_814_);
v___x_820_ = 1;
lean_inc_ref(v_env_818_);
v___x_821_ = l_Lean_Environment_contains(v_env_818_, v___x_819_, v___x_820_);
if (v___x_821_ == 0)
{
lean_object* v___x_822_; uint8_t v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; 
v___x_822_ = l_Lean_privateToUserName(v_declName_814_);
v___x_823_ = l_Lean_Environment_contains(v_env_818_, v___x_822_, v___x_820_);
v___x_824_ = lean_box(v___x_823_);
v___x_825_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_825_, 0, v___x_824_);
lean_ctor_set(v___x_825_, 1, v___y_816_);
return v___x_825_;
}
else
{
lean_object* v___x_826_; lean_object* v___x_827_; 
lean_dec_ref(v_env_818_);
lean_dec(v_declName_814_);
v___x_826_ = lean_box(v___x_821_);
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set(v___x_827_, 1, v___y_816_);
return v___x_827_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1___boxed(lean_object* v_env_828_, lean_object* v_declName_829_, lean_object* v___y_830_, lean_object* v___y_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1(v_env_828_, v_declName_829_, v___y_830_, v___y_831_);
lean_dec_ref(v___y_830_);
return v_res_832_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0(void){
_start:
{
lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_833_ = lean_box(1);
v___x_834_ = l_Lean_MessageData_ofFormat(v___x_833_);
return v___x_834_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3(void){
_start:
{
lean_object* v___x_838_; lean_object* v___x_839_; 
v___x_838_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__2));
v___x_839_ = l_Lean_MessageData_ofFormat(v___x_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21(lean_object* v_x_840_, lean_object* v_x_841_){
_start:
{
if (lean_obj_tag(v_x_841_) == 0)
{
return v_x_840_;
}
else
{
lean_object* v_head_842_; lean_object* v_tail_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_865_; 
v_head_842_ = lean_ctor_get(v_x_841_, 0);
v_tail_843_ = lean_ctor_get(v_x_841_, 1);
v_isSharedCheck_865_ = !lean_is_exclusive(v_x_841_);
if (v_isSharedCheck_865_ == 0)
{
v___x_845_ = v_x_841_;
v_isShared_846_ = v_isSharedCheck_865_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_tail_843_);
lean_inc(v_head_842_);
lean_dec(v_x_841_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_865_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v_before_847_; lean_object* v___x_849_; uint8_t v_isShared_850_; uint8_t v_isSharedCheck_863_; 
v_before_847_ = lean_ctor_get(v_head_842_, 0);
v_isSharedCheck_863_ = !lean_is_exclusive(v_head_842_);
if (v_isSharedCheck_863_ == 0)
{
lean_object* v_unused_864_; 
v_unused_864_ = lean_ctor_get(v_head_842_, 1);
lean_dec(v_unused_864_);
v___x_849_ = v_head_842_;
v_isShared_850_ = v_isSharedCheck_863_;
goto v_resetjp_848_;
}
else
{
lean_inc(v_before_847_);
lean_dec(v_head_842_);
v___x_849_ = lean_box(0);
v_isShared_850_ = v_isSharedCheck_863_;
goto v_resetjp_848_;
}
v_resetjp_848_:
{
lean_object* v___x_851_; lean_object* v___x_853_; 
v___x_851_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0);
if (v_isShared_850_ == 0)
{
lean_ctor_set_tag(v___x_849_, 7);
lean_ctor_set(v___x_849_, 1, v___x_851_);
lean_ctor_set(v___x_849_, 0, v_x_840_);
v___x_853_ = v___x_849_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v_x_840_);
lean_ctor_set(v_reuseFailAlloc_862_, 1, v___x_851_);
v___x_853_ = v_reuseFailAlloc_862_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
lean_object* v___x_854_; lean_object* v___x_856_; 
v___x_854_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__3);
if (v_isShared_846_ == 0)
{
lean_ctor_set_tag(v___x_845_, 7);
lean_ctor_set(v___x_845_, 1, v___x_854_);
lean_ctor_set(v___x_845_, 0, v___x_853_);
v___x_856_ = v___x_845_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v___x_853_);
lean_ctor_set(v_reuseFailAlloc_861_, 1, v___x_854_);
v___x_856_ = v_reuseFailAlloc_861_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_857_ = l_Lean_MessageData_ofSyntax(v_before_847_);
v___x_858_ = l_Lean_indentD(v___x_857_);
v___x_859_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_859_, 0, v___x_856_);
lean_ctor_set(v___x_859_, 1, v___x_858_);
v_x_840_ = v___x_859_;
v_x_841_ = v_tail_843_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20(lean_object* v_opts_866_, lean_object* v_opt_867_){
_start:
{
lean_object* v_name_868_; lean_object* v_defValue_869_; lean_object* v_map_870_; lean_object* v___x_871_; 
v_name_868_ = lean_ctor_get(v_opt_867_, 0);
v_defValue_869_ = lean_ctor_get(v_opt_867_, 1);
v_map_870_ = lean_ctor_get(v_opts_866_, 0);
v___x_871_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_870_, v_name_868_);
if (lean_obj_tag(v___x_871_) == 0)
{
uint8_t v___x_872_; 
v___x_872_ = lean_unbox(v_defValue_869_);
return v___x_872_;
}
else
{
lean_object* v_val_873_; 
v_val_873_ = lean_ctor_get(v___x_871_, 0);
lean_inc(v_val_873_);
lean_dec_ref_known(v___x_871_, 1);
if (lean_obj_tag(v_val_873_) == 1)
{
uint8_t v_v_874_; 
v_v_874_ = lean_ctor_get_uint8(v_val_873_, 0);
lean_dec_ref_known(v_val_873_, 0);
return v_v_874_;
}
else
{
uint8_t v___x_875_; 
lean_dec(v_val_873_);
v___x_875_ = lean_unbox(v_defValue_869_);
return v___x_875_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20___boxed(lean_object* v_opts_876_, lean_object* v_opt_877_){
_start:
{
uint8_t v_res_878_; lean_object* v_r_879_; 
v_res_878_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20(v_opts_876_, v_opt_877_);
lean_dec_ref(v_opt_877_);
lean_dec_ref(v_opts_876_);
v_r_879_ = lean_box(v_res_878_);
return v_r_879_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2(void){
_start:
{
lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_883_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__1));
v___x_884_ = l_Lean_MessageData_ofFormat(v___x_883_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg(lean_object* v_msgData_885_, lean_object* v_macroStack_886_, lean_object* v___y_887_){
_start:
{
lean_object* v_options_889_; lean_object* v___x_890_; uint8_t v___x_891_; 
v_options_889_ = lean_ctor_get(v___y_887_, 2);
v___x_890_ = l_Lean_Elab_pp_macroStack;
v___x_891_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__20(v_options_889_, v___x_890_);
if (v___x_891_ == 0)
{
lean_object* v___x_892_; 
lean_dec(v_macroStack_886_);
v___x_892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_892_, 0, v_msgData_885_);
return v___x_892_;
}
else
{
if (lean_obj_tag(v_macroStack_886_) == 0)
{
lean_object* v___x_893_; 
v___x_893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_893_, 0, v_msgData_885_);
return v___x_893_;
}
else
{
lean_object* v_head_894_; lean_object* v_after_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_910_; 
v_head_894_ = lean_ctor_get(v_macroStack_886_, 0);
lean_inc(v_head_894_);
v_after_895_ = lean_ctor_get(v_head_894_, 1);
v_isSharedCheck_910_ = !lean_is_exclusive(v_head_894_);
if (v_isSharedCheck_910_ == 0)
{
lean_object* v_unused_911_; 
v_unused_911_ = lean_ctor_get(v_head_894_, 0);
lean_dec(v_unused_911_);
v___x_897_ = v_head_894_;
v_isShared_898_ = v_isSharedCheck_910_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_after_895_);
lean_dec(v_head_894_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_910_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_899_; lean_object* v___x_901_; 
v___x_899_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21___closed__0);
if (v_isShared_898_ == 0)
{
lean_ctor_set_tag(v___x_897_, 7);
lean_ctor_set(v___x_897_, 1, v___x_899_);
lean_ctor_set(v___x_897_, 0, v_msgData_885_);
v___x_901_ = v___x_897_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_msgData_885_);
lean_ctor_set(v_reuseFailAlloc_909_, 1, v___x_899_);
v___x_901_ = v_reuseFailAlloc_909_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v_msgData_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_902_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___closed__2);
v___x_903_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_903_, 0, v___x_901_);
lean_ctor_set(v___x_903_, 1, v___x_902_);
v___x_904_ = l_Lean_MessageData_ofSyntax(v_after_895_);
v___x_905_ = l_Lean_indentD(v___x_904_);
v_msgData_906_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_906_, 0, v___x_903_);
lean_ctor_set(v_msgData_906_, 1, v___x_905_);
v___x_907_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17_spec__21(v_msgData_906_, v_macroStack_886_);
v___x_908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
return v___x_908_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg___boxed(lean_object* v_msgData_912_, lean_object* v_macroStack_913_, lean_object* v___y_914_, lean_object* v___y_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg(v_msgData_912_, v_macroStack_913_, v___y_914_);
lean_dec_ref(v___y_914_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg(lean_object* v_msg_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
lean_object* v_ref_925_; lean_object* v___x_926_; lean_object* v_a_927_; lean_object* v_macroStack_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_939_; 
v_ref_925_ = lean_ctor_get(v___y_922_, 5);
v___x_926_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(v_msg_917_, v___y_920_, v___y_921_, v___y_922_, v___y_923_);
v_a_927_ = lean_ctor_get(v___x_926_, 0);
lean_inc(v_a_927_);
lean_dec_ref(v___x_926_);
v_macroStack_928_ = lean_ctor_get(v___y_918_, 1);
v___x_929_ = l_Lean_Elab_getBetterRef(v_ref_925_, v_macroStack_928_);
lean_inc(v_macroStack_928_);
v___x_930_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg(v_a_927_, v_macroStack_928_, v___y_922_);
v_a_931_ = lean_ctor_get(v___x_930_, 0);
v_isSharedCheck_939_ = !lean_is_exclusive(v___x_930_);
if (v_isSharedCheck_939_ == 0)
{
v___x_933_ = v___x_930_;
v_isShared_934_ = v_isSharedCheck_939_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v___x_930_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_939_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_935_; lean_object* v___x_937_; 
v___x_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_935_, 0, v___x_929_);
lean_ctor_set(v___x_935_, 1, v_a_931_);
if (v_isShared_934_ == 0)
{
lean_ctor_set_tag(v___x_933_, 1);
lean_ctor_set(v___x_933_, 0, v___x_935_);
v___x_937_ = v___x_933_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v___x_935_);
v___x_937_ = v_reuseFailAlloc_938_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
return v___x_937_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg___boxed(lean_object* v_msg_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_){
_start:
{
lean_object* v_res_948_; 
v_res_948_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg(v_msg_940_, v___y_941_, v___y_942_, v___y_943_, v___y_944_, v___y_945_, v___y_946_);
lean_dec(v___y_946_);
lean_dec_ref(v___y_945_);
lean_dec(v___y_944_);
lean_dec_ref(v___y_943_);
lean_dec(v___y_942_);
lean_dec_ref(v___y_941_);
return v_res_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(lean_object* v_ref_949_, lean_object* v_msg_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_){
_start:
{
lean_object* v_fileName_958_; lean_object* v_fileMap_959_; lean_object* v_options_960_; lean_object* v_currRecDepth_961_; lean_object* v_maxRecDepth_962_; lean_object* v_ref_963_; lean_object* v_currNamespace_964_; lean_object* v_openDecls_965_; lean_object* v_initHeartbeats_966_; lean_object* v_maxHeartbeats_967_; lean_object* v_quotContext_968_; lean_object* v_currMacroScope_969_; uint8_t v_diag_970_; lean_object* v_cancelTk_x3f_971_; uint8_t v_suppressElabErrors_972_; lean_object* v_inheritedTraceOptions_973_; lean_object* v_ref_974_; lean_object* v___x_975_; lean_object* v___x_976_; 
v_fileName_958_ = lean_ctor_get(v___y_955_, 0);
v_fileMap_959_ = lean_ctor_get(v___y_955_, 1);
v_options_960_ = lean_ctor_get(v___y_955_, 2);
v_currRecDepth_961_ = lean_ctor_get(v___y_955_, 3);
v_maxRecDepth_962_ = lean_ctor_get(v___y_955_, 4);
v_ref_963_ = lean_ctor_get(v___y_955_, 5);
v_currNamespace_964_ = lean_ctor_get(v___y_955_, 6);
v_openDecls_965_ = lean_ctor_get(v___y_955_, 7);
v_initHeartbeats_966_ = lean_ctor_get(v___y_955_, 8);
v_maxHeartbeats_967_ = lean_ctor_get(v___y_955_, 9);
v_quotContext_968_ = lean_ctor_get(v___y_955_, 10);
v_currMacroScope_969_ = lean_ctor_get(v___y_955_, 11);
v_diag_970_ = lean_ctor_get_uint8(v___y_955_, sizeof(void*)*14);
v_cancelTk_x3f_971_ = lean_ctor_get(v___y_955_, 12);
v_suppressElabErrors_972_ = lean_ctor_get_uint8(v___y_955_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_973_ = lean_ctor_get(v___y_955_, 13);
v_ref_974_ = l_Lean_replaceRef(v_ref_949_, v_ref_963_);
lean_inc_ref(v_inheritedTraceOptions_973_);
lean_inc(v_cancelTk_x3f_971_);
lean_inc(v_currMacroScope_969_);
lean_inc(v_quotContext_968_);
lean_inc(v_maxHeartbeats_967_);
lean_inc(v_initHeartbeats_966_);
lean_inc(v_openDecls_965_);
lean_inc(v_currNamespace_964_);
lean_inc(v_maxRecDepth_962_);
lean_inc(v_currRecDepth_961_);
lean_inc_ref(v_options_960_);
lean_inc_ref(v_fileMap_959_);
lean_inc_ref(v_fileName_958_);
v___x_975_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_975_, 0, v_fileName_958_);
lean_ctor_set(v___x_975_, 1, v_fileMap_959_);
lean_ctor_set(v___x_975_, 2, v_options_960_);
lean_ctor_set(v___x_975_, 3, v_currRecDepth_961_);
lean_ctor_set(v___x_975_, 4, v_maxRecDepth_962_);
lean_ctor_set(v___x_975_, 5, v_ref_974_);
lean_ctor_set(v___x_975_, 6, v_currNamespace_964_);
lean_ctor_set(v___x_975_, 7, v_openDecls_965_);
lean_ctor_set(v___x_975_, 8, v_initHeartbeats_966_);
lean_ctor_set(v___x_975_, 9, v_maxHeartbeats_967_);
lean_ctor_set(v___x_975_, 10, v_quotContext_968_);
lean_ctor_set(v___x_975_, 11, v_currMacroScope_969_);
lean_ctor_set(v___x_975_, 12, v_cancelTk_x3f_971_);
lean_ctor_set(v___x_975_, 13, v_inheritedTraceOptions_973_);
lean_ctor_set_uint8(v___x_975_, sizeof(void*)*14, v_diag_970_);
lean_ctor_set_uint8(v___x_975_, sizeof(void*)*14 + 1, v_suppressElabErrors_972_);
v___x_976_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg(v_msg_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_, v___x_975_, v___y_956_);
lean_dec_ref_known(v___x_975_, 14);
return v___x_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg___boxed(lean_object* v_ref_977_, lean_object* v_msg_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_){
_start:
{
lean_object* v_res_986_; 
v_res_986_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(v_ref_977_, v_msg_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec(v_ref_977_);
return v_res_986_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; 
v___x_992_ = l_Lean_maxRecDepthErrorMessage;
v___x_993_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_993_, 0, v___x_992_);
return v___x_993_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4(void){
_start:
{
lean_object* v___x_994_; lean_object* v___x_995_; 
v___x_994_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__3);
v___x_995_ = l_Lean_MessageData_ofFormat(v___x_994_);
return v___x_995_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; 
v___x_996_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__4);
v___x_997_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__2));
v___x_998_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_998_, 0, v___x_997_);
lean_ctor_set(v___x_998_, 1, v___x_996_);
return v___x_998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg(lean_object* v_ref_999_){
_start:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_1001_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___closed__5);
v___x_1002_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1002_, 0, v_ref_999_);
lean_ctor_set(v___x_1002_, 1, v___x_1001_);
v___x_1003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1003_, 0, v___x_1002_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg___boxed(lean_object* v_ref_1004_, lean_object* v___y_1005_){
_start:
{
lean_object* v_res_1006_; 
v_res_1006_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg(v_ref_1004_);
return v_res_1006_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3(lean_object* v_as_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_){
_start:
{
if (lean_obj_tag(v_as_1007_) == 0)
{
lean_object* v___x_1015_; lean_object* v___x_1016_; 
v___x_1015_ = lean_box(0);
v___x_1016_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1016_, 0, v___x_1015_);
return v___x_1016_;
}
else
{
lean_object* v_options_1017_; uint8_t v_hasTrace_1018_; 
v_options_1017_ = lean_ctor_get(v___y_1012_, 2);
v_hasTrace_1018_ = lean_ctor_get_uint8(v_options_1017_, sizeof(void*)*1);
if (v_hasTrace_1018_ == 0)
{
lean_object* v_tail_1019_; 
v_tail_1019_ = lean_ctor_get(v_as_1007_, 1);
lean_inc(v_tail_1019_);
lean_dec_ref_known(v_as_1007_, 2);
v_as_1007_ = v_tail_1019_;
goto _start;
}
else
{
lean_object* v_head_1021_; lean_object* v_tail_1022_; lean_object* v_fst_1023_; lean_object* v_snd_1024_; lean_object* v_inheritedTraceOptions_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; uint8_t v___x_1028_; 
v_head_1021_ = lean_ctor_get(v_as_1007_, 0);
lean_inc(v_head_1021_);
v_tail_1022_ = lean_ctor_get(v_as_1007_, 1);
lean_inc(v_tail_1022_);
lean_dec_ref_known(v_as_1007_, 2);
v_fst_1023_ = lean_ctor_get(v_head_1021_, 0);
lean_inc_n(v_fst_1023_, 2);
v_snd_1024_ = lean_ctor_get(v_head_1021_, 1);
lean_inc(v_snd_1024_);
lean_dec(v_head_1021_);
v_inheritedTraceOptions_1025_ = lean_ctor_get(v___y_1012_, 13);
v___x_1026_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15));
v___x_1027_ = l_Lean_Name_append(v___x_1026_, v_fst_1023_);
v___x_1028_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1025_, v_options_1017_, v___x_1027_);
lean_dec(v___x_1027_);
if (v___x_1028_ == 0)
{
lean_dec(v_snd_1024_);
lean_dec(v_fst_1023_);
v_as_1007_ = v_tail_1022_;
goto _start;
}
else
{
lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; 
v___x_1030_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1030_, 0, v_snd_1024_);
v___x_1031_ = l_Lean_MessageData_ofFormat(v___x_1030_);
v___x_1032_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v_fst_1023_, v___x_1031_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1032_) == 0)
{
lean_dec_ref_known(v___x_1032_, 1);
v_as_1007_ = v_tail_1022_;
goto _start;
}
else
{
lean_dec(v_tail_1022_);
return v___x_1032_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3___boxed(lean_object* v_as_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v_res_1042_; 
v_res_1042_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3(v_as_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_, v___y_1040_);
lean_dec(v___y_1040_);
lean_dec_ref(v___y_1039_);
lean_dec(v___y_1038_);
lean_dec_ref(v___y_1037_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
return v_res_1042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0(lean_object* v_env_1043_, lean_object* v_stx_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v___x_1047_; 
v___x_1047_ = l_Lean_Elab_expandMacroImpl_x3f(v_env_1043_, v_stx_1044_, v___y_1045_, v___y_1046_);
if (lean_obj_tag(v___x_1047_) == 0)
{
lean_object* v_a_1048_; 
v_a_1048_ = lean_ctor_get(v___x_1047_, 0);
lean_inc(v_a_1048_);
if (lean_obj_tag(v_a_1048_) == 0)
{
lean_object* v_a_1049_; lean_object* v___x_1051_; uint8_t v_isShared_1052_; uint8_t v_isSharedCheck_1057_; 
v_a_1049_ = lean_ctor_get(v___x_1047_, 1);
v_isSharedCheck_1057_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1057_ == 0)
{
lean_object* v_unused_1058_; 
v_unused_1058_ = lean_ctor_get(v___x_1047_, 0);
lean_dec(v_unused_1058_);
v___x_1051_ = v___x_1047_;
v_isShared_1052_ = v_isSharedCheck_1057_;
goto v_resetjp_1050_;
}
else
{
lean_inc(v_a_1049_);
lean_dec(v___x_1047_);
v___x_1051_ = lean_box(0);
v_isShared_1052_ = v_isSharedCheck_1057_;
goto v_resetjp_1050_;
}
v_resetjp_1050_:
{
lean_object* v___x_1053_; lean_object* v___x_1055_; 
v___x_1053_ = lean_box(0);
if (v_isShared_1052_ == 0)
{
lean_ctor_set(v___x_1051_, 0, v___x_1053_);
v___x_1055_ = v___x_1051_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1056_; 
v_reuseFailAlloc_1056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1056_, 0, v___x_1053_);
lean_ctor_set(v_reuseFailAlloc_1056_, 1, v_a_1049_);
v___x_1055_ = v_reuseFailAlloc_1056_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
return v___x_1055_;
}
}
}
else
{
lean_object* v_val_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1087_; 
v_val_1059_ = lean_ctor_get(v_a_1048_, 0);
v_isSharedCheck_1087_ = !lean_is_exclusive(v_a_1048_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1061_ = v_a_1048_;
v_isShared_1062_ = v_isSharedCheck_1087_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_val_1059_);
lean_dec(v_a_1048_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1087_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v_snd_1063_; 
v_snd_1063_ = lean_ctor_get(v_val_1059_, 1);
lean_inc(v_snd_1063_);
lean_dec(v_val_1059_);
if (lean_obj_tag(v_snd_1063_) == 0)
{
lean_object* v_a_1064_; lean_object* v_a_1065_; lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1073_; 
lean_del_object(v___x_1061_);
v_a_1064_ = lean_ctor_get(v___x_1047_, 1);
lean_inc(v_a_1064_);
lean_dec_ref_known(v___x_1047_, 2);
v_a_1065_ = lean_ctor_get(v_snd_1063_, 0);
v_isSharedCheck_1073_ = !lean_is_exclusive(v_snd_1063_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1067_ = v_snd_1063_;
v_isShared_1068_ = v_isSharedCheck_1073_;
goto v_resetjp_1066_;
}
else
{
lean_inc(v_a_1065_);
lean_dec(v_snd_1063_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1073_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v___x_1070_; 
if (v_isShared_1068_ == 0)
{
v___x_1070_ = v___x_1067_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1072_; 
v_reuseFailAlloc_1072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1072_, 0, v_a_1065_);
v___x_1070_ = v_reuseFailAlloc_1072_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
lean_object* v___x_1071_; 
v___x_1071_ = lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(v___x_1070_, v_a_1064_);
lean_dec_ref(v___x_1070_);
return v___x_1071_;
}
}
}
else
{
lean_object* v_a_1074_; lean_object* v_a_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1086_; 
v_a_1074_ = lean_ctor_get(v___x_1047_, 1);
lean_inc(v_a_1074_);
lean_dec_ref_known(v___x_1047_, 2);
v_a_1075_ = lean_ctor_get(v_snd_1063_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v_snd_1063_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1077_ = v_snd_1063_;
v_isShared_1078_ = v_isSharedCheck_1086_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_a_1075_);
lean_dec(v_snd_1063_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1086_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v___x_1080_; 
if (v_isShared_1062_ == 0)
{
lean_ctor_set(v___x_1061_, 0, v_a_1075_);
v___x_1080_ = v___x_1061_;
goto v_reusejp_1079_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v_a_1075_);
v___x_1080_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1079_;
}
v_reusejp_1079_:
{
lean_object* v___x_1082_; 
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 0, v___x_1080_);
v___x_1082_ = v___x_1077_;
goto v_reusejp_1081_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v___x_1080_);
v___x_1082_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1081_;
}
v_reusejp_1081_:
{
lean_object* v___x_1083_; 
v___x_1083_ = lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___redArg(v___x_1082_, v_a_1074_);
lean_dec_ref(v___x_1082_);
return v___x_1083_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1088_; lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1096_; 
v_a_1088_ = lean_ctor_get(v___x_1047_, 0);
v_a_1089_ = lean_ctor_get(v___x_1047_, 1);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1091_ = v___x_1047_;
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_inc(v_a_1088_);
lean_dec(v___x_1047_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1094_; 
if (v_isShared_1092_ == 0)
{
v___x_1094_ = v___x_1091_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v_a_1088_);
lean_ctor_set(v_reuseFailAlloc_1095_, 1, v_a_1089_);
v___x_1094_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
return v___x_1094_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0___boxed(lean_object* v_env_1097_, lean_object* v_stx_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_){
_start:
{
lean_object* v_res_1101_; 
v_res_1101_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0(v_env_1097_, v_stx_1098_, v___y_1099_, v___y_1100_);
lean_dec_ref(v___y_1099_);
return v_res_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(lean_object* v_x_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
lean_object* v___x_1111_; lean_object* v_env_1112_; lean_object* v_options_1113_; lean_object* v_currRecDepth_1114_; lean_object* v_maxRecDepth_1115_; lean_object* v_ref_1116_; lean_object* v_currNamespace_1117_; lean_object* v_openDecls_1118_; lean_object* v_quotContext_1119_; lean_object* v_currMacroScope_1120_; lean_object* v___x_1121_; lean_object* v_nextMacroScope_1122_; lean_object* v___f_1123_; lean_object* v___f_1124_; lean_object* v___f_1125_; lean_object* v___f_1126_; lean_object* v___f_1127_; lean_object* v_methods_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; 
v___x_1111_ = lean_st_ref_get(v___y_1109_);
v_env_1112_ = lean_ctor_get(v___x_1111_, 0);
lean_inc_ref_n(v_env_1112_, 4);
lean_dec(v___x_1111_);
v_options_1113_ = lean_ctor_get(v___y_1108_, 2);
v_currRecDepth_1114_ = lean_ctor_get(v___y_1108_, 3);
v_maxRecDepth_1115_ = lean_ctor_get(v___y_1108_, 4);
v_ref_1116_ = lean_ctor_get(v___y_1108_, 5);
v_currNamespace_1117_ = lean_ctor_get(v___y_1108_, 6);
v_openDecls_1118_ = lean_ctor_get(v___y_1108_, 7);
v_quotContext_1119_ = lean_ctor_get(v___y_1108_, 10);
v_currMacroScope_1120_ = lean_ctor_get(v___y_1108_, 11);
v___x_1121_ = lean_st_ref_get(v___y_1109_);
v_nextMacroScope_1122_ = lean_ctor_get(v___x_1121_, 1);
lean_inc(v_nextMacroScope_1122_);
lean_dec(v___x_1121_);
v___f_1123_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1123_, 0, v_env_1112_);
v___f_1124_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_1124_, 0, v_env_1112_);
lean_inc_n(v_openDecls_1118_, 2);
lean_inc_n(v_currNamespace_1117_, 3);
v___f_1125_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__2___boxed), 6, 3);
lean_closure_set(v___f_1125_, 0, v_env_1112_);
lean_closure_set(v___f_1125_, 1, v_currNamespace_1117_);
lean_closure_set(v___f_1125_, 2, v_openDecls_1118_);
v___f_1126_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_1126_, 0, v_currNamespace_1117_);
lean_inc_ref(v_options_1113_);
v___f_1127_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___lam__4___boxed), 7, 4);
lean_closure_set(v___f_1127_, 0, v_env_1112_);
lean_closure_set(v___f_1127_, 1, v_options_1113_);
lean_closure_set(v___f_1127_, 2, v_currNamespace_1117_);
lean_closure_set(v___f_1127_, 3, v_openDecls_1118_);
v_methods_1128_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_methods_1128_, 0, v___f_1123_);
lean_ctor_set(v_methods_1128_, 1, v___f_1126_);
lean_ctor_set(v_methods_1128_, 2, v___f_1124_);
lean_ctor_set(v_methods_1128_, 3, v___f_1125_);
lean_ctor_set(v_methods_1128_, 4, v___f_1127_);
lean_inc(v_ref_1116_);
lean_inc(v_maxRecDepth_1115_);
lean_inc(v_currRecDepth_1114_);
lean_inc(v_currMacroScope_1120_);
lean_inc(v_quotContext_1119_);
v___x_1129_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1129_, 0, v_methods_1128_);
lean_ctor_set(v___x_1129_, 1, v_quotContext_1119_);
lean_ctor_set(v___x_1129_, 2, v_currMacroScope_1120_);
lean_ctor_set(v___x_1129_, 3, v_currRecDepth_1114_);
lean_ctor_set(v___x_1129_, 4, v_maxRecDepth_1115_);
lean_ctor_set(v___x_1129_, 5, v_ref_1116_);
v___x_1130_ = lean_box(0);
v___x_1131_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1131_, 0, v_nextMacroScope_1122_);
lean_ctor_set(v___x_1131_, 1, v___x_1130_);
lean_ctor_set(v___x_1131_, 2, v___x_1130_);
v___x_1132_ = lean_apply_2(v_x_1103_, v___x_1129_, v___x_1131_);
if (lean_obj_tag(v___x_1132_) == 0)
{
lean_object* v_a_1133_; lean_object* v_a_1134_; lean_object* v_macroScope_1135_; lean_object* v_traceMsgs_1136_; lean_object* v_expandedMacroDecls_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; 
v_a_1133_ = lean_ctor_get(v___x_1132_, 1);
lean_inc(v_a_1133_);
v_a_1134_ = lean_ctor_get(v___x_1132_, 0);
lean_inc(v_a_1134_);
lean_dec_ref_known(v___x_1132_, 2);
v_macroScope_1135_ = lean_ctor_get(v_a_1133_, 0);
lean_inc(v_macroScope_1135_);
v_traceMsgs_1136_ = lean_ctor_get(v_a_1133_, 1);
lean_inc(v_traceMsgs_1136_);
v_expandedMacroDecls_1137_ = lean_ctor_get(v_a_1133_, 2);
lean_inc(v_expandedMacroDecls_1137_);
lean_dec(v_a_1133_);
v___x_1138_ = lean_box(0);
v___x_1139_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg(v_expandedMacroDecls_1137_, v___x_1138_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
lean_dec(v_expandedMacroDecls_1137_);
if (lean_obj_tag(v___x_1139_) == 0)
{
lean_object* v___x_1140_; lean_object* v_env_1141_; lean_object* v_ngen_1142_; lean_object* v_auxDeclNGen_1143_; lean_object* v_traceState_1144_; lean_object* v_cache_1145_; lean_object* v_messages_1146_; lean_object* v_infoState_1147_; lean_object* v_snapshotTasks_1148_; lean_object* v___x_1150_; uint8_t v_isShared_1151_; uint8_t v_isSharedCheck_1174_; 
lean_dec_ref_known(v___x_1139_, 1);
v___x_1140_ = lean_st_ref_take(v___y_1109_);
v_env_1141_ = lean_ctor_get(v___x_1140_, 0);
v_ngen_1142_ = lean_ctor_get(v___x_1140_, 2);
v_auxDeclNGen_1143_ = lean_ctor_get(v___x_1140_, 3);
v_traceState_1144_ = lean_ctor_get(v___x_1140_, 4);
v_cache_1145_ = lean_ctor_get(v___x_1140_, 5);
v_messages_1146_ = lean_ctor_get(v___x_1140_, 6);
v_infoState_1147_ = lean_ctor_get(v___x_1140_, 7);
v_snapshotTasks_1148_ = lean_ctor_get(v___x_1140_, 8);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___x_1140_);
if (v_isSharedCheck_1174_ == 0)
{
lean_object* v_unused_1175_; 
v_unused_1175_ = lean_ctor_get(v___x_1140_, 1);
lean_dec(v_unused_1175_);
v___x_1150_ = v___x_1140_;
v_isShared_1151_ = v_isSharedCheck_1174_;
goto v_resetjp_1149_;
}
else
{
lean_inc(v_snapshotTasks_1148_);
lean_inc(v_infoState_1147_);
lean_inc(v_messages_1146_);
lean_inc(v_cache_1145_);
lean_inc(v_traceState_1144_);
lean_inc(v_auxDeclNGen_1143_);
lean_inc(v_ngen_1142_);
lean_inc(v_env_1141_);
lean_dec(v___x_1140_);
v___x_1150_ = lean_box(0);
v_isShared_1151_ = v_isSharedCheck_1174_;
goto v_resetjp_1149_;
}
v_resetjp_1149_:
{
lean_object* v___x_1153_; 
if (v_isShared_1151_ == 0)
{
lean_ctor_set(v___x_1150_, 1, v_macroScope_1135_);
v___x_1153_ = v___x_1150_;
goto v_reusejp_1152_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_env_1141_);
lean_ctor_set(v_reuseFailAlloc_1173_, 1, v_macroScope_1135_);
lean_ctor_set(v_reuseFailAlloc_1173_, 2, v_ngen_1142_);
lean_ctor_set(v_reuseFailAlloc_1173_, 3, v_auxDeclNGen_1143_);
lean_ctor_set(v_reuseFailAlloc_1173_, 4, v_traceState_1144_);
lean_ctor_set(v_reuseFailAlloc_1173_, 5, v_cache_1145_);
lean_ctor_set(v_reuseFailAlloc_1173_, 6, v_messages_1146_);
lean_ctor_set(v_reuseFailAlloc_1173_, 7, v_infoState_1147_);
lean_ctor_set(v_reuseFailAlloc_1173_, 8, v_snapshotTasks_1148_);
v___x_1153_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1152_;
}
v_reusejp_1152_:
{
lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; 
v___x_1154_ = lean_st_ref_set(v___y_1109_, v___x_1153_);
v___x_1155_ = l_List_reverse___redArg(v_traceMsgs_1136_);
v___x_1156_ = lp_mathlib_List_forM___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__3(v___x_1155_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
if (lean_obj_tag(v___x_1156_) == 0)
{
lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1163_; 
v_isSharedCheck_1163_ = !lean_is_exclusive(v___x_1156_);
if (v_isSharedCheck_1163_ == 0)
{
lean_object* v_unused_1164_; 
v_unused_1164_ = lean_ctor_get(v___x_1156_, 0);
lean_dec(v_unused_1164_);
v___x_1158_ = v___x_1156_;
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
else
{
lean_dec(v___x_1156_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1161_; 
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 0, v_a_1134_);
v___x_1161_ = v___x_1158_;
goto v_reusejp_1160_;
}
else
{
lean_object* v_reuseFailAlloc_1162_; 
v_reuseFailAlloc_1162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1162_, 0, v_a_1134_);
v___x_1161_ = v_reuseFailAlloc_1162_;
goto v_reusejp_1160_;
}
v_reusejp_1160_:
{
return v___x_1161_;
}
}
}
else
{
lean_object* v_a_1165_; lean_object* v___x_1167_; uint8_t v_isShared_1168_; uint8_t v_isSharedCheck_1172_; 
lean_dec(v_a_1134_);
v_a_1165_ = lean_ctor_get(v___x_1156_, 0);
v_isSharedCheck_1172_ = !lean_is_exclusive(v___x_1156_);
if (v_isSharedCheck_1172_ == 0)
{
v___x_1167_ = v___x_1156_;
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
else
{
lean_inc(v_a_1165_);
lean_dec(v___x_1156_);
v___x_1167_ = lean_box(0);
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
v_resetjp_1166_:
{
lean_object* v___x_1170_; 
if (v_isShared_1168_ == 0)
{
v___x_1170_ = v___x_1167_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_a_1165_);
v___x_1170_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
return v___x_1170_;
}
}
}
}
}
}
else
{
lean_object* v_a_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
lean_dec(v_traceMsgs_1136_);
lean_dec(v_macroScope_1135_);
lean_dec(v_a_1134_);
v_a_1176_ = lean_ctor_get(v___x_1139_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v___x_1139_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_a_1176_);
lean_dec(v___x_1139_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
lean_object* v___x_1181_; 
if (v_isShared_1179_ == 0)
{
v___x_1181_ = v___x_1178_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_a_1176_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
}
else
{
lean_object* v_a_1184_; 
v_a_1184_ = lean_ctor_get(v___x_1132_, 0);
lean_inc(v_a_1184_);
lean_dec_ref_known(v___x_1132_, 2);
if (lean_obj_tag(v_a_1184_) == 0)
{
lean_object* v_a_1185_; lean_object* v_a_1186_; lean_object* v___x_1187_; uint8_t v___x_1188_; 
v_a_1185_ = lean_ctor_get(v_a_1184_, 0);
lean_inc(v_a_1185_);
v_a_1186_ = lean_ctor_get(v_a_1184_, 1);
lean_inc_ref(v_a_1186_);
lean_dec_ref_known(v_a_1184_, 2);
v___x_1187_ = ((lean_object*)(lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___closed__0));
v___x_1188_ = lean_string_dec_eq(v_a_1186_, v___x_1187_);
if (v___x_1188_ == 0)
{
lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v___x_1189_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1189_, 0, v_a_1186_);
v___x_1190_ = l_Lean_MessageData_ofFormat(v___x_1189_);
v___x_1191_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(v_a_1185_, v___x_1190_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
lean_dec(v_a_1185_);
return v___x_1191_;
}
else
{
lean_object* v___x_1192_; 
lean_dec_ref(v_a_1186_);
v___x_1192_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg(v_a_1185_);
return v___x_1192_;
}
}
else
{
lean_object* v___x_1193_; 
v___x_1193_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg();
return v___x_1193_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg___boxed(lean_object* v_x_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v_x_1194_, v___y_1195_, v___y_1196_, v___y_1197_, v___y_1198_, v___y_1199_, v___y_1200_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v___y_1198_);
lean_dec_ref(v___y_1197_);
lean_dec(v___y_1196_);
lean_dec_ref(v___y_1195_);
return v_res_1202_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0(void){
_start:
{
lean_object* v___x_1203_; 
v___x_1203_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1203_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1(void){
_start:
{
lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1204_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__0);
v___x_1205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
return v___x_1205_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2(void){
_start:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; 
v___x_1206_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1);
v___x_1207_ = lean_unsigned_to_nat(0u);
v___x_1208_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1208_, 0, v___x_1207_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
lean_ctor_set(v___x_1208_, 2, v___x_1207_);
lean_ctor_set(v___x_1208_, 3, v___x_1207_);
lean_ctor_set(v___x_1208_, 4, v___x_1206_);
lean_ctor_set(v___x_1208_, 5, v___x_1206_);
lean_ctor_set(v___x_1208_, 6, v___x_1206_);
lean_ctor_set(v___x_1208_, 7, v___x_1206_);
lean_ctor_set(v___x_1208_, 8, v___x_1206_);
lean_ctor_set(v___x_1208_, 9, v___x_1206_);
return v___x_1208_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3(void){
_start:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1209_ = lean_unsigned_to_nat(32u);
v___x_1210_ = lean_mk_empty_array_with_capacity(v___x_1209_);
v___x_1211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
return v___x_1211_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4(void){
_start:
{
size_t v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; 
v___x_1212_ = ((size_t)5ULL);
v___x_1213_ = lean_unsigned_to_nat(0u);
v___x_1214_ = lean_unsigned_to_nat(32u);
v___x_1215_ = lean_mk_empty_array_with_capacity(v___x_1214_);
v___x_1216_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__3);
v___x_1217_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1217_, 0, v___x_1216_);
lean_ctor_set(v___x_1217_, 1, v___x_1215_);
lean_ctor_set(v___x_1217_, 2, v___x_1213_);
lean_ctor_set(v___x_1217_, 3, v___x_1213_);
lean_ctor_set_usize(v___x_1217_, 4, v___x_1212_);
return v___x_1217_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5(void){
_start:
{
lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; 
v___x_1218_ = lean_box(1);
v___x_1219_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__4);
v___x_1220_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__1);
v___x_1221_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1221_, 0, v___x_1220_);
lean_ctor_set(v___x_1221_, 1, v___x_1219_);
lean_ctor_set(v___x_1221_, 2, v___x_1218_);
return v___x_1221_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7(void){
_start:
{
lean_object* v___x_1223_; lean_object* v___x_1224_; 
v___x_1223_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__6));
v___x_1224_ = l_Lean_stringToMessageData(v___x_1223_);
return v___x_1224_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9(void){
_start:
{
lean_object* v___x_1226_; lean_object* v___x_1227_; 
v___x_1226_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__8));
v___x_1227_ = l_Lean_stringToMessageData(v___x_1226_);
return v___x_1227_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11(void){
_start:
{
lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___x_1229_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__10));
v___x_1230_ = l_Lean_stringToMessageData(v___x_1229_);
return v___x_1230_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13(void){
_start:
{
lean_object* v___x_1232_; lean_object* v___x_1233_; 
v___x_1232_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__12));
v___x_1233_ = l_Lean_stringToMessageData(v___x_1232_);
return v___x_1233_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15(void){
_start:
{
lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1235_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__14));
v___x_1236_ = l_Lean_stringToMessageData(v___x_1235_);
return v___x_1236_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17(void){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; 
v___x_1238_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__16));
v___x_1239_ = l_Lean_stringToMessageData(v___x_1238_);
return v___x_1239_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19(void){
_start:
{
lean_object* v___x_1241_; lean_object* v___x_1242_; 
v___x_1241_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__18));
v___x_1242_ = l_Lean_stringToMessageData(v___x_1241_);
return v___x_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg(lean_object* v_msg_1243_, lean_object* v_declHint_1244_, lean_object* v___y_1245_){
_start:
{
lean_object* v___x_1247_; lean_object* v_env_1248_; uint8_t v___x_1249_; 
v___x_1247_ = lean_st_ref_get(v___y_1245_);
v_env_1248_ = lean_ctor_get(v___x_1247_, 0);
lean_inc_ref(v_env_1248_);
lean_dec(v___x_1247_);
v___x_1249_ = l_Lean_Name_isAnonymous(v_declHint_1244_);
if (v___x_1249_ == 0)
{
uint8_t v_isExporting_1250_; 
v_isExporting_1250_ = lean_ctor_get_uint8(v_env_1248_, sizeof(void*)*8);
if (v_isExporting_1250_ == 0)
{
lean_object* v___x_1251_; 
lean_dec_ref(v_env_1248_);
lean_dec(v_declHint_1244_);
v___x_1251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1251_, 0, v_msg_1243_);
return v___x_1251_;
}
else
{
lean_object* v___x_1252_; uint8_t v___x_1253_; 
lean_inc_ref(v_env_1248_);
v___x_1252_ = l_Lean_Environment_setExporting(v_env_1248_, v___x_1249_);
lean_inc(v_declHint_1244_);
lean_inc_ref(v___x_1252_);
v___x_1253_ = l_Lean_Environment_contains(v___x_1252_, v_declHint_1244_, v_isExporting_1250_);
if (v___x_1253_ == 0)
{
lean_object* v___x_1254_; 
lean_dec_ref(v___x_1252_);
lean_dec_ref(v_env_1248_);
lean_dec(v_declHint_1244_);
v___x_1254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1254_, 0, v_msg_1243_);
return v___x_1254_;
}
else
{
lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v_c_1260_; lean_object* v___x_1261_; 
v___x_1255_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__2);
v___x_1256_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__5);
v___x_1257_ = l_Lean_Options_empty;
v___x_1258_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1258_, 0, v___x_1252_);
lean_ctor_set(v___x_1258_, 1, v___x_1255_);
lean_ctor_set(v___x_1258_, 2, v___x_1256_);
lean_ctor_set(v___x_1258_, 3, v___x_1257_);
lean_inc(v_declHint_1244_);
v___x_1259_ = l_Lean_MessageData_ofConstName(v_declHint_1244_, v___x_1249_);
v_c_1260_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_1260_, 0, v___x_1258_);
lean_ctor_set(v_c_1260_, 1, v___x_1259_);
v___x_1261_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1248_, v_declHint_1244_);
if (lean_obj_tag(v___x_1261_) == 0)
{
lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; 
lean_dec_ref(v_env_1248_);
lean_dec(v_declHint_1244_);
v___x_1262_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7);
v___x_1263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
lean_ctor_set(v___x_1263_, 1, v_c_1260_);
v___x_1264_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__9);
v___x_1265_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1265_, 0, v___x_1263_);
lean_ctor_set(v___x_1265_, 1, v___x_1264_);
v___x_1266_ = l_Lean_MessageData_note(v___x_1265_);
v___x_1267_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1267_, 0, v_msg_1243_);
lean_ctor_set(v___x_1267_, 1, v___x_1266_);
v___x_1268_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1268_, 0, v___x_1267_);
return v___x_1268_;
}
else
{
lean_object* v_val_1269_; lean_object* v___x_1271_; uint8_t v_isShared_1272_; uint8_t v_isSharedCheck_1304_; 
v_val_1269_ = lean_ctor_get(v___x_1261_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1261_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1271_ = v___x_1261_;
v_isShared_1272_ = v_isSharedCheck_1304_;
goto v_resetjp_1270_;
}
else
{
lean_inc(v_val_1269_);
lean_dec(v___x_1261_);
v___x_1271_ = lean_box(0);
v_isShared_1272_ = v_isSharedCheck_1304_;
goto v_resetjp_1270_;
}
v_resetjp_1270_:
{
lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v_mod_1276_; uint8_t v___x_1277_; 
v___x_1273_ = lean_box(0);
v___x_1274_ = l_Lean_Environment_header(v_env_1248_);
lean_dec_ref(v_env_1248_);
v___x_1275_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1274_);
v_mod_1276_ = lean_array_get(v___x_1273_, v___x_1275_, v_val_1269_);
lean_dec(v_val_1269_);
lean_dec_ref(v___x_1275_);
v___x_1277_ = l_Lean_isPrivateName(v_declHint_1244_);
lean_dec(v_declHint_1244_);
if (v___x_1277_ == 0)
{
lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1289_; 
v___x_1278_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__11);
v___x_1279_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1278_);
lean_ctor_set(v___x_1279_, 1, v_c_1260_);
v___x_1280_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__13);
v___x_1281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1279_);
lean_ctor_set(v___x_1281_, 1, v___x_1280_);
v___x_1282_ = l_Lean_MessageData_ofName(v_mod_1276_);
v___x_1283_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1283_, 0, v___x_1281_);
lean_ctor_set(v___x_1283_, 1, v___x_1282_);
v___x_1284_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__15);
v___x_1285_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1283_);
lean_ctor_set(v___x_1285_, 1, v___x_1284_);
v___x_1286_ = l_Lean_MessageData_note(v___x_1285_);
v___x_1287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1287_, 0, v_msg_1243_);
lean_ctor_set(v___x_1287_, 1, v___x_1286_);
if (v_isShared_1272_ == 0)
{
lean_ctor_set_tag(v___x_1271_, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1287_);
v___x_1289_ = v___x_1271_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1287_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
else
{
lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1302_; 
v___x_1291_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__7);
v___x_1292_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1291_);
lean_ctor_set(v___x_1292_, 1, v_c_1260_);
v___x_1293_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__17);
v___x_1294_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1294_, 0, v___x_1292_);
lean_ctor_set(v___x_1294_, 1, v___x_1293_);
v___x_1295_ = l_Lean_MessageData_ofName(v_mod_1276_);
v___x_1296_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1294_);
lean_ctor_set(v___x_1296_, 1, v___x_1295_);
v___x_1297_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___closed__19);
v___x_1298_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1296_);
lean_ctor_set(v___x_1298_, 1, v___x_1297_);
v___x_1299_ = l_Lean_MessageData_note(v___x_1298_);
v___x_1300_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1300_, 0, v_msg_1243_);
lean_ctor_set(v___x_1300_, 1, v___x_1299_);
if (v_isShared_1272_ == 0)
{
lean_ctor_set_tag(v___x_1271_, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1300_);
v___x_1302_ = v___x_1271_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 1, 0);
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
}
}
else
{
lean_object* v___x_1305_; 
lean_dec_ref(v_env_1248_);
lean_dec(v_declHint_1244_);
v___x_1305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1305_, 0, v_msg_1243_);
return v___x_1305_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg___boxed(lean_object* v_msg_1306_, lean_object* v_declHint_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_){
_start:
{
lean_object* v_res_1310_; 
v_res_1310_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg(v_msg_1306_, v_declHint_1307_, v___y_1308_);
lean_dec(v___y_1308_);
return v_res_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20(lean_object* v_msg_1311_, lean_object* v_declHint_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_){
_start:
{
lean_object* v___x_1320_; lean_object* v_a_1321_; lean_object* v___x_1323_; uint8_t v_isShared_1324_; uint8_t v_isSharedCheck_1330_; 
v___x_1320_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg(v_msg_1311_, v_declHint_1312_, v___y_1318_);
v_a_1321_ = lean_ctor_get(v___x_1320_, 0);
v_isSharedCheck_1330_ = !lean_is_exclusive(v___x_1320_);
if (v_isSharedCheck_1330_ == 0)
{
v___x_1323_ = v___x_1320_;
v_isShared_1324_ = v_isSharedCheck_1330_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_a_1321_);
lean_dec(v___x_1320_);
v___x_1323_ = lean_box(0);
v_isShared_1324_ = v_isSharedCheck_1330_;
goto v_resetjp_1322_;
}
v_resetjp_1322_:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1328_; 
v___x_1325_ = l_Lean_unknownIdentifierMessageTag;
v___x_1326_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1326_, 0, v___x_1325_);
lean_ctor_set(v___x_1326_, 1, v_a_1321_);
if (v_isShared_1324_ == 0)
{
lean_ctor_set(v___x_1323_, 0, v___x_1326_);
v___x_1328_ = v___x_1323_;
goto v_reusejp_1327_;
}
else
{
lean_object* v_reuseFailAlloc_1329_; 
v_reuseFailAlloc_1329_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1329_, 0, v___x_1326_);
v___x_1328_ = v_reuseFailAlloc_1329_;
goto v_reusejp_1327_;
}
v_reusejp_1327_:
{
return v___x_1328_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20___boxed(lean_object* v_msg_1331_, lean_object* v_declHint_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20(v_msg_1331_, v_declHint_1332_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_, v___y_1337_, v___y_1338_);
lean_dec(v___y_1338_);
lean_dec_ref(v___y_1337_);
lean_dec(v___y_1336_);
lean_dec_ref(v___y_1335_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg(lean_object* v_ref_1341_, lean_object* v_msg_1342_, lean_object* v_declHint_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
lean_object* v___x_1351_; lean_object* v_a_1352_; lean_object* v___x_1353_; 
v___x_1351_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20(v_msg_1342_, v_declHint_1343_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_);
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
lean_inc(v_a_1352_);
lean_dec_ref(v___x_1351_);
v___x_1353_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(v_ref_1341_, v_a_1352_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg___boxed(lean_object* v_ref_1354_, lean_object* v_msg_1355_, lean_object* v_declHint_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_){
_start:
{
lean_object* v_res_1364_; 
v_res_1364_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg(v_ref_1354_, v_msg_1355_, v_declHint_1356_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
lean_dec(v___y_1358_);
lean_dec_ref(v___y_1357_);
lean_dec(v_ref_1354_);
return v_res_1364_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1(void){
_start:
{
lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1366_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__0));
v___x_1367_ = l_Lean_stringToMessageData(v___x_1366_);
return v___x_1367_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3(void){
_start:
{
lean_object* v___x_1369_; lean_object* v___x_1370_; 
v___x_1369_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__2));
v___x_1370_ = l_Lean_stringToMessageData(v___x_1369_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg(lean_object* v_ref_1371_, lean_object* v_constName_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
lean_object* v___x_1380_; uint8_t v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; 
v___x_1380_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__1);
v___x_1381_ = 0;
lean_inc(v_constName_1372_);
v___x_1382_ = l_Lean_MessageData_ofConstName(v_constName_1372_, v___x_1381_);
v___x_1383_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1383_, 0, v___x_1380_);
lean_ctor_set(v___x_1383_, 1, v___x_1382_);
v___x_1384_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___closed__3);
v___x_1385_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1385_, 0, v___x_1383_);
lean_ctor_set(v___x_1385_, 1, v___x_1384_);
v___x_1386_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg(v_ref_1371_, v___x_1385_, v_constName_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg___boxed(lean_object* v_ref_1387_, lean_object* v_constName_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v_res_1396_; 
v_res_1396_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg(v_ref_1387_, v_constName_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v___y_1390_);
lean_dec_ref(v___y_1389_);
lean_dec(v_ref_1387_);
return v_res_1396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg(lean_object* v_constName_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_){
_start:
{
lean_object* v_ref_1405_; lean_object* v___x_1406_; 
v_ref_1405_ = lean_ctor_get(v___y_1402_, 5);
v___x_1406_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg(v_ref_1405_, v_constName_1397_, v___y_1398_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg___boxed(lean_object* v_constName_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_){
_start:
{
lean_object* v_res_1415_; 
v_res_1415_ = lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg(v_constName_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1412_);
lean_dec(v___y_1411_);
lean_dec_ref(v___y_1410_);
lean_dec(v___y_1409_);
lean_dec_ref(v___y_1408_);
return v_res_1415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0___boxed(lean_object* v_a_1425_, lean_object* v_fst_1426_, lean_object* v_s_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_){
_start:
{
lean_object* v_res_1435_; 
v_res_1435_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0(v_a_1425_, v_fst_1426_, v_s_1427_, v___y_1428_, v___y_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec(v___y_1431_);
lean_dec_ref(v___y_1430_);
lean_dec(v___y_1429_);
lean_dec_ref(v___y_1428_);
return v_res_1435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp(lean_object* v_ref_1445_, lean_object* v_f_1446_, lean_object* v_lhs_1447_, lean_object* v_rhs_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v___x_1456_; uint8_t v___x_1457_; lean_object* v___x_1458_; 
v___x_1456_ = ((lean_object*)(lp_mathlib_FBinopElab_prodSyntax___closed__14));
v___x_1457_ = 0;
lean_inc(v_f_1446_);
v___x_1458_ = l_Lean_Elab_Term_resolveId_x3f(v_f_1446_, v___x_1456_, v___x_1457_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_);
if (lean_obj_tag(v___x_1458_) == 0)
{
lean_object* v_a_1459_; 
v_a_1459_ = lean_ctor_get(v___x_1458_, 0);
lean_inc(v_a_1459_);
lean_dec_ref_known(v___x_1458_, 1);
if (lean_obj_tag(v_a_1459_) == 1)
{
lean_object* v_val_1460_; lean_object* v___x_1461_; 
lean_dec(v_f_1446_);
v_val_1460_ = lean_ctor_get(v_a_1459_, 0);
lean_inc(v_val_1460_);
lean_dec_ref_known(v_a_1459_, 1);
v___x_1461_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(v_lhs_1447_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_);
if (lean_obj_tag(v___x_1461_) == 0)
{
lean_object* v_a_1462_; lean_object* v___x_1463_; 
v_a_1462_ = lean_ctor_get(v___x_1461_, 0);
lean_inc(v_a_1462_);
lean_dec_ref_known(v___x_1461_, 1);
v___x_1463_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(v_rhs_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_);
if (lean_obj_tag(v___x_1463_) == 0)
{
lean_object* v_a_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1472_; 
v_a_1464_ = lean_ctor_get(v___x_1463_, 0);
v_isSharedCheck_1472_ = !lean_is_exclusive(v___x_1463_);
if (v_isSharedCheck_1472_ == 0)
{
v___x_1466_ = v___x_1463_;
v_isShared_1467_ = v_isSharedCheck_1472_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_a_1464_);
lean_dec(v___x_1463_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1472_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v___x_1468_; lean_object* v___x_1470_; 
v___x_1468_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1468_, 0, v_ref_1445_);
lean_ctor_set(v___x_1468_, 1, v_val_1460_);
lean_ctor_set(v___x_1468_, 2, v_a_1462_);
lean_ctor_set(v___x_1468_, 3, v_a_1464_);
if (v_isShared_1467_ == 0)
{
lean_ctor_set(v___x_1466_, 0, v___x_1468_);
v___x_1470_ = v___x_1466_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1471_; 
v_reuseFailAlloc_1471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1471_, 0, v___x_1468_);
v___x_1470_ = v_reuseFailAlloc_1471_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
return v___x_1470_;
}
}
}
else
{
lean_dec(v_a_1462_);
lean_dec(v_val_1460_);
lean_dec(v_ref_1445_);
return v___x_1463_;
}
}
else
{
lean_dec(v_val_1460_);
lean_dec(v_rhs_1448_);
lean_dec(v_ref_1445_);
return v___x_1461_;
}
}
else
{
lean_object* v___x_1473_; lean_object* v___x_1474_; 
lean_dec(v_a_1459_);
lean_dec(v_rhs_1448_);
lean_dec(v_lhs_1447_);
lean_dec(v_ref_1445_);
v___x_1473_ = l_Lean_Syntax_getId(v_f_1446_);
lean_dec(v_f_1446_);
v___x_1474_ = lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg(v___x_1473_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_);
return v___x_1474_;
}
}
else
{
lean_object* v_a_1475_; lean_object* v___x_1477_; uint8_t v_isShared_1478_; uint8_t v_isSharedCheck_1482_; 
lean_dec(v_rhs_1448_);
lean_dec(v_lhs_1447_);
lean_dec(v_f_1446_);
lean_dec(v_ref_1445_);
v_a_1475_ = lean_ctor_get(v___x_1458_, 0);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1458_);
if (v_isSharedCheck_1482_ == 0)
{
v___x_1477_ = v___x_1458_;
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
else
{
lean_inc(v_a_1475_);
lean_dec(v___x_1458_);
v___x_1477_ = lean_box(0);
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
v_resetjp_1476_:
{
lean_object* v___x_1480_; 
if (v_isShared_1478_ == 0)
{
v___x_1480_ = v___x_1477_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1481_; 
v_reuseFailAlloc_1481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1481_, 0, v_a_1475_);
v___x_1480_ = v_reuseFailAlloc_1481_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
return v___x_1480_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(lean_object* v_s_1483_, lean_object* v_a_1484_, lean_object* v_a_1485_, lean_object* v_a_1486_, lean_object* v_a_1487_, lean_object* v_a_1488_, lean_object* v_a_1489_){
_start:
{
lean_object* v___x_1491_; uint8_t v___x_1492_; 
v___x_1491_ = ((lean_object*)(lp_mathlib_FBinopElab_prodSyntax___closed__1));
lean_inc(v_s_1483_);
v___x_1492_ = l_Lean_Syntax_isOfKind(v_s_1483_, v___x_1491_);
if (v___x_1492_ == 0)
{
lean_object* v___x_1493_; uint8_t v___x_1494_; 
v___x_1493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__4));
lean_inc(v_s_1483_);
v___x_1494_ = l_Lean_Syntax_isOfKind(v_s_1483_, v___x_1493_);
if (v___x_1494_ == 0)
{
lean_object* v___x_1495_; lean_object* v_fileName_1496_; lean_object* v_fileMap_1497_; lean_object* v_options_1498_; lean_object* v_currRecDepth_1499_; lean_object* v_maxRecDepth_1500_; lean_object* v_ref_1501_; lean_object* v_currNamespace_1502_; lean_object* v_openDecls_1503_; lean_object* v_initHeartbeats_1504_; lean_object* v_maxHeartbeats_1505_; lean_object* v_quotContext_1506_; lean_object* v_currMacroScope_1507_; uint8_t v_diag_1508_; lean_object* v_cancelTk_x3f_1509_; uint8_t v_suppressElabErrors_1510_; lean_object* v_inheritedTraceOptions_1511_; lean_object* v_env_1512_; lean_object* v_ref_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1495_ = lean_st_ref_get(v_a_1489_);
v_fileName_1496_ = lean_ctor_get(v_a_1488_, 0);
v_fileMap_1497_ = lean_ctor_get(v_a_1488_, 1);
v_options_1498_ = lean_ctor_get(v_a_1488_, 2);
v_currRecDepth_1499_ = lean_ctor_get(v_a_1488_, 3);
v_maxRecDepth_1500_ = lean_ctor_get(v_a_1488_, 4);
v_ref_1501_ = lean_ctor_get(v_a_1488_, 5);
v_currNamespace_1502_ = lean_ctor_get(v_a_1488_, 6);
v_openDecls_1503_ = lean_ctor_get(v_a_1488_, 7);
v_initHeartbeats_1504_ = lean_ctor_get(v_a_1488_, 8);
v_maxHeartbeats_1505_ = lean_ctor_get(v_a_1488_, 9);
v_quotContext_1506_ = lean_ctor_get(v_a_1488_, 10);
v_currMacroScope_1507_ = lean_ctor_get(v_a_1488_, 11);
v_diag_1508_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14);
v_cancelTk_x3f_1509_ = lean_ctor_get(v_a_1488_, 12);
v_suppressElabErrors_1510_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1511_ = lean_ctor_get(v_a_1488_, 13);
v_env_1512_ = lean_ctor_get(v___x_1495_, 0);
lean_inc_ref(v_env_1512_);
lean_dec(v___x_1495_);
v_ref_1513_ = l_Lean_replaceRef(v_s_1483_, v_ref_1501_);
lean_inc_ref(v_inheritedTraceOptions_1511_);
lean_inc(v_cancelTk_x3f_1509_);
lean_inc(v_currMacroScope_1507_);
lean_inc(v_quotContext_1506_);
lean_inc(v_maxHeartbeats_1505_);
lean_inc(v_initHeartbeats_1504_);
lean_inc(v_openDecls_1503_);
lean_inc(v_currNamespace_1502_);
lean_inc(v_maxRecDepth_1500_);
lean_inc(v_currRecDepth_1499_);
lean_inc_ref(v_options_1498_);
lean_inc_ref(v_fileMap_1497_);
lean_inc_ref(v_fileName_1496_);
v___x_1514_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1514_, 0, v_fileName_1496_);
lean_ctor_set(v___x_1514_, 1, v_fileMap_1497_);
lean_ctor_set(v___x_1514_, 2, v_options_1498_);
lean_ctor_set(v___x_1514_, 3, v_currRecDepth_1499_);
lean_ctor_set(v___x_1514_, 4, v_maxRecDepth_1500_);
lean_ctor_set(v___x_1514_, 5, v_ref_1513_);
lean_ctor_set(v___x_1514_, 6, v_currNamespace_1502_);
lean_ctor_set(v___x_1514_, 7, v_openDecls_1503_);
lean_ctor_set(v___x_1514_, 8, v_initHeartbeats_1504_);
lean_ctor_set(v___x_1514_, 9, v_maxHeartbeats_1505_);
lean_ctor_set(v___x_1514_, 10, v_quotContext_1506_);
lean_ctor_set(v___x_1514_, 11, v_currMacroScope_1507_);
lean_ctor_set(v___x_1514_, 12, v_cancelTk_x3f_1509_);
lean_ctor_set(v___x_1514_, 13, v_inheritedTraceOptions_1511_);
lean_ctor_set_uint8(v___x_1514_, sizeof(void*)*14, v_diag_1508_);
lean_ctor_set_uint8(v___x_1514_, sizeof(void*)*14 + 1, v_suppressElabErrors_1510_);
lean_inc(v_s_1483_);
v___x_1515_ = lean_alloc_closure((void*)(l_Lean_Elab_expandMacroImpl_x3f___boxed), 4, 2);
lean_closure_set(v___x_1515_, 0, v_env_1512_);
lean_closure_set(v___x_1515_, 1, v_s_1483_);
v___x_1516_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1515_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1514_, v_a_1489_);
if (lean_obj_tag(v___x_1516_) == 0)
{
lean_object* v_a_1517_; 
v_a_1517_ = lean_ctor_get(v___x_1516_, 0);
lean_inc(v_a_1517_);
lean_dec_ref_known(v___x_1516_, 1);
if (lean_obj_tag(v_a_1517_) == 0)
{
lean_object* v___x_1518_; 
v___x_1518_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(v_s_1483_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1514_, v_a_1489_);
lean_dec_ref_known(v___x_1514_, 14);
return v___x_1518_;
}
else
{
lean_object* v_val_1519_; lean_object* v_fst_1520_; lean_object* v_snd_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v_val_1519_ = lean_ctor_get(v_a_1517_, 0);
lean_inc(v_val_1519_);
lean_dec_ref_known(v_a_1517_, 1);
v_fst_1520_ = lean_ctor_get(v_val_1519_, 0);
lean_inc(v_fst_1520_);
v_snd_1521_ = lean_ctor_get(v_val_1519_, 1);
lean_inc(v_snd_1521_);
lean_dec(v_val_1519_);
v___x_1522_ = lean_alloc_closure((void*)(lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___boxed), 4, 2);
lean_closure_set(v___x_1522_, 0, lean_box(0));
lean_closure_set(v___x_1522_, 1, v_snd_1521_);
v___x_1523_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1522_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1514_, v_a_1489_);
if (lean_obj_tag(v___x_1523_) == 0)
{
lean_object* v_a_1524_; lean_object* v___f_1525_; lean_object* v___x_1526_; 
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc_n(v_a_1524_, 2);
lean_dec_ref_known(v___x_1523_, 1);
lean_inc(v_s_1483_);
v___f_1525_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1525_, 0, v_a_1524_);
lean_closure_set(v___f_1525_, 1, v_fst_1520_);
lean_closure_set(v___f_1525_, 2, v_s_1483_);
v___x_1526_ = l_Lean_Elab_Term_withPushMacroExpansionStack___redArg(v_s_1483_, v_a_1524_, v___f_1525_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1514_, v_a_1489_);
lean_dec_ref_known(v___x_1514_, 14);
return v___x_1526_;
}
else
{
lean_object* v_a_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1534_; 
lean_dec(v_fst_1520_);
lean_dec_ref_known(v___x_1514_, 14);
lean_dec(v_s_1483_);
v_a_1527_ = lean_ctor_get(v___x_1523_, 0);
v_isSharedCheck_1534_ = !lean_is_exclusive(v___x_1523_);
if (v_isSharedCheck_1534_ == 0)
{
v___x_1529_ = v___x_1523_;
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_a_1527_);
lean_dec(v___x_1523_);
v___x_1529_ = lean_box(0);
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
v_resetjp_1528_:
{
lean_object* v___x_1532_; 
if (v_isShared_1530_ == 0)
{
v___x_1532_ = v___x_1529_;
goto v_reusejp_1531_;
}
else
{
lean_object* v_reuseFailAlloc_1533_; 
v_reuseFailAlloc_1533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1533_, 0, v_a_1527_);
v___x_1532_ = v_reuseFailAlloc_1533_;
goto v_reusejp_1531_;
}
v_reusejp_1531_:
{
return v___x_1532_;
}
}
}
}
}
else
{
lean_object* v_a_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1542_; 
lean_dec_ref_known(v___x_1514_, 14);
lean_dec(v_s_1483_);
v_a_1535_ = lean_ctor_get(v___x_1516_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1516_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1537_ = v___x_1516_;
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_a_1535_);
lean_dec(v___x_1516_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1540_; 
if (v_isShared_1538_ == 0)
{
v___x_1540_ = v___x_1537_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v_a_1535_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
}
}
else
{
lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; uint8_t v___x_1546_; 
v___x_1543_ = lean_unsigned_to_nat(0u);
v___x_1544_ = l_Lean_Syntax_getArg(v_s_1483_, v___x_1543_);
v___x_1545_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__6));
lean_inc(v___x_1544_);
v___x_1546_ = l_Lean_Syntax_isOfKind(v___x_1544_, v___x_1545_);
if (v___x_1546_ == 0)
{
lean_object* v___x_1547_; lean_object* v_fileName_1548_; lean_object* v_fileMap_1549_; lean_object* v_options_1550_; lean_object* v_currRecDepth_1551_; lean_object* v_maxRecDepth_1552_; lean_object* v_ref_1553_; lean_object* v_currNamespace_1554_; lean_object* v_openDecls_1555_; lean_object* v_initHeartbeats_1556_; lean_object* v_maxHeartbeats_1557_; lean_object* v_quotContext_1558_; lean_object* v_currMacroScope_1559_; uint8_t v_diag_1560_; lean_object* v_cancelTk_x3f_1561_; uint8_t v_suppressElabErrors_1562_; lean_object* v_inheritedTraceOptions_1563_; lean_object* v_env_1564_; lean_object* v_ref_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; 
lean_dec(v___x_1544_);
v___x_1547_ = lean_st_ref_get(v_a_1489_);
v_fileName_1548_ = lean_ctor_get(v_a_1488_, 0);
v_fileMap_1549_ = lean_ctor_get(v_a_1488_, 1);
v_options_1550_ = lean_ctor_get(v_a_1488_, 2);
v_currRecDepth_1551_ = lean_ctor_get(v_a_1488_, 3);
v_maxRecDepth_1552_ = lean_ctor_get(v_a_1488_, 4);
v_ref_1553_ = lean_ctor_get(v_a_1488_, 5);
v_currNamespace_1554_ = lean_ctor_get(v_a_1488_, 6);
v_openDecls_1555_ = lean_ctor_get(v_a_1488_, 7);
v_initHeartbeats_1556_ = lean_ctor_get(v_a_1488_, 8);
v_maxHeartbeats_1557_ = lean_ctor_get(v_a_1488_, 9);
v_quotContext_1558_ = lean_ctor_get(v_a_1488_, 10);
v_currMacroScope_1559_ = lean_ctor_get(v_a_1488_, 11);
v_diag_1560_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14);
v_cancelTk_x3f_1561_ = lean_ctor_get(v_a_1488_, 12);
v_suppressElabErrors_1562_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1563_ = lean_ctor_get(v_a_1488_, 13);
v_env_1564_ = lean_ctor_get(v___x_1547_, 0);
lean_inc_ref(v_env_1564_);
lean_dec(v___x_1547_);
v_ref_1565_ = l_Lean_replaceRef(v_s_1483_, v_ref_1553_);
lean_inc_ref(v_inheritedTraceOptions_1563_);
lean_inc(v_cancelTk_x3f_1561_);
lean_inc(v_currMacroScope_1559_);
lean_inc(v_quotContext_1558_);
lean_inc(v_maxHeartbeats_1557_);
lean_inc(v_initHeartbeats_1556_);
lean_inc(v_openDecls_1555_);
lean_inc(v_currNamespace_1554_);
lean_inc(v_maxRecDepth_1552_);
lean_inc(v_currRecDepth_1551_);
lean_inc_ref(v_options_1550_);
lean_inc_ref(v_fileMap_1549_);
lean_inc_ref(v_fileName_1548_);
v___x_1566_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1566_, 0, v_fileName_1548_);
lean_ctor_set(v___x_1566_, 1, v_fileMap_1549_);
lean_ctor_set(v___x_1566_, 2, v_options_1550_);
lean_ctor_set(v___x_1566_, 3, v_currRecDepth_1551_);
lean_ctor_set(v___x_1566_, 4, v_maxRecDepth_1552_);
lean_ctor_set(v___x_1566_, 5, v_ref_1565_);
lean_ctor_set(v___x_1566_, 6, v_currNamespace_1554_);
lean_ctor_set(v___x_1566_, 7, v_openDecls_1555_);
lean_ctor_set(v___x_1566_, 8, v_initHeartbeats_1556_);
lean_ctor_set(v___x_1566_, 9, v_maxHeartbeats_1557_);
lean_ctor_set(v___x_1566_, 10, v_quotContext_1558_);
lean_ctor_set(v___x_1566_, 11, v_currMacroScope_1559_);
lean_ctor_set(v___x_1566_, 12, v_cancelTk_x3f_1561_);
lean_ctor_set(v___x_1566_, 13, v_inheritedTraceOptions_1563_);
lean_ctor_set_uint8(v___x_1566_, sizeof(void*)*14, v_diag_1560_);
lean_ctor_set_uint8(v___x_1566_, sizeof(void*)*14 + 1, v_suppressElabErrors_1562_);
lean_inc(v_s_1483_);
v___x_1567_ = lean_alloc_closure((void*)(l_Lean_Elab_expandMacroImpl_x3f___boxed), 4, 2);
lean_closure_set(v___x_1567_, 0, v_env_1564_);
lean_closure_set(v___x_1567_, 1, v_s_1483_);
v___x_1568_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1567_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1566_, v_a_1489_);
if (lean_obj_tag(v___x_1568_) == 0)
{
lean_object* v_a_1569_; 
v_a_1569_ = lean_ctor_get(v___x_1568_, 0);
lean_inc(v_a_1569_);
lean_dec_ref_known(v___x_1568_, 1);
if (lean_obj_tag(v_a_1569_) == 0)
{
lean_object* v___x_1570_; 
v___x_1570_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(v_s_1483_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1566_, v_a_1489_);
lean_dec_ref_known(v___x_1566_, 14);
return v___x_1570_;
}
else
{
lean_object* v_val_1571_; lean_object* v_fst_1572_; lean_object* v_snd_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v_val_1571_ = lean_ctor_get(v_a_1569_, 0);
lean_inc(v_val_1571_);
lean_dec_ref_known(v_a_1569_, 1);
v_fst_1572_ = lean_ctor_get(v_val_1571_, 0);
lean_inc(v_fst_1572_);
v_snd_1573_ = lean_ctor_get(v_val_1571_, 1);
lean_inc(v_snd_1573_);
lean_dec(v_val_1571_);
v___x_1574_ = lean_alloc_closure((void*)(lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___boxed), 4, 2);
lean_closure_set(v___x_1574_, 0, lean_box(0));
lean_closure_set(v___x_1574_, 1, v_snd_1573_);
v___x_1575_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1574_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1566_, v_a_1489_);
if (lean_obj_tag(v___x_1575_) == 0)
{
lean_object* v_a_1576_; lean_object* v___f_1577_; lean_object* v___x_1578_; 
v_a_1576_ = lean_ctor_get(v___x_1575_, 0);
lean_inc_n(v_a_1576_, 2);
lean_dec_ref_known(v___x_1575_, 1);
lean_inc(v_s_1483_);
v___f_1577_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1577_, 0, v_a_1576_);
lean_closure_set(v___f_1577_, 1, v_fst_1572_);
lean_closure_set(v___f_1577_, 2, v_s_1483_);
v___x_1578_ = l_Lean_Elab_Term_withPushMacroExpansionStack___redArg(v_s_1483_, v_a_1576_, v___f_1577_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1566_, v_a_1489_);
lean_dec_ref_known(v___x_1566_, 14);
return v___x_1578_;
}
else
{
lean_object* v_a_1579_; lean_object* v___x_1581_; uint8_t v_isShared_1582_; uint8_t v_isSharedCheck_1586_; 
lean_dec(v_fst_1572_);
lean_dec_ref_known(v___x_1566_, 14);
lean_dec(v_s_1483_);
v_a_1579_ = lean_ctor_get(v___x_1575_, 0);
v_isSharedCheck_1586_ = !lean_is_exclusive(v___x_1575_);
if (v_isSharedCheck_1586_ == 0)
{
v___x_1581_ = v___x_1575_;
v_isShared_1582_ = v_isSharedCheck_1586_;
goto v_resetjp_1580_;
}
else
{
lean_inc(v_a_1579_);
lean_dec(v___x_1575_);
v___x_1581_ = lean_box(0);
v_isShared_1582_ = v_isSharedCheck_1586_;
goto v_resetjp_1580_;
}
v_resetjp_1580_:
{
lean_object* v___x_1584_; 
if (v_isShared_1582_ == 0)
{
v___x_1584_ = v___x_1581_;
goto v_reusejp_1583_;
}
else
{
lean_object* v_reuseFailAlloc_1585_; 
v_reuseFailAlloc_1585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1585_, 0, v_a_1579_);
v___x_1584_ = v_reuseFailAlloc_1585_;
goto v_reusejp_1583_;
}
v_reusejp_1583_:
{
return v___x_1584_;
}
}
}
}
}
else
{
lean_object* v_a_1587_; lean_object* v___x_1589_; uint8_t v_isShared_1590_; uint8_t v_isSharedCheck_1594_; 
lean_dec_ref_known(v___x_1566_, 14);
lean_dec(v_s_1483_);
v_a_1587_ = lean_ctor_get(v___x_1568_, 0);
v_isSharedCheck_1594_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1594_ == 0)
{
v___x_1589_ = v___x_1568_;
v_isShared_1590_ = v_isSharedCheck_1594_;
goto v_resetjp_1588_;
}
else
{
lean_inc(v_a_1587_);
lean_dec(v___x_1568_);
v___x_1589_ = lean_box(0);
v_isShared_1590_ = v_isSharedCheck_1594_;
goto v_resetjp_1588_;
}
v_resetjp_1588_:
{
lean_object* v___x_1592_; 
if (v_isShared_1590_ == 0)
{
v___x_1592_ = v___x_1589_;
goto v_reusejp_1591_;
}
else
{
lean_object* v_reuseFailAlloc_1593_; 
v_reuseFailAlloc_1593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1593_, 0, v_a_1587_);
v___x_1592_ = v_reuseFailAlloc_1593_;
goto v_reusejp_1591_;
}
v_reusejp_1591_:
{
return v___x_1592_;
}
}
}
}
else
{
lean_object* v___x_1595_; lean_object* v_h_1596_; lean_object* v___x_1597_; uint8_t v___x_1598_; 
v___x_1595_ = lean_unsigned_to_nat(1u);
v_h_1596_ = l_Lean_Syntax_getArg(v___x_1544_, v___x_1595_);
lean_dec(v___x_1544_);
v___x_1597_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___closed__8));
lean_inc(v_h_1596_);
v___x_1598_ = l_Lean_Syntax_isOfKind(v_h_1596_, v___x_1597_);
if (v___x_1598_ == 0)
{
lean_object* v___x_1599_; lean_object* v_fileName_1600_; lean_object* v_fileMap_1601_; lean_object* v_options_1602_; lean_object* v_currRecDepth_1603_; lean_object* v_maxRecDepth_1604_; lean_object* v_ref_1605_; lean_object* v_currNamespace_1606_; lean_object* v_openDecls_1607_; lean_object* v_initHeartbeats_1608_; lean_object* v_maxHeartbeats_1609_; lean_object* v_quotContext_1610_; lean_object* v_currMacroScope_1611_; uint8_t v_diag_1612_; lean_object* v_cancelTk_x3f_1613_; uint8_t v_suppressElabErrors_1614_; lean_object* v_inheritedTraceOptions_1615_; lean_object* v_env_1616_; lean_object* v_ref_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; 
lean_dec(v_h_1596_);
v___x_1599_ = lean_st_ref_get(v_a_1489_);
v_fileName_1600_ = lean_ctor_get(v_a_1488_, 0);
v_fileMap_1601_ = lean_ctor_get(v_a_1488_, 1);
v_options_1602_ = lean_ctor_get(v_a_1488_, 2);
v_currRecDepth_1603_ = lean_ctor_get(v_a_1488_, 3);
v_maxRecDepth_1604_ = lean_ctor_get(v_a_1488_, 4);
v_ref_1605_ = lean_ctor_get(v_a_1488_, 5);
v_currNamespace_1606_ = lean_ctor_get(v_a_1488_, 6);
v_openDecls_1607_ = lean_ctor_get(v_a_1488_, 7);
v_initHeartbeats_1608_ = lean_ctor_get(v_a_1488_, 8);
v_maxHeartbeats_1609_ = lean_ctor_get(v_a_1488_, 9);
v_quotContext_1610_ = lean_ctor_get(v_a_1488_, 10);
v_currMacroScope_1611_ = lean_ctor_get(v_a_1488_, 11);
v_diag_1612_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14);
v_cancelTk_x3f_1613_ = lean_ctor_get(v_a_1488_, 12);
v_suppressElabErrors_1614_ = lean_ctor_get_uint8(v_a_1488_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1615_ = lean_ctor_get(v_a_1488_, 13);
v_env_1616_ = lean_ctor_get(v___x_1599_, 0);
lean_inc_ref(v_env_1616_);
lean_dec(v___x_1599_);
v_ref_1617_ = l_Lean_replaceRef(v_s_1483_, v_ref_1605_);
lean_inc_ref(v_inheritedTraceOptions_1615_);
lean_inc(v_cancelTk_x3f_1613_);
lean_inc(v_currMacroScope_1611_);
lean_inc(v_quotContext_1610_);
lean_inc(v_maxHeartbeats_1609_);
lean_inc(v_initHeartbeats_1608_);
lean_inc(v_openDecls_1607_);
lean_inc(v_currNamespace_1606_);
lean_inc(v_maxRecDepth_1604_);
lean_inc(v_currRecDepth_1603_);
lean_inc_ref(v_options_1602_);
lean_inc_ref(v_fileMap_1601_);
lean_inc_ref(v_fileName_1600_);
v___x_1618_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1618_, 0, v_fileName_1600_);
lean_ctor_set(v___x_1618_, 1, v_fileMap_1601_);
lean_ctor_set(v___x_1618_, 2, v_options_1602_);
lean_ctor_set(v___x_1618_, 3, v_currRecDepth_1603_);
lean_ctor_set(v___x_1618_, 4, v_maxRecDepth_1604_);
lean_ctor_set(v___x_1618_, 5, v_ref_1617_);
lean_ctor_set(v___x_1618_, 6, v_currNamespace_1606_);
lean_ctor_set(v___x_1618_, 7, v_openDecls_1607_);
lean_ctor_set(v___x_1618_, 8, v_initHeartbeats_1608_);
lean_ctor_set(v___x_1618_, 9, v_maxHeartbeats_1609_);
lean_ctor_set(v___x_1618_, 10, v_quotContext_1610_);
lean_ctor_set(v___x_1618_, 11, v_currMacroScope_1611_);
lean_ctor_set(v___x_1618_, 12, v_cancelTk_x3f_1613_);
lean_ctor_set(v___x_1618_, 13, v_inheritedTraceOptions_1615_);
lean_ctor_set_uint8(v___x_1618_, sizeof(void*)*14, v_diag_1612_);
lean_ctor_set_uint8(v___x_1618_, sizeof(void*)*14 + 1, v_suppressElabErrors_1614_);
lean_inc(v_s_1483_);
v___x_1619_ = lean_alloc_closure((void*)(l_Lean_Elab_expandMacroImpl_x3f___boxed), 4, 2);
lean_closure_set(v___x_1619_, 0, v_env_1616_);
lean_closure_set(v___x_1619_, 1, v_s_1483_);
v___x_1620_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1619_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1618_, v_a_1489_);
if (lean_obj_tag(v___x_1620_) == 0)
{
lean_object* v_a_1621_; 
v_a_1621_ = lean_ctor_get(v___x_1620_, 0);
lean_inc(v_a_1621_);
lean_dec_ref_known(v___x_1620_, 1);
if (lean_obj_tag(v_a_1621_) == 0)
{
lean_object* v___x_1622_; 
v___x_1622_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(v_s_1483_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1618_, v_a_1489_);
lean_dec_ref_known(v___x_1618_, 14);
return v___x_1622_;
}
else
{
lean_object* v_val_1623_; lean_object* v_fst_1624_; lean_object* v_snd_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; 
v_val_1623_ = lean_ctor_get(v_a_1621_, 0);
lean_inc(v_val_1623_);
lean_dec_ref_known(v_a_1621_, 1);
v_fst_1624_ = lean_ctor_get(v_val_1623_, 0);
lean_inc(v_fst_1624_);
v_snd_1625_ = lean_ctor_get(v_val_1623_, 1);
lean_inc(v_snd_1625_);
lean_dec(v_val_1623_);
v___x_1626_ = lean_alloc_closure((void*)(lp_mathlib_liftExcept___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__1___boxed), 4, 2);
lean_closure_set(v___x_1626_, 0, lean_box(0));
lean_closure_set(v___x_1626_, 1, v_snd_1625_);
v___x_1627_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v___x_1626_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1618_, v_a_1489_);
if (lean_obj_tag(v___x_1627_) == 0)
{
lean_object* v_a_1628_; lean_object* v___f_1629_; lean_object* v___x_1630_; 
v_a_1628_ = lean_ctor_get(v___x_1627_, 0);
lean_inc_n(v_a_1628_, 2);
lean_dec_ref_known(v___x_1627_, 1);
lean_inc(v_s_1483_);
v___f_1629_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1629_, 0, v_a_1628_);
lean_closure_set(v___f_1629_, 1, v_fst_1624_);
lean_closure_set(v___f_1629_, 2, v_s_1483_);
v___x_1630_ = l_Lean_Elab_Term_withPushMacroExpansionStack___redArg(v_s_1483_, v_a_1628_, v___f_1629_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v___x_1618_, v_a_1489_);
lean_dec_ref_known(v___x_1618_, 14);
return v___x_1630_;
}
else
{
lean_object* v_a_1631_; lean_object* v___x_1633_; uint8_t v_isShared_1634_; uint8_t v_isSharedCheck_1638_; 
lean_dec(v_fst_1624_);
lean_dec_ref_known(v___x_1618_, 14);
lean_dec(v_s_1483_);
v_a_1631_ = lean_ctor_get(v___x_1627_, 0);
v_isSharedCheck_1638_ = !lean_is_exclusive(v___x_1627_);
if (v_isSharedCheck_1638_ == 0)
{
v___x_1633_ = v___x_1627_;
v_isShared_1634_ = v_isSharedCheck_1638_;
goto v_resetjp_1632_;
}
else
{
lean_inc(v_a_1631_);
lean_dec(v___x_1627_);
v___x_1633_ = lean_box(0);
v_isShared_1634_ = v_isSharedCheck_1638_;
goto v_resetjp_1632_;
}
v_resetjp_1632_:
{
lean_object* v___x_1636_; 
if (v_isShared_1634_ == 0)
{
v___x_1636_ = v___x_1633_;
goto v_reusejp_1635_;
}
else
{
lean_object* v_reuseFailAlloc_1637_; 
v_reuseFailAlloc_1637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1637_, 0, v_a_1631_);
v___x_1636_ = v_reuseFailAlloc_1637_;
goto v_reusejp_1635_;
}
v_reusejp_1635_:
{
return v___x_1636_;
}
}
}
}
}
else
{
lean_object* v_a_1639_; lean_object* v___x_1641_; uint8_t v_isShared_1642_; uint8_t v_isSharedCheck_1646_; 
lean_dec_ref_known(v___x_1618_, 14);
lean_dec(v_s_1483_);
v_a_1639_ = lean_ctor_get(v___x_1620_, 0);
v_isSharedCheck_1646_ = !lean_is_exclusive(v___x_1620_);
if (v_isSharedCheck_1646_ == 0)
{
v___x_1641_ = v___x_1620_;
v_isShared_1642_ = v_isSharedCheck_1646_;
goto v_resetjp_1640_;
}
else
{
lean_inc(v_a_1639_);
lean_dec(v___x_1620_);
v___x_1641_ = lean_box(0);
v_isShared_1642_ = v_isSharedCheck_1646_;
goto v_resetjp_1640_;
}
v_resetjp_1640_:
{
lean_object* v___x_1644_; 
if (v_isShared_1642_ == 0)
{
v___x_1644_ = v___x_1641_;
goto v_reusejp_1643_;
}
else
{
lean_object* v_reuseFailAlloc_1645_; 
v_reuseFailAlloc_1645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1645_, 0, v_a_1639_);
v___x_1644_ = v_reuseFailAlloc_1645_;
goto v_reusejp_1643_;
}
v_reusejp_1643_:
{
return v___x_1644_;
}
}
}
}
else
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; uint8_t v___x_1650_; 
v___x_1647_ = l_Lean_Syntax_getArg(v_s_1483_, v___x_1595_);
v___x_1648_ = l_Lean_TSyntax_getHygieneInfo(v_h_1596_);
lean_dec(v_h_1596_);
v___x_1649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1649_, 0, v___x_1648_);
lean_inc(v___x_1647_);
v___x_1650_ = l_Lean_Elab_Term_hasCDot(v___x_1647_, v___x_1649_);
lean_dec_ref_known(v___x_1649_, 1);
if (v___x_1650_ == 0)
{
lean_dec(v_s_1483_);
v_s_1483_ = v___x_1647_;
goto _start;
}
else
{
lean_object* v___x_1652_; 
lean_dec(v___x_1647_);
v___x_1652_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf(v_s_1483_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v_a_1488_, v_a_1489_);
return v___x_1652_;
}
}
}
}
}
else
{
lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; 
v___x_1653_ = lean_unsigned_to_nat(1u);
v___x_1654_ = l_Lean_Syntax_getArg(v_s_1483_, v___x_1653_);
v___x_1655_ = lean_unsigned_to_nat(2u);
v___x_1656_ = l_Lean_Syntax_getArg(v_s_1483_, v___x_1655_);
v___x_1657_ = lean_unsigned_to_nat(3u);
v___x_1658_ = l_Lean_Syntax_getArg(v_s_1483_, v___x_1657_);
v___x_1659_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp(v_s_1483_, v___x_1654_, v___x_1656_, v___x_1658_, v_a_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v_a_1488_, v_a_1489_);
return v___x_1659_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___lam__0(lean_object* v_a_1660_, lean_object* v_fst_1661_, lean_object* v_s_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_){
_start:
{
lean_object* v___x_1670_; 
lean_inc(v_a_1660_);
v___x_1670_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(v_a_1660_, v___y_1663_, v___y_1664_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1679_; 
v_a_1671_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1679_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1679_ == 0)
{
v___x_1673_ = v___x_1670_;
v_isShared_1674_ = v_isSharedCheck_1679_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1670_);
v___x_1673_ = lean_box(0);
v_isShared_1674_ = v_isSharedCheck_1679_;
goto v_resetjp_1672_;
}
v_resetjp_1672_:
{
lean_object* v___x_1675_; lean_object* v___x_1677_; 
v___x_1675_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_1675_, 0, v_fst_1661_);
lean_ctor_set(v___x_1675_, 1, v_s_1662_);
lean_ctor_set(v___x_1675_, 2, v_a_1660_);
lean_ctor_set(v___x_1675_, 3, v_a_1671_);
if (v_isShared_1674_ == 0)
{
lean_ctor_set(v___x_1673_, 0, v___x_1675_);
v___x_1677_ = v___x_1673_;
goto v_reusejp_1676_;
}
else
{
lean_object* v_reuseFailAlloc_1678_; 
v_reuseFailAlloc_1678_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1678_, 0, v___x_1675_);
v___x_1677_ = v_reuseFailAlloc_1678_;
goto v_reusejp_1676_;
}
v_reusejp_1676_:
{
return v___x_1677_;
}
}
}
else
{
lean_dec(v_s_1662_);
lean_dec(v_fst_1661_);
lean_dec(v_a_1660_);
return v___x_1670_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp___boxed(lean_object* v_ref_1680_, lean_object* v_f_1681_, lean_object* v_lhs_1682_, lean_object* v_rhs_1683_, lean_object* v_a_1684_, lean_object* v_a_1685_, lean_object* v_a_1686_, lean_object* v_a_1687_, lean_object* v_a_1688_, lean_object* v_a_1689_, lean_object* v_a_1690_){
_start:
{
lean_object* v_res_1691_; 
v_res_1691_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp(v_ref_1680_, v_f_1681_, v_lhs_1682_, v_rhs_1683_, v_a_1684_, v_a_1685_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
lean_dec(v_a_1689_);
lean_dec_ref(v_a_1688_);
lean_dec(v_a_1687_);
lean_dec_ref(v_a_1686_);
lean_dec(v_a_1685_);
lean_dec_ref(v_a_1684_);
return v_res_1691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go___boxed(lean_object* v_s_1692_, lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_){
_start:
{
lean_object* v_res_1700_; 
v_res_1700_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(v_s_1692_, v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_);
lean_dec(v_a_1698_);
lean_dec_ref(v_a_1697_);
lean_dec(v_a_1696_);
lean_dec_ref(v_a_1695_);
lean_dec(v_a_1694_);
lean_dec_ref(v_a_1693_);
return v_res_1700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5(lean_object* v_00_u03b1_1701_, lean_object* v_ref_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_){
_start:
{
lean_object* v___x_1710_; 
v___x_1710_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___redArg(v_ref_1702_);
return v___x_1710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5___boxed(lean_object* v_00_u03b1_1711_, lean_object* v_ref_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_){
_start:
{
lean_object* v_res_1720_; 
v_res_1720_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__5(v_00_u03b1_1711_, v_ref_1712_, v___y_1713_, v___y_1714_, v___y_1715_, v___y_1716_, v___y_1717_, v___y_1718_);
lean_dec(v___y_1718_);
lean_dec_ref(v___y_1717_);
lean_dec(v___y_1716_);
lean_dec_ref(v___y_1715_);
lean_dec(v___y_1714_);
lean_dec_ref(v___y_1713_);
return v_res_1720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6(lean_object* v_00_u03b1_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_){
_start:
{
lean_object* v___x_1729_; 
v___x_1729_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___redArg();
return v___x_1729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6___boxed(lean_object* v_00_u03b1_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_){
_start:
{
lean_object* v_res_1738_; 
v_res_1738_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__6(v_00_u03b1_1730_, v___y_1731_, v___y_1732_, v___y_1733_, v___y_1734_, v___y_1735_, v___y_1736_);
lean_dec(v___y_1736_);
lean_dec_ref(v___y_1735_);
lean_dec(v___y_1734_);
lean_dec_ref(v___y_1733_);
lean_dec(v___y_1732_);
lean_dec_ref(v___y_1731_);
return v_res_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0(lean_object* v_00_u03b1_1739_, lean_object* v_x_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_){
_start:
{
lean_object* v___x_1748_; 
v___x_1748_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___redArg(v_x_1740_, v___y_1741_, v___y_1742_, v___y_1743_, v___y_1744_, v___y_1745_, v___y_1746_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0___boxed(lean_object* v_00_u03b1_1749_, lean_object* v_x_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_){
_start:
{
lean_object* v_res_1758_; 
v_res_1758_ = lp_mathlib_Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0(v_00_u03b1_1749_, v_x_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_, v___y_1755_, v___y_1756_);
lean_dec(v___y_1756_);
lean_dec_ref(v___y_1755_);
lean_dec(v___y_1754_);
lean_dec_ref(v___y_1753_);
lean_dec(v___y_1752_);
lean_dec_ref(v___y_1751_);
return v_res_1758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3(lean_object* v_00_u03b1_1759_, lean_object* v_constName_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_){
_start:
{
lean_object* v___x_1768_; 
v___x_1768_ = lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___redArg(v_constName_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_, v___y_1765_, v___y_1766_);
return v___x_1768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3___boxed(lean_object* v_00_u03b1_1769_, lean_object* v_constName_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_){
_start:
{
lean_object* v_res_1778_; 
v_res_1778_ = lp_mathlib_Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3(v_00_u03b1_1769_, v_constName_1770_, v___y_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_);
lean_dec(v___y_1776_);
lean_dec_ref(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
lean_dec(v___y_1772_);
lean_dec_ref(v___y_1771_);
return v_res_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0(lean_object* v_cls_1779_, lean_object* v_msg_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_){
_start:
{
lean_object* v___x_1788_; 
v___x_1788_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v_cls_1779_, v_msg_1780_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_);
return v___x_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___boxed(lean_object* v_cls_1789_, lean_object* v_msg_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v_res_1798_; 
v_res_1798_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0(v_cls_1789_, v_msg_1790_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
lean_dec(v___y_1792_);
lean_dec_ref(v___y_1791_);
return v_res_1798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2(lean_object* v_as_1799_, lean_object* v_as_x27_1800_, lean_object* v_b_1801_, lean_object* v_a_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_){
_start:
{
lean_object* v___x_1810_; 
v___x_1810_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___redArg(v_as_x27_1800_, v_b_1801_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_, v___y_1807_, v___y_1808_);
return v___x_1810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2___boxed(lean_object* v_as_1811_, lean_object* v_as_x27_1812_, lean_object* v_b_1813_, lean_object* v_a_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v_res_1822_; 
v_res_1822_ = lp_mathlib_List_forIn_x27_loop___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__2(v_as_1811_, v_as_x27_1812_, v_b_1813_, v_a_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_, v___y_1820_);
lean_dec(v___y_1820_);
lean_dec_ref(v___y_1819_);
lean_dec(v___y_1818_);
lean_dec_ref(v___y_1817_);
lean_dec(v___y_1816_);
lean_dec_ref(v___y_1815_);
lean_dec(v_as_x27_1812_);
lean_dec(v_as_1811_);
return v_res_1822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4(lean_object* v_00_u03b1_1823_, lean_object* v_ref_1824_, lean_object* v_msg_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_){
_start:
{
lean_object* v___x_1833_; 
v___x_1833_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___redArg(v_ref_1824_, v_msg_1825_, v___y_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_, v___y_1831_);
return v___x_1833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4___boxed(lean_object* v_00_u03b1_1834_, lean_object* v_ref_1835_, lean_object* v_msg_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_){
_start:
{
lean_object* v_res_1844_; 
v_res_1844_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4(v_00_u03b1_1834_, v_ref_1835_, v_msg_1836_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_, v___y_1841_, v___y_1842_);
lean_dec(v___y_1842_);
lean_dec_ref(v___y_1841_);
lean_dec(v___y_1840_);
lean_dec_ref(v___y_1839_);
lean_dec(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec(v_ref_1835_);
return v_res_1844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10(lean_object* v_00_u03b1_1845_, lean_object* v_ref_1846_, lean_object* v_constName_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_){
_start:
{
lean_object* v___x_1855_; 
v___x_1855_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___redArg(v_ref_1846_, v_constName_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_);
return v___x_1855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10___boxed(lean_object* v_00_u03b1_1856_, lean_object* v_ref_1857_, lean_object* v_constName_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_){
_start:
{
lean_object* v_res_1866_; 
v_res_1866_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10(v_00_u03b1_1856_, v_ref_1857_, v_constName_1858_, v___y_1859_, v___y_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_);
lean_dec(v___y_1864_);
lean_dec_ref(v___y_1863_);
lean_dec(v___y_1862_);
lean_dec_ref(v___y_1861_);
lean_dec(v___y_1860_);
lean_dec_ref(v___y_1859_);
lean_dec(v_ref_1857_);
return v_res_1866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7(lean_object* v_00_u03b2_1867_, lean_object* v_m_1868_, lean_object* v_a_1869_){
_start:
{
lean_object* v___x_1870_; 
v___x_1870_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___redArg(v_m_1868_, v_a_1869_);
return v___x_1870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7___boxed(lean_object* v_00_u03b2_1871_, lean_object* v_m_1872_, lean_object* v_a_1873_){
_start:
{
lean_object* v_res_1874_; 
v_res_1874_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7(v_00_u03b2_1871_, v_m_1872_, v_a_1873_);
lean_dec(v_a_1873_);
lean_dec_ref(v_m_1872_);
return v_res_1874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11(lean_object* v_00_u03b1_1875_, lean_object* v_msg_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_, lean_object* v___y_1882_){
_start:
{
lean_object* v___x_1884_; 
v___x_1884_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___redArg(v_msg_1876_, v___y_1877_, v___y_1878_, v___y_1879_, v___y_1880_, v___y_1881_, v___y_1882_);
return v___x_1884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11___boxed(lean_object* v_00_u03b1_1885_, lean_object* v_msg_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_){
_start:
{
lean_object* v_res_1894_; 
v_res_1894_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11(v_00_u03b1_1885_, v_msg_1886_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_, v___y_1891_, v___y_1892_);
lean_dec(v___y_1892_);
lean_dec_ref(v___y_1891_);
lean_dec(v___y_1890_);
lean_dec_ref(v___y_1889_);
lean_dec(v___y_1888_);
lean_dec_ref(v___y_1887_);
return v_res_1894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16(lean_object* v_00_u03b1_1895_, lean_object* v_ref_1896_, lean_object* v_msg_1897_, lean_object* v_declHint_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v___x_1906_; 
v___x_1906_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___redArg(v_ref_1896_, v_msg_1897_, v_declHint_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_);
return v___x_1906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16___boxed(lean_object* v_00_u03b1_1907_, lean_object* v_ref_1908_, lean_object* v_msg_1909_, lean_object* v_declHint_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v_res_1918_; 
v_res_1918_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16(v_00_u03b1_1907_, v_ref_1908_, v_msg_1909_, v_declHint_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_, v___y_1916_);
lean_dec(v___y_1916_);
lean_dec_ref(v___y_1915_);
lean_dec(v___y_1914_);
lean_dec_ref(v___y_1913_);
lean_dec(v___y_1912_);
lean_dec_ref(v___y_1911_);
lean_dec(v_ref_1908_);
return v_res_1918_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9(lean_object* v_00_u03b2_1919_, lean_object* v_x_1920_, lean_object* v_x_1921_){
_start:
{
uint8_t v___x_1922_; 
v___x_1922_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___redArg(v_x_1920_, v_x_1921_);
return v___x_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9___boxed(lean_object* v_00_u03b2_1923_, lean_object* v_x_1924_, lean_object* v_x_1925_){
_start:
{
uint8_t v_res_1926_; lean_object* v_r_1927_; 
v_res_1926_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9(v_00_u03b2_1923_, v_x_1924_, v_x_1925_);
lean_dec_ref(v_x_1925_);
lean_dec_ref(v_x_1924_);
v_r_1927_ = lean_box(v_res_1926_);
return v_r_1927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12(lean_object* v_00_u03b2_1928_, lean_object* v_a_1929_, lean_object* v_x_1930_){
_start:
{
lean_object* v___x_1931_; 
v___x_1931_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___redArg(v_a_1929_, v_x_1930_);
return v___x_1931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12___boxed(lean_object* v_00_u03b2_1932_, lean_object* v_a_1933_, lean_object* v_x_1934_){
_start:
{
lean_object* v_res_1935_; 
v_res_1935_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__7_spec__12(v_00_u03b2_1932_, v_a_1933_, v_x_1934_);
lean_dec(v_x_1934_);
lean_dec(v_a_1933_);
return v_res_1935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17(lean_object* v_msgData_1936_, lean_object* v_macroStack_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_){
_start:
{
lean_object* v___x_1945_; 
v___x_1945_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___redArg(v_msgData_1936_, v_macroStack_1937_, v___y_1942_);
return v___x_1945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17___boxed(lean_object* v_msgData_1946_, lean_object* v_macroStack_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_){
_start:
{
lean_object* v_res_1955_; 
v_res_1955_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__4_spec__11_spec__17(v_msgData_1946_, v_macroStack_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_, v___y_1953_);
lean_dec(v___y_1953_);
lean_dec_ref(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
return v_res_1955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24(lean_object* v_msg_1956_, lean_object* v_declHint_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_){
_start:
{
lean_object* v___x_1965_; 
v___x_1965_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___redArg(v_msg_1956_, v_declHint_1957_, v___y_1963_);
return v___x_1965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24___boxed(lean_object* v_msg_1966_, lean_object* v_declHint_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_){
_start:
{
lean_object* v_res_1975_; 
v_res_1975_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processBinOp_spec__3_spec__10_spec__16_spec__20_spec__24(v_msg_1966_, v_declHint_1967_, v___y_1968_, v___y_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec(v___y_1971_);
lean_dec_ref(v___y_1970_);
lean_dec(v___y_1969_);
lean_dec_ref(v___y_1968_);
return v_res_1975_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14(lean_object* v_00_u03b2_1976_, lean_object* v_x_1977_, size_t v_x_1978_, lean_object* v_x_1979_){
_start:
{
uint8_t v___x_1980_; 
v___x_1980_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___redArg(v_x_1977_, v_x_1978_, v_x_1979_);
return v___x_1980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14___boxed(lean_object* v_00_u03b2_1981_, lean_object* v_x_1982_, lean_object* v_x_1983_, lean_object* v_x_1984_){
_start:
{
size_t v_x_28478__boxed_1985_; uint8_t v_res_1986_; lean_object* v_r_1987_; 
v_x_28478__boxed_1985_ = lean_unbox_usize(v_x_1983_);
lean_dec(v_x_1983_);
v_res_1986_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14(v_00_u03b2_1981_, v_x_1982_, v_x_28478__boxed_1985_, v_x_1984_);
lean_dec_ref(v_x_1984_);
lean_dec_ref(v_x_1982_);
v_r_1987_ = lean_box(v_res_1986_);
return v_r_1987_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19(lean_object* v_00_u03b2_1988_, lean_object* v_keys_1989_, lean_object* v_vals_1990_, lean_object* v_heq_1991_, lean_object* v_i_1992_, lean_object* v_k_1993_){
_start:
{
uint8_t v___x_1994_; 
v___x_1994_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___redArg(v_keys_1989_, v_i_1992_, v_k_1993_);
return v___x_1994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19___boxed(lean_object* v_00_u03b2_1995_, lean_object* v_keys_1996_, lean_object* v_vals_1997_, lean_object* v_heq_1998_, lean_object* v_i_1999_, lean_object* v_k_2000_){
_start:
{
uint8_t v_res_2001_; lean_object* v_r_2002_; 
v_res_2001_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5_spec__9_spec__14_spec__19(v_00_u03b2_1995_, v_keys_1996_, v_vals_1997_, v_heq_1998_, v_i_1999_, v_k_2000_);
lean_dec_ref(v_k_2000_);
lean_dec_ref(v_vals_1997_);
lean_dec_ref(v_keys_1996_);
v_r_2002_ = lean_box(v_res_2001_);
return v_r_2002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree(lean_object* v_s_2003_, lean_object* v_a_2004_, lean_object* v_a_2005_, lean_object* v_a_2006_, lean_object* v_a_2007_, lean_object* v_a_2008_, lean_object* v_a_2009_){
_start:
{
lean_object* v___x_2011_; 
v___x_2011_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go(v_s_2003_, v_a_2004_, v_a_2005_, v_a_2006_, v_a_2007_, v_a_2008_, v_a_2009_);
if (lean_obj_tag(v___x_2011_) == 0)
{
lean_object* v_a_2012_; uint8_t v___x_2013_; uint8_t v___x_2014_; lean_object* v___x_2015_; 
v_a_2012_ = lean_ctor_get(v___x_2011_, 0);
lean_inc(v_a_2012_);
lean_dec_ref_known(v___x_2011_, 1);
v___x_2013_ = 0;
v___x_2014_ = 0;
v___x_2015_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_2013_, v___x_2014_, v_a_2004_, v_a_2005_, v_a_2006_, v_a_2007_, v_a_2008_, v_a_2009_);
if (lean_obj_tag(v___x_2015_) == 0)
{
lean_object* v___x_2017_; uint8_t v_isShared_2018_; uint8_t v_isSharedCheck_2022_; 
v_isSharedCheck_2022_ = !lean_is_exclusive(v___x_2015_);
if (v_isSharedCheck_2022_ == 0)
{
lean_object* v_unused_2023_; 
v_unused_2023_ = lean_ctor_get(v___x_2015_, 0);
lean_dec(v_unused_2023_);
v___x_2017_ = v___x_2015_;
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
else
{
lean_dec(v___x_2015_);
v___x_2017_ = lean_box(0);
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
v_resetjp_2016_:
{
lean_object* v___x_2020_; 
if (v_isShared_2018_ == 0)
{
lean_ctor_set(v___x_2017_, 0, v_a_2012_);
v___x_2020_ = v___x_2017_;
goto v_reusejp_2019_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v_a_2012_);
v___x_2020_ = v_reuseFailAlloc_2021_;
goto v_reusejp_2019_;
}
v_reusejp_2019_:
{
return v___x_2020_;
}
}
}
else
{
lean_object* v_a_2024_; lean_object* v___x_2026_; uint8_t v_isShared_2027_; uint8_t v_isSharedCheck_2031_; 
lean_dec(v_a_2012_);
v_a_2024_ = lean_ctor_get(v___x_2015_, 0);
v_isSharedCheck_2031_ = !lean_is_exclusive(v___x_2015_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_2026_ = v___x_2015_;
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
else
{
lean_inc(v_a_2024_);
lean_dec(v___x_2015_);
v___x_2026_ = lean_box(0);
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
v_resetjp_2025_:
{
lean_object* v___x_2029_; 
if (v_isShared_2027_ == 0)
{
v___x_2029_ = v___x_2026_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2030_; 
v_reuseFailAlloc_2030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2030_, 0, v_a_2024_);
v___x_2029_ = v_reuseFailAlloc_2030_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
return v___x_2029_;
}
}
}
}
else
{
return v___x_2011_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree___boxed(lean_object* v_s_2032_, lean_object* v_a_2033_, lean_object* v_a_2034_, lean_object* v_a_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_, lean_object* v_a_2039_){
_start:
{
lean_object* v_res_2040_; 
v_res_2040_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree(v_s_2032_, v_a_2033_, v_a_2034_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
lean_dec(v_a_2038_);
lean_dec_ref(v_a_2037_);
lean_dec(v_a_2036_);
lean_dec_ref(v_a_2035_);
lean_dec(v_a_2034_);
lean_dec_ref(v_a_2033_);
return v_res_2040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0(lean_object* v_nilFn_2048_, lean_object* v_consFn_2049_, lean_object* v_x_2050_){
_start:
{
if (lean_obj_tag(v_x_2050_) == 0)
{
lean_dec_ref(v_consFn_2049_);
lean_inc_ref(v_nilFn_2048_);
return v_nilFn_2048_;
}
else
{
lean_object* v_head_2051_; lean_object* v_tail_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; 
v_head_2051_ = lean_ctor_get(v_x_2050_, 0);
lean_inc(v_head_2051_);
v_tail_2052_ = lean_ctor_get(v_x_2050_, 1);
lean_inc(v_tail_2052_);
lean_dec_ref_known(v_x_2050_, 2);
v___x_2053_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_head_2051_);
lean_inc_ref(v_consFn_2049_);
v___x_2054_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0(v_nilFn_2048_, v_consFn_2049_, v_tail_2052_);
v___x_2055_ = l_Lean_mkAppB(v_consFn_2049_, v___x_2053_, v___x_2054_);
return v___x_2055_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0___boxed(lean_object* v_nilFn_2056_, lean_object* v_consFn_2057_, lean_object* v_x_2058_){
_start:
{
lean_object* v_res_2059_; 
v_res_2059_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0(v_nilFn_2056_, v_consFn_2057_, v_x_2058_);
lean_dec_ref(v_nilFn_2056_);
return v_res_2059_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3(void){
_start:
{
lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; 
v___x_2066_ = lean_box(0);
v___x_2067_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__2));
v___x_2068_ = l_Lean_Expr_const___override(v___x_2067_, v___x_2066_);
return v___x_2068_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6(void){
_start:
{
lean_object* v___x_2073_; lean_object* v___x_2074_; lean_object* v_type_2075_; 
v___x_2073_ = lean_box(0);
v___x_2074_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__5));
v_type_2075_ = l_Lean_Expr_const___override(v___x_2074_, v___x_2073_);
return v_type_2075_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11(void){
_start:
{
lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; 
v___x_2084_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10));
v___x_2085_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__9));
v___x_2086_ = l_Lean_mkConst(v___x_2085_, v___x_2084_);
return v___x_2086_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14(void){
_start:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; 
v___x_2091_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10));
v___x_2092_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__13));
v___x_2093_ = l_Lean_mkConst(v___x_2092_, v___x_2091_);
return v___x_2093_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15(void){
_start:
{
lean_object* v_type_2094_; lean_object* v___x_2095_; lean_object* v_nil_2096_; 
v_type_2094_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6);
v___x_2095_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__14);
v_nil_2096_ = l_Lean_Expr_app___override(v___x_2095_, v_type_2094_);
return v_nil_2096_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18(void){
_start:
{
lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; 
v___x_2101_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10));
v___x_2102_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__17));
v___x_2103_ = l_Lean_mkConst(v___x_2102_, v___x_2101_);
return v___x_2103_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19(void){
_start:
{
lean_object* v_type_2104_; lean_object* v___x_2105_; lean_object* v_cons_2106_; 
v_type_2104_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6);
v___x_2105_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__18);
v_cons_2106_ = l_Lean_Expr_app___override(v___x_2105_, v_type_2104_);
return v_cons_2106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_instToExprSRec_toExpr(lean_object* v_x_2107_){
_start:
{
lean_object* v_name_2108_; lean_object* v_args_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v_type_2113_; lean_object* v___x_2114_; lean_object* v_nil_2115_; lean_object* v_cons_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; 
v_name_2108_ = lean_ctor_get(v_x_2107_, 0);
lean_inc(v_name_2108_);
v_args_2109_ = lean_ctor_get(v_x_2107_, 1);
lean_inc_ref(v_args_2109_);
lean_dec_ref(v_x_2107_);
v___x_2110_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__3);
v___x_2111_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_name_2108_);
v___x_2112_ = l_Lean_Expr_app___override(v___x_2110_, v___x_2111_);
v_type_2113_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__6);
v___x_2114_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__11);
v_nil_2115_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__15);
v_cons_2116_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19, &lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19_once, _init_lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__19);
v___x_2117_ = lean_array_to_list(v_args_2109_);
v___x_2118_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00FBinopElab_instToExprSRec_toExpr_spec__0(v_nil_2115_, v_cons_2116_, v___x_2117_);
v___x_2119_ = l_Lean_mkAppB(v___x_2114_, v_type_2113_, v___x_2118_);
v___x_2120_ = l_Lean_Expr_app___override(v___x_2112_, v___x_2119_);
return v___x_2120_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec___closed__2(void){
_start:
{
lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; 
v___x_2125_ = lean_box(0);
v___x_2126_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec___closed__1));
v___x_2127_ = l_Lean_Expr_const___override(v___x_2126_, v___x_2125_);
return v___x_2127_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec___closed__3(void){
_start:
{
lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; 
v___x_2128_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec___closed__2, &lp_mathlib_FBinopElab_instToExprSRec___closed__2_once, _init_lp_mathlib_FBinopElab_instToExprSRec___closed__2);
v___x_2129_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec___closed__0));
v___x_2130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2130_, 0, v___x_2129_);
lean_ctor_set(v___x_2130_, 1, v___x_2128_);
return v___x_2130_;
}
}
static lean_object* _init_lp_mathlib_FBinopElab_instToExprSRec(void){
_start:
{
lean_object* v___x_2131_; 
v___x_2131_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec___closed__3, &lp_mathlib_FBinopElab_instToExprSRec___closed__3_once, _init_lp_mathlib_FBinopElab_instToExprSRec___closed__3);
return v___x_2131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg(lean_object* v_range_2132_, lean_object* v_b_2133_, lean_object* v_i_2134_){
_start:
{
lean_object* v_stop_2136_; lean_object* v_step_2137_; uint8_t v___x_2138_; 
v_stop_2136_ = lean_ctor_get(v_range_2132_, 1);
v_step_2137_ = lean_ctor_get(v_range_2132_, 2);
v___x_2138_ = lean_nat_dec_lt(v_i_2134_, v_stop_2136_);
if (v___x_2138_ == 0)
{
lean_object* v___x_2139_; 
lean_dec(v_i_2134_);
v___x_2139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2139_, 0, v_b_2133_);
return v___x_2139_;
}
else
{
lean_object* v_fst_2140_; lean_object* v_snd_2141_; lean_object* v___x_2143_; uint8_t v_isShared_2144_; uint8_t v_isSharedCheck_2162_; 
v_fst_2140_ = lean_ctor_get(v_b_2133_, 0);
v_snd_2141_ = lean_ctor_get(v_b_2133_, 1);
v_isSharedCheck_2162_ = !lean_is_exclusive(v_b_2133_);
if (v_isSharedCheck_2162_ == 0)
{
v___x_2143_ = v_b_2133_;
v_isShared_2144_ = v_isSharedCheck_2162_;
goto v_resetjp_2142_;
}
else
{
lean_inc(v_snd_2141_);
lean_inc(v_fst_2140_);
lean_dec(v_b_2133_);
v___x_2143_ = lean_box(0);
v_isShared_2144_ = v_isSharedCheck_2162_;
goto v_resetjp_2142_;
}
v_resetjp_2142_:
{
lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; uint8_t v___x_2150_; 
v___x_2145_ = l_Lean_Meta_instInhabitedParamInfo_default;
v___x_2146_ = lean_array_get_size(v_snd_2141_);
v___x_2147_ = lean_unsigned_to_nat(1u);
v___x_2148_ = lean_nat_sub(v___x_2146_, v___x_2147_);
v___x_2149_ = lean_array_get_borrowed(v___x_2145_, v_snd_2141_, v___x_2148_);
lean_dec(v___x_2148_);
v___x_2150_ = l_Lean_Meta_ParamInfo_isInstImplicit(v___x_2149_);
if (v___x_2150_ == 0)
{
lean_object* v___x_2152_; 
lean_dec(v_i_2134_);
if (v_isShared_2144_ == 0)
{
v___x_2152_ = v___x_2143_;
goto v_reusejp_2151_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v_fst_2140_);
lean_ctor_set(v_reuseFailAlloc_2154_, 1, v_snd_2141_);
v___x_2152_ = v_reuseFailAlloc_2154_;
goto v_reusejp_2151_;
}
v_reusejp_2151_:
{
lean_object* v___x_2153_; 
v___x_2153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2153_, 0, v___x_2152_);
return v___x_2153_;
}
}
else
{
lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2158_; 
v___x_2155_ = lean_array_pop(v_fst_2140_);
v___x_2156_ = lean_array_pop(v_snd_2141_);
if (v_isShared_2144_ == 0)
{
lean_ctor_set(v___x_2143_, 1, v___x_2156_);
lean_ctor_set(v___x_2143_, 0, v___x_2155_);
v___x_2158_ = v___x_2143_;
goto v_reusejp_2157_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v___x_2155_);
lean_ctor_set(v_reuseFailAlloc_2161_, 1, v___x_2156_);
v___x_2158_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2157_;
}
v_reusejp_2157_:
{
lean_object* v___x_2159_; 
v___x_2159_ = lean_nat_add(v_i_2134_, v_step_2137_);
lean_dec(v_i_2134_);
v_b_2133_ = v___x_2158_;
v_i_2134_ = v___x_2159_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg___boxed(lean_object* v_range_2163_, lean_object* v_b_2164_, lean_object* v_i_2165_, lean_object* v___y_2166_){
_start:
{
lean_object* v_res_2167_; 
v_res_2167_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg(v_range_2163_, v_b_2164_, v_i_2165_);
lean_dec_ref(v_range_2163_);
return v_res_2167_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0(void){
_start:
{
lean_object* v___x_2168_; lean_object* v_dummy_2169_; 
v___x_2168_ = lean_box(0);
v_dummy_2169_ = l_Lean_Expr_sort___override(v___x_2168_);
return v_dummy_2169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(lean_object* v_e_2170_, lean_object* v_a_2171_, lean_object* v_a_2172_, lean_object* v_a_2173_, lean_object* v_a_2174_, lean_object* v_a_2175_, lean_object* v_a_2176_){
_start:
{
switch(lean_obj_tag(v_e_2170_))
{
case 8:
{
lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; 
v___x_2178_ = l_Lean_Expr_letBody_x21(v_e_2170_);
v___x_2179_ = l_Lean_Expr_letValue_x21(v_e_2170_);
lean_dec_ref_known(v_e_2170_, 4);
v___x_2180_ = lean_expr_instantiate1(v___x_2178_, v___x_2179_);
lean_dec_ref(v___x_2179_);
lean_dec_ref(v___x_2178_);
v_e_2170_ = v___x_2180_;
goto _start;
}
case 10:
{
lean_object* v_expr_2182_; 
v_expr_2182_ = lean_ctor_get(v_e_2170_, 1);
lean_inc_ref(v_expr_2182_);
lean_dec_ref_known(v_e_2170_, 2);
v_e_2170_ = v_expr_2182_;
goto _start;
}
case 5:
{
lean_object* v_f_2184_; 
v_f_2184_ = l_Lean_Expr_getAppFn(v_e_2170_);
if (lean_obj_tag(v_f_2184_) == 4)
{
lean_object* v_declName_2185_; lean_object* v_dummy_2186_; lean_object* v_nargs_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v_args_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; 
v_declName_2185_ = lean_ctor_get(v_f_2184_, 0);
lean_inc(v_declName_2185_);
v_dummy_2186_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___closed__0);
v_nargs_2187_ = l_Lean_Expr_getAppNumArgs(v_e_2170_);
lean_inc(v_nargs_2187_);
v___x_2188_ = lean_mk_array(v_nargs_2187_, v_dummy_2186_);
v___x_2189_ = lean_unsigned_to_nat(1u);
v___x_2190_ = lean_nat_sub(v_nargs_2187_, v___x_2189_);
lean_dec(v_nargs_2187_);
v_args_2191_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_e_2170_, v___x_2188_, v___x_2190_);
v___x_2192_ = lean_array_get_size(v_args_2191_);
v___x_2193_ = l_Lean_Meta_getFunInfoNArgs(v_f_2184_, v___x_2192_, v_a_2173_, v_a_2174_, v_a_2175_, v_a_2176_);
if (lean_obj_tag(v___x_2193_) == 0)
{
lean_object* v_a_2194_; lean_object* v_paramInfo_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2253_; 
v_a_2194_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2194_);
lean_dec_ref_known(v___x_2193_, 1);
v_paramInfo_2195_ = lean_ctor_get(v_a_2194_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v_a_2194_);
if (v_isSharedCheck_2253_ == 0)
{
lean_object* v_unused_2254_; 
v_unused_2254_ = lean_ctor_get(v_a_2194_, 1);
lean_dec(v_unused_2254_);
v___x_2197_ = v_a_2194_;
v_isShared_2198_ = v_isSharedCheck_2253_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_paramInfo_2195_);
lean_dec(v_a_2194_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2253_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2203_; 
v___x_2199_ = lean_unsigned_to_nat(0u);
v___x_2200_ = lean_nat_sub(v___x_2192_, v___x_2189_);
v___x_2201_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2201_, 0, v___x_2199_);
lean_ctor_set(v___x_2201_, 1, v___x_2200_);
lean_ctor_set(v___x_2201_, 2, v___x_2189_);
if (v_isShared_2198_ == 0)
{
lean_ctor_set(v___x_2197_, 1, v_paramInfo_2195_);
lean_ctor_set(v___x_2197_, 0, v_args_2191_);
v___x_2203_ = v___x_2197_;
goto v_reusejp_2202_;
}
else
{
lean_object* v_reuseFailAlloc_2252_; 
v_reuseFailAlloc_2252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2252_, 0, v_args_2191_);
lean_ctor_set(v_reuseFailAlloc_2252_, 1, v_paramInfo_2195_);
v___x_2203_ = v_reuseFailAlloc_2252_;
goto v_reusejp_2202_;
}
v_reusejp_2202_:
{
lean_object* v___x_2204_; 
v___x_2204_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg(v___x_2201_, v___x_2203_, v___x_2199_);
lean_dec_ref_known(v___x_2201_, 3);
if (lean_obj_tag(v___x_2204_) == 0)
{
lean_object* v_a_2205_; lean_object* v_fst_2206_; lean_object* v___x_2208_; uint8_t v_isShared_2209_; uint8_t v_isSharedCheck_2242_; 
v_a_2205_ = lean_ctor_get(v___x_2204_, 0);
lean_inc(v_a_2205_);
lean_dec_ref_known(v___x_2204_, 1);
v_fst_2206_ = lean_ctor_get(v_a_2205_, 0);
v_isSharedCheck_2242_ = !lean_is_exclusive(v_a_2205_);
if (v_isSharedCheck_2242_ == 0)
{
lean_object* v_unused_2243_; 
v_unused_2243_ = lean_ctor_get(v_a_2205_, 1);
lean_dec(v_unused_2243_);
v___x_2208_ = v_a_2205_;
v_isShared_2209_ = v_isSharedCheck_2242_;
goto v_resetjp_2207_;
}
else
{
lean_inc(v_fst_2206_);
lean_dec(v_a_2205_);
v___x_2208_ = lean_box(0);
v_isShared_2209_ = v_isSharedCheck_2242_;
goto v_resetjp_2207_;
}
v_resetjp_2207_:
{
lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; 
v___x_2210_ = l_Lean_instInhabitedExpr;
v___x_2211_ = lean_array_get_size(v_fst_2206_);
v___x_2212_ = lean_nat_sub(v___x_2211_, v___x_2189_);
v___x_2213_ = lean_array_get(v___x_2210_, v_fst_2206_, v___x_2212_);
lean_dec(v___x_2212_);
lean_inc(v___x_2213_);
v___x_2214_ = l_Lean_Meta_isType(v___x_2213_, v_a_2173_, v_a_2174_, v_a_2175_, v_a_2176_);
if (lean_obj_tag(v___x_2214_) == 0)
{
lean_object* v_a_2215_; lean_object* v___x_2217_; uint8_t v_isShared_2218_; uint8_t v_isSharedCheck_2233_; 
v_a_2215_ = lean_ctor_get(v___x_2214_, 0);
v_isSharedCheck_2233_ = !lean_is_exclusive(v___x_2214_);
if (v_isSharedCheck_2233_ == 0)
{
v___x_2217_ = v___x_2214_;
v_isShared_2218_ = v_isSharedCheck_2233_;
goto v_resetjp_2216_;
}
else
{
lean_inc(v_a_2215_);
lean_dec(v___x_2214_);
v___x_2217_ = lean_box(0);
v_isShared_2218_ = v_isSharedCheck_2233_;
goto v_resetjp_2216_;
}
v_resetjp_2216_:
{
uint8_t v___x_2219_; 
v___x_2219_ = lean_unbox(v_a_2215_);
lean_dec(v_a_2215_);
if (v___x_2219_ == 0)
{
lean_object* v___x_2220_; lean_object* v___x_2222_; 
lean_dec(v___x_2213_);
lean_del_object(v___x_2208_);
lean_dec(v_fst_2206_);
lean_dec(v_declName_2185_);
v___x_2220_ = lean_box(0);
if (v_isShared_2218_ == 0)
{
lean_ctor_set(v___x_2217_, 0, v___x_2220_);
v___x_2222_ = v___x_2217_;
goto v_reusejp_2221_;
}
else
{
lean_object* v_reuseFailAlloc_2223_; 
v_reuseFailAlloc_2223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2223_, 0, v___x_2220_);
v___x_2222_ = v_reuseFailAlloc_2223_;
goto v_reusejp_2221_;
}
v_reusejp_2221_:
{
return v___x_2222_;
}
}
else
{
lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2227_; 
v___x_2224_ = lean_array_pop(v_fst_2206_);
v___x_2225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2225_, 0, v_declName_2185_);
lean_ctor_set(v___x_2225_, 1, v___x_2224_);
if (v_isShared_2209_ == 0)
{
lean_ctor_set(v___x_2208_, 1, v___x_2213_);
lean_ctor_set(v___x_2208_, 0, v___x_2225_);
v___x_2227_ = v___x_2208_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2232_; 
v_reuseFailAlloc_2232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2232_, 0, v___x_2225_);
lean_ctor_set(v_reuseFailAlloc_2232_, 1, v___x_2213_);
v___x_2227_ = v_reuseFailAlloc_2232_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
lean_object* v___x_2228_; lean_object* v___x_2230_; 
v___x_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2228_, 0, v___x_2227_);
if (v_isShared_2218_ == 0)
{
lean_ctor_set(v___x_2217_, 0, v___x_2228_);
v___x_2230_ = v___x_2217_;
goto v_reusejp_2229_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v___x_2228_);
v___x_2230_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2229_;
}
v_reusejp_2229_:
{
return v___x_2230_;
}
}
}
}
}
else
{
lean_object* v_a_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2241_; 
lean_dec(v___x_2213_);
lean_del_object(v___x_2208_);
lean_dec(v_fst_2206_);
lean_dec(v_declName_2185_);
v_a_2234_ = lean_ctor_get(v___x_2214_, 0);
v_isSharedCheck_2241_ = !lean_is_exclusive(v___x_2214_);
if (v_isSharedCheck_2241_ == 0)
{
v___x_2236_ = v___x_2214_;
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_a_2234_);
lean_dec(v___x_2214_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
v_resetjp_2235_:
{
lean_object* v___x_2239_; 
if (v_isShared_2237_ == 0)
{
v___x_2239_ = v___x_2236_;
goto v_reusejp_2238_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v_a_2234_);
v___x_2239_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2238_;
}
v_reusejp_2238_:
{
return v___x_2239_;
}
}
}
}
}
else
{
lean_object* v_a_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2251_; 
lean_dec(v_declName_2185_);
v_a_2244_ = lean_ctor_get(v___x_2204_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v___x_2204_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2246_ = v___x_2204_;
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_a_2244_);
lean_dec(v___x_2204_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2249_; 
if (v_isShared_2247_ == 0)
{
v___x_2249_ = v___x_2246_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v_a_2244_);
v___x_2249_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
return v___x_2249_;
}
}
}
}
}
}
else
{
lean_object* v_a_2255_; lean_object* v___x_2257_; uint8_t v_isShared_2258_; uint8_t v_isSharedCheck_2262_; 
lean_dec_ref(v_args_2191_);
lean_dec(v_declName_2185_);
v_a_2255_ = lean_ctor_get(v___x_2193_, 0);
v_isSharedCheck_2262_ = !lean_is_exclusive(v___x_2193_);
if (v_isSharedCheck_2262_ == 0)
{
v___x_2257_ = v___x_2193_;
v_isShared_2258_ = v_isSharedCheck_2262_;
goto v_resetjp_2256_;
}
else
{
lean_inc(v_a_2255_);
lean_dec(v___x_2193_);
v___x_2257_ = lean_box(0);
v_isShared_2258_ = v_isSharedCheck_2262_;
goto v_resetjp_2256_;
}
v_resetjp_2256_:
{
lean_object* v___x_2260_; 
if (v_isShared_2258_ == 0)
{
v___x_2260_ = v___x_2257_;
goto v_reusejp_2259_;
}
else
{
lean_object* v_reuseFailAlloc_2261_; 
v_reuseFailAlloc_2261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2261_, 0, v_a_2255_);
v___x_2260_ = v_reuseFailAlloc_2261_;
goto v_reusejp_2259_;
}
v_reusejp_2259_:
{
return v___x_2260_;
}
}
}
}
else
{
lean_object* v___x_2263_; lean_object* v___x_2264_; 
lean_dec_ref(v_f_2184_);
lean_dec_ref_known(v_e_2170_, 2);
v___x_2263_ = lean_box(0);
v___x_2264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2264_, 0, v___x_2263_);
return v___x_2264_;
}
}
default: 
{
lean_object* v___x_2265_; lean_object* v___x_2266_; 
lean_dec_ref(v_e_2170_);
v___x_2265_ = lean_box(0);
v___x_2266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2266_, 0, v___x_2265_);
return v___x_2266_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS___boxed(lean_object* v_e_2267_, lean_object* v_a_2268_, lean_object* v_a_2269_, lean_object* v_a_2270_, lean_object* v_a_2271_, lean_object* v_a_2272_, lean_object* v_a_2273_, lean_object* v_a_2274_){
_start:
{
lean_object* v_res_2275_; 
v_res_2275_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(v_e_2267_, v_a_2268_, v_a_2269_, v_a_2270_, v_a_2271_, v_a_2272_, v_a_2273_);
lean_dec(v_a_2273_);
lean_dec_ref(v_a_2272_);
lean_dec(v_a_2271_);
lean_dec_ref(v_a_2270_);
lean_dec(v_a_2269_);
lean_dec_ref(v_a_2268_);
return v_res_2275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0(lean_object* v_range_2276_, lean_object* v_b_2277_, lean_object* v_i_2278_, lean_object* v_hs_2279_, lean_object* v_hl_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_){
_start:
{
lean_object* v___x_2288_; 
v___x_2288_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___redArg(v_range_2276_, v_b_2277_, v_i_2278_);
return v___x_2288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0___boxed(lean_object* v_range_2289_, lean_object* v_b_2290_, lean_object* v_i_2291_, lean_object* v_hs_2292_, lean_object* v_hl_2293_, lean_object* v___y_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_){
_start:
{
lean_object* v_res_2301_; 
v_res_2301_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS_spec__0(v_range_2289_, v_b_2290_, v_i_2291_, v_hs_2292_, v_hl_2293_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec_ref(v_range_2289_);
return v_res_2301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0(size_t v_sz_2302_, size_t v_i_2303_, lean_object* v_bs_2304_){
_start:
{
uint8_t v___x_2305_; 
v___x_2305_ = lean_usize_dec_lt(v_i_2303_, v_sz_2302_);
if (v___x_2305_ == 0)
{
return v_bs_2304_;
}
else
{
lean_object* v_v_2306_; lean_object* v___x_2307_; lean_object* v_bs_x27_2308_; lean_object* v___x_2309_; size_t v___x_2310_; size_t v___x_2311_; lean_object* v___x_2312_; 
v_v_2306_ = lean_array_uget(v_bs_2304_, v_i_2303_);
v___x_2307_ = lean_unsigned_to_nat(0u);
v_bs_x27_2308_ = lean_array_uset(v_bs_2304_, v_i_2303_, v___x_2307_);
v___x_2309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2309_, 0, v_v_2306_);
v___x_2310_ = ((size_t)1ULL);
v___x_2311_ = lean_usize_add(v_i_2303_, v___x_2310_);
v___x_2312_ = lean_array_uset(v_bs_x27_2308_, v_i_2303_, v___x_2309_);
v_i_2303_ = v___x_2311_;
v_bs_2304_ = v___x_2312_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0___boxed(lean_object* v_sz_2314_, lean_object* v_i_2315_, lean_object* v_bs_2316_){
_start:
{
size_t v_sz_boxed_2317_; size_t v_i_boxed_2318_; lean_object* v_res_2319_; 
v_sz_boxed_2317_ = lean_unbox_usize(v_sz_2314_);
lean_dec(v_sz_2314_);
v_i_boxed_2318_ = lean_unbox_usize(v_i_2315_);
lean_dec(v_i_2315_);
v_res_2319_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0(v_sz_boxed_2317_, v_i_boxed_2318_, v_bs_2316_);
return v_res_2319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(lean_object* v_S_2322_, lean_object* v_x_2323_, lean_object* v_a_2324_, lean_object* v_a_2325_, lean_object* v_a_2326_, lean_object* v_a_2327_, lean_object* v_a_2328_, lean_object* v_a_2329_){
_start:
{
lean_object* v___y_2332_; uint8_t v___y_2333_; lean_object* v_a_2338_; lean_object* v_name_2341_; lean_object* v_args_2342_; lean_object* v___x_2343_; 
v_name_2341_ = lean_ctor_get(v_S_2322_, 0);
lean_inc(v_name_2341_);
v_args_2342_ = lean_ctor_get(v_S_2322_, 1);
lean_inc_ref(v_args_2342_);
lean_dec_ref(v_S_2322_);
v___x_2343_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_name_2341_, v_a_2326_, v_a_2327_, v_a_2328_, v_a_2329_);
if (lean_obj_tag(v___x_2343_) == 0)
{
lean_object* v_a_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; size_t v_sz_2347_; size_t v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; uint8_t v___x_2351_; uint8_t v___x_2352_; lean_object* v___x_2353_; 
v_a_2344_ = lean_ctor_get(v___x_2343_, 0);
lean_inc(v_a_2344_);
lean_dec_ref_known(v___x_2343_, 1);
v___x_2345_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___closed__0));
v___x_2346_ = lean_array_push(v_args_2342_, v_x_2323_);
v_sz_2347_ = lean_array_size(v___x_2346_);
v___x_2348_ = ((size_t)0ULL);
v___x_2349_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS_spec__0(v_sz_2347_, v___x_2348_, v___x_2346_);
v___x_2350_ = lean_box(0);
v___x_2351_ = 1;
v___x_2352_ = 0;
v___x_2353_ = l_Lean_Elab_Term_elabAppArgs(v_a_2344_, v___x_2345_, v___x_2349_, v___x_2350_, v___x_2351_, v___x_2352_, v___x_2351_, v_a_2324_, v_a_2325_, v_a_2326_, v_a_2327_, v_a_2328_, v_a_2329_);
if (lean_obj_tag(v___x_2353_) == 0)
{
lean_object* v_a_2354_; lean_object* v___x_2356_; uint8_t v_isShared_2357_; uint8_t v_isSharedCheck_2371_; 
v_a_2354_ = lean_ctor_get(v___x_2353_, 0);
v_isSharedCheck_2371_ = !lean_is_exclusive(v___x_2353_);
if (v_isSharedCheck_2371_ == 0)
{
v___x_2356_ = v___x_2353_;
v_isShared_2357_ = v_isSharedCheck_2371_;
goto v_resetjp_2355_;
}
else
{
lean_inc(v_a_2354_);
lean_dec(v___x_2353_);
v___x_2356_ = lean_box(0);
v_isShared_2357_ = v_isSharedCheck_2371_;
goto v_resetjp_2355_;
}
v_resetjp_2355_:
{
lean_object* v___x_2358_; 
v___x_2358_ = l_Lean_Elab_Term_elabAppArgs(v_a_2354_, v___x_2345_, v___x_2345_, v___x_2350_, v___x_2352_, v___x_2352_, v___x_2351_, v_a_2324_, v_a_2325_, v_a_2326_, v_a_2327_, v_a_2328_, v_a_2329_);
if (lean_obj_tag(v___x_2358_) == 0)
{
lean_object* v_a_2359_; lean_object* v___x_2361_; uint8_t v_isShared_2362_; uint8_t v_isSharedCheck_2369_; 
v_a_2359_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2369_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2369_ == 0)
{
v___x_2361_ = v___x_2358_;
v_isShared_2362_ = v_isSharedCheck_2369_;
goto v_resetjp_2360_;
}
else
{
lean_inc(v_a_2359_);
lean_dec(v___x_2358_);
v___x_2361_ = lean_box(0);
v_isShared_2362_ = v_isSharedCheck_2369_;
goto v_resetjp_2360_;
}
v_resetjp_2360_:
{
lean_object* v___x_2364_; 
if (v_isShared_2357_ == 0)
{
lean_ctor_set_tag(v___x_2356_, 1);
lean_ctor_set(v___x_2356_, 0, v_a_2359_);
v___x_2364_ = v___x_2356_;
goto v_reusejp_2363_;
}
else
{
lean_object* v_reuseFailAlloc_2368_; 
v_reuseFailAlloc_2368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2368_, 0, v_a_2359_);
v___x_2364_ = v_reuseFailAlloc_2368_;
goto v_reusejp_2363_;
}
v_reusejp_2363_:
{
lean_object* v___x_2366_; 
if (v_isShared_2362_ == 0)
{
lean_ctor_set(v___x_2361_, 0, v___x_2364_);
v___x_2366_ = v___x_2361_;
goto v_reusejp_2365_;
}
else
{
lean_object* v_reuseFailAlloc_2367_; 
v_reuseFailAlloc_2367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2367_, 0, v___x_2364_);
v___x_2366_ = v_reuseFailAlloc_2367_;
goto v_reusejp_2365_;
}
v_reusejp_2365_:
{
return v___x_2366_;
}
}
}
}
else
{
lean_object* v_a_2370_; 
lean_del_object(v___x_2356_);
v_a_2370_ = lean_ctor_get(v___x_2358_, 0);
lean_inc(v_a_2370_);
lean_dec_ref_known(v___x_2358_, 1);
v_a_2338_ = v_a_2370_;
goto v___jp_2337_;
}
}
}
else
{
lean_object* v_a_2372_; 
v_a_2372_ = lean_ctor_get(v___x_2353_, 0);
lean_inc(v_a_2372_);
lean_dec_ref_known(v___x_2353_, 1);
v_a_2338_ = v_a_2372_;
goto v___jp_2337_;
}
}
else
{
lean_object* v_a_2373_; 
lean_dec_ref(v_args_2342_);
lean_dec_ref(v_x_2323_);
v_a_2373_ = lean_ctor_get(v___x_2343_, 0);
lean_inc(v_a_2373_);
lean_dec_ref_known(v___x_2343_, 1);
v_a_2338_ = v_a_2373_;
goto v___jp_2337_;
}
v___jp_2331_:
{
if (v___y_2333_ == 0)
{
lean_object* v___x_2334_; lean_object* v___x_2335_; 
lean_dec_ref(v___y_2332_);
v___x_2334_ = lean_box(0);
v___x_2335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2335_, 0, v___x_2334_);
return v___x_2335_;
}
else
{
lean_object* v___x_2336_; 
v___x_2336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2336_, 0, v___y_2332_);
return v___x_2336_;
}
}
v___jp_2337_:
{
uint8_t v___x_2339_; 
v___x_2339_ = l_Lean_Exception_isInterrupt(v_a_2338_);
if (v___x_2339_ == 0)
{
uint8_t v___x_2340_; 
lean_inc_ref(v_a_2338_);
v___x_2340_ = l_Lean_Exception_isRuntime(v_a_2338_);
v___y_2332_ = v_a_2338_;
v___y_2333_ = v___x_2340_;
goto v___jp_2331_;
}
else
{
v___y_2332_ = v_a_2338_;
v___y_2333_ = v___x_2339_;
goto v___jp_2331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___boxed(lean_object* v_S_2374_, lean_object* v_x_2375_, lean_object* v_a_2376_, lean_object* v_a_2377_, lean_object* v_a_2378_, lean_object* v_a_2379_, lean_object* v_a_2380_, lean_object* v_a_2381_, lean_object* v_a_2382_){
_start:
{
lean_object* v_res_2383_; 
v_res_2383_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_S_2374_, v_x_2375_, v_a_2376_, v_a_2377_, v_a_2378_, v_a_2379_, v_a_2380_, v_a_2381_);
lean_dec(v_a_2381_);
lean_dec_ref(v_a_2380_);
lean_dec(v_a_2379_);
lean_dec_ref(v_a_2378_);
lean_dec(v_a_2377_);
lean_dec_ref(v_a_2376_);
return v_res_2383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0(lean_object* v_val_2384_, lean_object* v_v_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_){
_start:
{
lean_object* v___x_2393_; 
v___x_2393_ = l_Lean_Meta_coerceSimple_x3f(v_v_2385_, v_val_2384_, v___y_2388_, v___y_2389_, v___y_2390_, v___y_2391_);
if (lean_obj_tag(v___x_2393_) == 0)
{
lean_object* v_a_2394_; lean_object* v___x_2396_; uint8_t v_isShared_2397_; uint8_t v_isSharedCheck_2408_; 
v_a_2394_ = lean_ctor_get(v___x_2393_, 0);
v_isSharedCheck_2408_ = !lean_is_exclusive(v___x_2393_);
if (v_isSharedCheck_2408_ == 0)
{
v___x_2396_ = v___x_2393_;
v_isShared_2397_ = v_isSharedCheck_2408_;
goto v_resetjp_2395_;
}
else
{
lean_inc(v_a_2394_);
lean_dec(v___x_2393_);
v___x_2396_ = lean_box(0);
v_isShared_2397_ = v_isSharedCheck_2408_;
goto v_resetjp_2395_;
}
v_resetjp_2395_:
{
if (lean_obj_tag(v_a_2394_) == 1)
{
uint8_t v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2401_; 
lean_dec_ref_known(v_a_2394_, 1);
v___x_2398_ = 1;
v___x_2399_ = lean_box(v___x_2398_);
if (v_isShared_2397_ == 0)
{
lean_ctor_set(v___x_2396_, 0, v___x_2399_);
v___x_2401_ = v___x_2396_;
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
else
{
uint8_t v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2406_; 
lean_dec(v_a_2394_);
v___x_2403_ = 0;
v___x_2404_ = lean_box(v___x_2403_);
if (v_isShared_2397_ == 0)
{
lean_ctor_set(v___x_2396_, 0, v___x_2404_);
v___x_2406_ = v___x_2396_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v___x_2404_);
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
else
{
lean_object* v_a_2409_; lean_object* v___x_2411_; uint8_t v_isShared_2412_; uint8_t v_isSharedCheck_2416_; 
v_a_2409_ = lean_ctor_get(v___x_2393_, 0);
v_isSharedCheck_2416_ = !lean_is_exclusive(v___x_2393_);
if (v_isSharedCheck_2416_ == 0)
{
v___x_2411_ = v___x_2393_;
v_isShared_2412_ = v_isSharedCheck_2416_;
goto v_resetjp_2410_;
}
else
{
lean_inc(v_a_2409_);
lean_dec(v___x_2393_);
v___x_2411_ = lean_box(0);
v_isShared_2412_ = v_isSharedCheck_2416_;
goto v_resetjp_2410_;
}
v_resetjp_2410_:
{
lean_object* v___x_2414_; 
if (v_isShared_2412_ == 0)
{
v___x_2414_ = v___x_2411_;
goto v_reusejp_2413_;
}
else
{
lean_object* v_reuseFailAlloc_2415_; 
v_reuseFailAlloc_2415_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2415_, 0, v_a_2409_);
v___x_2414_ = v_reuseFailAlloc_2415_;
goto v_reusejp_2413_;
}
v_reusejp_2413_:
{
return v___x_2414_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0___boxed(lean_object* v_val_2417_, lean_object* v_v_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_res_2426_; 
v_res_2426_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0(v_val_2417_, v_v_2418_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
lean_dec(v___y_2424_);
lean_dec_ref(v___y_2423_);
lean_dec(v___y_2422_);
lean_dec_ref(v___y_2421_);
lean_dec(v___y_2420_);
lean_dec_ref(v___y_2419_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0(lean_object* v_k_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_, lean_object* v_b_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_){
_start:
{
lean_object* v___x_2436_; 
lean_inc(v___y_2434_);
lean_inc_ref(v___y_2433_);
lean_inc(v___y_2432_);
lean_inc_ref(v___y_2431_);
lean_inc(v___y_2429_);
lean_inc_ref(v___y_2428_);
v___x_2436_ = lean_apply_8(v_k_2427_, v_b_2430_, v___y_2428_, v___y_2429_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_, lean_box(0));
return v___x_2436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v_b_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_){
_start:
{
lean_object* v_res_2446_; 
v_res_2446_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0(v_k_2437_, v___y_2438_, v___y_2439_, v_b_2440_, v___y_2441_, v___y_2442_, v___y_2443_, v___y_2444_);
lean_dec(v___y_2444_);
lean_dec_ref(v___y_2443_);
lean_dec(v___y_2442_);
lean_dec_ref(v___y_2441_);
lean_dec(v___y_2439_);
lean_dec_ref(v___y_2438_);
return v_res_2446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg(lean_object* v_name_2447_, uint8_t v_bi_2448_, lean_object* v_type_2449_, lean_object* v_k_2450_, uint8_t v_kind_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_){
_start:
{
lean_object* v___f_2459_; lean_object* v___x_2460_; 
lean_inc(v___y_2453_);
lean_inc_ref(v___y_2452_);
v___f_2459_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_2459_, 0, v_k_2450_);
lean_closure_set(v___f_2459_, 1, v___y_2452_);
lean_closure_set(v___f_2459_, 2, v___y_2453_);
v___x_2460_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_2447_, v_bi_2448_, v_type_2449_, v___f_2459_, v_kind_2451_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2460_) == 0)
{
return v___x_2460_;
}
else
{
lean_object* v_a_2461_; lean_object* v___x_2463_; uint8_t v_isShared_2464_; uint8_t v_isSharedCheck_2468_; 
v_a_2461_ = lean_ctor_get(v___x_2460_, 0);
v_isSharedCheck_2468_ = !lean_is_exclusive(v___x_2460_);
if (v_isSharedCheck_2468_ == 0)
{
v___x_2463_ = v___x_2460_;
v_isShared_2464_ = v_isSharedCheck_2468_;
goto v_resetjp_2462_;
}
else
{
lean_inc(v_a_2461_);
lean_dec(v___x_2460_);
v___x_2463_ = lean_box(0);
v_isShared_2464_ = v_isSharedCheck_2468_;
goto v_resetjp_2462_;
}
v_resetjp_2462_:
{
lean_object* v___x_2466_; 
if (v_isShared_2464_ == 0)
{
v___x_2466_ = v___x_2463_;
goto v_reusejp_2465_;
}
else
{
lean_object* v_reuseFailAlloc_2467_; 
v_reuseFailAlloc_2467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2467_, 0, v_a_2461_);
v___x_2466_ = v_reuseFailAlloc_2467_;
goto v_reusejp_2465_;
}
v_reusejp_2465_:
{
return v___x_2466_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg___boxed(lean_object* v_name_2469_, lean_object* v_bi_2470_, lean_object* v_type_2471_, lean_object* v_k_2472_, lean_object* v_kind_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_){
_start:
{
uint8_t v_bi_boxed_2481_; uint8_t v_kind_boxed_2482_; lean_object* v_res_2483_; 
v_bi_boxed_2481_ = lean_unbox(v_bi_2470_);
v_kind_boxed_2482_ = lean_unbox(v_kind_2473_);
v_res_2483_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg(v_name_2469_, v_bi_boxed_2481_, v_type_2471_, v_k_2472_, v_kind_boxed_2482_, v___y_2474_, v___y_2475_, v___y_2476_, v___y_2477_, v___y_2478_, v___y_2479_);
lean_dec(v___y_2479_);
lean_dec_ref(v___y_2478_);
lean_dec(v___y_2477_);
lean_dec_ref(v___y_2476_);
lean_dec(v___y_2475_);
lean_dec_ref(v___y_2474_);
return v_res_2483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg(lean_object* v_name_2484_, lean_object* v_type_2485_, lean_object* v_k_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_, lean_object* v___y_2492_){
_start:
{
uint8_t v___x_2494_; uint8_t v___x_2495_; lean_object* v___x_2496_; 
v___x_2494_ = 0;
v___x_2495_ = 0;
v___x_2496_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg(v_name_2484_, v___x_2494_, v_type_2485_, v_k_2486_, v___x_2495_, v___y_2487_, v___y_2488_, v___y_2489_, v___y_2490_, v___y_2491_, v___y_2492_);
return v___x_2496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg___boxed(lean_object* v_name_2497_, lean_object* v_type_2498_, lean_object* v_k_2499_, lean_object* v___y_2500_, lean_object* v___y_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_){
_start:
{
lean_object* v_res_2507_; 
v_res_2507_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg(v_name_2497_, v_type_2498_, v_k_2499_, v___y_2500_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_);
lean_dec(v___y_2505_);
lean_dec_ref(v___y_2504_);
lean_dec(v___y_2503_);
lean_dec_ref(v___y_2502_);
lean_dec(v___y_2501_);
lean_dec_ref(v___y_2500_);
return v_res_2507_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2(void){
_start:
{
lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; 
v___x_2511_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_2512_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15));
v___x_2513_ = l_Lean_Name_append(v___x_2512_, v___x_2511_);
return v___x_2513_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4(void){
_start:
{
lean_object* v___x_2515_; lean_object* v___x_2516_; 
v___x_2515_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__3));
v___x_2516_ = l_Lean_stringToMessageData(v___x_2515_);
return v___x_2516_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6(void){
_start:
{
lean_object* v___x_2518_; lean_object* v___x_2519_; 
v___x_2518_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__5));
v___x_2519_ = l_Lean_stringToMessageData(v___x_2518_);
return v___x_2519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS(lean_object* v_fromS_2520_, lean_object* v_toS_2521_, lean_object* v_x_2522_, lean_object* v_a_2523_, lean_object* v_a_2524_, lean_object* v_a_2525_, lean_object* v_a_2526_, lean_object* v_a_2527_, lean_object* v_a_2528_){
_start:
{
lean_object* v___x_2530_; 
lean_inc_ref(v_x_2522_);
v___x_2530_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_fromS_2520_, v_x_2522_, v_a_2523_, v_a_2524_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_);
if (lean_obj_tag(v___x_2530_) == 0)
{
lean_object* v_a_2531_; lean_object* v___x_2533_; uint8_t v_isShared_2534_; uint8_t v_isSharedCheck_2593_; 
v_a_2531_ = lean_ctor_get(v___x_2530_, 0);
v_isSharedCheck_2593_ = !lean_is_exclusive(v___x_2530_);
if (v_isSharedCheck_2593_ == 0)
{
v___x_2533_ = v___x_2530_;
v_isShared_2534_ = v_isSharedCheck_2593_;
goto v_resetjp_2532_;
}
else
{
lean_inc(v_a_2531_);
lean_dec(v___x_2530_);
v___x_2533_ = lean_box(0);
v_isShared_2534_ = v_isSharedCheck_2593_;
goto v_resetjp_2532_;
}
v_resetjp_2532_:
{
if (lean_obj_tag(v_a_2531_) == 1)
{
lean_object* v_val_2535_; lean_object* v___x_2536_; 
lean_del_object(v___x_2533_);
v_val_2535_ = lean_ctor_get(v_a_2531_, 0);
lean_inc(v_val_2535_);
lean_dec_ref_known(v_a_2531_, 1);
v___x_2536_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_toS_2521_, v_x_2522_, v_a_2523_, v_a_2524_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_);
if (lean_obj_tag(v___x_2536_) == 0)
{
lean_object* v_a_2537_; lean_object* v___x_2539_; uint8_t v_isShared_2540_; uint8_t v_isSharedCheck_2579_; 
v_a_2537_ = lean_ctor_get(v___x_2536_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2536_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2539_ = v___x_2536_;
v_isShared_2540_ = v_isSharedCheck_2579_;
goto v_resetjp_2538_;
}
else
{
lean_inc(v_a_2537_);
lean_dec(v___x_2536_);
v___x_2539_ = lean_box(0);
v_isShared_2540_ = v_isSharedCheck_2579_;
goto v_resetjp_2538_;
}
v_resetjp_2538_:
{
if (lean_obj_tag(v_a_2537_) == 1)
{
lean_object* v_options_2541_; lean_object* v_val_2542_; lean_object* v_inheritedTraceOptions_2543_; uint8_t v_hasTrace_2544_; lean_object* v___f_2545_; lean_object* v___y_2547_; lean_object* v___y_2548_; lean_object* v___y_2549_; lean_object* v___y_2550_; lean_object* v___y_2551_; lean_object* v___y_2552_; 
lean_del_object(v___x_2539_);
v_options_2541_ = lean_ctor_get(v_a_2527_, 2);
v_val_2542_ = lean_ctor_get(v_a_2537_, 0);
lean_inc_n(v_val_2542_, 2);
lean_dec_ref_known(v_a_2537_, 1);
v_inheritedTraceOptions_2543_ = lean_ctor_get(v_a_2527_, 13);
v_hasTrace_2544_ = lean_ctor_get_uint8(v_options_2541_, sizeof(void*)*1);
v___f_2545_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___lam__0___boxed), 9, 1);
lean_closure_set(v___f_2545_, 0, v_val_2542_);
if (v_hasTrace_2544_ == 0)
{
lean_dec(v_val_2542_);
v___y_2547_ = v_a_2523_;
v___y_2548_ = v_a_2524_;
v___y_2549_ = v_a_2525_;
v___y_2550_ = v_a_2526_;
v___y_2551_ = v_a_2527_;
v___y_2552_ = v_a_2528_;
goto v___jp_2546_;
}
else
{
lean_object* v___x_2555_; lean_object* v___x_2556_; uint8_t v___x_2557_; 
v___x_2555_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_2556_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2);
v___x_2557_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2543_, v_options_2541_, v___x_2556_);
if (v___x_2557_ == 0)
{
lean_dec(v_val_2542_);
v___y_2547_ = v_a_2523_;
v___y_2548_ = v_a_2524_;
v___y_2549_ = v_a_2525_;
v___y_2550_ = v_a_2526_;
v___y_2551_ = v_a_2527_;
v___y_2552_ = v_a_2528_;
goto v___jp_2546_;
}
else
{
lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; 
v___x_2558_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__4);
lean_inc(v_val_2535_);
v___x_2559_ = l_Lean_MessageData_ofExpr(v_val_2535_);
v___x_2560_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2560_, 0, v___x_2558_);
lean_ctor_set(v___x_2560_, 1, v___x_2559_);
v___x_2561_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__6);
v___x_2562_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2562_, 0, v___x_2560_);
lean_ctor_set(v___x_2562_, 1, v___x_2561_);
v___x_2563_ = l_Lean_MessageData_ofExpr(v_val_2542_);
v___x_2564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2564_, 0, v___x_2562_);
lean_ctor_set(v___x_2564_, 1, v___x_2563_);
v___x_2565_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_2555_, v___x_2564_, v_a_2525_, v_a_2526_, v_a_2527_, v_a_2528_);
if (lean_obj_tag(v___x_2565_) == 0)
{
lean_dec_ref_known(v___x_2565_, 1);
v___y_2547_ = v_a_2523_;
v___y_2548_ = v_a_2524_;
v___y_2549_ = v_a_2525_;
v___y_2550_ = v_a_2526_;
v___y_2551_ = v_a_2527_;
v___y_2552_ = v_a_2528_;
goto v___jp_2546_;
}
else
{
lean_object* v_a_2566_; lean_object* v___x_2568_; uint8_t v_isShared_2569_; uint8_t v_isSharedCheck_2573_; 
lean_dec_ref(v___f_2545_);
lean_dec(v_val_2535_);
v_a_2566_ = lean_ctor_get(v___x_2565_, 0);
v_isSharedCheck_2573_ = !lean_is_exclusive(v___x_2565_);
if (v_isSharedCheck_2573_ == 0)
{
v___x_2568_ = v___x_2565_;
v_isShared_2569_ = v_isSharedCheck_2573_;
goto v_resetjp_2567_;
}
else
{
lean_inc(v_a_2566_);
lean_dec(v___x_2565_);
v___x_2568_ = lean_box(0);
v_isShared_2569_ = v_isSharedCheck_2573_;
goto v_resetjp_2567_;
}
v_resetjp_2567_:
{
lean_object* v___x_2571_; 
if (v_isShared_2569_ == 0)
{
v___x_2571_ = v___x_2568_;
goto v_reusejp_2570_;
}
else
{
lean_object* v_reuseFailAlloc_2572_; 
v_reuseFailAlloc_2572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2572_, 0, v_a_2566_);
v___x_2571_ = v_reuseFailAlloc_2572_;
goto v_reusejp_2570_;
}
v_reusejp_2570_:
{
return v___x_2571_;
}
}
}
}
}
v___jp_2546_:
{
lean_object* v___x_2553_; lean_object* v___x_2554_; 
v___x_2553_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__1));
v___x_2554_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg(v___x_2553_, v_val_2535_, v___f_2545_, v___y_2547_, v___y_2548_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_);
return v___x_2554_;
}
}
else
{
uint8_t v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2577_; 
lean_dec(v_a_2537_);
lean_dec(v_val_2535_);
v___x_2574_ = 0;
v___x_2575_ = lean_box(v___x_2574_);
if (v_isShared_2540_ == 0)
{
lean_ctor_set(v___x_2539_, 0, v___x_2575_);
v___x_2577_ = v___x_2539_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v___x_2575_);
v___x_2577_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
return v___x_2577_;
}
}
}
}
else
{
lean_object* v_a_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2587_; 
lean_dec(v_val_2535_);
v_a_2580_ = lean_ctor_get(v___x_2536_, 0);
v_isSharedCheck_2587_ = !lean_is_exclusive(v___x_2536_);
if (v_isSharedCheck_2587_ == 0)
{
v___x_2582_ = v___x_2536_;
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_a_2580_);
lean_dec(v___x_2536_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2583_ == 0)
{
v___x_2585_ = v___x_2582_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2586_; 
v_reuseFailAlloc_2586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2586_, 0, v_a_2580_);
v___x_2585_ = v_reuseFailAlloc_2586_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
return v___x_2585_;
}
}
}
}
else
{
uint8_t v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2591_; 
lean_dec(v_a_2531_);
lean_dec_ref(v_x_2522_);
lean_dec_ref(v_toS_2521_);
v___x_2588_ = 0;
v___x_2589_ = lean_box(v___x_2588_);
if (v_isShared_2534_ == 0)
{
lean_ctor_set(v___x_2533_, 0, v___x_2589_);
v___x_2591_ = v___x_2533_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2592_; 
v_reuseFailAlloc_2592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2592_, 0, v___x_2589_);
v___x_2591_ = v_reuseFailAlloc_2592_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
return v___x_2591_;
}
}
}
}
else
{
lean_object* v_a_2594_; lean_object* v___x_2596_; uint8_t v_isShared_2597_; uint8_t v_isSharedCheck_2601_; 
lean_dec_ref(v_x_2522_);
lean_dec_ref(v_toS_2521_);
v_a_2594_ = lean_ctor_get(v___x_2530_, 0);
v_isSharedCheck_2601_ = !lean_is_exclusive(v___x_2530_);
if (v_isSharedCheck_2601_ == 0)
{
v___x_2596_ = v___x_2530_;
v_isShared_2597_ = v_isSharedCheck_2601_;
goto v_resetjp_2595_;
}
else
{
lean_inc(v_a_2594_);
lean_dec(v___x_2530_);
v___x_2596_ = lean_box(0);
v_isShared_2597_ = v_isSharedCheck_2601_;
goto v_resetjp_2595_;
}
v_resetjp_2595_:
{
lean_object* v___x_2599_; 
if (v_isShared_2597_ == 0)
{
v___x_2599_ = v___x_2596_;
goto v_reusejp_2598_;
}
else
{
lean_object* v_reuseFailAlloc_2600_; 
v_reuseFailAlloc_2600_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2600_, 0, v_a_2594_);
v___x_2599_ = v_reuseFailAlloc_2600_;
goto v_reusejp_2598_;
}
v_reusejp_2598_:
{
return v___x_2599_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___boxed(lean_object* v_fromS_2602_, lean_object* v_toS_2603_, lean_object* v_x_2604_, lean_object* v_a_2605_, lean_object* v_a_2606_, lean_object* v_a_2607_, lean_object* v_a_2608_, lean_object* v_a_2609_, lean_object* v_a_2610_, lean_object* v_a_2611_){
_start:
{
lean_object* v_res_2612_; 
v_res_2612_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS(v_fromS_2602_, v_toS_2603_, v_x_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_, v_a_2610_);
lean_dec(v_a_2610_);
lean_dec_ref(v_a_2609_);
lean_dec(v_a_2608_);
lean_dec_ref(v_a_2607_);
lean_dec(v_a_2606_);
lean_dec_ref(v_a_2605_);
return v_res_2612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0(lean_object* v_00_u03b1_2613_, lean_object* v_name_2614_, uint8_t v_bi_2615_, lean_object* v_type_2616_, lean_object* v_k_2617_, uint8_t v_kind_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_, lean_object* v___y_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_){
_start:
{
lean_object* v___x_2626_; 
v___x_2626_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___redArg(v_name_2614_, v_bi_2615_, v_type_2616_, v_k_2617_, v_kind_2618_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_, v___y_2623_, v___y_2624_);
return v___x_2626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2627_, lean_object* v_name_2628_, lean_object* v_bi_2629_, lean_object* v_type_2630_, lean_object* v_k_2631_, lean_object* v_kind_2632_, lean_object* v___y_2633_, lean_object* v___y_2634_, lean_object* v___y_2635_, lean_object* v___y_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_){
_start:
{
uint8_t v_bi_boxed_2640_; uint8_t v_kind_boxed_2641_; lean_object* v_res_2642_; 
v_bi_boxed_2640_ = lean_unbox(v_bi_2629_);
v_kind_boxed_2641_ = lean_unbox(v_kind_2632_);
v_res_2642_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0_spec__0(v_00_u03b1_2627_, v_name_2628_, v_bi_boxed_2640_, v_type_2630_, v_k_2631_, v_kind_boxed_2641_, v___y_2633_, v___y_2634_, v___y_2635_, v___y_2636_, v___y_2637_, v___y_2638_);
lean_dec(v___y_2638_);
lean_dec_ref(v___y_2637_);
lean_dec(v___y_2636_);
lean_dec_ref(v___y_2635_);
lean_dec(v___y_2634_);
lean_dec_ref(v___y_2633_);
return v_res_2642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0(lean_object* v_00_u03b1_2643_, lean_object* v_name_2644_, lean_object* v_type_2645_, lean_object* v_k_2646_, lean_object* v___y_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_){
_start:
{
lean_object* v___x_2654_; 
v___x_2654_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___redArg(v_name_2644_, v_type_2645_, v_k_2646_, v___y_2647_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_);
return v___x_2654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0___boxed(lean_object* v_00_u03b1_2655_, lean_object* v_name_2656_, lean_object* v_type_2657_, lean_object* v_k_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_){
_start:
{
lean_object* v_res_2666_; 
v_res_2666_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS_spec__0(v_00_u03b1_2655_, v_name_2656_, v_type_2657_, v_k_2658_, v___y_2659_, v___y_2660_, v___y_2661_, v___y_2662_, v___y_2663_, v___y_2664_);
lean_dec(v___y_2664_);
lean_dec_ref(v___y_2663_);
lean_dec(v___y_2662_);
lean_dec_ref(v___y_2661_);
lean_dec(v___y_2660_);
lean_dec_ref(v___y_2659_);
return v_res_2666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg(lean_object* v_e_2667_, lean_object* v___y_2668_){
_start:
{
uint8_t v___x_2670_; 
v___x_2670_ = l_Lean_Expr_hasMVar(v_e_2667_);
if (v___x_2670_ == 0)
{
lean_object* v___x_2671_; 
v___x_2671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2671_, 0, v_e_2667_);
return v___x_2671_;
}
else
{
lean_object* v___x_2672_; lean_object* v_mctx_2673_; lean_object* v___x_2674_; lean_object* v_fst_2675_; lean_object* v_snd_2676_; lean_object* v___x_2677_; lean_object* v_cache_2678_; lean_object* v_zetaDeltaFVarIds_2679_; lean_object* v_postponed_2680_; lean_object* v_diag_2681_; lean_object* v___x_2683_; uint8_t v_isShared_2684_; uint8_t v_isSharedCheck_2690_; 
v___x_2672_ = lean_st_ref_get(v___y_2668_);
v_mctx_2673_ = lean_ctor_get(v___x_2672_, 0);
lean_inc_ref(v_mctx_2673_);
lean_dec(v___x_2672_);
v___x_2674_ = l_Lean_instantiateMVarsCore(v_mctx_2673_, v_e_2667_);
v_fst_2675_ = lean_ctor_get(v___x_2674_, 0);
lean_inc(v_fst_2675_);
v_snd_2676_ = lean_ctor_get(v___x_2674_, 1);
lean_inc(v_snd_2676_);
lean_dec_ref(v___x_2674_);
v___x_2677_ = lean_st_ref_take(v___y_2668_);
v_cache_2678_ = lean_ctor_get(v___x_2677_, 1);
v_zetaDeltaFVarIds_2679_ = lean_ctor_get(v___x_2677_, 2);
v_postponed_2680_ = lean_ctor_get(v___x_2677_, 3);
v_diag_2681_ = lean_ctor_get(v___x_2677_, 4);
v_isSharedCheck_2690_ = !lean_is_exclusive(v___x_2677_);
if (v_isSharedCheck_2690_ == 0)
{
lean_object* v_unused_2691_; 
v_unused_2691_ = lean_ctor_get(v___x_2677_, 0);
lean_dec(v_unused_2691_);
v___x_2683_ = v___x_2677_;
v_isShared_2684_ = v_isSharedCheck_2690_;
goto v_resetjp_2682_;
}
else
{
lean_inc(v_diag_2681_);
lean_inc(v_postponed_2680_);
lean_inc(v_zetaDeltaFVarIds_2679_);
lean_inc(v_cache_2678_);
lean_dec(v___x_2677_);
v___x_2683_ = lean_box(0);
v_isShared_2684_ = v_isSharedCheck_2690_;
goto v_resetjp_2682_;
}
v_resetjp_2682_:
{
lean_object* v___x_2686_; 
if (v_isShared_2684_ == 0)
{
lean_ctor_set(v___x_2683_, 0, v_snd_2676_);
v___x_2686_ = v___x_2683_;
goto v_reusejp_2685_;
}
else
{
lean_object* v_reuseFailAlloc_2689_; 
v_reuseFailAlloc_2689_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2689_, 0, v_snd_2676_);
lean_ctor_set(v_reuseFailAlloc_2689_, 1, v_cache_2678_);
lean_ctor_set(v_reuseFailAlloc_2689_, 2, v_zetaDeltaFVarIds_2679_);
lean_ctor_set(v_reuseFailAlloc_2689_, 3, v_postponed_2680_);
lean_ctor_set(v_reuseFailAlloc_2689_, 4, v_diag_2681_);
v___x_2686_ = v_reuseFailAlloc_2689_;
goto v_reusejp_2685_;
}
v_reusejp_2685_:
{
lean_object* v___x_2687_; lean_object* v___x_2688_; 
v___x_2687_ = lean_st_ref_set(v___y_2668_, v___x_2686_);
v___x_2688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2688_, 0, v_fst_2675_);
return v___x_2688_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg___boxed(lean_object* v_e_2692_, lean_object* v___y_2693_, lean_object* v___y_2694_){
_start:
{
lean_object* v_res_2695_; 
v_res_2695_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg(v_e_2692_, v___y_2693_);
lean_dec(v___y_2693_);
return v_res_2695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0(lean_object* v_e_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_, lean_object* v___y_2700_, lean_object* v___y_2701_, lean_object* v___y_2702_, lean_object* v___y_2703_){
_start:
{
lean_object* v___x_2705_; 
v___x_2705_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg(v_e_2696_, v___y_2701_);
return v___x_2705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___boxed(lean_object* v_e_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_, lean_object* v___y_2714_){
_start:
{
lean_object* v_res_2715_; 
v_res_2715_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0(v_e_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
lean_dec(v___y_2713_);
lean_dec_ref(v___y_2712_);
lean_dec(v___y_2711_);
lean_dec_ref(v___y_2710_);
lean_dec(v___y_2709_);
lean_dec_ref(v___y_2708_);
lean_dec(v___y_2707_);
return v_res_2715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0(lean_object* v_k_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_, lean_object* v___y_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_){
_start:
{
lean_object* v___x_2725_; 
lean_inc(v___y_2719_);
lean_inc_ref(v___y_2718_);
lean_inc(v___y_2717_);
v___x_2725_ = lean_apply_8(v_k_2716_, v___y_2717_, v___y_2718_, v___y_2719_, v___y_2720_, v___y_2721_, v___y_2722_, v___y_2723_, lean_box(0));
return v___x_2725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0___boxed(lean_object* v_k_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_, lean_object* v___y_2731_, lean_object* v___y_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_){
_start:
{
lean_object* v_res_2735_; 
v_res_2735_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0(v_k_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_, v___y_2731_, v___y_2732_, v___y_2733_);
lean_dec(v___y_2729_);
lean_dec_ref(v___y_2728_);
lean_dec(v___y_2727_);
return v_res_2735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg(lean_object* v_k_2736_, uint8_t v_allowLevelAssignments_2737_, lean_object* v___y_2738_, lean_object* v___y_2739_, lean_object* v___y_2740_, lean_object* v___y_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_){
_start:
{
lean_object* v___f_2746_; lean_object* v___x_2747_; 
lean_inc(v___y_2740_);
lean_inc_ref(v___y_2739_);
lean_inc(v___y_2738_);
v___f_2746_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_2746_, 0, v_k_2736_);
lean_closure_set(v___f_2746_, 1, v___y_2738_);
lean_closure_set(v___f_2746_, 2, v___y_2739_);
lean_closure_set(v___f_2746_, 3, v___y_2740_);
v___x_2747_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_2737_, v___f_2746_, v___y_2741_, v___y_2742_, v___y_2743_, v___y_2744_);
if (lean_obj_tag(v___x_2747_) == 0)
{
return v___x_2747_;
}
else
{
lean_object* v_a_2748_; lean_object* v___x_2750_; uint8_t v_isShared_2751_; uint8_t v_isSharedCheck_2755_; 
v_a_2748_ = lean_ctor_get(v___x_2747_, 0);
v_isSharedCheck_2755_ = !lean_is_exclusive(v___x_2747_);
if (v_isSharedCheck_2755_ == 0)
{
v___x_2750_ = v___x_2747_;
v_isShared_2751_ = v_isSharedCheck_2755_;
goto v_resetjp_2749_;
}
else
{
lean_inc(v_a_2748_);
lean_dec(v___x_2747_);
v___x_2750_ = lean_box(0);
v_isShared_2751_ = v_isSharedCheck_2755_;
goto v_resetjp_2749_;
}
v_resetjp_2749_:
{
lean_object* v___x_2753_; 
if (v_isShared_2751_ == 0)
{
v___x_2753_ = v___x_2750_;
goto v_reusejp_2752_;
}
else
{
lean_object* v_reuseFailAlloc_2754_; 
v_reuseFailAlloc_2754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2754_, 0, v_a_2748_);
v___x_2753_ = v_reuseFailAlloc_2754_;
goto v_reusejp_2752_;
}
v_reusejp_2752_:
{
return v___x_2753_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg___boxed(lean_object* v_k_2756_, lean_object* v_allowLevelAssignments_2757_, lean_object* v___y_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_, lean_object* v___y_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2766_; lean_object* v_res_2767_; 
v_allowLevelAssignments_boxed_2766_ = lean_unbox(v_allowLevelAssignments_2757_);
v_res_2767_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg(v_k_2756_, v_allowLevelAssignments_boxed_2766_, v___y_2758_, v___y_2759_, v___y_2760_, v___y_2761_, v___y_2762_, v___y_2763_, v___y_2764_);
lean_dec(v___y_2764_);
lean_dec_ref(v___y_2763_);
lean_dec(v___y_2762_);
lean_dec_ref(v___y_2761_);
lean_dec(v___y_2760_);
lean_dec_ref(v___y_2759_);
lean_dec(v___y_2758_);
return v_res_2767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1(lean_object* v_00_u03b1_2768_, lean_object* v_k_2769_, uint8_t v_allowLevelAssignments_2770_, lean_object* v___y_2771_, lean_object* v___y_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_){
_start:
{
lean_object* v___x_2779_; 
v___x_2779_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg(v_k_2769_, v_allowLevelAssignments_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___boxed(lean_object* v_00_u03b1_2780_, lean_object* v_k_2781_, lean_object* v_allowLevelAssignments_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2791_; lean_object* v_res_2792_; 
v_allowLevelAssignments_boxed_2791_ = lean_unbox(v_allowLevelAssignments_2782_);
v_res_2792_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1(v_00_u03b1_2780_, v_k_2781_, v_allowLevelAssignments_boxed_2791_, v___y_2783_, v___y_2784_, v___y_2785_, v___y_2786_, v___y_2787_, v___y_2788_, v___y_2789_);
lean_dec(v___y_2789_);
lean_dec_ref(v___y_2788_);
lean_dec(v___y_2787_);
lean_dec_ref(v___y_2786_);
lean_dec(v___y_2785_);
lean_dec_ref(v___y_2784_);
lean_dec(v___y_2783_);
return v_res_2792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0(lean_object* v_val_2793_, lean_object* v_a_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_){
_start:
{
lean_object* v___x_2803_; 
v___x_2803_ = l_Lean_Meta_isExprDefEqGuarded(v_val_2793_, v_a_2794_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_);
return v___x_2803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0___boxed(lean_object* v_val_2804_, lean_object* v_a_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_){
_start:
{
lean_object* v_res_2814_; 
v_res_2814_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0(v_val_2804_, v_a_2805_, v___y_2806_, v___y_2807_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_);
lean_dec(v___y_2812_);
lean_dec_ref(v___y_2811_);
lean_dec(v___y_2810_);
lean_dec_ref(v___y_2809_);
lean_dec(v___y_2808_);
lean_dec_ref(v___y_2807_);
lean_dec(v___y_2806_);
return v_res_2814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg(lean_object* v_cls_2815_, lean_object* v_msg_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_){
_start:
{
lean_object* v_ref_2822_; lean_object* v___x_2823_; lean_object* v_a_2824_; lean_object* v___x_2826_; uint8_t v_isShared_2827_; uint8_t v_isSharedCheck_2868_; 
v_ref_2822_ = lean_ctor_get(v___y_2819_, 5);
v___x_2823_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0_spec__3(v_msg_2816_, v___y_2817_, v___y_2818_, v___y_2819_, v___y_2820_);
v_a_2824_ = lean_ctor_get(v___x_2823_, 0);
v_isSharedCheck_2868_ = !lean_is_exclusive(v___x_2823_);
if (v_isSharedCheck_2868_ == 0)
{
v___x_2826_ = v___x_2823_;
v_isShared_2827_ = v_isSharedCheck_2868_;
goto v_resetjp_2825_;
}
else
{
lean_inc(v_a_2824_);
lean_dec(v___x_2823_);
v___x_2826_ = lean_box(0);
v_isShared_2827_ = v_isSharedCheck_2868_;
goto v_resetjp_2825_;
}
v_resetjp_2825_:
{
lean_object* v___x_2828_; lean_object* v_traceState_2829_; lean_object* v_env_2830_; lean_object* v_nextMacroScope_2831_; lean_object* v_ngen_2832_; lean_object* v_auxDeclNGen_2833_; lean_object* v_cache_2834_; lean_object* v_messages_2835_; lean_object* v_infoState_2836_; lean_object* v_snapshotTasks_2837_; lean_object* v___x_2839_; uint8_t v_isShared_2840_; uint8_t v_isSharedCheck_2867_; 
v___x_2828_ = lean_st_ref_take(v___y_2820_);
v_traceState_2829_ = lean_ctor_get(v___x_2828_, 4);
v_env_2830_ = lean_ctor_get(v___x_2828_, 0);
v_nextMacroScope_2831_ = lean_ctor_get(v___x_2828_, 1);
v_ngen_2832_ = lean_ctor_get(v___x_2828_, 2);
v_auxDeclNGen_2833_ = lean_ctor_get(v___x_2828_, 3);
v_cache_2834_ = lean_ctor_get(v___x_2828_, 5);
v_messages_2835_ = lean_ctor_get(v___x_2828_, 6);
v_infoState_2836_ = lean_ctor_get(v___x_2828_, 7);
v_snapshotTasks_2837_ = lean_ctor_get(v___x_2828_, 8);
v_isSharedCheck_2867_ = !lean_is_exclusive(v___x_2828_);
if (v_isSharedCheck_2867_ == 0)
{
v___x_2839_ = v___x_2828_;
v_isShared_2840_ = v_isSharedCheck_2867_;
goto v_resetjp_2838_;
}
else
{
lean_inc(v_snapshotTasks_2837_);
lean_inc(v_infoState_2836_);
lean_inc(v_messages_2835_);
lean_inc(v_cache_2834_);
lean_inc(v_traceState_2829_);
lean_inc(v_auxDeclNGen_2833_);
lean_inc(v_ngen_2832_);
lean_inc(v_nextMacroScope_2831_);
lean_inc(v_env_2830_);
lean_dec(v___x_2828_);
v___x_2839_ = lean_box(0);
v_isShared_2840_ = v_isSharedCheck_2867_;
goto v_resetjp_2838_;
}
v_resetjp_2838_:
{
uint64_t v_tid_2841_; lean_object* v_traces_2842_; lean_object* v___x_2844_; uint8_t v_isShared_2845_; uint8_t v_isSharedCheck_2866_; 
v_tid_2841_ = lean_ctor_get_uint64(v_traceState_2829_, sizeof(void*)*1);
v_traces_2842_ = lean_ctor_get(v_traceState_2829_, 0);
v_isSharedCheck_2866_ = !lean_is_exclusive(v_traceState_2829_);
if (v_isSharedCheck_2866_ == 0)
{
v___x_2844_ = v_traceState_2829_;
v_isShared_2845_ = v_isSharedCheck_2866_;
goto v_resetjp_2843_;
}
else
{
lean_inc(v_traces_2842_);
lean_dec(v_traceState_2829_);
v___x_2844_ = lean_box(0);
v_isShared_2845_ = v_isSharedCheck_2866_;
goto v_resetjp_2843_;
}
v_resetjp_2843_:
{
lean_object* v___x_2846_; double v___x_2847_; uint8_t v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2856_; 
v___x_2846_ = lean_box(0);
v___x_2847_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__0);
v___x_2848_ = 0;
v___x_2849_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__1));
v___x_2850_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2850_, 0, v_cls_2815_);
lean_ctor_set(v___x_2850_, 1, v___x_2846_);
lean_ctor_set(v___x_2850_, 2, v___x_2849_);
lean_ctor_set_float(v___x_2850_, sizeof(void*)*3, v___x_2847_);
lean_ctor_set_float(v___x_2850_, sizeof(void*)*3 + 8, v___x_2847_);
lean_ctor_set_uint8(v___x_2850_, sizeof(void*)*3 + 16, v___x_2848_);
v___x_2851_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg___closed__2));
v___x_2852_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2852_, 0, v___x_2850_);
lean_ctor_set(v___x_2852_, 1, v_a_2824_);
lean_ctor_set(v___x_2852_, 2, v___x_2851_);
lean_inc(v_ref_2822_);
v___x_2853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2853_, 0, v_ref_2822_);
lean_ctor_set(v___x_2853_, 1, v___x_2852_);
v___x_2854_ = l_Lean_PersistentArray_push___redArg(v_traces_2842_, v___x_2853_);
if (v_isShared_2845_ == 0)
{
lean_ctor_set(v___x_2844_, 0, v___x_2854_);
v___x_2856_ = v___x_2844_;
goto v_reusejp_2855_;
}
else
{
lean_object* v_reuseFailAlloc_2865_; 
v_reuseFailAlloc_2865_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2865_, 0, v___x_2854_);
lean_ctor_set_uint64(v_reuseFailAlloc_2865_, sizeof(void*)*1, v_tid_2841_);
v___x_2856_ = v_reuseFailAlloc_2865_;
goto v_reusejp_2855_;
}
v_reusejp_2855_:
{
lean_object* v___x_2858_; 
if (v_isShared_2840_ == 0)
{
lean_ctor_set(v___x_2839_, 4, v___x_2856_);
v___x_2858_ = v___x_2839_;
goto v_reusejp_2857_;
}
else
{
lean_object* v_reuseFailAlloc_2864_; 
v_reuseFailAlloc_2864_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2864_, 0, v_env_2830_);
lean_ctor_set(v_reuseFailAlloc_2864_, 1, v_nextMacroScope_2831_);
lean_ctor_set(v_reuseFailAlloc_2864_, 2, v_ngen_2832_);
lean_ctor_set(v_reuseFailAlloc_2864_, 3, v_auxDeclNGen_2833_);
lean_ctor_set(v_reuseFailAlloc_2864_, 4, v___x_2856_);
lean_ctor_set(v_reuseFailAlloc_2864_, 5, v_cache_2834_);
lean_ctor_set(v_reuseFailAlloc_2864_, 6, v_messages_2835_);
lean_ctor_set(v_reuseFailAlloc_2864_, 7, v_infoState_2836_);
lean_ctor_set(v_reuseFailAlloc_2864_, 8, v_snapshotTasks_2837_);
v___x_2858_ = v_reuseFailAlloc_2864_;
goto v_reusejp_2857_;
}
v_reusejp_2857_:
{
lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2862_; 
v___x_2859_ = lean_st_ref_set(v___y_2820_, v___x_2858_);
v___x_2860_ = lean_box(0);
if (v_isShared_2827_ == 0)
{
lean_ctor_set(v___x_2826_, 0, v___x_2860_);
v___x_2862_ = v___x_2826_;
goto v_reusejp_2861_;
}
else
{
lean_object* v_reuseFailAlloc_2863_; 
v_reuseFailAlloc_2863_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2863_, 0, v___x_2860_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg___boxed(lean_object* v_cls_2869_, lean_object* v_msg_2870_, lean_object* v___y_2871_, lean_object* v___y_2872_, lean_object* v___y_2873_, lean_object* v___y_2874_, lean_object* v___y_2875_){
_start:
{
lean_object* v_res_2876_; 
v_res_2876_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg(v_cls_2869_, v_msg_2870_, v___y_2871_, v___y_2872_, v___y_2873_, v___y_2874_);
lean_dec(v___y_2874_);
lean_dec_ref(v___y_2873_);
lean_dec(v___y_2872_);
lean_dec_ref(v___y_2871_);
return v_res_2876_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1(void){
_start:
{
lean_object* v___x_2878_; lean_object* v___x_2879_; 
v___x_2878_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__0));
v___x_2879_ = l_Lean_stringToMessageData(v___x_2878_);
return v___x_2879_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3(void){
_start:
{
lean_object* v___x_2881_; lean_object* v___x_2882_; 
v___x_2881_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__2));
v___x_2882_ = l_Lean_stringToMessageData(v___x_2881_);
return v___x_2882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go(lean_object* v_t_2883_, lean_object* v_a_2884_, lean_object* v_a_2885_, lean_object* v_a_2886_, lean_object* v_a_2887_, lean_object* v_a_2888_, lean_object* v_a_2889_, lean_object* v_a_2890_){
_start:
{
lean_object* v___x_2892_; uint8_t v_hasUncomparable_2893_; 
v___x_2892_ = lean_st_ref_get(v_a_2884_);
v_hasUncomparable_2893_ = lean_ctor_get_uint8(v___x_2892_, sizeof(void*)*1);
lean_dec(v___x_2892_);
if (v_hasUncomparable_2893_ == 0)
{
switch(lean_obj_tag(v_t_2883_))
{
case 0:
{
lean_object* v_val_2894_; lean_object* v___x_2895_; 
v_val_2894_ = lean_ctor_get(v_t_2883_, 2);
lean_inc_ref(v_val_2894_);
lean_dec_ref_known(v_t_2883_, 3);
lean_inc(v_a_2890_);
lean_inc_ref(v_a_2889_);
lean_inc(v_a_2888_);
lean_inc_ref(v_a_2887_);
v___x_2895_ = lean_infer_type(v_val_2894_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2895_) == 0)
{
lean_object* v_a_2896_; lean_object* v___x_2897_; lean_object* v_a_2898_; lean_object* v___x_2899_; 
v_a_2896_ = lean_ctor_get(v___x_2895_, 0);
lean_inc(v_a_2896_);
lean_dec_ref_known(v___x_2895_, 1);
v___x_2897_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__0___redArg(v_a_2896_, v_a_2888_);
v_a_2898_ = lean_ctor_get(v___x_2897_, 0);
lean_inc_n(v_a_2898_, 2);
lean_dec_ref(v___x_2897_);
v___x_2899_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(v_a_2898_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2899_) == 0)
{
lean_object* v_a_2900_; lean_object* v___x_2902_; uint8_t v_isShared_2903_; uint8_t v_isSharedCheck_3068_; 
v_a_2900_ = lean_ctor_get(v___x_2899_, 0);
v_isSharedCheck_3068_ = !lean_is_exclusive(v___x_2899_);
if (v_isSharedCheck_3068_ == 0)
{
v___x_2902_ = v___x_2899_;
v_isShared_2903_ = v_isSharedCheck_3068_;
goto v_resetjp_2901_;
}
else
{
lean_inc(v_a_2900_);
lean_dec(v___x_2899_);
v___x_2902_ = lean_box(0);
v_isShared_2903_ = v_isSharedCheck_3068_;
goto v_resetjp_2901_;
}
v_resetjp_2901_:
{
if (lean_obj_tag(v_a_2900_) == 1)
{
lean_object* v_val_2904_; lean_object* v___x_2906_; uint8_t v_isShared_2907_; uint8_t v_isSharedCheck_3063_; 
v_val_2904_ = lean_ctor_get(v_a_2900_, 0);
v_isSharedCheck_3063_ = !lean_is_exclusive(v_a_2900_);
if (v_isSharedCheck_3063_ == 0)
{
v___x_2906_ = v_a_2900_;
v_isShared_2907_ = v_isSharedCheck_3063_;
goto v_resetjp_2905_;
}
else
{
lean_inc(v_val_2904_);
lean_dec(v_a_2900_);
v___x_2906_ = lean_box(0);
v_isShared_2907_ = v_isSharedCheck_3063_;
goto v_resetjp_2905_;
}
v_resetjp_2905_:
{
lean_object* v_fst_2908_; lean_object* v_snd_2909_; lean_object* v___x_2911_; uint8_t v_isShared_2912_; uint8_t v_isSharedCheck_3062_; 
v_fst_2908_ = lean_ctor_get(v_val_2904_, 0);
v_snd_2909_ = lean_ctor_get(v_val_2904_, 1);
v_isSharedCheck_3062_ = !lean_is_exclusive(v_val_2904_);
if (v_isSharedCheck_3062_ == 0)
{
v___x_2911_ = v_val_2904_;
v_isShared_2912_ = v_isSharedCheck_3062_;
goto v_resetjp_2910_;
}
else
{
lean_inc(v_snd_2909_);
lean_inc(v_fst_2908_);
lean_dec(v_val_2904_);
v___x_2911_ = lean_box(0);
v_isShared_2912_ = v_isSharedCheck_3062_;
goto v_resetjp_2910_;
}
v_resetjp_2910_:
{
lean_object* v___x_2913_; lean_object* v_maxS_x3f_2914_; 
v___x_2913_ = lean_st_ref_get(v_a_2884_);
v_maxS_x3f_2914_ = lean_ctor_get(v___x_2913_, 0);
lean_inc(v_maxS_x3f_2914_);
lean_dec(v___x_2913_);
if (lean_obj_tag(v_maxS_x3f_2914_) == 0)
{
lean_object* v___x_2915_; uint8_t v_hasUncomparable_2916_; lean_object* v___x_2918_; uint8_t v_isShared_2919_; uint8_t v_isSharedCheck_2931_; 
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_a_2898_);
v___x_2915_ = lean_st_ref_take(v_a_2884_);
v_hasUncomparable_2916_ = lean_ctor_get_uint8(v___x_2915_, sizeof(void*)*1);
v_isSharedCheck_2931_ = !lean_is_exclusive(v___x_2915_);
if (v_isSharedCheck_2931_ == 0)
{
lean_object* v_unused_2932_; 
v_unused_2932_ = lean_ctor_get(v___x_2915_, 0);
lean_dec(v_unused_2932_);
v___x_2918_ = v___x_2915_;
v_isShared_2919_ = v_isSharedCheck_2931_;
goto v_resetjp_2917_;
}
else
{
lean_dec(v___x_2915_);
v___x_2918_ = lean_box(0);
v_isShared_2919_ = v_isSharedCheck_2931_;
goto v_resetjp_2917_;
}
v_resetjp_2917_:
{
lean_object* v___x_2921_; 
if (v_isShared_2907_ == 0)
{
lean_ctor_set(v___x_2906_, 0, v_fst_2908_);
v___x_2921_ = v___x_2906_;
goto v_reusejp_2920_;
}
else
{
lean_object* v_reuseFailAlloc_2930_; 
v_reuseFailAlloc_2930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2930_, 0, v_fst_2908_);
v___x_2921_ = v_reuseFailAlloc_2930_;
goto v_reusejp_2920_;
}
v_reusejp_2920_:
{
lean_object* v___x_2923_; 
if (v_isShared_2919_ == 0)
{
lean_ctor_set(v___x_2918_, 0, v___x_2921_);
v___x_2923_ = v___x_2918_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v___x_2921_);
lean_ctor_set_uint8(v_reuseFailAlloc_2929_, sizeof(void*)*1, v_hasUncomparable_2916_);
v___x_2923_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
lean_object* v___x_2924_; lean_object* v___x_2925_; lean_object* v___x_2927_; 
v___x_2924_ = lean_st_ref_set(v_a_2884_, v___x_2923_);
v___x_2925_ = lean_box(0);
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 0, v___x_2925_);
v___x_2927_ = v___x_2902_;
goto v_reusejp_2926_;
}
else
{
lean_object* v_reuseFailAlloc_2928_; 
v_reuseFailAlloc_2928_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2928_, 0, v___x_2925_);
v___x_2927_ = v_reuseFailAlloc_2928_;
goto v_reusejp_2926_;
}
v_reusejp_2926_:
{
return v___x_2927_;
}
}
}
}
}
else
{
lean_object* v_val_2933_; lean_object* v___x_2934_; 
lean_del_object(v___x_2906_);
lean_del_object(v___x_2902_);
v_val_2933_ = lean_ctor_get(v_maxS_x3f_2914_, 0);
lean_inc_n(v_val_2933_, 2);
lean_dec_ref_known(v_maxS_x3f_2914_, 1);
lean_inc(v_snd_2909_);
v___x_2934_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_val_2933_, v_snd_2909_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2934_) == 0)
{
lean_object* v_a_2935_; lean_object* v___x_2937_; uint8_t v_isShared_2938_; uint8_t v_isSharedCheck_3053_; 
v_a_2935_ = lean_ctor_get(v___x_2934_, 0);
v_isSharedCheck_3053_ = !lean_is_exclusive(v___x_2934_);
if (v_isSharedCheck_3053_ == 0)
{
v___x_2937_ = v___x_2934_;
v_isShared_2938_ = v_isSharedCheck_3053_;
goto v_resetjp_2936_;
}
else
{
lean_inc(v_a_2935_);
lean_dec(v___x_2934_);
v___x_2937_ = lean_box(0);
v_isShared_2938_ = v_isSharedCheck_3053_;
goto v_resetjp_2936_;
}
v_resetjp_2936_:
{
if (lean_obj_tag(v_a_2935_) == 1)
{
lean_object* v_val_2939_; lean_object* v___x_2941_; uint8_t v_isShared_2942_; uint8_t v_isSharedCheck_3048_; 
lean_del_object(v___x_2937_);
v_val_2939_ = lean_ctor_get(v_a_2935_, 0);
v_isSharedCheck_3048_ = !lean_is_exclusive(v_a_2935_);
if (v_isSharedCheck_3048_ == 0)
{
v___x_2941_ = v_a_2935_;
v_isShared_2942_ = v_isSharedCheck_3048_;
goto v_resetjp_2940_;
}
else
{
lean_inc(v_val_2939_);
lean_dec(v_a_2935_);
v___x_2941_ = lean_box(0);
v_isShared_2942_ = v_isSharedCheck_3048_;
goto v_resetjp_2940_;
}
v_resetjp_2940_:
{
lean_object* v___f_2943_; lean_object* v___x_2944_; 
lean_inc(v_a_2898_);
lean_inc(v_val_2939_);
v___f_2943_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___lam__0___boxed), 10, 2);
lean_closure_set(v___f_2943_, 0, v_val_2939_);
lean_closure_set(v___f_2943_, 1, v_a_2898_);
v___x_2944_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__1___redArg(v___f_2943_, v_hasUncomparable_2893_, v_a_2884_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2944_) == 0)
{
lean_object* v_a_2945_; lean_object* v___x_2947_; uint8_t v_isShared_2948_; uint8_t v_isSharedCheck_3039_; 
v_a_2945_ = lean_ctor_get(v___x_2944_, 0);
v_isSharedCheck_3039_ = !lean_is_exclusive(v___x_2944_);
if (v_isSharedCheck_3039_ == 0)
{
v___x_2947_ = v___x_2944_;
v_isShared_2948_ = v_isSharedCheck_3039_;
goto v_resetjp_2946_;
}
else
{
lean_inc(v_a_2945_);
lean_dec(v___x_2944_);
v___x_2947_ = lean_box(0);
v_isShared_2948_ = v_isSharedCheck_3039_;
goto v_resetjp_2946_;
}
v_resetjp_2946_:
{
uint8_t v___x_2949_; 
v___x_2949_ = lean_unbox(v_a_2945_);
lean_dec(v_a_2945_);
if (v___x_2949_ == 0)
{
lean_object* v___x_2950_; 
lean_del_object(v___x_2947_);
lean_inc(v_snd_2909_);
lean_inc(v_val_2933_);
lean_inc(v_fst_2908_);
v___x_2950_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS(v_fst_2908_, v_val_2933_, v_snd_2909_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2950_) == 0)
{
lean_object* v_a_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_3026_; 
v_a_2951_ = lean_ctor_get(v___x_2950_, 0);
v_isSharedCheck_3026_ = !lean_is_exclusive(v___x_2950_);
if (v_isSharedCheck_3026_ == 0)
{
v___x_2953_ = v___x_2950_;
v_isShared_2954_ = v_isSharedCheck_3026_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_a_2951_);
lean_dec(v___x_2950_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_3026_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
uint8_t v___x_2955_; 
v___x_2955_ = lean_unbox(v_a_2951_);
lean_dec(v_a_2951_);
if (v___x_2955_ == 0)
{
lean_object* v___x_2956_; 
lean_del_object(v___x_2953_);
lean_inc(v_fst_2908_);
v___x_2956_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS(v_val_2933_, v_fst_2908_, v_snd_2909_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2956_) == 0)
{
lean_object* v_a_2957_; lean_object* v___x_2959_; uint8_t v_isShared_2960_; uint8_t v_isSharedCheck_3013_; 
v_a_2957_ = lean_ctor_get(v___x_2956_, 0);
v_isSharedCheck_3013_ = !lean_is_exclusive(v___x_2956_);
if (v_isSharedCheck_3013_ == 0)
{
v___x_2959_ = v___x_2956_;
v_isShared_2960_ = v_isSharedCheck_3013_;
goto v_resetjp_2958_;
}
else
{
lean_inc(v_a_2957_);
lean_dec(v___x_2956_);
v___x_2959_ = lean_box(0);
v_isShared_2960_ = v_isSharedCheck_3013_;
goto v_resetjp_2958_;
}
v_resetjp_2958_:
{
uint8_t v___x_2961_; 
v___x_2961_ = lean_unbox(v_a_2957_);
lean_dec(v_a_2957_);
if (v___x_2961_ == 0)
{
lean_object* v_options_2962_; lean_object* v_inheritedTraceOptions_2963_; uint8_t v_hasTrace_2964_; uint8_t v___x_2965_; lean_object* v___y_2967_; 
lean_del_object(v___x_2941_);
lean_dec(v_fst_2908_);
v_options_2962_ = lean_ctor_get(v_a_2889_, 2);
v_inheritedTraceOptions_2963_ = lean_ctor_get(v_a_2889_, 13);
v_hasTrace_2964_ = lean_ctor_get_uint8(v_options_2962_, sizeof(void*)*1);
v___x_2965_ = 1;
if (v_hasTrace_2964_ == 0)
{
lean_dec(v_val_2939_);
lean_del_object(v___x_2911_);
lean_dec(v_a_2898_);
v___y_2967_ = v_a_2884_;
goto v___jp_2966_;
}
else
{
lean_object* v___x_2982_; lean_object* v___x_2983_; uint8_t v___x_2984_; 
v___x_2982_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_2983_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2);
v___x_2984_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2963_, v_options_2962_, v___x_2983_);
if (v___x_2984_ == 0)
{
lean_dec(v_val_2939_);
lean_del_object(v___x_2911_);
lean_dec(v_a_2898_);
v___y_2967_ = v_a_2884_;
goto v___jp_2966_;
}
else
{
lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2988_; 
v___x_2985_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__1);
v___x_2986_ = l_Lean_MessageData_ofExpr(v_val_2939_);
if (v_isShared_2912_ == 0)
{
lean_ctor_set_tag(v___x_2911_, 7);
lean_ctor_set(v___x_2911_, 1, v___x_2986_);
lean_ctor_set(v___x_2911_, 0, v___x_2985_);
v___x_2988_ = v___x_2911_;
goto v_reusejp_2987_;
}
else
{
lean_object* v_reuseFailAlloc_2994_; 
v_reuseFailAlloc_2994_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2994_, 0, v___x_2985_);
lean_ctor_set(v_reuseFailAlloc_2994_, 1, v___x_2986_);
v___x_2988_ = v_reuseFailAlloc_2994_;
goto v_reusejp_2987_;
}
v_reusejp_2987_:
{
lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; lean_object* v___x_2992_; lean_object* v___x_2993_; 
v___x_2989_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___closed__3);
v___x_2990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2990_, 0, v___x_2988_);
lean_ctor_set(v___x_2990_, 1, v___x_2989_);
v___x_2991_ = l_Lean_MessageData_ofExpr(v_a_2898_);
v___x_2992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2992_, 0, v___x_2990_);
lean_ctor_set(v___x_2992_, 1, v___x_2991_);
v___x_2993_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg(v___x_2982_, v___x_2992_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_2993_) == 0)
{
lean_dec_ref_known(v___x_2993_, 1);
v___y_2967_ = v_a_2884_;
goto v___jp_2966_;
}
else
{
lean_del_object(v___x_2959_);
return v___x_2993_;
}
}
}
}
v___jp_2966_:
{
lean_object* v___x_2968_; lean_object* v_maxS_x3f_2969_; lean_object* v___x_2971_; uint8_t v_isShared_2972_; uint8_t v_isSharedCheck_2981_; 
v___x_2968_ = lean_st_ref_take(v___y_2967_);
v_maxS_x3f_2969_ = lean_ctor_get(v___x_2968_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2968_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2971_ = v___x_2968_;
v_isShared_2972_ = v_isSharedCheck_2981_;
goto v_resetjp_2970_;
}
else
{
lean_inc(v_maxS_x3f_2969_);
lean_dec(v___x_2968_);
v___x_2971_ = lean_box(0);
v_isShared_2972_ = v_isSharedCheck_2981_;
goto v_resetjp_2970_;
}
v_resetjp_2970_:
{
lean_object* v___x_2974_; 
if (v_isShared_2972_ == 0)
{
v___x_2974_ = v___x_2971_;
goto v_reusejp_2973_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_maxS_x3f_2969_);
v___x_2974_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2973_;
}
v_reusejp_2973_:
{
lean_object* v___x_2975_; lean_object* v___x_2976_; lean_object* v___x_2978_; 
lean_ctor_set_uint8(v___x_2974_, sizeof(void*)*1, v___x_2965_);
v___x_2975_ = lean_st_ref_set(v___y_2967_, v___x_2974_);
v___x_2976_ = lean_box(0);
if (v_isShared_2960_ == 0)
{
lean_ctor_set(v___x_2959_, 0, v___x_2976_);
v___x_2978_ = v___x_2959_;
goto v_reusejp_2977_;
}
else
{
lean_object* v_reuseFailAlloc_2979_; 
v_reuseFailAlloc_2979_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2979_, 0, v___x_2976_);
v___x_2978_ = v_reuseFailAlloc_2979_;
goto v_reusejp_2977_;
}
v_reusejp_2977_:
{
return v___x_2978_;
}
}
}
}
}
else
{
lean_object* v___x_2995_; uint8_t v_hasUncomparable_2996_; lean_object* v___x_2998_; uint8_t v_isShared_2999_; uint8_t v_isSharedCheck_3011_; 
lean_dec(v_val_2939_);
lean_del_object(v___x_2911_);
lean_dec(v_a_2898_);
v___x_2995_ = lean_st_ref_take(v_a_2884_);
v_hasUncomparable_2996_ = lean_ctor_get_uint8(v___x_2995_, sizeof(void*)*1);
v_isSharedCheck_3011_ = !lean_is_exclusive(v___x_2995_);
if (v_isSharedCheck_3011_ == 0)
{
lean_object* v_unused_3012_; 
v_unused_3012_ = lean_ctor_get(v___x_2995_, 0);
lean_dec(v_unused_3012_);
v___x_2998_ = v___x_2995_;
v_isShared_2999_ = v_isSharedCheck_3011_;
goto v_resetjp_2997_;
}
else
{
lean_dec(v___x_2995_);
v___x_2998_ = lean_box(0);
v_isShared_2999_ = v_isSharedCheck_3011_;
goto v_resetjp_2997_;
}
v_resetjp_2997_:
{
lean_object* v___x_3001_; 
if (v_isShared_2942_ == 0)
{
lean_ctor_set(v___x_2941_, 0, v_fst_2908_);
v___x_3001_ = v___x_2941_;
goto v_reusejp_3000_;
}
else
{
lean_object* v_reuseFailAlloc_3010_; 
v_reuseFailAlloc_3010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3010_, 0, v_fst_2908_);
v___x_3001_ = v_reuseFailAlloc_3010_;
goto v_reusejp_3000_;
}
v_reusejp_3000_:
{
lean_object* v___x_3003_; 
if (v_isShared_2999_ == 0)
{
lean_ctor_set(v___x_2998_, 0, v___x_3001_);
v___x_3003_ = v___x_2998_;
goto v_reusejp_3002_;
}
else
{
lean_object* v_reuseFailAlloc_3009_; 
v_reuseFailAlloc_3009_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_3009_, 0, v___x_3001_);
lean_ctor_set_uint8(v_reuseFailAlloc_3009_, sizeof(void*)*1, v_hasUncomparable_2996_);
v___x_3003_ = v_reuseFailAlloc_3009_;
goto v_reusejp_3002_;
}
v_reusejp_3002_:
{
lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3007_; 
v___x_3004_ = lean_st_ref_set(v_a_2884_, v___x_3003_);
v___x_3005_ = lean_box(0);
if (v_isShared_2960_ == 0)
{
lean_ctor_set(v___x_2959_, 0, v___x_3005_);
v___x_3007_ = v___x_2959_;
goto v_reusejp_3006_;
}
else
{
lean_object* v_reuseFailAlloc_3008_; 
v_reuseFailAlloc_3008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3008_, 0, v___x_3005_);
v___x_3007_ = v_reuseFailAlloc_3008_;
goto v_reusejp_3006_;
}
v_reusejp_3006_:
{
return v___x_3007_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3014_; lean_object* v___x_3016_; uint8_t v_isShared_3017_; uint8_t v_isSharedCheck_3021_; 
lean_del_object(v___x_2941_);
lean_dec(v_val_2939_);
lean_del_object(v___x_2911_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v_a_3014_ = lean_ctor_get(v___x_2956_, 0);
v_isSharedCheck_3021_ = !lean_is_exclusive(v___x_2956_);
if (v_isSharedCheck_3021_ == 0)
{
v___x_3016_ = v___x_2956_;
v_isShared_3017_ = v_isSharedCheck_3021_;
goto v_resetjp_3015_;
}
else
{
lean_inc(v_a_3014_);
lean_dec(v___x_2956_);
v___x_3016_ = lean_box(0);
v_isShared_3017_ = v_isSharedCheck_3021_;
goto v_resetjp_3015_;
}
v_resetjp_3015_:
{
lean_object* v___x_3019_; 
if (v_isShared_3017_ == 0)
{
v___x_3019_ = v___x_3016_;
goto v_reusejp_3018_;
}
else
{
lean_object* v_reuseFailAlloc_3020_; 
v_reuseFailAlloc_3020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3020_, 0, v_a_3014_);
v___x_3019_ = v_reuseFailAlloc_3020_;
goto v_reusejp_3018_;
}
v_reusejp_3018_:
{
return v___x_3019_;
}
}
}
}
else
{
lean_object* v___x_3022_; lean_object* v___x_3024_; 
lean_del_object(v___x_2941_);
lean_dec(v_val_2939_);
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v___x_3022_ = lean_box(0);
if (v_isShared_2954_ == 0)
{
lean_ctor_set(v___x_2953_, 0, v___x_3022_);
v___x_3024_ = v___x_2953_;
goto v_reusejp_3023_;
}
else
{
lean_object* v_reuseFailAlloc_3025_; 
v_reuseFailAlloc_3025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3025_, 0, v___x_3022_);
v___x_3024_ = v_reuseFailAlloc_3025_;
goto v_reusejp_3023_;
}
v_reusejp_3023_:
{
return v___x_3024_;
}
}
}
}
else
{
lean_object* v_a_3027_; lean_object* v___x_3029_; uint8_t v_isShared_3030_; uint8_t v_isSharedCheck_3034_; 
lean_del_object(v___x_2941_);
lean_dec(v_val_2939_);
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v_a_3027_ = lean_ctor_get(v___x_2950_, 0);
v_isSharedCheck_3034_ = !lean_is_exclusive(v___x_2950_);
if (v_isSharedCheck_3034_ == 0)
{
v___x_3029_ = v___x_2950_;
v_isShared_3030_ = v_isSharedCheck_3034_;
goto v_resetjp_3028_;
}
else
{
lean_inc(v_a_3027_);
lean_dec(v___x_2950_);
v___x_3029_ = lean_box(0);
v_isShared_3030_ = v_isSharedCheck_3034_;
goto v_resetjp_3028_;
}
v_resetjp_3028_:
{
lean_object* v___x_3032_; 
if (v_isShared_3030_ == 0)
{
v___x_3032_ = v___x_3029_;
goto v_reusejp_3031_;
}
else
{
lean_object* v_reuseFailAlloc_3033_; 
v_reuseFailAlloc_3033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3033_, 0, v_a_3027_);
v___x_3032_ = v_reuseFailAlloc_3033_;
goto v_reusejp_3031_;
}
v_reusejp_3031_:
{
return v___x_3032_;
}
}
}
}
else
{
lean_object* v___x_3035_; lean_object* v___x_3037_; 
lean_del_object(v___x_2941_);
lean_dec(v_val_2939_);
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v___x_3035_ = lean_box(0);
if (v_isShared_2948_ == 0)
{
lean_ctor_set(v___x_2947_, 0, v___x_3035_);
v___x_3037_ = v___x_2947_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v___x_3035_);
v___x_3037_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
return v___x_3037_;
}
}
}
}
else
{
lean_object* v_a_3040_; lean_object* v___x_3042_; uint8_t v_isShared_3043_; uint8_t v_isSharedCheck_3047_; 
lean_del_object(v___x_2941_);
lean_dec(v_val_2939_);
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v_a_3040_ = lean_ctor_get(v___x_2944_, 0);
v_isSharedCheck_3047_ = !lean_is_exclusive(v___x_2944_);
if (v_isSharedCheck_3047_ == 0)
{
v___x_3042_ = v___x_2944_;
v_isShared_3043_ = v_isSharedCheck_3047_;
goto v_resetjp_3041_;
}
else
{
lean_inc(v_a_3040_);
lean_dec(v___x_2944_);
v___x_3042_ = lean_box(0);
v_isShared_3043_ = v_isSharedCheck_3047_;
goto v_resetjp_3041_;
}
v_resetjp_3041_:
{
lean_object* v___x_3045_; 
if (v_isShared_3043_ == 0)
{
v___x_3045_ = v___x_3042_;
goto v_reusejp_3044_;
}
else
{
lean_object* v_reuseFailAlloc_3046_; 
v_reuseFailAlloc_3046_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3046_, 0, v_a_3040_);
v___x_3045_ = v_reuseFailAlloc_3046_;
goto v_reusejp_3044_;
}
v_reusejp_3044_:
{
return v___x_3045_;
}
}
}
}
}
else
{
lean_object* v___x_3049_; lean_object* v___x_3051_; 
lean_dec(v_a_2935_);
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v___x_3049_ = lean_box(0);
if (v_isShared_2938_ == 0)
{
lean_ctor_set(v___x_2937_, 0, v___x_3049_);
v___x_3051_ = v___x_2937_;
goto v_reusejp_3050_;
}
else
{
lean_object* v_reuseFailAlloc_3052_; 
v_reuseFailAlloc_3052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3052_, 0, v___x_3049_);
v___x_3051_ = v_reuseFailAlloc_3052_;
goto v_reusejp_3050_;
}
v_reusejp_3050_:
{
return v___x_3051_;
}
}
}
}
else
{
lean_object* v_a_3054_; lean_object* v___x_3056_; uint8_t v_isShared_3057_; uint8_t v_isSharedCheck_3061_; 
lean_dec(v_val_2933_);
lean_del_object(v___x_2911_);
lean_dec(v_snd_2909_);
lean_dec(v_fst_2908_);
lean_dec(v_a_2898_);
v_a_3054_ = lean_ctor_get(v___x_2934_, 0);
v_isSharedCheck_3061_ = !lean_is_exclusive(v___x_2934_);
if (v_isSharedCheck_3061_ == 0)
{
v___x_3056_ = v___x_2934_;
v_isShared_3057_ = v_isSharedCheck_3061_;
goto v_resetjp_3055_;
}
else
{
lean_inc(v_a_3054_);
lean_dec(v___x_2934_);
v___x_3056_ = lean_box(0);
v_isShared_3057_ = v_isSharedCheck_3061_;
goto v_resetjp_3055_;
}
v_resetjp_3055_:
{
lean_object* v___x_3059_; 
if (v_isShared_3057_ == 0)
{
v___x_3059_ = v___x_3056_;
goto v_reusejp_3058_;
}
else
{
lean_object* v_reuseFailAlloc_3060_; 
v_reuseFailAlloc_3060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3060_, 0, v_a_3054_);
v___x_3059_ = v_reuseFailAlloc_3060_;
goto v_reusejp_3058_;
}
v_reusejp_3058_:
{
return v___x_3059_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3064_; lean_object* v___x_3066_; 
lean_dec(v_a_2900_);
lean_dec(v_a_2898_);
v___x_3064_ = lean_box(0);
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 0, v___x_3064_);
v___x_3066_ = v___x_2902_;
goto v_reusejp_3065_;
}
else
{
lean_object* v_reuseFailAlloc_3067_; 
v_reuseFailAlloc_3067_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3067_, 0, v___x_3064_);
v___x_3066_ = v_reuseFailAlloc_3067_;
goto v_reusejp_3065_;
}
v_reusejp_3065_:
{
return v___x_3066_;
}
}
}
}
else
{
lean_object* v_a_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3076_; 
lean_dec(v_a_2898_);
v_a_3069_ = lean_ctor_get(v___x_2899_, 0);
v_isSharedCheck_3076_ = !lean_is_exclusive(v___x_2899_);
if (v_isSharedCheck_3076_ == 0)
{
v___x_3071_ = v___x_2899_;
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
else
{
lean_inc(v_a_3069_);
lean_dec(v___x_2899_);
v___x_3071_ = lean_box(0);
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
v_resetjp_3070_:
{
lean_object* v___x_3074_; 
if (v_isShared_3072_ == 0)
{
v___x_3074_ = v___x_3071_;
goto v_reusejp_3073_;
}
else
{
lean_object* v_reuseFailAlloc_3075_; 
v_reuseFailAlloc_3075_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3075_, 0, v_a_3069_);
v___x_3074_ = v_reuseFailAlloc_3075_;
goto v_reusejp_3073_;
}
v_reusejp_3073_:
{
return v___x_3074_;
}
}
}
}
else
{
lean_object* v_a_3077_; lean_object* v___x_3079_; uint8_t v_isShared_3080_; uint8_t v_isSharedCheck_3084_; 
v_a_3077_ = lean_ctor_get(v___x_2895_, 0);
v_isSharedCheck_3084_ = !lean_is_exclusive(v___x_2895_);
if (v_isSharedCheck_3084_ == 0)
{
v___x_3079_ = v___x_2895_;
v_isShared_3080_ = v_isSharedCheck_3084_;
goto v_resetjp_3078_;
}
else
{
lean_inc(v_a_3077_);
lean_dec(v___x_2895_);
v___x_3079_ = lean_box(0);
v_isShared_3080_ = v_isSharedCheck_3084_;
goto v_resetjp_3078_;
}
v_resetjp_3078_:
{
lean_object* v___x_3082_; 
if (v_isShared_3080_ == 0)
{
v___x_3082_ = v___x_3079_;
goto v_reusejp_3081_;
}
else
{
lean_object* v_reuseFailAlloc_3083_; 
v_reuseFailAlloc_3083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3083_, 0, v_a_3077_);
v___x_3082_ = v_reuseFailAlloc_3083_;
goto v_reusejp_3081_;
}
v_reusejp_3081_:
{
return v___x_3082_;
}
}
}
}
case 1:
{
lean_object* v_lhs_3085_; lean_object* v_rhs_3086_; lean_object* v___x_3087_; 
v_lhs_3085_ = lean_ctor_get(v_t_2883_, 2);
lean_inc_ref(v_lhs_3085_);
v_rhs_3086_ = lean_ctor_get(v_t_2883_, 3);
lean_inc_ref(v_rhs_3086_);
lean_dec_ref_known(v_t_2883_, 4);
v___x_3087_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go(v_lhs_3085_, v_a_2884_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_, v_a_2890_);
if (lean_obj_tag(v___x_3087_) == 0)
{
lean_dec_ref_known(v___x_3087_, 1);
v_t_2883_ = v_rhs_3086_;
goto _start;
}
else
{
lean_dec_ref(v_rhs_3086_);
return v___x_3087_;
}
}
default: 
{
lean_object* v_nested_3089_; 
v_nested_3089_ = lean_ctor_get(v_t_2883_, 3);
lean_inc_ref(v_nested_3089_);
lean_dec_ref_known(v_t_2883_, 4);
v_t_2883_ = v_nested_3089_;
goto _start;
}
}
}
else
{
lean_object* v___x_3091_; lean_object* v___x_3092_; 
lean_dec_ref(v_t_2883_);
v___x_3091_ = lean_box(0);
v___x_3092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3092_, 0, v___x_3091_);
return v___x_3092_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go___boxed(lean_object* v_t_3093_, lean_object* v_a_3094_, lean_object* v_a_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_, lean_object* v_a_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_){
_start:
{
lean_object* v_res_3102_; 
v_res_3102_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go(v_t_3093_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_);
lean_dec(v_a_3100_);
lean_dec_ref(v_a_3099_);
lean_dec(v_a_3098_);
lean_dec_ref(v_a_3097_);
lean_dec(v_a_3096_);
lean_dec_ref(v_a_3095_);
lean_dec(v_a_3094_);
return v_res_3102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2(lean_object* v_cls_3103_, lean_object* v_msg_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_, lean_object* v___y_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_){
_start:
{
lean_object* v___x_3113_; 
v___x_3113_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___redArg(v_cls_3103_, v_msg_3104_, v___y_3108_, v___y_3109_, v___y_3110_, v___y_3111_);
return v___x_3113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2___boxed(lean_object* v_cls_3114_, lean_object* v_msg_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_){
_start:
{
lean_object* v_res_3124_; 
v_res_3124_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go_spec__2(v_cls_3114_, v_msg_3115_, v___y_3116_, v___y_3117_, v___y_3118_, v___y_3119_, v___y_3120_, v___y_3121_, v___y_3122_);
lean_dec(v___y_3122_);
lean_dec_ref(v___y_3121_);
lean_dec(v___y_3120_);
lean_dec_ref(v___y_3119_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
lean_dec(v___y_3116_);
return v_res_3124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(lean_object* v_e_3125_, lean_object* v___y_3126_){
_start:
{
uint8_t v___x_3128_; 
v___x_3128_ = l_Lean_Expr_hasMVar(v_e_3125_);
if (v___x_3128_ == 0)
{
lean_object* v___x_3129_; 
v___x_3129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3129_, 0, v_e_3125_);
return v___x_3129_;
}
else
{
lean_object* v___x_3130_; lean_object* v_mctx_3131_; lean_object* v___x_3132_; lean_object* v_fst_3133_; lean_object* v_snd_3134_; lean_object* v___x_3135_; lean_object* v_cache_3136_; lean_object* v_zetaDeltaFVarIds_3137_; lean_object* v_postponed_3138_; lean_object* v_diag_3139_; lean_object* v___x_3141_; uint8_t v_isShared_3142_; uint8_t v_isSharedCheck_3148_; 
v___x_3130_ = lean_st_ref_get(v___y_3126_);
v_mctx_3131_ = lean_ctor_get(v___x_3130_, 0);
lean_inc_ref(v_mctx_3131_);
lean_dec(v___x_3130_);
v___x_3132_ = l_Lean_instantiateMVarsCore(v_mctx_3131_, v_e_3125_);
v_fst_3133_ = lean_ctor_get(v___x_3132_, 0);
lean_inc(v_fst_3133_);
v_snd_3134_ = lean_ctor_get(v___x_3132_, 1);
lean_inc(v_snd_3134_);
lean_dec_ref(v___x_3132_);
v___x_3135_ = lean_st_ref_take(v___y_3126_);
v_cache_3136_ = lean_ctor_get(v___x_3135_, 1);
v_zetaDeltaFVarIds_3137_ = lean_ctor_get(v___x_3135_, 2);
v_postponed_3138_ = lean_ctor_get(v___x_3135_, 3);
v_diag_3139_ = lean_ctor_get(v___x_3135_, 4);
v_isSharedCheck_3148_ = !lean_is_exclusive(v___x_3135_);
if (v_isSharedCheck_3148_ == 0)
{
lean_object* v_unused_3149_; 
v_unused_3149_ = lean_ctor_get(v___x_3135_, 0);
lean_dec(v_unused_3149_);
v___x_3141_ = v___x_3135_;
v_isShared_3142_ = v_isSharedCheck_3148_;
goto v_resetjp_3140_;
}
else
{
lean_inc(v_diag_3139_);
lean_inc(v_postponed_3138_);
lean_inc(v_zetaDeltaFVarIds_3137_);
lean_inc(v_cache_3136_);
lean_dec(v___x_3135_);
v___x_3141_ = lean_box(0);
v_isShared_3142_ = v_isSharedCheck_3148_;
goto v_resetjp_3140_;
}
v_resetjp_3140_:
{
lean_object* v___x_3144_; 
if (v_isShared_3142_ == 0)
{
lean_ctor_set(v___x_3141_, 0, v_snd_3134_);
v___x_3144_ = v___x_3141_;
goto v_reusejp_3143_;
}
else
{
lean_object* v_reuseFailAlloc_3147_; 
v_reuseFailAlloc_3147_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3147_, 0, v_snd_3134_);
lean_ctor_set(v_reuseFailAlloc_3147_, 1, v_cache_3136_);
lean_ctor_set(v_reuseFailAlloc_3147_, 2, v_zetaDeltaFVarIds_3137_);
lean_ctor_set(v_reuseFailAlloc_3147_, 3, v_postponed_3138_);
lean_ctor_set(v_reuseFailAlloc_3147_, 4, v_diag_3139_);
v___x_3144_ = v_reuseFailAlloc_3147_;
goto v_reusejp_3143_;
}
v_reusejp_3143_:
{
lean_object* v___x_3145_; lean_object* v___x_3146_; 
v___x_3145_ = lean_st_ref_set(v___y_3126_, v___x_3144_);
v___x_3146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3146_, 0, v_fst_3133_);
return v___x_3146_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg___boxed(lean_object* v_e_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_){
_start:
{
lean_object* v_res_3153_; 
v_res_3153_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(v_e_3150_, v___y_3151_);
lean_dec(v___y_3151_);
return v_res_3153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0(lean_object* v_e_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_){
_start:
{
lean_object* v___x_3162_; 
v___x_3162_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(v_e_3154_, v___y_3158_);
return v___x_3162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___boxed(lean_object* v_e_3163_, lean_object* v___y_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_, lean_object* v___y_3167_, lean_object* v___y_3168_, lean_object* v___y_3169_, lean_object* v___y_3170_){
_start:
{
lean_object* v_res_3171_; 
v_res_3171_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0(v_e_3163_, v___y_3164_, v___y_3165_, v___y_3166_, v___y_3167_, v___y_3168_, v___y_3169_);
lean_dec(v___y_3169_);
lean_dec_ref(v___y_3168_);
lean_dec(v___y_3167_);
lean_dec_ref(v___y_3166_);
lean_dec(v___y_3165_);
lean_dec_ref(v___y_3164_);
return v_res_3171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze(lean_object* v_t_3172_, lean_object* v_expectedType_x3f_3173_, lean_object* v_a_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_, lean_object* v_a_3178_, lean_object* v_a_3179_){
_start:
{
lean_object* v_maxS_x3f_3182_; lean_object* v___y_3183_; lean_object* v___y_3184_; lean_object* v___y_3185_; lean_object* v___y_3186_; lean_object* v___y_3187_; lean_object* v___y_3188_; 
if (lean_obj_tag(v_expectedType_x3f_3173_) == 0)
{
lean_object* v___x_3211_; 
v___x_3211_ = lean_box(0);
v_maxS_x3f_3182_ = v___x_3211_;
v___y_3183_ = v_a_3174_;
v___y_3184_ = v_a_3175_;
v___y_3185_ = v_a_3176_;
v___y_3186_ = v_a_3177_;
v___y_3187_ = v_a_3178_;
v___y_3188_ = v_a_3179_;
goto v___jp_3181_;
}
else
{
lean_object* v_val_3212_; lean_object* v___x_3213_; lean_object* v_a_3214_; lean_object* v___x_3215_; 
v_val_3212_ = lean_ctor_get(v_expectedType_x3f_3173_, 0);
lean_inc(v_val_3212_);
lean_dec_ref_known(v_expectedType_x3f_3173_, 1);
v___x_3213_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(v_val_3212_, v_a_3177_);
v_a_3214_ = lean_ctor_get(v___x_3213_, 0);
lean_inc(v_a_3214_);
lean_dec_ref(v___x_3213_);
v___x_3215_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(v_a_3214_, v_a_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_, v_a_3179_);
if (lean_obj_tag(v___x_3215_) == 0)
{
lean_object* v_a_3216_; 
v_a_3216_ = lean_ctor_get(v___x_3215_, 0);
lean_inc(v_a_3216_);
lean_dec_ref_known(v___x_3215_, 1);
if (lean_obj_tag(v_a_3216_) == 1)
{
lean_object* v_val_3217_; lean_object* v___x_3219_; uint8_t v_isShared_3220_; uint8_t v_isSharedCheck_3225_; 
v_val_3217_ = lean_ctor_get(v_a_3216_, 0);
v_isSharedCheck_3225_ = !lean_is_exclusive(v_a_3216_);
if (v_isSharedCheck_3225_ == 0)
{
v___x_3219_ = v_a_3216_;
v_isShared_3220_ = v_isSharedCheck_3225_;
goto v_resetjp_3218_;
}
else
{
lean_inc(v_val_3217_);
lean_dec(v_a_3216_);
v___x_3219_ = lean_box(0);
v_isShared_3220_ = v_isSharedCheck_3225_;
goto v_resetjp_3218_;
}
v_resetjp_3218_:
{
lean_object* v_fst_3221_; lean_object* v___x_3223_; 
v_fst_3221_ = lean_ctor_get(v_val_3217_, 0);
lean_inc(v_fst_3221_);
lean_dec(v_val_3217_);
if (v_isShared_3220_ == 0)
{
lean_ctor_set(v___x_3219_, 0, v_fst_3221_);
v___x_3223_ = v___x_3219_;
goto v_reusejp_3222_;
}
else
{
lean_object* v_reuseFailAlloc_3224_; 
v_reuseFailAlloc_3224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3224_, 0, v_fst_3221_);
v___x_3223_ = v_reuseFailAlloc_3224_;
goto v_reusejp_3222_;
}
v_reusejp_3222_:
{
v_maxS_x3f_3182_ = v___x_3223_;
v___y_3183_ = v_a_3174_;
v___y_3184_ = v_a_3175_;
v___y_3185_ = v_a_3176_;
v___y_3186_ = v_a_3177_;
v___y_3187_ = v_a_3178_;
v___y_3188_ = v_a_3179_;
goto v___jp_3181_;
}
}
}
else
{
lean_object* v___x_3226_; 
lean_dec(v_a_3216_);
v___x_3226_ = lean_box(0);
v_maxS_x3f_3182_ = v___x_3226_;
v___y_3183_ = v_a_3174_;
v___y_3184_ = v_a_3175_;
v___y_3185_ = v_a_3176_;
v___y_3186_ = v_a_3177_;
v___y_3187_ = v_a_3178_;
v___y_3188_ = v_a_3179_;
goto v___jp_3181_;
}
}
else
{
lean_object* v_a_3227_; lean_object* v___x_3229_; uint8_t v_isShared_3230_; uint8_t v_isSharedCheck_3234_; 
lean_dec_ref(v_t_3172_);
v_a_3227_ = lean_ctor_get(v___x_3215_, 0);
v_isSharedCheck_3234_ = !lean_is_exclusive(v___x_3215_);
if (v_isSharedCheck_3234_ == 0)
{
v___x_3229_ = v___x_3215_;
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
else
{
lean_inc(v_a_3227_);
lean_dec(v___x_3215_);
v___x_3229_ = lean_box(0);
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
v_resetjp_3228_:
{
lean_object* v___x_3232_; 
if (v_isShared_3230_ == 0)
{
v___x_3232_ = v___x_3229_;
goto v_reusejp_3231_;
}
else
{
lean_object* v_reuseFailAlloc_3233_; 
v_reuseFailAlloc_3233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3233_, 0, v_a_3227_);
v___x_3232_ = v_reuseFailAlloc_3233_;
goto v_reusejp_3231_;
}
v_reusejp_3231_:
{
return v___x_3232_;
}
}
}
}
v___jp_3181_:
{
uint8_t v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; 
v___x_3189_ = 0;
v___x_3190_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_3190_, 0, v_maxS_x3f_3182_);
lean_ctor_set_uint8(v___x_3190_, sizeof(void*)*1, v___x_3189_);
v___x_3191_ = lean_st_mk_ref(v___x_3190_);
v___x_3192_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_go(v_t_3172_, v___x_3191_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
if (lean_obj_tag(v___x_3192_) == 0)
{
lean_object* v___x_3194_; uint8_t v_isShared_3195_; uint8_t v_isSharedCheck_3201_; 
v_isSharedCheck_3201_ = !lean_is_exclusive(v___x_3192_);
if (v_isSharedCheck_3201_ == 0)
{
lean_object* v_unused_3202_; 
v_unused_3202_ = lean_ctor_get(v___x_3192_, 0);
lean_dec(v_unused_3202_);
v___x_3194_ = v___x_3192_;
v_isShared_3195_ = v_isSharedCheck_3201_;
goto v_resetjp_3193_;
}
else
{
lean_dec(v___x_3192_);
v___x_3194_ = lean_box(0);
v_isShared_3195_ = v_isSharedCheck_3201_;
goto v_resetjp_3193_;
}
v_resetjp_3193_:
{
lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3199_; 
v___x_3196_ = lean_st_ref_get(v___x_3191_);
v___x_3197_ = lean_st_ref_get(v___x_3191_);
lean_dec(v___x_3191_);
lean_dec(v___x_3197_);
if (v_isShared_3195_ == 0)
{
lean_ctor_set(v___x_3194_, 0, v___x_3196_);
v___x_3199_ = v___x_3194_;
goto v_reusejp_3198_;
}
else
{
lean_object* v_reuseFailAlloc_3200_; 
v_reuseFailAlloc_3200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3200_, 0, v___x_3196_);
v___x_3199_ = v_reuseFailAlloc_3200_;
goto v_reusejp_3198_;
}
v_reusejp_3198_:
{
return v___x_3199_;
}
}
}
else
{
lean_object* v_a_3203_; lean_object* v___x_3205_; uint8_t v_isShared_3206_; uint8_t v_isSharedCheck_3210_; 
lean_dec(v___x_3191_);
v_a_3203_ = lean_ctor_get(v___x_3192_, 0);
v_isSharedCheck_3210_ = !lean_is_exclusive(v___x_3192_);
if (v_isSharedCheck_3210_ == 0)
{
v___x_3205_ = v___x_3192_;
v_isShared_3206_ = v_isSharedCheck_3210_;
goto v_resetjp_3204_;
}
else
{
lean_inc(v_a_3203_);
lean_dec(v___x_3192_);
v___x_3205_ = lean_box(0);
v_isShared_3206_ = v_isSharedCheck_3210_;
goto v_resetjp_3204_;
}
v_resetjp_3204_:
{
lean_object* v___x_3208_; 
if (v_isShared_3206_ == 0)
{
v___x_3208_ = v___x_3205_;
goto v_reusejp_3207_;
}
else
{
lean_object* v_reuseFailAlloc_3209_; 
v_reuseFailAlloc_3209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3209_, 0, v_a_3203_);
v___x_3208_ = v_reuseFailAlloc_3209_;
goto v_reusejp_3207_;
}
v_reusejp_3207_:
{
return v___x_3208_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze___boxed(lean_object* v_t_3235_, lean_object* v_expectedType_x3f_3236_, lean_object* v_a_3237_, lean_object* v_a_3238_, lean_object* v_a_3239_, lean_object* v_a_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_){
_start:
{
lean_object* v_res_3244_; 
v_res_3244_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze(v_t_3235_, v_expectedType_x3f_3236_, v_a_3237_, v_a_3238_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_);
lean_dec(v_a_3242_);
lean_dec_ref(v_a_3241_);
lean_dec(v_a_3240_);
lean_dec_ref(v_a_3239_);
lean_dec(v_a_3238_);
lean_dec_ref(v_a_3237_);
return v_res_3244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp(lean_object* v_f_3245_, lean_object* v_lhs_3246_, lean_object* v_rhs_3247_, lean_object* v_a_3248_, lean_object* v_a_3249_, lean_object* v_a_3250_, lean_object* v_a_3251_, lean_object* v_a_3252_, lean_object* v_a_3253_){
_start:
{
lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; lean_object* v___x_3258_; lean_object* v___x_3259_; lean_object* v___x_3260_; lean_object* v___x_3261_; lean_object* v___x_3262_; uint8_t v___x_3263_; lean_object* v___x_3264_; 
v___x_3255_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS___closed__0));
v___x_3256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3256_, 0, v_lhs_3246_);
v___x_3257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3257_, 0, v_rhs_3247_);
v___x_3258_ = lean_unsigned_to_nat(2u);
v___x_3259_ = lean_mk_empty_array_with_capacity(v___x_3258_);
v___x_3260_ = lean_array_push(v___x_3259_, v___x_3256_);
v___x_3261_ = lean_array_push(v___x_3260_, v___x_3257_);
v___x_3262_ = lean_box(0);
v___x_3263_ = 0;
v___x_3264_ = l_Lean_Elab_Term_elabAppArgs(v_f_3245_, v___x_3255_, v___x_3261_, v___x_3262_, v___x_3263_, v___x_3263_, v___x_3263_, v_a_3248_, v_a_3249_, v_a_3250_, v_a_3251_, v_a_3252_, v_a_3253_);
return v___x_3264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp___boxed(lean_object* v_f_3265_, lean_object* v_lhs_3266_, lean_object* v_rhs_3267_, lean_object* v_a_3268_, lean_object* v_a_3269_, lean_object* v_a_3270_, lean_object* v_a_3271_, lean_object* v_a_3272_, lean_object* v_a_3273_, lean_object* v_a_3274_){
_start:
{
lean_object* v_res_3275_; 
v_res_3275_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp(v_f_3265_, v_lhs_3266_, v_rhs_3267_, v_a_3268_, v_a_3269_, v_a_3270_, v_a_3271_, v_a_3272_, v_a_3273_);
lean_dec(v_a_3273_);
lean_dec_ref(v_a_3272_);
lean_dec(v_a_3271_);
lean_dec_ref(v_a_3270_);
lean_dec(v_a_3269_);
lean_dec_ref(v_a_3268_);
return v_res_3275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0(lean_object* v___y_3276_, lean_object* v_mkInfoTree_3277_, lean_object* v___y_3278_, lean_object* v___y_3279_, lean_object* v___y_3280_, lean_object* v___y_3281_, lean_object* v___y_3282_, lean_object* v_a_3283_, lean_object* v_a_x3f_3284_){
_start:
{
lean_object* v___x_3286_; lean_object* v_infoState_3287_; lean_object* v_trees_3288_; lean_object* v___x_3289_; 
v___x_3286_ = lean_st_ref_get(v___y_3276_);
v_infoState_3287_ = lean_ctor_get(v___x_3286_, 7);
lean_inc_ref(v_infoState_3287_);
lean_dec(v___x_3286_);
v_trees_3288_ = lean_ctor_get(v_infoState_3287_, 2);
lean_inc_ref(v_trees_3288_);
lean_dec_ref(v_infoState_3287_);
lean_inc(v___y_3276_);
lean_inc_ref(v___y_3282_);
lean_inc(v___y_3281_);
lean_inc_ref(v___y_3280_);
lean_inc(v___y_3279_);
lean_inc_ref(v___y_3278_);
v___x_3289_ = lean_apply_8(v_mkInfoTree_3277_, v_trees_3288_, v___y_3278_, v___y_3279_, v___y_3280_, v___y_3281_, v___y_3282_, v___y_3276_, lean_box(0));
if (lean_obj_tag(v___x_3289_) == 0)
{
lean_object* v_a_3290_; lean_object* v___x_3292_; uint8_t v_isShared_3293_; uint8_t v_isSharedCheck_3328_; 
v_a_3290_ = lean_ctor_get(v___x_3289_, 0);
v_isSharedCheck_3328_ = !lean_is_exclusive(v___x_3289_);
if (v_isSharedCheck_3328_ == 0)
{
v___x_3292_ = v___x_3289_;
v_isShared_3293_ = v_isSharedCheck_3328_;
goto v_resetjp_3291_;
}
else
{
lean_inc(v_a_3290_);
lean_dec(v___x_3289_);
v___x_3292_ = lean_box(0);
v_isShared_3293_ = v_isSharedCheck_3328_;
goto v_resetjp_3291_;
}
v_resetjp_3291_:
{
lean_object* v___x_3294_; lean_object* v_infoState_3295_; lean_object* v_env_3296_; lean_object* v_nextMacroScope_3297_; lean_object* v_ngen_3298_; lean_object* v_auxDeclNGen_3299_; lean_object* v_traceState_3300_; lean_object* v_cache_3301_; lean_object* v_messages_3302_; lean_object* v_snapshotTasks_3303_; lean_object* v___x_3305_; uint8_t v_isShared_3306_; uint8_t v_isSharedCheck_3327_; 
v___x_3294_ = lean_st_ref_take(v___y_3276_);
v_infoState_3295_ = lean_ctor_get(v___x_3294_, 7);
v_env_3296_ = lean_ctor_get(v___x_3294_, 0);
v_nextMacroScope_3297_ = lean_ctor_get(v___x_3294_, 1);
v_ngen_3298_ = lean_ctor_get(v___x_3294_, 2);
v_auxDeclNGen_3299_ = lean_ctor_get(v___x_3294_, 3);
v_traceState_3300_ = lean_ctor_get(v___x_3294_, 4);
v_cache_3301_ = lean_ctor_get(v___x_3294_, 5);
v_messages_3302_ = lean_ctor_get(v___x_3294_, 6);
v_snapshotTasks_3303_ = lean_ctor_get(v___x_3294_, 8);
v_isSharedCheck_3327_ = !lean_is_exclusive(v___x_3294_);
if (v_isSharedCheck_3327_ == 0)
{
v___x_3305_ = v___x_3294_;
v_isShared_3306_ = v_isSharedCheck_3327_;
goto v_resetjp_3304_;
}
else
{
lean_inc(v_snapshotTasks_3303_);
lean_inc(v_infoState_3295_);
lean_inc(v_messages_3302_);
lean_inc(v_cache_3301_);
lean_inc(v_traceState_3300_);
lean_inc(v_auxDeclNGen_3299_);
lean_inc(v_ngen_3298_);
lean_inc(v_nextMacroScope_3297_);
lean_inc(v_env_3296_);
lean_dec(v___x_3294_);
v___x_3305_ = lean_box(0);
v_isShared_3306_ = v_isSharedCheck_3327_;
goto v_resetjp_3304_;
}
v_resetjp_3304_:
{
uint8_t v_enabled_3307_; lean_object* v_assignment_3308_; lean_object* v_lazyAssignment_3309_; lean_object* v___x_3311_; uint8_t v_isShared_3312_; uint8_t v_isSharedCheck_3325_; 
v_enabled_3307_ = lean_ctor_get_uint8(v_infoState_3295_, sizeof(void*)*3);
v_assignment_3308_ = lean_ctor_get(v_infoState_3295_, 0);
v_lazyAssignment_3309_ = lean_ctor_get(v_infoState_3295_, 1);
v_isSharedCheck_3325_ = !lean_is_exclusive(v_infoState_3295_);
if (v_isSharedCheck_3325_ == 0)
{
lean_object* v_unused_3326_; 
v_unused_3326_ = lean_ctor_get(v_infoState_3295_, 2);
lean_dec(v_unused_3326_);
v___x_3311_ = v_infoState_3295_;
v_isShared_3312_ = v_isSharedCheck_3325_;
goto v_resetjp_3310_;
}
else
{
lean_inc(v_lazyAssignment_3309_);
lean_inc(v_assignment_3308_);
lean_dec(v_infoState_3295_);
v___x_3311_ = lean_box(0);
v_isShared_3312_ = v_isSharedCheck_3325_;
goto v_resetjp_3310_;
}
v_resetjp_3310_:
{
lean_object* v___x_3313_; lean_object* v___x_3315_; 
v___x_3313_ = l_Lean_PersistentArray_push___redArg(v_a_3283_, v_a_3290_);
if (v_isShared_3312_ == 0)
{
lean_ctor_set(v___x_3311_, 2, v___x_3313_);
v___x_3315_ = v___x_3311_;
goto v_reusejp_3314_;
}
else
{
lean_object* v_reuseFailAlloc_3324_; 
v_reuseFailAlloc_3324_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_3324_, 0, v_assignment_3308_);
lean_ctor_set(v_reuseFailAlloc_3324_, 1, v_lazyAssignment_3309_);
lean_ctor_set(v_reuseFailAlloc_3324_, 2, v___x_3313_);
lean_ctor_set_uint8(v_reuseFailAlloc_3324_, sizeof(void*)*3, v_enabled_3307_);
v___x_3315_ = v_reuseFailAlloc_3324_;
goto v_reusejp_3314_;
}
v_reusejp_3314_:
{
lean_object* v___x_3317_; 
if (v_isShared_3306_ == 0)
{
lean_ctor_set(v___x_3305_, 7, v___x_3315_);
v___x_3317_ = v___x_3305_;
goto v_reusejp_3316_;
}
else
{
lean_object* v_reuseFailAlloc_3323_; 
v_reuseFailAlloc_3323_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3323_, 0, v_env_3296_);
lean_ctor_set(v_reuseFailAlloc_3323_, 1, v_nextMacroScope_3297_);
lean_ctor_set(v_reuseFailAlloc_3323_, 2, v_ngen_3298_);
lean_ctor_set(v_reuseFailAlloc_3323_, 3, v_auxDeclNGen_3299_);
lean_ctor_set(v_reuseFailAlloc_3323_, 4, v_traceState_3300_);
lean_ctor_set(v_reuseFailAlloc_3323_, 5, v_cache_3301_);
lean_ctor_set(v_reuseFailAlloc_3323_, 6, v_messages_3302_);
lean_ctor_set(v_reuseFailAlloc_3323_, 7, v___x_3315_);
lean_ctor_set(v_reuseFailAlloc_3323_, 8, v_snapshotTasks_3303_);
v___x_3317_ = v_reuseFailAlloc_3323_;
goto v_reusejp_3316_;
}
v_reusejp_3316_:
{
lean_object* v___x_3318_; lean_object* v___x_3319_; lean_object* v___x_3321_; 
v___x_3318_ = lean_st_ref_set(v___y_3276_, v___x_3317_);
v___x_3319_ = lean_box(0);
if (v_isShared_3293_ == 0)
{
lean_ctor_set(v___x_3292_, 0, v___x_3319_);
v___x_3321_ = v___x_3292_;
goto v_reusejp_3320_;
}
else
{
lean_object* v_reuseFailAlloc_3322_; 
v_reuseFailAlloc_3322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3322_, 0, v___x_3319_);
v___x_3321_ = v_reuseFailAlloc_3322_;
goto v_reusejp_3320_;
}
v_reusejp_3320_:
{
return v___x_3321_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3329_; lean_object* v___x_3331_; uint8_t v_isShared_3332_; uint8_t v_isSharedCheck_3336_; 
lean_dec_ref(v_a_3283_);
v_a_3329_ = lean_ctor_get(v___x_3289_, 0);
v_isSharedCheck_3336_ = !lean_is_exclusive(v___x_3289_);
if (v_isSharedCheck_3336_ == 0)
{
v___x_3331_ = v___x_3289_;
v_isShared_3332_ = v_isSharedCheck_3336_;
goto v_resetjp_3330_;
}
else
{
lean_inc(v_a_3329_);
lean_dec(v___x_3289_);
v___x_3331_ = lean_box(0);
v_isShared_3332_ = v_isSharedCheck_3336_;
goto v_resetjp_3330_;
}
v_resetjp_3330_:
{
lean_object* v___x_3334_; 
if (v_isShared_3332_ == 0)
{
v___x_3334_ = v___x_3331_;
goto v_reusejp_3333_;
}
else
{
lean_object* v_reuseFailAlloc_3335_; 
v_reuseFailAlloc_3335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3335_, 0, v_a_3329_);
v___x_3334_ = v_reuseFailAlloc_3335_;
goto v_reusejp_3333_;
}
v_reusejp_3333_:
{
return v___x_3334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object* v___y_3337_, lean_object* v_mkInfoTree_3338_, lean_object* v___y_3339_, lean_object* v___y_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_, lean_object* v_a_3344_, lean_object* v_a_x3f_3345_, lean_object* v___y_3346_){
_start:
{
lean_object* v_res_3347_; 
v_res_3347_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0(v___y_3337_, v_mkInfoTree_3338_, v___y_3339_, v___y_3340_, v___y_3341_, v___y_3342_, v___y_3343_, v_a_3344_, v_a_x3f_3345_);
lean_dec(v_a_x3f_3345_);
lean_dec_ref(v___y_3343_);
lean_dec(v___y_3342_);
lean_dec_ref(v___y_3341_);
lean_dec(v___y_3340_);
lean_dec_ref(v___y_3339_);
lean_dec(v___y_3337_);
return v_res_3347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg(lean_object* v_x_3348_, lean_object* v_mkInfoTree_3349_, lean_object* v___y_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_, lean_object* v___y_3355_){
_start:
{
lean_object* v___x_3357_; lean_object* v_infoState_3358_; uint8_t v_enabled_3359_; 
v___x_3357_ = lean_st_ref_get(v___y_3355_);
v_infoState_3358_ = lean_ctor_get(v___x_3357_, 7);
lean_inc_ref(v_infoState_3358_);
lean_dec(v___x_3357_);
v_enabled_3359_ = lean_ctor_get_uint8(v_infoState_3358_, sizeof(void*)*3);
lean_dec_ref(v_infoState_3358_);
if (v_enabled_3359_ == 0)
{
lean_object* v___x_3360_; 
lean_dec_ref(v_mkInfoTree_3349_);
lean_inc(v___y_3355_);
lean_inc_ref(v___y_3354_);
lean_inc(v___y_3353_);
lean_inc_ref(v___y_3352_);
lean_inc(v___y_3351_);
lean_inc_ref(v___y_3350_);
v___x_3360_ = lean_apply_7(v_x_3348_, v___y_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v___y_3355_, lean_box(0));
return v___x_3360_;
}
else
{
lean_object* v___x_3361_; lean_object* v_a_3362_; lean_object* v_r_3363_; 
v___x_3361_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_processLeaf_spec__0___redArg(v___y_3355_);
v_a_3362_ = lean_ctor_get(v___x_3361_, 0);
lean_inc(v_a_3362_);
lean_dec_ref(v___x_3361_);
lean_inc(v___y_3355_);
lean_inc_ref(v___y_3354_);
lean_inc(v___y_3353_);
lean_inc_ref(v___y_3352_);
lean_inc(v___y_3351_);
lean_inc_ref(v___y_3350_);
v_r_3363_ = lean_apply_7(v_x_3348_, v___y_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v___y_3355_, lean_box(0));
if (lean_obj_tag(v_r_3363_) == 0)
{
lean_object* v_a_3364_; lean_object* v___x_3366_; uint8_t v_isShared_3367_; uint8_t v_isSharedCheck_3388_; 
v_a_3364_ = lean_ctor_get(v_r_3363_, 0);
v_isSharedCheck_3388_ = !lean_is_exclusive(v_r_3363_);
if (v_isSharedCheck_3388_ == 0)
{
v___x_3366_ = v_r_3363_;
v_isShared_3367_ = v_isSharedCheck_3388_;
goto v_resetjp_3365_;
}
else
{
lean_inc(v_a_3364_);
lean_dec(v_r_3363_);
v___x_3366_ = lean_box(0);
v_isShared_3367_ = v_isSharedCheck_3388_;
goto v_resetjp_3365_;
}
v_resetjp_3365_:
{
lean_object* v___x_3369_; 
lean_inc(v_a_3364_);
if (v_isShared_3367_ == 0)
{
lean_ctor_set_tag(v___x_3366_, 1);
v___x_3369_ = v___x_3366_;
goto v_reusejp_3368_;
}
else
{
lean_object* v_reuseFailAlloc_3387_; 
v_reuseFailAlloc_3387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3387_, 0, v_a_3364_);
v___x_3369_ = v_reuseFailAlloc_3387_;
goto v_reusejp_3368_;
}
v_reusejp_3368_:
{
lean_object* v___x_3370_; 
v___x_3370_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0(v___y_3355_, v_mkInfoTree_3349_, v___y_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v_a_3362_, v___x_3369_);
lean_dec_ref(v___x_3369_);
if (lean_obj_tag(v___x_3370_) == 0)
{
lean_object* v___x_3372_; uint8_t v_isShared_3373_; uint8_t v_isSharedCheck_3377_; 
v_isSharedCheck_3377_ = !lean_is_exclusive(v___x_3370_);
if (v_isSharedCheck_3377_ == 0)
{
lean_object* v_unused_3378_; 
v_unused_3378_ = lean_ctor_get(v___x_3370_, 0);
lean_dec(v_unused_3378_);
v___x_3372_ = v___x_3370_;
v_isShared_3373_ = v_isSharedCheck_3377_;
goto v_resetjp_3371_;
}
else
{
lean_dec(v___x_3370_);
v___x_3372_ = lean_box(0);
v_isShared_3373_ = v_isSharedCheck_3377_;
goto v_resetjp_3371_;
}
v_resetjp_3371_:
{
lean_object* v___x_3375_; 
if (v_isShared_3373_ == 0)
{
lean_ctor_set(v___x_3372_, 0, v_a_3364_);
v___x_3375_ = v___x_3372_;
goto v_reusejp_3374_;
}
else
{
lean_object* v_reuseFailAlloc_3376_; 
v_reuseFailAlloc_3376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3376_, 0, v_a_3364_);
v___x_3375_ = v_reuseFailAlloc_3376_;
goto v_reusejp_3374_;
}
v_reusejp_3374_:
{
return v___x_3375_;
}
}
}
else
{
lean_object* v_a_3379_; lean_object* v___x_3381_; uint8_t v_isShared_3382_; uint8_t v_isSharedCheck_3386_; 
lean_dec(v_a_3364_);
v_a_3379_ = lean_ctor_get(v___x_3370_, 0);
v_isSharedCheck_3386_ = !lean_is_exclusive(v___x_3370_);
if (v_isSharedCheck_3386_ == 0)
{
v___x_3381_ = v___x_3370_;
v_isShared_3382_ = v_isSharedCheck_3386_;
goto v_resetjp_3380_;
}
else
{
lean_inc(v_a_3379_);
lean_dec(v___x_3370_);
v___x_3381_ = lean_box(0);
v_isShared_3382_ = v_isSharedCheck_3386_;
goto v_resetjp_3380_;
}
v_resetjp_3380_:
{
lean_object* v___x_3384_; 
if (v_isShared_3382_ == 0)
{
v___x_3384_ = v___x_3381_;
goto v_reusejp_3383_;
}
else
{
lean_object* v_reuseFailAlloc_3385_; 
v_reuseFailAlloc_3385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3385_, 0, v_a_3379_);
v___x_3384_ = v_reuseFailAlloc_3385_;
goto v_reusejp_3383_;
}
v_reusejp_3383_:
{
return v___x_3384_;
}
}
}
}
}
}
else
{
lean_object* v_a_3389_; lean_object* v___x_3390_; lean_object* v___x_3391_; 
v_a_3389_ = lean_ctor_get(v_r_3363_, 0);
lean_inc(v_a_3389_);
lean_dec_ref_known(v_r_3363_, 1);
v___x_3390_ = lean_box(0);
v___x_3391_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___lam__0(v___y_3355_, v_mkInfoTree_3349_, v___y_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v_a_3362_, v___x_3390_);
if (lean_obj_tag(v___x_3391_) == 0)
{
lean_object* v___x_3393_; uint8_t v_isShared_3394_; uint8_t v_isSharedCheck_3398_; 
v_isSharedCheck_3398_ = !lean_is_exclusive(v___x_3391_);
if (v_isSharedCheck_3398_ == 0)
{
lean_object* v_unused_3399_; 
v_unused_3399_ = lean_ctor_get(v___x_3391_, 0);
lean_dec(v_unused_3399_);
v___x_3393_ = v___x_3391_;
v_isShared_3394_ = v_isSharedCheck_3398_;
goto v_resetjp_3392_;
}
else
{
lean_dec(v___x_3391_);
v___x_3393_ = lean_box(0);
v_isShared_3394_ = v_isSharedCheck_3398_;
goto v_resetjp_3392_;
}
v_resetjp_3392_:
{
lean_object* v___x_3396_; 
if (v_isShared_3394_ == 0)
{
lean_ctor_set_tag(v___x_3393_, 1);
lean_ctor_set(v___x_3393_, 0, v_a_3389_);
v___x_3396_ = v___x_3393_;
goto v_reusejp_3395_;
}
else
{
lean_object* v_reuseFailAlloc_3397_; 
v_reuseFailAlloc_3397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3397_, 0, v_a_3389_);
v___x_3396_ = v_reuseFailAlloc_3397_;
goto v_reusejp_3395_;
}
v_reusejp_3395_:
{
return v___x_3396_;
}
}
}
else
{
lean_object* v_a_3400_; lean_object* v___x_3402_; uint8_t v_isShared_3403_; uint8_t v_isSharedCheck_3407_; 
lean_dec(v_a_3389_);
v_a_3400_ = lean_ctor_get(v___x_3391_, 0);
v_isSharedCheck_3407_ = !lean_is_exclusive(v___x_3391_);
if (v_isSharedCheck_3407_ == 0)
{
v___x_3402_ = v___x_3391_;
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
else
{
lean_inc(v_a_3400_);
lean_dec(v___x_3391_);
v___x_3402_ = lean_box(0);
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
v_resetjp_3401_:
{
lean_object* v___x_3405_; 
if (v_isShared_3403_ == 0)
{
v___x_3405_ = v___x_3402_;
goto v_reusejp_3404_;
}
else
{
lean_object* v_reuseFailAlloc_3406_; 
v_reuseFailAlloc_3406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3406_, 0, v_a_3400_);
v___x_3405_ = v_reuseFailAlloc_3406_;
goto v_reusejp_3404_;
}
v_reusejp_3404_:
{
return v___x_3405_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_3408_, lean_object* v_mkInfoTree_3409_, lean_object* v___y_3410_, lean_object* v___y_3411_, lean_object* v___y_3412_, lean_object* v___y_3413_, lean_object* v___y_3414_, lean_object* v___y_3415_, lean_object* v___y_3416_){
_start:
{
lean_object* v_res_3417_; 
v_res_3417_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg(v_x_3408_, v_mkInfoTree_3409_, v___y_3410_, v___y_3411_, v___y_3412_, v___y_3413_, v___y_3414_, v___y_3415_);
lean_dec(v___y_3415_);
lean_dec_ref(v___y_3414_);
lean_dec(v___y_3413_);
lean_dec_ref(v___y_3412_);
lean_dec(v___y_3411_);
lean_dec_ref(v___y_3410_);
return v_res_3417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0(lean_object* v_stx_3418_, lean_object* v_output_3419_, lean_object* v_trees_3420_, lean_object* v___y_3421_, lean_object* v___y_3422_, lean_object* v___y_3423_, lean_object* v___y_3424_, lean_object* v___y_3425_, lean_object* v___y_3426_){
_start:
{
lean_object* v_lctx_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; 
v_lctx_3428_ = lean_ctor_get(v___y_3423_, 2);
lean_inc_ref(v_lctx_3428_);
v___x_3429_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3429_, 0, v_lctx_3428_);
lean_ctor_set(v___x_3429_, 1, v_stx_3418_);
lean_ctor_set(v___x_3429_, 2, v_output_3419_);
v___x_3430_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_3430_, 0, v___x_3429_);
v___x_3431_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3431_, 0, v___x_3430_);
lean_ctor_set(v___x_3431_, 1, v_trees_3420_);
v___x_3432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3432_, 0, v___x_3431_);
return v___x_3432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_stx_3433_, lean_object* v_output_3434_, lean_object* v_trees_3435_, lean_object* v___y_3436_, lean_object* v___y_3437_, lean_object* v___y_3438_, lean_object* v___y_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_, lean_object* v___y_3442_){
_start:
{
lean_object* v_res_3443_; 
v_res_3443_ = lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0(v_stx_3433_, v_output_3434_, v_trees_3435_, v___y_3436_, v___y_3437_, v___y_3438_, v___y_3439_, v___y_3440_, v___y_3441_);
lean_dec(v___y_3441_);
lean_dec_ref(v___y_3440_);
lean_dec(v___y_3439_);
lean_dec_ref(v___y_3438_);
lean_dec(v___y_3437_);
lean_dec_ref(v___y_3436_);
return v_res_3443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg(lean_object* v_stx_3444_, lean_object* v_output_3445_, lean_object* v_x_3446_, lean_object* v___y_3447_, lean_object* v___y_3448_, lean_object* v___y_3449_, lean_object* v___y_3450_, lean_object* v___y_3451_, lean_object* v___y_3452_){
_start:
{
lean_object* v___f_3454_; lean_object* v___x_3455_; 
v___f_3454_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___lam__0___boxed), 10, 2);
lean_closure_set(v___f_3454_, 0, v_stx_3444_);
lean_closure_set(v___f_3454_, 1, v_output_3445_);
v___x_3455_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg(v_x_3446_, v___f_3454_, v___y_3447_, v___y_3448_, v___y_3449_, v___y_3450_, v___y_3451_, v___y_3452_);
return v___x_3455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg___boxed(lean_object* v_stx_3456_, lean_object* v_output_3457_, lean_object* v_x_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_, lean_object* v___y_3461_, lean_object* v___y_3462_, lean_object* v___y_3463_, lean_object* v___y_3464_, lean_object* v___y_3465_){
_start:
{
lean_object* v_res_3466_; 
v_res_3466_ = lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg(v_stx_3456_, v_output_3457_, v_x_3458_, v___y_3459_, v___y_3460_, v___y_3461_, v___y_3462_, v___y_3463_, v___y_3464_);
lean_dec(v___y_3464_);
lean_dec_ref(v___y_3463_);
lean_dec(v___y_3462_);
lean_dec_ref(v___y_3461_);
lean_dec(v___y_3460_);
lean_dec_ref(v___y_3459_);
return v_res_3466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg(lean_object* v_beforeStx_3467_, lean_object* v_afterStx_3468_, lean_object* v_x_3469_, lean_object* v___y_3470_, lean_object* v___y_3471_, lean_object* v___y_3472_, lean_object* v___y_3473_, lean_object* v___y_3474_, lean_object* v___y_3475_){
_start:
{
lean_object* v___x_3477_; lean_object* v___x_3478_; 
lean_inc(v_afterStx_3468_);
lean_inc(v_beforeStx_3467_);
v___x_3477_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_withPushMacroExpansionStack___boxed), 11, 4);
lean_closure_set(v___x_3477_, 0, lean_box(0));
lean_closure_set(v___x_3477_, 1, v_beforeStx_3467_);
lean_closure_set(v___x_3477_, 2, v_afterStx_3468_);
lean_closure_set(v___x_3477_, 3, v_x_3469_);
v___x_3478_ = lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg(v_beforeStx_3467_, v_afterStx_3468_, v___x_3477_, v___y_3470_, v___y_3471_, v___y_3472_, v___y_3473_, v___y_3474_, v___y_3475_);
if (lean_obj_tag(v___x_3478_) == 0)
{
lean_object* v_a_3479_; lean_object* v___x_3481_; uint8_t v_isShared_3482_; uint8_t v_isSharedCheck_3486_; 
v_a_3479_ = lean_ctor_get(v___x_3478_, 0);
v_isSharedCheck_3486_ = !lean_is_exclusive(v___x_3478_);
if (v_isSharedCheck_3486_ == 0)
{
v___x_3481_ = v___x_3478_;
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
else
{
lean_inc(v_a_3479_);
lean_dec(v___x_3478_);
v___x_3481_ = lean_box(0);
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
v_resetjp_3480_:
{
lean_object* v___x_3484_; 
if (v_isShared_3482_ == 0)
{
v___x_3484_ = v___x_3481_;
goto v_reusejp_3483_;
}
else
{
lean_object* v_reuseFailAlloc_3485_; 
v_reuseFailAlloc_3485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3485_, 0, v_a_3479_);
v___x_3484_ = v_reuseFailAlloc_3485_;
goto v_reusejp_3483_;
}
v_reusejp_3483_:
{
return v___x_3484_;
}
}
}
else
{
lean_object* v_a_3487_; lean_object* v___x_3489_; uint8_t v_isShared_3490_; uint8_t v_isSharedCheck_3494_; 
v_a_3487_ = lean_ctor_get(v___x_3478_, 0);
v_isSharedCheck_3494_ = !lean_is_exclusive(v___x_3478_);
if (v_isSharedCheck_3494_ == 0)
{
v___x_3489_ = v___x_3478_;
v_isShared_3490_ = v_isSharedCheck_3494_;
goto v_resetjp_3488_;
}
else
{
lean_inc(v_a_3487_);
lean_dec(v___x_3478_);
v___x_3489_ = lean_box(0);
v_isShared_3490_ = v_isSharedCheck_3494_;
goto v_resetjp_3488_;
}
v_resetjp_3488_:
{
lean_object* v___x_3492_; 
if (v_isShared_3490_ == 0)
{
v___x_3492_ = v___x_3489_;
goto v_reusejp_3491_;
}
else
{
lean_object* v_reuseFailAlloc_3493_; 
v_reuseFailAlloc_3493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3493_, 0, v_a_3487_);
v___x_3492_ = v_reuseFailAlloc_3493_;
goto v_reusejp_3491_;
}
v_reusejp_3491_:
{
return v___x_3492_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg___boxed(lean_object* v_beforeStx_3495_, lean_object* v_afterStx_3496_, lean_object* v_x_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_, lean_object* v___y_3502_, lean_object* v___y_3503_, lean_object* v___y_3504_){
_start:
{
lean_object* v_res_3505_; 
v_res_3505_ = lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg(v_beforeStx_3495_, v_afterStx_3496_, v_x_3497_, v___y_3498_, v___y_3499_, v___y_3500_, v___y_3501_, v___y_3502_, v___y_3503_);
lean_dec(v___y_3503_);
lean_dec_ref(v___y_3502_);
lean_dec(v___y_3501_);
lean_dec_ref(v___y_3500_);
lean_dec(v___y_3499_);
lean_dec_ref(v___y_3498_);
return v_res_3505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0(lean_object* v_00_u03b1_3506_, lean_object* v_beforeStx_3507_, lean_object* v_afterStx_3508_, lean_object* v_x_3509_, lean_object* v___y_3510_, lean_object* v___y_3511_, lean_object* v___y_3512_, lean_object* v___y_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_){
_start:
{
lean_object* v___x_3517_; 
v___x_3517_ = lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___redArg(v_beforeStx_3507_, v_afterStx_3508_, v_x_3509_, v___y_3510_, v___y_3511_, v___y_3512_, v___y_3513_, v___y_3514_, v___y_3515_);
return v___x_3517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___boxed(lean_object* v_00_u03b1_3518_, lean_object* v_beforeStx_3519_, lean_object* v_afterStx_3520_, lean_object* v_x_3521_, lean_object* v___y_3522_, lean_object* v___y_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_, lean_object* v___y_3527_, lean_object* v___y_3528_){
_start:
{
lean_object* v_res_3529_; 
v_res_3529_ = lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0(v_00_u03b1_3518_, v_beforeStx_3519_, v_afterStx_3520_, v_x_3521_, v___y_3522_, v___y_3523_, v___y_3524_, v___y_3525_, v___y_3526_, v___y_3527_);
lean_dec(v___y_3527_);
lean_dec_ref(v___y_3526_);
lean_dec(v___y_3525_);
lean_dec_ref(v___y_3524_);
lean_dec(v___y_3523_);
lean_dec_ref(v___y_3522_);
return v_res_3529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0___boxed(lean_object* v_lhs_3530_, lean_object* v_rhs_3531_, lean_object* v_f_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_, lean_object* v___y_3535_, lean_object* v___y_3536_, lean_object* v___y_3537_, lean_object* v___y_3538_, lean_object* v___y_3539_){
_start:
{
lean_object* v_res_3540_; 
v_res_3540_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0(v_lhs_3530_, v_rhs_3531_, v_f_3532_, v___y_3533_, v___y_3534_, v___y_3535_, v___y_3536_, v___y_3537_, v___y_3538_);
lean_dec(v___y_3538_);
lean_dec_ref(v___y_3537_);
lean_dec(v___y_3536_);
lean_dec_ref(v___y_3535_);
lean_dec(v___y_3534_);
lean_dec_ref(v___y_3533_);
return v_res_3540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___boxed(lean_object* v_t_3541_, lean_object* v_a_3542_, lean_object* v_a_3543_, lean_object* v_a_3544_, lean_object* v_a_3545_, lean_object* v_a_3546_, lean_object* v_a_3547_, lean_object* v_a_3548_){
_start:
{
lean_object* v_res_3549_; 
v_res_3549_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(v_t_3541_, v_a_3542_, v_a_3543_, v_a_3544_, v_a_3545_, v_a_3546_, v_a_3547_);
lean_dec(v_a_3547_);
lean_dec_ref(v_a_3546_);
lean_dec(v_a_3545_);
lean_dec_ref(v_a_3544_);
lean_dec(v_a_3543_);
lean_dec_ref(v_a_3542_);
return v_res_3549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(lean_object* v_t_3550_, lean_object* v_a_3551_, lean_object* v_a_3552_, lean_object* v_a_3553_, lean_object* v_a_3554_, lean_object* v_a_3555_, lean_object* v_a_3556_){
_start:
{
switch(lean_obj_tag(v_t_3550_))
{
case 0:
{
lean_object* v_infoTrees_3558_; lean_object* v_val_3559_; lean_object* v___x_3560_; lean_object* v_infoState_3561_; lean_object* v_env_3562_; lean_object* v_nextMacroScope_3563_; lean_object* v_ngen_3564_; lean_object* v_auxDeclNGen_3565_; lean_object* v_traceState_3566_; lean_object* v_cache_3567_; lean_object* v_messages_3568_; lean_object* v_snapshotTasks_3569_; lean_object* v___x_3571_; uint8_t v_isShared_3572_; uint8_t v_isSharedCheck_3590_; 
v_infoTrees_3558_ = lean_ctor_get(v_t_3550_, 1);
lean_inc_ref(v_infoTrees_3558_);
v_val_3559_ = lean_ctor_get(v_t_3550_, 2);
lean_inc_ref(v_val_3559_);
lean_dec_ref_known(v_t_3550_, 3);
v___x_3560_ = lean_st_ref_take(v_a_3556_);
v_infoState_3561_ = lean_ctor_get(v___x_3560_, 7);
v_env_3562_ = lean_ctor_get(v___x_3560_, 0);
v_nextMacroScope_3563_ = lean_ctor_get(v___x_3560_, 1);
v_ngen_3564_ = lean_ctor_get(v___x_3560_, 2);
v_auxDeclNGen_3565_ = lean_ctor_get(v___x_3560_, 3);
v_traceState_3566_ = lean_ctor_get(v___x_3560_, 4);
v_cache_3567_ = lean_ctor_get(v___x_3560_, 5);
v_messages_3568_ = lean_ctor_get(v___x_3560_, 6);
v_snapshotTasks_3569_ = lean_ctor_get(v___x_3560_, 8);
v_isSharedCheck_3590_ = !lean_is_exclusive(v___x_3560_);
if (v_isSharedCheck_3590_ == 0)
{
v___x_3571_ = v___x_3560_;
v_isShared_3572_ = v_isSharedCheck_3590_;
goto v_resetjp_3570_;
}
else
{
lean_inc(v_snapshotTasks_3569_);
lean_inc(v_infoState_3561_);
lean_inc(v_messages_3568_);
lean_inc(v_cache_3567_);
lean_inc(v_traceState_3566_);
lean_inc(v_auxDeclNGen_3565_);
lean_inc(v_ngen_3564_);
lean_inc(v_nextMacroScope_3563_);
lean_inc(v_env_3562_);
lean_dec(v___x_3560_);
v___x_3571_ = lean_box(0);
v_isShared_3572_ = v_isSharedCheck_3590_;
goto v_resetjp_3570_;
}
v_resetjp_3570_:
{
uint8_t v_enabled_3573_; lean_object* v_assignment_3574_; lean_object* v_lazyAssignment_3575_; lean_object* v_trees_3576_; lean_object* v___x_3578_; uint8_t v_isShared_3579_; uint8_t v_isSharedCheck_3589_; 
v_enabled_3573_ = lean_ctor_get_uint8(v_infoState_3561_, sizeof(void*)*3);
v_assignment_3574_ = lean_ctor_get(v_infoState_3561_, 0);
v_lazyAssignment_3575_ = lean_ctor_get(v_infoState_3561_, 1);
v_trees_3576_ = lean_ctor_get(v_infoState_3561_, 2);
v_isSharedCheck_3589_ = !lean_is_exclusive(v_infoState_3561_);
if (v_isSharedCheck_3589_ == 0)
{
v___x_3578_ = v_infoState_3561_;
v_isShared_3579_ = v_isSharedCheck_3589_;
goto v_resetjp_3577_;
}
else
{
lean_inc(v_trees_3576_);
lean_inc(v_lazyAssignment_3575_);
lean_inc(v_assignment_3574_);
lean_dec(v_infoState_3561_);
v___x_3578_ = lean_box(0);
v_isShared_3579_ = v_isSharedCheck_3589_;
goto v_resetjp_3577_;
}
v_resetjp_3577_:
{
lean_object* v___x_3580_; lean_object* v___x_3582_; 
v___x_3580_ = l_Lean_PersistentArray_append___redArg(v_trees_3576_, v_infoTrees_3558_);
lean_dec_ref(v_infoTrees_3558_);
if (v_isShared_3579_ == 0)
{
lean_ctor_set(v___x_3578_, 2, v___x_3580_);
v___x_3582_ = v___x_3578_;
goto v_reusejp_3581_;
}
else
{
lean_object* v_reuseFailAlloc_3588_; 
v_reuseFailAlloc_3588_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_3588_, 0, v_assignment_3574_);
lean_ctor_set(v_reuseFailAlloc_3588_, 1, v_lazyAssignment_3575_);
lean_ctor_set(v_reuseFailAlloc_3588_, 2, v___x_3580_);
lean_ctor_set_uint8(v_reuseFailAlloc_3588_, sizeof(void*)*3, v_enabled_3573_);
v___x_3582_ = v_reuseFailAlloc_3588_;
goto v_reusejp_3581_;
}
v_reusejp_3581_:
{
lean_object* v___x_3584_; 
if (v_isShared_3572_ == 0)
{
lean_ctor_set(v___x_3571_, 7, v___x_3582_);
v___x_3584_ = v___x_3571_;
goto v_reusejp_3583_;
}
else
{
lean_object* v_reuseFailAlloc_3587_; 
v_reuseFailAlloc_3587_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3587_, 0, v_env_3562_);
lean_ctor_set(v_reuseFailAlloc_3587_, 1, v_nextMacroScope_3563_);
lean_ctor_set(v_reuseFailAlloc_3587_, 2, v_ngen_3564_);
lean_ctor_set(v_reuseFailAlloc_3587_, 3, v_auxDeclNGen_3565_);
lean_ctor_set(v_reuseFailAlloc_3587_, 4, v_traceState_3566_);
lean_ctor_set(v_reuseFailAlloc_3587_, 5, v_cache_3567_);
lean_ctor_set(v_reuseFailAlloc_3587_, 6, v_messages_3568_);
lean_ctor_set(v_reuseFailAlloc_3587_, 7, v___x_3582_);
lean_ctor_set(v_reuseFailAlloc_3587_, 8, v_snapshotTasks_3569_);
v___x_3584_ = v_reuseFailAlloc_3587_;
goto v_reusejp_3583_;
}
v_reusejp_3583_:
{
lean_object* v___x_3585_; lean_object* v___x_3586_; 
v___x_3585_ = lean_st_ref_set(v_a_3556_, v___x_3584_);
v___x_3586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3586_, 0, v_val_3559_);
return v___x_3586_;
}
}
}
}
}
case 1:
{
lean_object* v_ref_3591_; lean_object* v_f_3592_; lean_object* v_lhs_3593_; lean_object* v_rhs_3594_; lean_object* v_fileName_3595_; lean_object* v_fileMap_3596_; lean_object* v_options_3597_; lean_object* v_currRecDepth_3598_; lean_object* v_maxRecDepth_3599_; lean_object* v_ref_3600_; lean_object* v_currNamespace_3601_; lean_object* v_openDecls_3602_; lean_object* v_initHeartbeats_3603_; lean_object* v_maxHeartbeats_3604_; lean_object* v_quotContext_3605_; lean_object* v_currMacroScope_3606_; uint8_t v_diag_3607_; lean_object* v_cancelTk_x3f_3608_; uint8_t v_suppressElabErrors_3609_; lean_object* v_inheritedTraceOptions_3610_; lean_object* v___f_3611_; lean_object* v___x_3612_; lean_object* v___x_3613_; uint8_t v___x_3614_; lean_object* v_ref_3615_; lean_object* v___x_3616_; lean_object* v___x_3617_; 
v_ref_3591_ = lean_ctor_get(v_t_3550_, 0);
lean_inc(v_ref_3591_);
v_f_3592_ = lean_ctor_get(v_t_3550_, 1);
lean_inc_ref(v_f_3592_);
v_lhs_3593_ = lean_ctor_get(v_t_3550_, 2);
lean_inc_ref(v_lhs_3593_);
v_rhs_3594_ = lean_ctor_get(v_t_3550_, 3);
lean_inc_ref(v_rhs_3594_);
lean_dec_ref_known(v_t_3550_, 4);
v_fileName_3595_ = lean_ctor_get(v_a_3555_, 0);
v_fileMap_3596_ = lean_ctor_get(v_a_3555_, 1);
v_options_3597_ = lean_ctor_get(v_a_3555_, 2);
v_currRecDepth_3598_ = lean_ctor_get(v_a_3555_, 3);
v_maxRecDepth_3599_ = lean_ctor_get(v_a_3555_, 4);
v_ref_3600_ = lean_ctor_get(v_a_3555_, 5);
v_currNamespace_3601_ = lean_ctor_get(v_a_3555_, 6);
v_openDecls_3602_ = lean_ctor_get(v_a_3555_, 7);
v_initHeartbeats_3603_ = lean_ctor_get(v_a_3555_, 8);
v_maxHeartbeats_3604_ = lean_ctor_get(v_a_3555_, 9);
v_quotContext_3605_ = lean_ctor_get(v_a_3555_, 10);
v_currMacroScope_3606_ = lean_ctor_get(v_a_3555_, 11);
v_diag_3607_ = lean_ctor_get_uint8(v_a_3555_, sizeof(void*)*14);
v_cancelTk_x3f_3608_ = lean_ctor_get(v_a_3555_, 12);
v_suppressElabErrors_3609_ = lean_ctor_get_uint8(v_a_3555_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3610_ = lean_ctor_get(v_a_3555_, 13);
v___f_3611_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0___boxed), 10, 3);
lean_closure_set(v___f_3611_, 0, v_lhs_3593_);
lean_closure_set(v___f_3611_, 1, v_rhs_3594_);
lean_closure_set(v___f_3611_, 2, v_f_3592_);
v___x_3612_ = lean_box(0);
v___x_3613_ = lean_box(0);
v___x_3614_ = 0;
v_ref_3615_ = l_Lean_replaceRef(v_ref_3591_, v_ref_3600_);
lean_inc_ref(v_inheritedTraceOptions_3610_);
lean_inc(v_cancelTk_x3f_3608_);
lean_inc(v_currMacroScope_3606_);
lean_inc(v_quotContext_3605_);
lean_inc(v_maxHeartbeats_3604_);
lean_inc(v_initHeartbeats_3603_);
lean_inc(v_openDecls_3602_);
lean_inc(v_currNamespace_3601_);
lean_inc(v_maxRecDepth_3599_);
lean_inc(v_currRecDepth_3598_);
lean_inc_ref(v_options_3597_);
lean_inc_ref(v_fileMap_3596_);
lean_inc_ref(v_fileName_3595_);
v___x_3616_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3616_, 0, v_fileName_3595_);
lean_ctor_set(v___x_3616_, 1, v_fileMap_3596_);
lean_ctor_set(v___x_3616_, 2, v_options_3597_);
lean_ctor_set(v___x_3616_, 3, v_currRecDepth_3598_);
lean_ctor_set(v___x_3616_, 4, v_maxRecDepth_3599_);
lean_ctor_set(v___x_3616_, 5, v_ref_3615_);
lean_ctor_set(v___x_3616_, 6, v_currNamespace_3601_);
lean_ctor_set(v___x_3616_, 7, v_openDecls_3602_);
lean_ctor_set(v___x_3616_, 8, v_initHeartbeats_3603_);
lean_ctor_set(v___x_3616_, 9, v_maxHeartbeats_3604_);
lean_ctor_set(v___x_3616_, 10, v_quotContext_3605_);
lean_ctor_set(v___x_3616_, 11, v_currMacroScope_3606_);
lean_ctor_set(v___x_3616_, 12, v_cancelTk_x3f_3608_);
lean_ctor_set(v___x_3616_, 13, v_inheritedTraceOptions_3610_);
lean_ctor_set_uint8(v___x_3616_, sizeof(void*)*14, v_diag_3607_);
lean_ctor_set_uint8(v___x_3616_, sizeof(void*)*14 + 1, v_suppressElabErrors_3609_);
v___x_3617_ = l_Lean_Elab_Term_withTermInfoContext_x27(v___x_3612_, v_ref_3591_, v___f_3611_, v___x_3613_, v___x_3613_, v___x_3614_, v___x_3614_, v_a_3551_, v_a_3552_, v_a_3553_, v_a_3554_, v___x_3616_, v_a_3556_);
lean_dec_ref_known(v___x_3616_, 14);
return v___x_3617_;
}
default: 
{
lean_object* v_macroName_3618_; lean_object* v_stx_3619_; lean_object* v_stx_x27_3620_; lean_object* v_nested_3621_; lean_object* v_fileName_3622_; lean_object* v_fileMap_3623_; lean_object* v_options_3624_; lean_object* v_currRecDepth_3625_; lean_object* v_maxRecDepth_3626_; lean_object* v_ref_3627_; lean_object* v_currNamespace_3628_; lean_object* v_openDecls_3629_; lean_object* v_initHeartbeats_3630_; lean_object* v_maxHeartbeats_3631_; lean_object* v_quotContext_3632_; lean_object* v_currMacroScope_3633_; uint8_t v_diag_3634_; lean_object* v_cancelTk_x3f_3635_; uint8_t v_suppressElabErrors_3636_; lean_object* v_inheritedTraceOptions_3637_; lean_object* v___x_3638_; lean_object* v___x_3639_; lean_object* v___x_3640_; uint8_t v___x_3641_; lean_object* v_ref_3642_; lean_object* v___x_3643_; lean_object* v___x_3644_; 
v_macroName_3618_ = lean_ctor_get(v_t_3550_, 0);
lean_inc(v_macroName_3618_);
v_stx_3619_ = lean_ctor_get(v_t_3550_, 1);
lean_inc_n(v_stx_3619_, 2);
v_stx_x27_3620_ = lean_ctor_get(v_t_3550_, 2);
lean_inc(v_stx_x27_3620_);
v_nested_3621_ = lean_ctor_get(v_t_3550_, 3);
lean_inc_ref(v_nested_3621_);
lean_dec_ref_known(v_t_3550_, 4);
v_fileName_3622_ = lean_ctor_get(v_a_3555_, 0);
v_fileMap_3623_ = lean_ctor_get(v_a_3555_, 1);
v_options_3624_ = lean_ctor_get(v_a_3555_, 2);
v_currRecDepth_3625_ = lean_ctor_get(v_a_3555_, 3);
v_maxRecDepth_3626_ = lean_ctor_get(v_a_3555_, 4);
v_ref_3627_ = lean_ctor_get(v_a_3555_, 5);
v_currNamespace_3628_ = lean_ctor_get(v_a_3555_, 6);
v_openDecls_3629_ = lean_ctor_get(v_a_3555_, 7);
v_initHeartbeats_3630_ = lean_ctor_get(v_a_3555_, 8);
v_maxHeartbeats_3631_ = lean_ctor_get(v_a_3555_, 9);
v_quotContext_3632_ = lean_ctor_get(v_a_3555_, 10);
v_currMacroScope_3633_ = lean_ctor_get(v_a_3555_, 11);
v_diag_3634_ = lean_ctor_get_uint8(v_a_3555_, sizeof(void*)*14);
v_cancelTk_x3f_3635_ = lean_ctor_get(v_a_3555_, 12);
v_suppressElabErrors_3636_ = lean_ctor_get_uint8(v_a_3555_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3637_ = lean_ctor_get(v_a_3555_, 13);
v___x_3638_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___boxed), 8, 1);
lean_closure_set(v___x_3638_, 0, v_nested_3621_);
v___x_3639_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0___boxed), 11, 4);
lean_closure_set(v___x_3639_, 0, lean_box(0));
lean_closure_set(v___x_3639_, 1, v_stx_3619_);
lean_closure_set(v___x_3639_, 2, v_stx_x27_3620_);
lean_closure_set(v___x_3639_, 3, v___x_3638_);
v___x_3640_ = lean_box(0);
v___x_3641_ = 0;
v_ref_3642_ = l_Lean_replaceRef(v_stx_3619_, v_ref_3627_);
lean_inc_ref(v_inheritedTraceOptions_3637_);
lean_inc(v_cancelTk_x3f_3635_);
lean_inc(v_currMacroScope_3633_);
lean_inc(v_quotContext_3632_);
lean_inc(v_maxHeartbeats_3631_);
lean_inc(v_initHeartbeats_3630_);
lean_inc(v_openDecls_3629_);
lean_inc(v_currNamespace_3628_);
lean_inc(v_maxRecDepth_3626_);
lean_inc(v_currRecDepth_3625_);
lean_inc_ref(v_options_3624_);
lean_inc_ref(v_fileMap_3623_);
lean_inc_ref(v_fileName_3622_);
v___x_3643_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3643_, 0, v_fileName_3622_);
lean_ctor_set(v___x_3643_, 1, v_fileMap_3623_);
lean_ctor_set(v___x_3643_, 2, v_options_3624_);
lean_ctor_set(v___x_3643_, 3, v_currRecDepth_3625_);
lean_ctor_set(v___x_3643_, 4, v_maxRecDepth_3626_);
lean_ctor_set(v___x_3643_, 5, v_ref_3642_);
lean_ctor_set(v___x_3643_, 6, v_currNamespace_3628_);
lean_ctor_set(v___x_3643_, 7, v_openDecls_3629_);
lean_ctor_set(v___x_3643_, 8, v_initHeartbeats_3630_);
lean_ctor_set(v___x_3643_, 9, v_maxHeartbeats_3631_);
lean_ctor_set(v___x_3643_, 10, v_quotContext_3632_);
lean_ctor_set(v___x_3643_, 11, v_currMacroScope_3633_);
lean_ctor_set(v___x_3643_, 12, v_cancelTk_x3f_3635_);
lean_ctor_set(v___x_3643_, 13, v_inheritedTraceOptions_3637_);
lean_ctor_set_uint8(v___x_3643_, sizeof(void*)*14, v_diag_3634_);
lean_ctor_set_uint8(v___x_3643_, sizeof(void*)*14 + 1, v_suppressElabErrors_3636_);
v___x_3644_ = l_Lean_Elab_Term_withTermInfoContext_x27(v_macroName_3618_, v_stx_3619_, v___x_3639_, v___x_3640_, v___x_3640_, v___x_3641_, v___x_3641_, v_a_3551_, v_a_3552_, v_a_3553_, v_a_3554_, v___x_3643_, v_a_3556_);
lean_dec_ref_known(v___x_3643_, 14);
return v___x_3644_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore___lam__0(lean_object* v_lhs_3645_, lean_object* v_rhs_3646_, lean_object* v_f_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_, lean_object* v___y_3651_, lean_object* v___y_3652_, lean_object* v___y_3653_){
_start:
{
lean_object* v___x_3655_; 
v___x_3655_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(v_lhs_3645_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_, v___y_3652_, v___y_3653_);
if (lean_obj_tag(v___x_3655_) == 0)
{
lean_object* v_a_3656_; lean_object* v___x_3657_; 
v_a_3656_ = lean_ctor_get(v___x_3655_, 0);
lean_inc(v_a_3656_);
lean_dec_ref_known(v___x_3655_, 1);
v___x_3657_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(v_rhs_3646_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_, v___y_3652_, v___y_3653_);
if (lean_obj_tag(v___x_3657_) == 0)
{
lean_object* v_a_3658_; lean_object* v___x_3659_; 
v_a_3658_ = lean_ctor_get(v___x_3657_, 0);
lean_inc(v_a_3658_);
lean_dec_ref_known(v___x_3657_, 1);
v___x_3659_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_mkBinOp(v_f_3647_, v_a_3656_, v_a_3658_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_, v___y_3652_, v___y_3653_);
return v___x_3659_;
}
else
{
lean_dec(v_a_3656_);
lean_dec_ref(v_f_3647_);
return v___x_3657_;
}
}
else
{
lean_dec_ref(v_f_3647_);
lean_dec_ref(v_rhs_3646_);
return v___x_3655_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0(lean_object* v_00_u03b1_3660_, lean_object* v_stx_3661_, lean_object* v_output_3662_, lean_object* v_x_3663_, lean_object* v___y_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_){
_start:
{
lean_object* v___x_3671_; 
v___x_3671_ = lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___redArg(v_stx_3661_, v_output_3662_, v_x_3663_, v___y_3664_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
return v___x_3671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0___boxed(lean_object* v_00_u03b1_3672_, lean_object* v_stx_3673_, lean_object* v_output_3674_, lean_object* v_x_3675_, lean_object* v___y_3676_, lean_object* v___y_3677_, lean_object* v___y_3678_, lean_object* v___y_3679_, lean_object* v___y_3680_, lean_object* v___y_3681_, lean_object* v___y_3682_){
_start:
{
lean_object* v_res_3683_; 
v_res_3683_ = lp_mathlib_Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0(v_00_u03b1_3672_, v_stx_3673_, v_output_3674_, v_x_3675_, v___y_3676_, v___y_3677_, v___y_3678_, v___y_3679_, v___y_3680_, v___y_3681_);
lean_dec(v___y_3681_);
lean_dec_ref(v___y_3680_);
lean_dec(v___y_3679_);
lean_dec_ref(v___y_3678_);
lean_dec(v___y_3677_);
lean_dec_ref(v___y_3676_);
return v_res_3683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_3684_, lean_object* v_x_3685_, lean_object* v_mkInfoTree_3686_, lean_object* v___y_3687_, lean_object* v___y_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_, lean_object* v___y_3692_){
_start:
{
lean_object* v___x_3694_; 
v___x_3694_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___redArg(v_x_3685_, v_mkInfoTree_3686_, v___y_3687_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_, v___y_3692_);
return v___x_3694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_3695_, lean_object* v_x_3696_, lean_object* v_mkInfoTree_3697_, lean_object* v___y_3698_, lean_object* v___y_3699_, lean_object* v___y_3700_, lean_object* v___y_3701_, lean_object* v___y_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_){
_start:
{
lean_object* v_res_3705_; 
v_res_3705_ = lp_mathlib_Lean_Elab_withInfoTreeContext___at___00Lean_Elab_withMacroExpansionInfo___at___00Lean_Elab_Term_withMacroExpansion___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore_spec__0_spec__0_spec__1(v_00_u03b1_3695_, v_x_3696_, v_mkInfoTree_3697_, v___y_3698_, v___y_3699_, v___y_3700_, v___y_3701_, v___y_3702_, v___y_3703_);
lean_dec(v___y_3703_);
lean_dec_ref(v___y_3702_);
lean_dec(v___y_3701_);
lean_dec_ref(v___y_3700_);
lean_dec(v___y_3699_);
lean_dec_ref(v___y_3698_);
return v_res_3705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(lean_object* v___x_3706_, lean_object* v___y_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_){
_start:
{
lean_object* v_options_3714_; uint8_t v_hasTrace_3715_; 
v_options_3714_ = lean_ctor_get(v___y_3711_, 2);
v_hasTrace_3715_ = lean_ctor_get_uint8(v_options_3714_, sizeof(void*)*1);
if (v_hasTrace_3715_ == 0)
{
lean_object* v___x_3716_; lean_object* v___x_3717_; 
lean_dec(v___x_3706_);
v___x_3716_ = lean_box(v_hasTrace_3715_);
v___x_3717_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3717_, 0, v___x_3716_);
return v___x_3717_;
}
else
{
lean_object* v_inheritedTraceOptions_3718_; lean_object* v___x_3719_; lean_object* v___x_3720_; uint8_t v___x_3721_; lean_object* v___x_3722_; lean_object* v___x_3723_; 
v_inheritedTraceOptions_3718_ = lean_ctor_get(v___y_3711_, 13);
v___x_3719_ = ((lean_object*)(lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__15));
v___x_3720_ = l_Lean_Name_append(v___x_3719_, v___x_3706_);
v___x_3721_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3718_, v_options_3714_, v___x_3720_);
lean_dec(v___x_3720_);
v___x_3722_ = lean_box(v___x_3721_);
v___x_3723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3723_, 0, v___x_3722_);
return v___x_3723_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0___boxed(lean_object* v___x_3724_, lean_object* v___y_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_, lean_object* v___y_3729_, lean_object* v___y_3730_, lean_object* v___y_3731_){
_start:
{
lean_object* v_res_3732_; 
v_res_3732_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3724_, v___y_3725_, v___y_3726_, v___y_3727_, v___y_3728_, v___y_3729_, v___y_3730_);
lean_dec(v___y_3730_);
lean_dec_ref(v___y_3729_);
lean_dec(v___y_3728_);
lean_dec_ref(v___y_3727_);
lean_dec(v___y_3726_);
lean_dec_ref(v___y_3725_);
return v_res_3732_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1(void){
_start:
{
lean_object* v___x_3734_; lean_object* v___x_3735_; 
v___x_3734_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__0));
v___x_3735_ = l_Lean_stringToMessageData(v___x_3734_);
return v___x_3735_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_3737_; lean_object* v___x_3738_; 
v___x_3737_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__2));
v___x_3738_ = l_Lean_stringToMessageData(v___x_3737_);
return v___x_3738_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5(void){
_start:
{
lean_object* v___x_3740_; lean_object* v___x_3741_; 
v___x_3740_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__4));
v___x_3741_ = l_Lean_stringToMessageData(v___x_3740_);
return v___x_3741_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7(void){
_start:
{
lean_object* v___x_3743_; lean_object* v___x_3744_; 
v___x_3743_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__6));
v___x_3744_ = l_Lean_stringToMessageData(v___x_3743_);
return v___x_3744_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9(void){
_start:
{
lean_object* v___x_3746_; lean_object* v___x_3747_; 
v___x_3746_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__8));
v___x_3747_ = l_Lean_stringToMessageData(v___x_3746_);
return v___x_3747_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11(void){
_start:
{
lean_object* v___x_3749_; lean_object* v___x_3750_; 
v___x_3749_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__10));
v___x_3750_ = l_Lean_stringToMessageData(v___x_3749_);
return v___x_3750_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13(void){
_start:
{
lean_object* v___x_3752_; lean_object* v___x_3753_; 
v___x_3752_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__12));
v___x_3753_ = l_Lean_stringToMessageData(v___x_3752_);
return v___x_3753_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15(void){
_start:
{
lean_object* v___x_3755_; lean_object* v___x_3756_; 
v___x_3755_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__14));
v___x_3756_ = l_Lean_stringToMessageData(v___x_3755_);
return v___x_3756_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17(void){
_start:
{
lean_object* v___x_3758_; lean_object* v___x_3759_; 
v___x_3758_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__16));
v___x_3759_ = l_Lean_stringToMessageData(v___x_3758_);
return v___x_3759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1___boxed(lean_object* v_maxS_3760_, lean_object* v_nested_3761_, lean_object* v_macroName_3762_, lean_object* v_stx_3763_, lean_object* v_stx_x27_3764_, lean_object* v___y_3765_, lean_object* v___y_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_){
_start:
{
lean_object* v_res_3772_; 
v_res_3772_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1(v_maxS_3760_, v_nested_3761_, v_macroName_3762_, v_stx_3763_, v_stx_x27_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_, v___y_3769_, v___y_3770_);
lean_dec(v___y_3770_);
lean_dec_ref(v___y_3769_);
lean_dec(v___y_3768_);
lean_dec_ref(v___y_3767_);
lean_dec(v___y_3766_);
lean_dec_ref(v___y_3765_);
return v_res_3772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(lean_object* v_maxS_3773_, lean_object* v_t_3774_, lean_object* v_a_3775_, lean_object* v_a_3776_, lean_object* v_a_3777_, lean_object* v_a_3778_, lean_object* v_a_3779_, lean_object* v_a_3780_){
_start:
{
switch(lean_obj_tag(v_t_3774_))
{
case 0:
{
lean_object* v_ref_3782_; lean_object* v_infoTrees_3783_; lean_object* v_val_3784_; lean_object* v___y_3786_; lean_object* v___y_3787_; lean_object* v___y_3788_; lean_object* v___y_3789_; lean_object* v___y_3790_; lean_object* v_fileName_3791_; lean_object* v_fileMap_3792_; lean_object* v_options_3793_; lean_object* v_currRecDepth_3794_; lean_object* v_maxRecDepth_3795_; lean_object* v_ref_3796_; lean_object* v_currNamespace_3797_; lean_object* v_openDecls_3798_; lean_object* v_initHeartbeats_3799_; lean_object* v_maxHeartbeats_3800_; lean_object* v_quotContext_3801_; lean_object* v_currMacroScope_3802_; uint8_t v_diag_3803_; lean_object* v_cancelTk_x3f_3804_; uint8_t v_suppressElabErrors_3805_; lean_object* v_inheritedTraceOptions_3806_; lean_object* v___y_3807_; lean_object* v___x_3829_; 
v_ref_3782_ = lean_ctor_get(v_t_3774_, 0);
v_infoTrees_3783_ = lean_ctor_get(v_t_3774_, 1);
v_val_3784_ = lean_ctor_get(v_t_3774_, 2);
lean_inc(v_a_3780_);
lean_inc_ref(v_a_3779_);
lean_inc(v_a_3778_);
lean_inc_ref(v_a_3777_);
lean_inc_ref(v_val_3784_);
v___x_3829_ = lean_infer_type(v_val_3784_, v_a_3777_, v_a_3778_, v_a_3779_, v_a_3780_);
if (lean_obj_tag(v___x_3829_) == 0)
{
lean_object* v_a_3830_; lean_object* v___x_3831_; lean_object* v_a_3832_; lean_object* v___y_3834_; lean_object* v___y_3835_; lean_object* v___y_3836_; lean_object* v___y_3837_; lean_object* v___y_3838_; lean_object* v___x_3856_; lean_object* v___y_3858_; lean_object* v___y_3859_; lean_object* v___y_3860_; lean_object* v___y_3861_; lean_object* v___y_3862_; lean_object* v___y_3863_; lean_object* v___y_3864_; lean_object* v___y_3938_; lean_object* v___y_3939_; lean_object* v___y_3940_; lean_object* v___y_3941_; lean_object* v___y_3942_; lean_object* v___y_3943_; lean_object* v___x_4101_; lean_object* v_a_4102_; uint8_t v___x_4103_; 
v_a_3830_ = lean_ctor_get(v___x_3829_, 0);
lean_inc(v_a_3830_);
lean_dec_ref_known(v___x_3829_, 1);
v___x_3831_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze_spec__0___redArg(v_a_3830_, v_a_3778_);
v_a_3832_ = lean_ctor_get(v___x_3831_, 0);
lean_inc(v_a_3832_);
lean_dec_ref(v___x_3831_);
v___x_3856_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
v___x_4101_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3856_, v_a_3775_, v_a_3776_, v_a_3777_, v_a_3778_, v_a_3779_, v_a_3780_);
v_a_4102_ = lean_ctor_get(v___x_4101_, 0);
lean_inc(v_a_4102_);
lean_dec_ref(v___x_4101_);
v___x_4103_ = lean_unbox(v_a_4102_);
lean_dec(v_a_4102_);
if (v___x_4103_ == 0)
{
v___y_3938_ = v_a_3775_;
v___y_3939_ = v_a_3776_;
v___y_3940_ = v_a_3777_;
v___y_3941_ = v_a_3778_;
v___y_3942_ = v_a_3779_;
v___y_3943_ = v_a_3780_;
goto v___jp_3937_;
}
else
{
lean_object* v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; lean_object* v___x_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; lean_object* v___x_4110_; lean_object* v___x_4111_; 
v___x_4104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__17);
lean_inc_ref(v_val_3784_);
v___x_4105_ = l_Lean_MessageData_ofExpr(v_val_3784_);
v___x_4106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4106_, 0, v___x_4104_);
lean_ctor_set(v___x_4106_, 1, v___x_4105_);
v___x_4107_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3);
v___x_4108_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4108_, 0, v___x_4106_);
lean_ctor_set(v___x_4108_, 1, v___x_4107_);
lean_inc(v_a_3832_);
v___x_4109_ = l_Lean_MessageData_ofExpr(v_a_3832_);
v___x_4110_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4110_, 0, v___x_4108_);
lean_ctor_set(v___x_4110_, 1, v___x_4109_);
v___x_4111_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_4110_, v_a_3777_, v_a_3778_, v_a_3779_, v_a_3780_);
if (lean_obj_tag(v___x_4111_) == 0)
{
lean_dec_ref_known(v___x_4111_, 1);
v___y_3938_ = v_a_3775_;
v___y_3939_ = v_a_3776_;
v___y_3940_ = v_a_3777_;
v___y_3941_ = v_a_3778_;
v___y_3942_ = v_a_3779_;
v___y_3943_ = v_a_3780_;
goto v___jp_3937_;
}
else
{
lean_object* v_a_4112_; lean_object* v___x_4114_; uint8_t v_isShared_4115_; uint8_t v_isSharedCheck_4119_; 
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
lean_dec_ref(v_maxS_3773_);
v_a_4112_ = lean_ctor_get(v___x_4111_, 0);
v_isSharedCheck_4119_ = !lean_is_exclusive(v___x_4111_);
if (v_isSharedCheck_4119_ == 0)
{
v___x_4114_ = v___x_4111_;
v_isShared_4115_ = v_isSharedCheck_4119_;
goto v_resetjp_4113_;
}
else
{
lean_inc(v_a_4112_);
lean_dec(v___x_4111_);
v___x_4114_ = lean_box(0);
v_isShared_4115_ = v_isSharedCheck_4119_;
goto v_resetjp_4113_;
}
v_resetjp_4113_:
{
lean_object* v___x_4117_; 
if (v_isShared_4115_ == 0)
{
v___x_4117_ = v___x_4114_;
goto v_reusejp_4116_;
}
else
{
lean_object* v_reuseFailAlloc_4118_; 
v_reuseFailAlloc_4118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4118_, 0, v_a_4112_);
v___x_4117_ = v_reuseFailAlloc_4118_;
goto v_reusejp_4116_;
}
v_reusejp_4116_:
{
return v___x_4117_;
}
}
}
}
v___jp_3833_:
{
lean_object* v___x_3839_; 
v___x_3839_ = l_Lean_Meta_isExprDefEqGuarded(v___y_3834_, v_a_3832_, v___y_3835_, v___y_3836_, v___y_3837_, v___y_3838_);
if (lean_obj_tag(v___x_3839_) == 0)
{
lean_object* v___x_3841_; uint8_t v_isShared_3842_; uint8_t v_isSharedCheck_3846_; 
v_isSharedCheck_3846_ = !lean_is_exclusive(v___x_3839_);
if (v_isSharedCheck_3846_ == 0)
{
lean_object* v_unused_3847_; 
v_unused_3847_ = lean_ctor_get(v___x_3839_, 0);
lean_dec(v_unused_3847_);
v___x_3841_ = v___x_3839_;
v_isShared_3842_ = v_isSharedCheck_3846_;
goto v_resetjp_3840_;
}
else
{
lean_dec(v___x_3839_);
v___x_3841_ = lean_box(0);
v_isShared_3842_ = v_isSharedCheck_3846_;
goto v_resetjp_3840_;
}
v_resetjp_3840_:
{
lean_object* v___x_3844_; 
if (v_isShared_3842_ == 0)
{
lean_ctor_set(v___x_3841_, 0, v_t_3774_);
v___x_3844_ = v___x_3841_;
goto v_reusejp_3843_;
}
else
{
lean_object* v_reuseFailAlloc_3845_; 
v_reuseFailAlloc_3845_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3845_, 0, v_t_3774_);
v___x_3844_ = v_reuseFailAlloc_3845_;
goto v_reusejp_3843_;
}
v_reusejp_3843_:
{
return v___x_3844_;
}
}
}
else
{
lean_object* v_a_3848_; lean_object* v___x_3850_; uint8_t v_isShared_3851_; uint8_t v_isSharedCheck_3855_; 
lean_dec_ref_known(v_t_3774_, 3);
v_a_3848_ = lean_ctor_get(v___x_3839_, 0);
v_isSharedCheck_3855_ = !lean_is_exclusive(v___x_3839_);
if (v_isSharedCheck_3855_ == 0)
{
v___x_3850_ = v___x_3839_;
v_isShared_3851_ = v_isSharedCheck_3855_;
goto v_resetjp_3849_;
}
else
{
lean_inc(v_a_3848_);
lean_dec(v___x_3839_);
v___x_3850_ = lean_box(0);
v_isShared_3851_ = v_isSharedCheck_3855_;
goto v_resetjp_3849_;
}
v_resetjp_3849_:
{
lean_object* v___x_3853_; 
if (v_isShared_3851_ == 0)
{
v___x_3853_ = v___x_3850_;
goto v_reusejp_3852_;
}
else
{
lean_object* v_reuseFailAlloc_3854_; 
v_reuseFailAlloc_3854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3854_, 0, v_a_3848_);
v___x_3853_ = v_reuseFailAlloc_3854_;
goto v_reusejp_3852_;
}
v_reusejp_3852_:
{
return v___x_3853_;
}
}
}
}
v___jp_3857_:
{
lean_object* v___x_3865_; 
lean_inc(v_a_3832_);
lean_inc_ref(v___y_3858_);
v___x_3865_ = l_Lean_Meta_isExprDefEqGuarded(v___y_3858_, v_a_3832_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3865_) == 0)
{
lean_object* v_a_3866_; lean_object* v___x_3868_; uint8_t v_isShared_3869_; uint8_t v_isSharedCheck_3928_; 
v_a_3866_ = lean_ctor_get(v___x_3865_, 0);
v_isSharedCheck_3928_ = !lean_is_exclusive(v___x_3865_);
if (v_isSharedCheck_3928_ == 0)
{
v___x_3868_ = v___x_3865_;
v_isShared_3869_ = v_isSharedCheck_3928_;
goto v_resetjp_3867_;
}
else
{
lean_inc(v_a_3866_);
lean_dec(v___x_3865_);
v___x_3868_ = lean_box(0);
v_isShared_3869_ = v_isSharedCheck_3928_;
goto v_resetjp_3867_;
}
v_resetjp_3867_:
{
uint8_t v___x_3870_; 
v___x_3870_ = lean_unbox(v_a_3866_);
lean_dec(v_a_3866_);
if (v___x_3870_ == 0)
{
lean_object* v_options_3871_; uint8_t v_hasTrace_3872_; 
lean_inc_ref(v_val_3784_);
lean_inc_ref(v_infoTrees_3783_);
lean_inc(v_ref_3782_);
lean_del_object(v___x_3868_);
lean_dec_ref_known(v_t_3774_, 3);
v_options_3871_ = lean_ctor_get(v___y_3863_, 2);
v_hasTrace_3872_ = lean_ctor_get_uint8(v_options_3871_, sizeof(void*)*1);
if (v_hasTrace_3872_ == 0)
{
lean_object* v_fileName_3873_; lean_object* v_fileMap_3874_; lean_object* v_currRecDepth_3875_; lean_object* v_maxRecDepth_3876_; lean_object* v_ref_3877_; lean_object* v_currNamespace_3878_; lean_object* v_openDecls_3879_; lean_object* v_initHeartbeats_3880_; lean_object* v_maxHeartbeats_3881_; lean_object* v_quotContext_3882_; lean_object* v_currMacroScope_3883_; uint8_t v_diag_3884_; lean_object* v_cancelTk_x3f_3885_; uint8_t v_suppressElabErrors_3886_; lean_object* v_inheritedTraceOptions_3887_; 
lean_dec(v_a_3832_);
v_fileName_3873_ = lean_ctor_get(v___y_3863_, 0);
v_fileMap_3874_ = lean_ctor_get(v___y_3863_, 1);
v_currRecDepth_3875_ = lean_ctor_get(v___y_3863_, 3);
v_maxRecDepth_3876_ = lean_ctor_get(v___y_3863_, 4);
v_ref_3877_ = lean_ctor_get(v___y_3863_, 5);
v_currNamespace_3878_ = lean_ctor_get(v___y_3863_, 6);
v_openDecls_3879_ = lean_ctor_get(v___y_3863_, 7);
v_initHeartbeats_3880_ = lean_ctor_get(v___y_3863_, 8);
v_maxHeartbeats_3881_ = lean_ctor_get(v___y_3863_, 9);
v_quotContext_3882_ = lean_ctor_get(v___y_3863_, 10);
v_currMacroScope_3883_ = lean_ctor_get(v___y_3863_, 11);
v_diag_3884_ = lean_ctor_get_uint8(v___y_3863_, sizeof(void*)*14);
v_cancelTk_x3f_3885_ = lean_ctor_get(v___y_3863_, 12);
v_suppressElabErrors_3886_ = lean_ctor_get_uint8(v___y_3863_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3887_ = lean_ctor_get(v___y_3863_, 13);
v___y_3786_ = v___y_3858_;
v___y_3787_ = v___y_3859_;
v___y_3788_ = v___y_3860_;
v___y_3789_ = v___y_3861_;
v___y_3790_ = v___y_3862_;
v_fileName_3791_ = v_fileName_3873_;
v_fileMap_3792_ = v_fileMap_3874_;
v_options_3793_ = v_options_3871_;
v_currRecDepth_3794_ = v_currRecDepth_3875_;
v_maxRecDepth_3795_ = v_maxRecDepth_3876_;
v_ref_3796_ = v_ref_3877_;
v_currNamespace_3797_ = v_currNamespace_3878_;
v_openDecls_3798_ = v_openDecls_3879_;
v_initHeartbeats_3799_ = v_initHeartbeats_3880_;
v_maxHeartbeats_3800_ = v_maxHeartbeats_3881_;
v_quotContext_3801_ = v_quotContext_3882_;
v_currMacroScope_3802_ = v_currMacroScope_3883_;
v_diag_3803_ = v_diag_3884_;
v_cancelTk_x3f_3804_ = v_cancelTk_x3f_3885_;
v_suppressElabErrors_3805_ = v_suppressElabErrors_3886_;
v_inheritedTraceOptions_3806_ = v_inheritedTraceOptions_3887_;
v___y_3807_ = v___y_3864_;
goto v___jp_3785_;
}
else
{
lean_object* v_fileName_3888_; lean_object* v_fileMap_3889_; lean_object* v_currRecDepth_3890_; lean_object* v_maxRecDepth_3891_; lean_object* v_ref_3892_; lean_object* v_currNamespace_3893_; lean_object* v_openDecls_3894_; lean_object* v_initHeartbeats_3895_; lean_object* v_maxHeartbeats_3896_; lean_object* v_quotContext_3897_; lean_object* v_currMacroScope_3898_; uint8_t v_diag_3899_; lean_object* v_cancelTk_x3f_3900_; uint8_t v_suppressElabErrors_3901_; lean_object* v_inheritedTraceOptions_3902_; lean_object* v___x_3903_; uint8_t v___x_3904_; 
v_fileName_3888_ = lean_ctor_get(v___y_3863_, 0);
v_fileMap_3889_ = lean_ctor_get(v___y_3863_, 1);
v_currRecDepth_3890_ = lean_ctor_get(v___y_3863_, 3);
v_maxRecDepth_3891_ = lean_ctor_get(v___y_3863_, 4);
v_ref_3892_ = lean_ctor_get(v___y_3863_, 5);
v_currNamespace_3893_ = lean_ctor_get(v___y_3863_, 6);
v_openDecls_3894_ = lean_ctor_get(v___y_3863_, 7);
v_initHeartbeats_3895_ = lean_ctor_get(v___y_3863_, 8);
v_maxHeartbeats_3896_ = lean_ctor_get(v___y_3863_, 9);
v_quotContext_3897_ = lean_ctor_get(v___y_3863_, 10);
v_currMacroScope_3898_ = lean_ctor_get(v___y_3863_, 11);
v_diag_3899_ = lean_ctor_get_uint8(v___y_3863_, sizeof(void*)*14);
v_cancelTk_x3f_3900_ = lean_ctor_get(v___y_3863_, 12);
v_suppressElabErrors_3901_ = lean_ctor_get_uint8(v___y_3863_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3902_ = lean_ctor_get(v___y_3863_, 13);
v___x_3903_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2);
v___x_3904_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3902_, v_options_3871_, v___x_3903_);
if (v___x_3904_ == 0)
{
lean_dec(v_a_3832_);
v___y_3786_ = v___y_3858_;
v___y_3787_ = v___y_3859_;
v___y_3788_ = v___y_3860_;
v___y_3789_ = v___y_3861_;
v___y_3790_ = v___y_3862_;
v_fileName_3791_ = v_fileName_3888_;
v_fileMap_3792_ = v_fileMap_3889_;
v_options_3793_ = v_options_3871_;
v_currRecDepth_3794_ = v_currRecDepth_3890_;
v_maxRecDepth_3795_ = v_maxRecDepth_3891_;
v_ref_3796_ = v_ref_3892_;
v_currNamespace_3797_ = v_currNamespace_3893_;
v_openDecls_3798_ = v_openDecls_3894_;
v_initHeartbeats_3799_ = v_initHeartbeats_3895_;
v_maxHeartbeats_3800_ = v_maxHeartbeats_3896_;
v_quotContext_3801_ = v_quotContext_3897_;
v_currMacroScope_3802_ = v_currMacroScope_3898_;
v_diag_3803_ = v_diag_3899_;
v_cancelTk_x3f_3804_ = v_cancelTk_x3f_3900_;
v_suppressElabErrors_3805_ = v_suppressElabErrors_3901_;
v_inheritedTraceOptions_3806_ = v_inheritedTraceOptions_3902_;
v___y_3807_ = v___y_3864_;
goto v___jp_3785_;
}
else
{
lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; 
v___x_3905_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__1);
lean_inc_ref(v_val_3784_);
v___x_3906_ = l_Lean_MessageData_ofExpr(v_val_3784_);
v___x_3907_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3907_, 0, v___x_3905_);
lean_ctor_set(v___x_3907_, 1, v___x_3906_);
v___x_3908_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__3);
v___x_3909_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3909_, 0, v___x_3907_);
lean_ctor_set(v___x_3909_, 1, v___x_3908_);
v___x_3910_ = l_Lean_MessageData_ofExpr(v_a_3832_);
v___x_3911_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3911_, 0, v___x_3909_);
lean_ctor_set(v___x_3911_, 1, v___x_3910_);
v___x_3912_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__5);
v___x_3913_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3913_, 0, v___x_3911_);
lean_ctor_set(v___x_3913_, 1, v___x_3912_);
lean_inc_ref(v___y_3858_);
v___x_3914_ = l_Lean_MessageData_ofExpr(v___y_3858_);
v___x_3915_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3915_, 0, v___x_3913_);
lean_ctor_set(v___x_3915_, 1, v___x_3914_);
v___x_3916_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_3915_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_);
if (lean_obj_tag(v___x_3916_) == 0)
{
lean_dec_ref_known(v___x_3916_, 1);
v___y_3786_ = v___y_3858_;
v___y_3787_ = v___y_3859_;
v___y_3788_ = v___y_3860_;
v___y_3789_ = v___y_3861_;
v___y_3790_ = v___y_3862_;
v_fileName_3791_ = v_fileName_3888_;
v_fileMap_3792_ = v_fileMap_3889_;
v_options_3793_ = v_options_3871_;
v_currRecDepth_3794_ = v_currRecDepth_3890_;
v_maxRecDepth_3795_ = v_maxRecDepth_3891_;
v_ref_3796_ = v_ref_3892_;
v_currNamespace_3797_ = v_currNamespace_3893_;
v_openDecls_3798_ = v_openDecls_3894_;
v_initHeartbeats_3799_ = v_initHeartbeats_3895_;
v_maxHeartbeats_3800_ = v_maxHeartbeats_3896_;
v_quotContext_3801_ = v_quotContext_3897_;
v_currMacroScope_3802_ = v_currMacroScope_3898_;
v_diag_3803_ = v_diag_3899_;
v_cancelTk_x3f_3804_ = v_cancelTk_x3f_3900_;
v_suppressElabErrors_3805_ = v_suppressElabErrors_3901_;
v_inheritedTraceOptions_3806_ = v_inheritedTraceOptions_3902_;
v___y_3807_ = v___y_3864_;
goto v___jp_3785_;
}
else
{
lean_object* v_a_3917_; lean_object* v___x_3919_; uint8_t v_isShared_3920_; uint8_t v_isSharedCheck_3924_; 
lean_dec_ref(v___y_3858_);
lean_dec_ref(v_val_3784_);
lean_dec_ref(v_infoTrees_3783_);
lean_dec(v_ref_3782_);
v_a_3917_ = lean_ctor_get(v___x_3916_, 0);
v_isSharedCheck_3924_ = !lean_is_exclusive(v___x_3916_);
if (v_isSharedCheck_3924_ == 0)
{
v___x_3919_ = v___x_3916_;
v_isShared_3920_ = v_isSharedCheck_3924_;
goto v_resetjp_3918_;
}
else
{
lean_inc(v_a_3917_);
lean_dec(v___x_3916_);
v___x_3919_ = lean_box(0);
v_isShared_3920_ = v_isSharedCheck_3924_;
goto v_resetjp_3918_;
}
v_resetjp_3918_:
{
lean_object* v___x_3922_; 
if (v_isShared_3920_ == 0)
{
v___x_3922_ = v___x_3919_;
goto v_reusejp_3921_;
}
else
{
lean_object* v_reuseFailAlloc_3923_; 
v_reuseFailAlloc_3923_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3923_, 0, v_a_3917_);
v___x_3922_ = v_reuseFailAlloc_3923_;
goto v_reusejp_3921_;
}
v_reusejp_3921_:
{
return v___x_3922_;
}
}
}
}
}
}
else
{
lean_object* v___x_3926_; 
lean_dec_ref(v___y_3858_);
lean_dec(v_a_3832_);
if (v_isShared_3869_ == 0)
{
lean_ctor_set(v___x_3868_, 0, v_t_3774_);
v___x_3926_ = v___x_3868_;
goto v_reusejp_3925_;
}
else
{
lean_object* v_reuseFailAlloc_3927_; 
v_reuseFailAlloc_3927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3927_, 0, v_t_3774_);
v___x_3926_ = v_reuseFailAlloc_3927_;
goto v_reusejp_3925_;
}
v_reusejp_3925_:
{
return v___x_3926_;
}
}
}
}
else
{
lean_object* v_a_3929_; lean_object* v___x_3931_; uint8_t v_isShared_3932_; uint8_t v_isSharedCheck_3936_; 
lean_dec_ref(v___y_3858_);
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
v_a_3929_ = lean_ctor_get(v___x_3865_, 0);
v_isSharedCheck_3936_ = !lean_is_exclusive(v___x_3865_);
if (v_isSharedCheck_3936_ == 0)
{
v___x_3931_ = v___x_3865_;
v_isShared_3932_ = v_isSharedCheck_3936_;
goto v_resetjp_3930_;
}
else
{
lean_inc(v_a_3929_);
lean_dec(v___x_3865_);
v___x_3931_ = lean_box(0);
v_isShared_3932_ = v_isSharedCheck_3936_;
goto v_resetjp_3930_;
}
v_resetjp_3930_:
{
lean_object* v___x_3934_; 
if (v_isShared_3932_ == 0)
{
v___x_3934_ = v___x_3931_;
goto v_reusejp_3933_;
}
else
{
lean_object* v_reuseFailAlloc_3935_; 
v_reuseFailAlloc_3935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3935_, 0, v_a_3929_);
v___x_3934_ = v_reuseFailAlloc_3935_;
goto v_reusejp_3933_;
}
v_reusejp_3933_:
{
return v___x_3934_;
}
}
}
}
v___jp_3937_:
{
lean_object* v___x_3944_; 
lean_inc(v_a_3832_);
v___x_3944_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_extractS(v_a_3832_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_3944_) == 0)
{
lean_object* v_a_3945_; 
v_a_3945_ = lean_ctor_get(v___x_3944_, 0);
lean_inc(v_a_3945_);
lean_dec_ref_known(v___x_3944_, 1);
if (lean_obj_tag(v_a_3945_) == 1)
{
lean_object* v_val_3946_; lean_object* v_snd_3947_; lean_object* v___x_3949_; uint8_t v_isShared_3950_; uint8_t v_isSharedCheck_4020_; 
v_val_3946_ = lean_ctor_get(v_a_3945_, 0);
lean_inc(v_val_3946_);
lean_dec_ref_known(v_a_3945_, 1);
v_snd_3947_ = lean_ctor_get(v_val_3946_, 1);
v_isSharedCheck_4020_ = !lean_is_exclusive(v_val_3946_);
if (v_isSharedCheck_4020_ == 0)
{
lean_object* v_unused_4021_; 
v_unused_4021_ = lean_ctor_get(v_val_3946_, 0);
lean_dec(v_unused_4021_);
v___x_3949_ = v_val_3946_;
v_isShared_3950_ = v_isSharedCheck_4020_;
goto v_resetjp_3948_;
}
else
{
lean_inc(v_snd_3947_);
lean_dec(v_val_3946_);
v___x_3949_ = lean_box(0);
v_isShared_3950_ = v_isSharedCheck_4020_;
goto v_resetjp_3948_;
}
v_resetjp_3948_:
{
lean_object* v___x_3951_; 
lean_inc(v_snd_3947_);
lean_inc_ref(v_maxS_3773_);
v___x_3951_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_maxS_3773_, v_snd_3947_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_3951_) == 0)
{
lean_object* v_a_3952_; 
v_a_3952_ = lean_ctor_get(v___x_3951_, 0);
lean_inc(v_a_3952_);
lean_dec_ref_known(v___x_3951_, 1);
if (lean_obj_tag(v_a_3952_) == 1)
{
lean_object* v_val_3953_; lean_object* v___x_3954_; lean_object* v_a_3955_; uint8_t v___x_3956_; 
lean_dec(v_snd_3947_);
lean_dec_ref(v_maxS_3773_);
v_val_3953_ = lean_ctor_get(v_a_3952_, 0);
lean_inc(v_val_3953_);
lean_dec_ref_known(v_a_3952_, 1);
v___x_3954_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3856_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
v_a_3955_ = lean_ctor_get(v___x_3954_, 0);
lean_inc(v_a_3955_);
lean_dec_ref(v___x_3954_);
v___x_3956_ = lean_unbox(v_a_3955_);
lean_dec(v_a_3955_);
if (v___x_3956_ == 0)
{
lean_del_object(v___x_3949_);
v___y_3858_ = v_val_3953_;
v___y_3859_ = v___y_3938_;
v___y_3860_ = v___y_3939_;
v___y_3861_ = v___y_3940_;
v___y_3862_ = v___y_3941_;
v___y_3863_ = v___y_3942_;
v___y_3864_ = v___y_3943_;
goto v___jp_3857_;
}
else
{
lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3960_; 
lean_inc(v_a_3832_);
v___x_3957_ = l_Lean_MessageData_ofExpr(v_a_3832_);
v___x_3958_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7);
if (v_isShared_3950_ == 0)
{
lean_ctor_set_tag(v___x_3949_, 7);
lean_ctor_set(v___x_3949_, 1, v___x_3958_);
lean_ctor_set(v___x_3949_, 0, v___x_3957_);
v___x_3960_ = v___x_3949_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3972_; 
v_reuseFailAlloc_3972_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3972_, 0, v___x_3957_);
lean_ctor_set(v_reuseFailAlloc_3972_, 1, v___x_3958_);
v___x_3960_ = v_reuseFailAlloc_3972_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; 
lean_inc(v_val_3953_);
v___x_3961_ = l_Lean_MessageData_ofExpr(v_val_3953_);
v___x_3962_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3962_, 0, v___x_3960_);
lean_ctor_set(v___x_3962_, 1, v___x_3961_);
v___x_3963_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_3962_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_3963_) == 0)
{
lean_dec_ref_known(v___x_3963_, 1);
v___y_3858_ = v_val_3953_;
v___y_3859_ = v___y_3938_;
v___y_3860_ = v___y_3939_;
v___y_3861_ = v___y_3940_;
v___y_3862_ = v___y_3941_;
v___y_3863_ = v___y_3942_;
v___y_3864_ = v___y_3943_;
goto v___jp_3857_;
}
else
{
lean_object* v_a_3964_; lean_object* v___x_3966_; uint8_t v_isShared_3967_; uint8_t v_isSharedCheck_3971_; 
lean_dec(v_val_3953_);
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
v_a_3964_ = lean_ctor_get(v___x_3963_, 0);
v_isSharedCheck_3971_ = !lean_is_exclusive(v___x_3963_);
if (v_isSharedCheck_3971_ == 0)
{
v___x_3966_ = v___x_3963_;
v_isShared_3967_ = v_isSharedCheck_3971_;
goto v_resetjp_3965_;
}
else
{
lean_inc(v_a_3964_);
lean_dec(v___x_3963_);
v___x_3966_ = lean_box(0);
v_isShared_3967_ = v_isSharedCheck_3971_;
goto v_resetjp_3965_;
}
v_resetjp_3965_:
{
lean_object* v___x_3969_; 
if (v_isShared_3967_ == 0)
{
v___x_3969_ = v___x_3966_;
goto v_reusejp_3968_;
}
else
{
lean_object* v_reuseFailAlloc_3970_; 
v_reuseFailAlloc_3970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3970_, 0, v_a_3964_);
v___x_3969_ = v_reuseFailAlloc_3970_;
goto v_reusejp_3968_;
}
v_reusejp_3968_:
{
return v___x_3969_;
}
}
}
}
}
}
else
{
lean_object* v___x_3973_; lean_object* v_a_3974_; lean_object* v___x_3976_; uint8_t v_isShared_3977_; uint8_t v_isSharedCheck_4011_; 
lean_dec(v_a_3952_);
lean_dec(v_a_3832_);
v___x_3973_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3856_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
v_a_3974_ = lean_ctor_get(v___x_3973_, 0);
v_isSharedCheck_4011_ = !lean_is_exclusive(v___x_3973_);
if (v_isSharedCheck_4011_ == 0)
{
v___x_3976_ = v___x_3973_;
v_isShared_3977_ = v_isSharedCheck_4011_;
goto v_resetjp_3975_;
}
else
{
lean_inc(v_a_3974_);
lean_dec(v___x_3973_);
v___x_3976_ = lean_box(0);
v_isShared_3977_ = v_isSharedCheck_4011_;
goto v_resetjp_3975_;
}
v_resetjp_3975_:
{
uint8_t v___x_3978_; 
v___x_3978_ = lean_unbox(v_a_3974_);
lean_dec(v_a_3974_);
if (v___x_3978_ == 0)
{
lean_object* v___x_3980_; 
lean_del_object(v___x_3949_);
lean_dec(v_snd_3947_);
lean_dec_ref(v_maxS_3773_);
if (v_isShared_3977_ == 0)
{
lean_ctor_set(v___x_3976_, 0, v_t_3774_);
v___x_3980_ = v___x_3976_;
goto v_reusejp_3979_;
}
else
{
lean_object* v_reuseFailAlloc_3981_; 
v_reuseFailAlloc_3981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3981_, 0, v_t_3774_);
v___x_3980_ = v_reuseFailAlloc_3981_;
goto v_reusejp_3979_;
}
v_reusejp_3979_:
{
return v___x_3980_;
}
}
else
{
lean_object* v___x_3982_; lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v___x_3986_; 
lean_del_object(v___x_3976_);
v___x_3982_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__9);
v___x_3983_ = lp_mathlib_FBinopElab_instToExprSRec_toExpr(v_maxS_3773_);
v___x_3984_ = l_Lean_MessageData_ofExpr(v___x_3983_);
if (v_isShared_3950_ == 0)
{
lean_ctor_set_tag(v___x_3949_, 7);
lean_ctor_set(v___x_3949_, 1, v___x_3984_);
lean_ctor_set(v___x_3949_, 0, v___x_3982_);
v___x_3986_ = v___x_3949_;
goto v_reusejp_3985_;
}
else
{
lean_object* v_reuseFailAlloc_4010_; 
v_reuseFailAlloc_4010_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4010_, 0, v___x_3982_);
lean_ctor_set(v_reuseFailAlloc_4010_, 1, v___x_3984_);
v___x_3986_ = v_reuseFailAlloc_4010_;
goto v_reusejp_3985_;
}
v_reusejp_3985_:
{
lean_object* v___x_3987_; lean_object* v___x_3988_; lean_object* v___x_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; 
v___x_3987_ = lean_obj_once(&lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20, &lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20_once, _init_lp_mathlib___private_Lean_ExtraModUses_0__Lean_recordExtraModUseCore___at___00Lean_recordExtraModUseFromDecl___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__1_spec__5___closed__20);
v___x_3988_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3988_, 0, v___x_3986_);
lean_ctor_set(v___x_3988_, 1, v___x_3987_);
v___x_3989_ = l_Lean_MessageData_ofExpr(v_snd_3947_);
v___x_3990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3990_, 0, v___x_3988_);
lean_ctor_set(v___x_3990_, 1, v___x_3989_);
v___x_3991_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__11);
v___x_3992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3992_, 0, v___x_3990_);
lean_ctor_set(v___x_3992_, 1, v___x_3991_);
v___x_3993_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_3992_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_3993_) == 0)
{
lean_object* v___x_3995_; uint8_t v_isShared_3996_; uint8_t v_isSharedCheck_4000_; 
v_isSharedCheck_4000_ = !lean_is_exclusive(v___x_3993_);
if (v_isSharedCheck_4000_ == 0)
{
lean_object* v_unused_4001_; 
v_unused_4001_ = lean_ctor_get(v___x_3993_, 0);
lean_dec(v_unused_4001_);
v___x_3995_ = v___x_3993_;
v_isShared_3996_ = v_isSharedCheck_4000_;
goto v_resetjp_3994_;
}
else
{
lean_dec(v___x_3993_);
v___x_3995_ = lean_box(0);
v_isShared_3996_ = v_isSharedCheck_4000_;
goto v_resetjp_3994_;
}
v_resetjp_3994_:
{
lean_object* v___x_3998_; 
if (v_isShared_3996_ == 0)
{
lean_ctor_set(v___x_3995_, 0, v_t_3774_);
v___x_3998_ = v___x_3995_;
goto v_reusejp_3997_;
}
else
{
lean_object* v_reuseFailAlloc_3999_; 
v_reuseFailAlloc_3999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3999_, 0, v_t_3774_);
v___x_3998_ = v_reuseFailAlloc_3999_;
goto v_reusejp_3997_;
}
v_reusejp_3997_:
{
return v___x_3998_;
}
}
}
else
{
lean_object* v_a_4002_; lean_object* v___x_4004_; uint8_t v_isShared_4005_; uint8_t v_isSharedCheck_4009_; 
lean_dec_ref_known(v_t_3774_, 3);
v_a_4002_ = lean_ctor_get(v___x_3993_, 0);
v_isSharedCheck_4009_ = !lean_is_exclusive(v___x_3993_);
if (v_isSharedCheck_4009_ == 0)
{
v___x_4004_ = v___x_3993_;
v_isShared_4005_ = v_isSharedCheck_4009_;
goto v_resetjp_4003_;
}
else
{
lean_inc(v_a_4002_);
lean_dec(v___x_3993_);
v___x_4004_ = lean_box(0);
v_isShared_4005_ = v_isSharedCheck_4009_;
goto v_resetjp_4003_;
}
v_resetjp_4003_:
{
lean_object* v___x_4007_; 
if (v_isShared_4005_ == 0)
{
v___x_4007_ = v___x_4004_;
goto v_reusejp_4006_;
}
else
{
lean_object* v_reuseFailAlloc_4008_; 
v_reuseFailAlloc_4008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4008_, 0, v_a_4002_);
v___x_4007_ = v_reuseFailAlloc_4008_;
goto v_reusejp_4006_;
}
v_reusejp_4006_:
{
return v___x_4007_;
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
lean_object* v_a_4012_; lean_object* v___x_4014_; uint8_t v_isShared_4015_; uint8_t v_isSharedCheck_4019_; 
lean_del_object(v___x_3949_);
lean_dec(v_snd_3947_);
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
lean_dec_ref(v_maxS_3773_);
v_a_4012_ = lean_ctor_get(v___x_3951_, 0);
v_isSharedCheck_4019_ = !lean_is_exclusive(v___x_3951_);
if (v_isSharedCheck_4019_ == 0)
{
v___x_4014_ = v___x_3951_;
v_isShared_4015_ = v_isSharedCheck_4019_;
goto v_resetjp_4013_;
}
else
{
lean_inc(v_a_4012_);
lean_dec(v___x_3951_);
v___x_4014_ = lean_box(0);
v_isShared_4015_ = v_isSharedCheck_4019_;
goto v_resetjp_4013_;
}
v_resetjp_4013_:
{
lean_object* v___x_4017_; 
if (v_isShared_4015_ == 0)
{
v___x_4017_ = v___x_4014_;
goto v_reusejp_4016_;
}
else
{
lean_object* v_reuseFailAlloc_4018_; 
v_reuseFailAlloc_4018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4018_, 0, v_a_4012_);
v___x_4017_ = v_reuseFailAlloc_4018_;
goto v_reusejp_4016_;
}
v_reusejp_4016_:
{
return v___x_4017_;
}
}
}
}
}
else
{
lean_object* v___x_4022_; uint8_t v___x_4023_; lean_object* v___x_4024_; lean_object* v___x_4025_; 
lean_dec(v_a_3945_);
v___x_4022_ = lean_box(0);
v___x_4023_ = 0;
v___x_4024_ = lean_box(0);
v___x_4025_ = l_Lean_Meta_mkFreshExprMVar(v___x_4022_, v___x_4023_, v___x_4024_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_4025_) == 0)
{
lean_object* v_a_4026_; lean_object* v___x_4027_; 
v_a_4026_ = lean_ctor_get(v___x_4025_, 0);
lean_inc(v_a_4026_);
lean_dec_ref_known(v___x_4025_, 1);
v___x_4027_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyS(v_maxS_3773_, v_a_4026_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_4027_) == 0)
{
lean_object* v_a_4028_; 
v_a_4028_ = lean_ctor_get(v___x_4027_, 0);
lean_inc(v_a_4028_);
lean_dec_ref_known(v___x_4027_, 1);
if (lean_obj_tag(v_a_4028_) == 1)
{
lean_object* v_val_4029_; lean_object* v___x_4030_; lean_object* v_a_4031_; uint8_t v___x_4032_; 
v_val_4029_ = lean_ctor_get(v_a_4028_, 0);
lean_inc(v_val_4029_);
lean_dec_ref_known(v_a_4028_, 1);
v___x_4030_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3856_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
v_a_4031_ = lean_ctor_get(v___x_4030_, 0);
lean_inc(v_a_4031_);
lean_dec_ref(v___x_4030_);
v___x_4032_ = lean_unbox(v_a_4031_);
lean_dec(v_a_4031_);
if (v___x_4032_ == 0)
{
v___y_3834_ = v_val_4029_;
v___y_3835_ = v___y_3940_;
v___y_3836_ = v___y_3941_;
v___y_3837_ = v___y_3942_;
v___y_3838_ = v___y_3943_;
goto v___jp_3833_;
}
else
{
lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; lean_object* v___x_4040_; 
v___x_4033_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__13);
lean_inc(v_val_4029_);
v___x_4034_ = l_Lean_MessageData_ofExpr(v_val_4029_);
v___x_4035_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4035_, 0, v___x_4033_);
lean_ctor_set(v___x_4035_, 1, v___x_4034_);
v___x_4036_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__7);
v___x_4037_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4037_, 0, v___x_4035_);
lean_ctor_set(v___x_4037_, 1, v___x_4036_);
lean_inc(v_a_3832_);
v___x_4038_ = l_Lean_MessageData_ofExpr(v_a_3832_);
v___x_4039_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4039_, 0, v___x_4037_);
lean_ctor_set(v___x_4039_, 1, v___x_4038_);
v___x_4040_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_4039_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_4040_) == 0)
{
lean_dec_ref_known(v___x_4040_, 1);
v___y_3834_ = v_val_4029_;
v___y_3835_ = v___y_3940_;
v___y_3836_ = v___y_3941_;
v___y_3837_ = v___y_3942_;
v___y_3838_ = v___y_3943_;
goto v___jp_3833_;
}
else
{
lean_object* v_a_4041_; lean_object* v___x_4043_; uint8_t v_isShared_4044_; uint8_t v_isSharedCheck_4048_; 
lean_dec(v_val_4029_);
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
v_a_4041_ = lean_ctor_get(v___x_4040_, 0);
v_isSharedCheck_4048_ = !lean_is_exclusive(v___x_4040_);
if (v_isSharedCheck_4048_ == 0)
{
v___x_4043_ = v___x_4040_;
v_isShared_4044_ = v_isSharedCheck_4048_;
goto v_resetjp_4042_;
}
else
{
lean_inc(v_a_4041_);
lean_dec(v___x_4040_);
v___x_4043_ = lean_box(0);
v_isShared_4044_ = v_isSharedCheck_4048_;
goto v_resetjp_4042_;
}
v_resetjp_4042_:
{
lean_object* v___x_4046_; 
if (v_isShared_4044_ == 0)
{
v___x_4046_ = v___x_4043_;
goto v_reusejp_4045_;
}
else
{
lean_object* v_reuseFailAlloc_4047_; 
v_reuseFailAlloc_4047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4047_, 0, v_a_4041_);
v___x_4046_ = v_reuseFailAlloc_4047_;
goto v_reusejp_4045_;
}
v_reusejp_4045_:
{
return v___x_4046_;
}
}
}
}
}
else
{
lean_object* v___x_4049_; lean_object* v_a_4050_; lean_object* v___x_4052_; uint8_t v_isShared_4053_; uint8_t v_isSharedCheck_4076_; 
lean_dec(v_a_4028_);
lean_dec(v_a_3832_);
v___x_4049_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__0(v___x_3856_, v___y_3938_, v___y_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
v_a_4050_ = lean_ctor_get(v___x_4049_, 0);
v_isSharedCheck_4076_ = !lean_is_exclusive(v___x_4049_);
if (v_isSharedCheck_4076_ == 0)
{
v___x_4052_ = v___x_4049_;
v_isShared_4053_ = v_isSharedCheck_4076_;
goto v_resetjp_4051_;
}
else
{
lean_inc(v_a_4050_);
lean_dec(v___x_4049_);
v___x_4052_ = lean_box(0);
v_isShared_4053_ = v_isSharedCheck_4076_;
goto v_resetjp_4051_;
}
v_resetjp_4051_:
{
uint8_t v___x_4054_; 
v___x_4054_ = lean_unbox(v_a_4050_);
lean_dec(v_a_4050_);
if (v___x_4054_ == 0)
{
lean_object* v___x_4056_; 
if (v_isShared_4053_ == 0)
{
lean_ctor_set(v___x_4052_, 0, v_t_3774_);
v___x_4056_ = v___x_4052_;
goto v_reusejp_4055_;
}
else
{
lean_object* v_reuseFailAlloc_4057_; 
v_reuseFailAlloc_4057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4057_, 0, v_t_3774_);
v___x_4056_ = v_reuseFailAlloc_4057_;
goto v_reusejp_4055_;
}
v_reusejp_4055_:
{
return v___x_4056_;
}
}
else
{
lean_object* v___x_4058_; lean_object* v___x_4059_; 
lean_del_object(v___x_4052_);
v___x_4058_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___closed__15);
v___x_4059_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_3856_, v___x_4058_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_4059_) == 0)
{
lean_object* v___x_4061_; uint8_t v_isShared_4062_; uint8_t v_isSharedCheck_4066_; 
v_isSharedCheck_4066_ = !lean_is_exclusive(v___x_4059_);
if (v_isSharedCheck_4066_ == 0)
{
lean_object* v_unused_4067_; 
v_unused_4067_ = lean_ctor_get(v___x_4059_, 0);
lean_dec(v_unused_4067_);
v___x_4061_ = v___x_4059_;
v_isShared_4062_ = v_isSharedCheck_4066_;
goto v_resetjp_4060_;
}
else
{
lean_dec(v___x_4059_);
v___x_4061_ = lean_box(0);
v_isShared_4062_ = v_isSharedCheck_4066_;
goto v_resetjp_4060_;
}
v_resetjp_4060_:
{
lean_object* v___x_4064_; 
if (v_isShared_4062_ == 0)
{
lean_ctor_set(v___x_4061_, 0, v_t_3774_);
v___x_4064_ = v___x_4061_;
goto v_reusejp_4063_;
}
else
{
lean_object* v_reuseFailAlloc_4065_; 
v_reuseFailAlloc_4065_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4065_, 0, v_t_3774_);
v___x_4064_ = v_reuseFailAlloc_4065_;
goto v_reusejp_4063_;
}
v_reusejp_4063_:
{
return v___x_4064_;
}
}
}
else
{
lean_object* v_a_4068_; lean_object* v___x_4070_; uint8_t v_isShared_4071_; uint8_t v_isSharedCheck_4075_; 
lean_dec_ref_known(v_t_3774_, 3);
v_a_4068_ = lean_ctor_get(v___x_4059_, 0);
v_isSharedCheck_4075_ = !lean_is_exclusive(v___x_4059_);
if (v_isSharedCheck_4075_ == 0)
{
v___x_4070_ = v___x_4059_;
v_isShared_4071_ = v_isSharedCheck_4075_;
goto v_resetjp_4069_;
}
else
{
lean_inc(v_a_4068_);
lean_dec(v___x_4059_);
v___x_4070_ = lean_box(0);
v_isShared_4071_ = v_isSharedCheck_4075_;
goto v_resetjp_4069_;
}
v_resetjp_4069_:
{
lean_object* v___x_4073_; 
if (v_isShared_4071_ == 0)
{
v___x_4073_ = v___x_4070_;
goto v_reusejp_4072_;
}
else
{
lean_object* v_reuseFailAlloc_4074_; 
v_reuseFailAlloc_4074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4074_, 0, v_a_4068_);
v___x_4073_ = v_reuseFailAlloc_4074_;
goto v_reusejp_4072_;
}
v_reusejp_4072_:
{
return v___x_4073_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4077_; lean_object* v___x_4079_; uint8_t v_isShared_4080_; uint8_t v_isSharedCheck_4084_; 
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
v_a_4077_ = lean_ctor_get(v___x_4027_, 0);
v_isSharedCheck_4084_ = !lean_is_exclusive(v___x_4027_);
if (v_isSharedCheck_4084_ == 0)
{
v___x_4079_ = v___x_4027_;
v_isShared_4080_ = v_isSharedCheck_4084_;
goto v_resetjp_4078_;
}
else
{
lean_inc(v_a_4077_);
lean_dec(v___x_4027_);
v___x_4079_ = lean_box(0);
v_isShared_4080_ = v_isSharedCheck_4084_;
goto v_resetjp_4078_;
}
v_resetjp_4078_:
{
lean_object* v___x_4082_; 
if (v_isShared_4080_ == 0)
{
v___x_4082_ = v___x_4079_;
goto v_reusejp_4081_;
}
else
{
lean_object* v_reuseFailAlloc_4083_; 
v_reuseFailAlloc_4083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4083_, 0, v_a_4077_);
v___x_4082_ = v_reuseFailAlloc_4083_;
goto v_reusejp_4081_;
}
v_reusejp_4081_:
{
return v___x_4082_;
}
}
}
}
else
{
lean_object* v_a_4085_; lean_object* v___x_4087_; uint8_t v_isShared_4088_; uint8_t v_isSharedCheck_4092_; 
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
lean_dec_ref(v_maxS_3773_);
v_a_4085_ = lean_ctor_get(v___x_4025_, 0);
v_isSharedCheck_4092_ = !lean_is_exclusive(v___x_4025_);
if (v_isSharedCheck_4092_ == 0)
{
v___x_4087_ = v___x_4025_;
v_isShared_4088_ = v_isSharedCheck_4092_;
goto v_resetjp_4086_;
}
else
{
lean_inc(v_a_4085_);
lean_dec(v___x_4025_);
v___x_4087_ = lean_box(0);
v_isShared_4088_ = v_isSharedCheck_4092_;
goto v_resetjp_4086_;
}
v_resetjp_4086_:
{
lean_object* v___x_4090_; 
if (v_isShared_4088_ == 0)
{
v___x_4090_ = v___x_4087_;
goto v_reusejp_4089_;
}
else
{
lean_object* v_reuseFailAlloc_4091_; 
v_reuseFailAlloc_4091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4091_, 0, v_a_4085_);
v___x_4090_ = v_reuseFailAlloc_4091_;
goto v_reusejp_4089_;
}
v_reusejp_4089_:
{
return v___x_4090_;
}
}
}
}
}
else
{
lean_object* v_a_4093_; lean_object* v___x_4095_; uint8_t v_isShared_4096_; uint8_t v_isSharedCheck_4100_; 
lean_dec(v_a_3832_);
lean_dec_ref_known(v_t_3774_, 3);
lean_dec_ref(v_maxS_3773_);
v_a_4093_ = lean_ctor_get(v___x_3944_, 0);
v_isSharedCheck_4100_ = !lean_is_exclusive(v___x_3944_);
if (v_isSharedCheck_4100_ == 0)
{
v___x_4095_ = v___x_3944_;
v_isShared_4096_ = v_isSharedCheck_4100_;
goto v_resetjp_4094_;
}
else
{
lean_inc(v_a_4093_);
lean_dec(v___x_3944_);
v___x_4095_ = lean_box(0);
v_isShared_4096_ = v_isSharedCheck_4100_;
goto v_resetjp_4094_;
}
v_resetjp_4094_:
{
lean_object* v___x_4098_; 
if (v_isShared_4096_ == 0)
{
v___x_4098_ = v___x_4095_;
goto v_reusejp_4097_;
}
else
{
lean_object* v_reuseFailAlloc_4099_; 
v_reuseFailAlloc_4099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4099_, 0, v_a_4093_);
v___x_4098_ = v_reuseFailAlloc_4099_;
goto v_reusejp_4097_;
}
v_reusejp_4097_:
{
return v___x_4098_;
}
}
}
}
}
else
{
lean_object* v_a_4120_; lean_object* v___x_4122_; uint8_t v_isShared_4123_; uint8_t v_isSharedCheck_4127_; 
lean_dec_ref_known(v_t_3774_, 3);
lean_dec_ref(v_maxS_3773_);
v_a_4120_ = lean_ctor_get(v___x_3829_, 0);
v_isSharedCheck_4127_ = !lean_is_exclusive(v___x_3829_);
if (v_isSharedCheck_4127_ == 0)
{
v___x_4122_ = v___x_3829_;
v_isShared_4123_ = v_isSharedCheck_4127_;
goto v_resetjp_4121_;
}
else
{
lean_inc(v_a_4120_);
lean_dec(v___x_3829_);
v___x_4122_ = lean_box(0);
v_isShared_4123_ = v_isSharedCheck_4127_;
goto v_resetjp_4121_;
}
v_resetjp_4121_:
{
lean_object* v___x_4125_; 
if (v_isShared_4123_ == 0)
{
v___x_4125_ = v___x_4122_;
goto v_reusejp_4124_;
}
else
{
lean_object* v_reuseFailAlloc_4126_; 
v_reuseFailAlloc_4126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4126_, 0, v_a_4120_);
v___x_4125_ = v_reuseFailAlloc_4126_;
goto v_reusejp_4124_;
}
v_reusejp_4124_:
{
return v___x_4125_;
}
}
}
v___jp_3785_:
{
lean_object* v___x_3808_; lean_object* v_ref_3809_; lean_object* v___x_3810_; lean_object* v___x_3811_; 
v___x_3808_ = lean_box(0);
v_ref_3809_ = l_Lean_replaceRef(v_ref_3782_, v_ref_3796_);
lean_inc_ref(v_inheritedTraceOptions_3806_);
lean_inc(v_cancelTk_x3f_3804_);
lean_inc(v_currMacroScope_3802_);
lean_inc(v_quotContext_3801_);
lean_inc(v_maxHeartbeats_3800_);
lean_inc(v_initHeartbeats_3799_);
lean_inc(v_openDecls_3798_);
lean_inc(v_currNamespace_3797_);
lean_inc(v_maxRecDepth_3795_);
lean_inc(v_currRecDepth_3794_);
lean_inc_ref(v_options_3793_);
lean_inc_ref(v_fileMap_3792_);
lean_inc_ref(v_fileName_3791_);
v___x_3810_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3810_, 0, v_fileName_3791_);
lean_ctor_set(v___x_3810_, 1, v_fileMap_3792_);
lean_ctor_set(v___x_3810_, 2, v_options_3793_);
lean_ctor_set(v___x_3810_, 3, v_currRecDepth_3794_);
lean_ctor_set(v___x_3810_, 4, v_maxRecDepth_3795_);
lean_ctor_set(v___x_3810_, 5, v_ref_3809_);
lean_ctor_set(v___x_3810_, 6, v_currNamespace_3797_);
lean_ctor_set(v___x_3810_, 7, v_openDecls_3798_);
lean_ctor_set(v___x_3810_, 8, v_initHeartbeats_3799_);
lean_ctor_set(v___x_3810_, 9, v_maxHeartbeats_3800_);
lean_ctor_set(v___x_3810_, 10, v_quotContext_3801_);
lean_ctor_set(v___x_3810_, 11, v_currMacroScope_3802_);
lean_ctor_set(v___x_3810_, 12, v_cancelTk_x3f_3804_);
lean_ctor_set(v___x_3810_, 13, v_inheritedTraceOptions_3806_);
lean_ctor_set_uint8(v___x_3810_, sizeof(void*)*14, v_diag_3803_);
lean_ctor_set_uint8(v___x_3810_, sizeof(void*)*14 + 1, v_suppressElabErrors_3805_);
v___x_3811_ = l_Lean_Elab_Term_mkCoe(v___y_3786_, v_val_3784_, v___x_3808_, v___x_3808_, v___x_3808_, v___x_3808_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___x_3810_, v___y_3807_);
lean_dec_ref_known(v___x_3810_, 14);
if (lean_obj_tag(v___x_3811_) == 0)
{
lean_object* v_a_3812_; lean_object* v___x_3814_; uint8_t v_isShared_3815_; uint8_t v_isSharedCheck_3820_; 
v_a_3812_ = lean_ctor_get(v___x_3811_, 0);
v_isSharedCheck_3820_ = !lean_is_exclusive(v___x_3811_);
if (v_isSharedCheck_3820_ == 0)
{
v___x_3814_ = v___x_3811_;
v_isShared_3815_ = v_isSharedCheck_3820_;
goto v_resetjp_3813_;
}
else
{
lean_inc(v_a_3812_);
lean_dec(v___x_3811_);
v___x_3814_ = lean_box(0);
v_isShared_3815_ = v_isSharedCheck_3820_;
goto v_resetjp_3813_;
}
v_resetjp_3813_:
{
lean_object* v___x_3816_; lean_object* v___x_3818_; 
v___x_3816_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3816_, 0, v_ref_3782_);
lean_ctor_set(v___x_3816_, 1, v_infoTrees_3783_);
lean_ctor_set(v___x_3816_, 2, v_a_3812_);
if (v_isShared_3815_ == 0)
{
lean_ctor_set(v___x_3814_, 0, v___x_3816_);
v___x_3818_ = v___x_3814_;
goto v_reusejp_3817_;
}
else
{
lean_object* v_reuseFailAlloc_3819_; 
v_reuseFailAlloc_3819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3819_, 0, v___x_3816_);
v___x_3818_ = v_reuseFailAlloc_3819_;
goto v_reusejp_3817_;
}
v_reusejp_3817_:
{
return v___x_3818_;
}
}
}
else
{
lean_object* v_a_3821_; lean_object* v___x_3823_; uint8_t v_isShared_3824_; uint8_t v_isSharedCheck_3828_; 
lean_dec_ref(v_infoTrees_3783_);
lean_dec(v_ref_3782_);
v_a_3821_ = lean_ctor_get(v___x_3811_, 0);
v_isSharedCheck_3828_ = !lean_is_exclusive(v___x_3811_);
if (v_isSharedCheck_3828_ == 0)
{
v___x_3823_ = v___x_3811_;
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
else
{
lean_inc(v_a_3821_);
lean_dec(v___x_3811_);
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
case 1:
{
lean_object* v_ref_4128_; lean_object* v_f_4129_; lean_object* v_lhs_4130_; lean_object* v_rhs_4131_; lean_object* v___x_4133_; uint8_t v_isShared_4134_; uint8_t v_isSharedCheck_4149_; 
v_ref_4128_ = lean_ctor_get(v_t_3774_, 0);
v_f_4129_ = lean_ctor_get(v_t_3774_, 1);
v_lhs_4130_ = lean_ctor_get(v_t_3774_, 2);
v_rhs_4131_ = lean_ctor_get(v_t_3774_, 3);
v_isSharedCheck_4149_ = !lean_is_exclusive(v_t_3774_);
if (v_isSharedCheck_4149_ == 0)
{
v___x_4133_ = v_t_3774_;
v_isShared_4134_ = v_isSharedCheck_4149_;
goto v_resetjp_4132_;
}
else
{
lean_inc(v_rhs_4131_);
lean_inc(v_lhs_4130_);
lean_inc(v_f_4129_);
lean_inc(v_ref_4128_);
lean_dec(v_t_3774_);
v___x_4133_ = lean_box(0);
v_isShared_4134_ = v_isSharedCheck_4149_;
goto v_resetjp_4132_;
}
v_resetjp_4132_:
{
lean_object* v___x_4135_; 
lean_inc_ref(v_maxS_3773_);
v___x_4135_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_3773_, v_lhs_4130_, v_a_3775_, v_a_3776_, v_a_3777_, v_a_3778_, v_a_3779_, v_a_3780_);
if (lean_obj_tag(v___x_4135_) == 0)
{
lean_object* v_a_4136_; lean_object* v___x_4137_; 
v_a_4136_ = lean_ctor_get(v___x_4135_, 0);
lean_inc(v_a_4136_);
lean_dec_ref_known(v___x_4135_, 1);
v___x_4137_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_3773_, v_rhs_4131_, v_a_3775_, v_a_3776_, v_a_3777_, v_a_3778_, v_a_3779_, v_a_3780_);
if (lean_obj_tag(v___x_4137_) == 0)
{
lean_object* v_a_4138_; lean_object* v___x_4140_; uint8_t v_isShared_4141_; uint8_t v_isSharedCheck_4148_; 
v_a_4138_ = lean_ctor_get(v___x_4137_, 0);
v_isSharedCheck_4148_ = !lean_is_exclusive(v___x_4137_);
if (v_isSharedCheck_4148_ == 0)
{
v___x_4140_ = v___x_4137_;
v_isShared_4141_ = v_isSharedCheck_4148_;
goto v_resetjp_4139_;
}
else
{
lean_inc(v_a_4138_);
lean_dec(v___x_4137_);
v___x_4140_ = lean_box(0);
v_isShared_4141_ = v_isSharedCheck_4148_;
goto v_resetjp_4139_;
}
v_resetjp_4139_:
{
lean_object* v___x_4143_; 
if (v_isShared_4134_ == 0)
{
lean_ctor_set(v___x_4133_, 3, v_a_4138_);
lean_ctor_set(v___x_4133_, 2, v_a_4136_);
v___x_4143_ = v___x_4133_;
goto v_reusejp_4142_;
}
else
{
lean_object* v_reuseFailAlloc_4147_; 
v_reuseFailAlloc_4147_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4147_, 0, v_ref_4128_);
lean_ctor_set(v_reuseFailAlloc_4147_, 1, v_f_4129_);
lean_ctor_set(v_reuseFailAlloc_4147_, 2, v_a_4136_);
lean_ctor_set(v_reuseFailAlloc_4147_, 3, v_a_4138_);
v___x_4143_ = v_reuseFailAlloc_4147_;
goto v_reusejp_4142_;
}
v_reusejp_4142_:
{
lean_object* v___x_4145_; 
if (v_isShared_4141_ == 0)
{
lean_ctor_set(v___x_4140_, 0, v___x_4143_);
v___x_4145_ = v___x_4140_;
goto v_reusejp_4144_;
}
else
{
lean_object* v_reuseFailAlloc_4146_; 
v_reuseFailAlloc_4146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4146_, 0, v___x_4143_);
v___x_4145_ = v_reuseFailAlloc_4146_;
goto v_reusejp_4144_;
}
v_reusejp_4144_:
{
return v___x_4145_;
}
}
}
}
else
{
lean_dec(v_a_4136_);
lean_del_object(v___x_4133_);
lean_dec_ref(v_f_4129_);
lean_dec(v_ref_4128_);
return v___x_4137_;
}
}
else
{
lean_del_object(v___x_4133_);
lean_dec_ref(v_rhs_4131_);
lean_dec_ref(v_f_4129_);
lean_dec(v_ref_4128_);
lean_dec_ref(v_maxS_3773_);
return v___x_4135_;
}
}
}
default: 
{
lean_object* v_macroName_4150_; lean_object* v_stx_4151_; lean_object* v_stx_x27_4152_; lean_object* v_nested_4153_; lean_object* v_fileName_4154_; lean_object* v_fileMap_4155_; lean_object* v_options_4156_; lean_object* v_currRecDepth_4157_; lean_object* v_maxRecDepth_4158_; lean_object* v_ref_4159_; lean_object* v_currNamespace_4160_; lean_object* v_openDecls_4161_; lean_object* v_initHeartbeats_4162_; lean_object* v_maxHeartbeats_4163_; lean_object* v_quotContext_4164_; lean_object* v_currMacroScope_4165_; uint8_t v_diag_4166_; lean_object* v_cancelTk_x3f_4167_; uint8_t v_suppressElabErrors_4168_; lean_object* v_inheritedTraceOptions_4169_; lean_object* v___f_4170_; lean_object* v_ref_4171_; lean_object* v___x_4172_; lean_object* v___x_4173_; 
v_macroName_4150_ = lean_ctor_get(v_t_3774_, 0);
lean_inc(v_macroName_4150_);
v_stx_4151_ = lean_ctor_get(v_t_3774_, 1);
lean_inc_n(v_stx_4151_, 2);
v_stx_x27_4152_ = lean_ctor_get(v_t_3774_, 2);
lean_inc_n(v_stx_x27_4152_, 2);
v_nested_4153_ = lean_ctor_get(v_t_3774_, 3);
lean_inc_ref(v_nested_4153_);
lean_dec_ref_known(v_t_3774_, 4);
v_fileName_4154_ = lean_ctor_get(v_a_3779_, 0);
v_fileMap_4155_ = lean_ctor_get(v_a_3779_, 1);
v_options_4156_ = lean_ctor_get(v_a_3779_, 2);
v_currRecDepth_4157_ = lean_ctor_get(v_a_3779_, 3);
v_maxRecDepth_4158_ = lean_ctor_get(v_a_3779_, 4);
v_ref_4159_ = lean_ctor_get(v_a_3779_, 5);
v_currNamespace_4160_ = lean_ctor_get(v_a_3779_, 6);
v_openDecls_4161_ = lean_ctor_get(v_a_3779_, 7);
v_initHeartbeats_4162_ = lean_ctor_get(v_a_3779_, 8);
v_maxHeartbeats_4163_ = lean_ctor_get(v_a_3779_, 9);
v_quotContext_4164_ = lean_ctor_get(v_a_3779_, 10);
v_currMacroScope_4165_ = lean_ctor_get(v_a_3779_, 11);
v_diag_4166_ = lean_ctor_get_uint8(v_a_3779_, sizeof(void*)*14);
v_cancelTk_x3f_4167_ = lean_ctor_get(v_a_3779_, 12);
v_suppressElabErrors_4168_ = lean_ctor_get_uint8(v_a_3779_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4169_ = lean_ctor_get(v_a_3779_, 13);
v___f_4170_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1___boxed), 12, 5);
lean_closure_set(v___f_4170_, 0, v_maxS_3773_);
lean_closure_set(v___f_4170_, 1, v_nested_4153_);
lean_closure_set(v___f_4170_, 2, v_macroName_4150_);
lean_closure_set(v___f_4170_, 3, v_stx_4151_);
lean_closure_set(v___f_4170_, 4, v_stx_x27_4152_);
v_ref_4171_ = l_Lean_replaceRef(v_stx_4151_, v_ref_4159_);
lean_inc_ref(v_inheritedTraceOptions_4169_);
lean_inc(v_cancelTk_x3f_4167_);
lean_inc(v_currMacroScope_4165_);
lean_inc(v_quotContext_4164_);
lean_inc(v_maxHeartbeats_4163_);
lean_inc(v_initHeartbeats_4162_);
lean_inc(v_openDecls_4161_);
lean_inc(v_currNamespace_4160_);
lean_inc(v_maxRecDepth_4158_);
lean_inc(v_currRecDepth_4157_);
lean_inc_ref(v_options_4156_);
lean_inc_ref(v_fileMap_4155_);
lean_inc_ref(v_fileName_4154_);
v___x_4172_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4172_, 0, v_fileName_4154_);
lean_ctor_set(v___x_4172_, 1, v_fileMap_4155_);
lean_ctor_set(v___x_4172_, 2, v_options_4156_);
lean_ctor_set(v___x_4172_, 3, v_currRecDepth_4157_);
lean_ctor_set(v___x_4172_, 4, v_maxRecDepth_4158_);
lean_ctor_set(v___x_4172_, 5, v_ref_4171_);
lean_ctor_set(v___x_4172_, 6, v_currNamespace_4160_);
lean_ctor_set(v___x_4172_, 7, v_openDecls_4161_);
lean_ctor_set(v___x_4172_, 8, v_initHeartbeats_4162_);
lean_ctor_set(v___x_4172_, 9, v_maxHeartbeats_4163_);
lean_ctor_set(v___x_4172_, 10, v_quotContext_4164_);
lean_ctor_set(v___x_4172_, 11, v_currMacroScope_4165_);
lean_ctor_set(v___x_4172_, 12, v_cancelTk_x3f_4167_);
lean_ctor_set(v___x_4172_, 13, v_inheritedTraceOptions_4169_);
lean_ctor_set_uint8(v___x_4172_, sizeof(void*)*14, v_diag_4166_);
lean_ctor_set_uint8(v___x_4172_, sizeof(void*)*14 + 1, v_suppressElabErrors_4168_);
v___x_4173_ = l_Lean_Elab_Term_withPushMacroExpansionStack___redArg(v_stx_4151_, v_stx_x27_4152_, v___f_4170_, v_a_3775_, v_a_3776_, v_a_3777_, v_a_3778_, v___x_4172_, v_a_3780_);
lean_dec_ref_known(v___x_4172_, 14);
return v___x_4173_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___lam__1(lean_object* v_maxS_4174_, lean_object* v_nested_4175_, lean_object* v_macroName_4176_, lean_object* v_stx_4177_, lean_object* v_stx_x27_4178_, lean_object* v___y_4179_, lean_object* v___y_4180_, lean_object* v___y_4181_, lean_object* v___y_4182_, lean_object* v___y_4183_, lean_object* v___y_4184_){
_start:
{
lean_object* v___x_4186_; 
v___x_4186_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_4174_, v_nested_4175_, v___y_4179_, v___y_4180_, v___y_4181_, v___y_4182_, v___y_4183_, v___y_4184_);
if (lean_obj_tag(v___x_4186_) == 0)
{
lean_object* v_a_4187_; lean_object* v___x_4189_; uint8_t v_isShared_4190_; uint8_t v_isSharedCheck_4195_; 
v_a_4187_ = lean_ctor_get(v___x_4186_, 0);
v_isSharedCheck_4195_ = !lean_is_exclusive(v___x_4186_);
if (v_isSharedCheck_4195_ == 0)
{
v___x_4189_ = v___x_4186_;
v_isShared_4190_ = v_isSharedCheck_4195_;
goto v_resetjp_4188_;
}
else
{
lean_inc(v_a_4187_);
lean_dec(v___x_4186_);
v___x_4189_ = lean_box(0);
v_isShared_4190_ = v_isSharedCheck_4195_;
goto v_resetjp_4188_;
}
v_resetjp_4188_:
{
lean_object* v___x_4191_; lean_object* v___x_4193_; 
v___x_4191_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_4191_, 0, v_macroName_4176_);
lean_ctor_set(v___x_4191_, 1, v_stx_4177_);
lean_ctor_set(v___x_4191_, 2, v_stx_x27_4178_);
lean_ctor_set(v___x_4191_, 3, v_a_4187_);
if (v_isShared_4190_ == 0)
{
lean_ctor_set(v___x_4189_, 0, v___x_4191_);
v___x_4193_ = v___x_4189_;
goto v_reusejp_4192_;
}
else
{
lean_object* v_reuseFailAlloc_4194_; 
v_reuseFailAlloc_4194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4194_, 0, v___x_4191_);
v___x_4193_ = v_reuseFailAlloc_4194_;
goto v_reusejp_4192_;
}
v_reusejp_4192_:
{
return v___x_4193_;
}
}
}
else
{
lean_dec(v_stx_x27_4178_);
lean_dec(v_stx_4177_);
lean_dec(v_macroName_4176_);
return v___x_4186_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg___boxed(lean_object* v_maxS_4196_, lean_object* v_t_4197_, lean_object* v_a_4198_, lean_object* v_a_4199_, lean_object* v_a_4200_, lean_object* v_a_4201_, lean_object* v_a_4202_, lean_object* v_a_4203_, lean_object* v_a_4204_){
_start:
{
lean_object* v_res_4205_; 
v_res_4205_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_4196_, v_t_4197_, v_a_4198_, v_a_4199_, v_a_4200_, v_a_4201_, v_a_4202_, v_a_4203_);
lean_dec(v_a_4203_);
lean_dec_ref(v_a_4202_);
lean_dec(v_a_4201_);
lean_dec_ref(v_a_4200_);
lean_dec(v_a_4199_);
lean_dec_ref(v_a_4198_);
return v_res_4205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go(lean_object* v_maxS_4206_, lean_object* v_t_4207_, lean_object* v_f_x3f_4208_, lean_object* v_a_4209_, lean_object* v_a_4210_, lean_object* v_a_4211_, lean_object* v_a_4212_, lean_object* v_a_4213_, lean_object* v_a_4214_){
_start:
{
lean_object* v___x_4216_; 
v___x_4216_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_4206_, v_t_4207_, v_a_4209_, v_a_4210_, v_a_4211_, v_a_4212_, v_a_4213_, v_a_4214_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___boxed(lean_object* v_maxS_4217_, lean_object* v_t_4218_, lean_object* v_f_x3f_4219_, lean_object* v_a_4220_, lean_object* v_a_4221_, lean_object* v_a_4222_, lean_object* v_a_4223_, lean_object* v_a_4224_, lean_object* v_a_4225_, lean_object* v_a_4226_){
_start:
{
lean_object* v_res_4227_; 
v_res_4227_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go(v_maxS_4217_, v_t_4218_, v_f_x3f_4219_, v_a_4220_, v_a_4221_, v_a_4222_, v_a_4223_, v_a_4224_, v_a_4225_);
lean_dec(v_a_4225_);
lean_dec_ref(v_a_4224_);
lean_dec(v_a_4223_);
lean_dec_ref(v_a_4222_);
lean_dec(v_a_4221_);
lean_dec_ref(v_a_4220_);
lean_dec(v_f_x3f_4219_);
return v_res_4227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe(lean_object* v_t_4228_, lean_object* v_maxS_4229_, lean_object* v_a_4230_, lean_object* v_a_4231_, lean_object* v_a_4232_, lean_object* v_a_4233_, lean_object* v_a_4234_, lean_object* v_a_4235_){
_start:
{
lean_object* v___x_4237_; 
v___x_4237_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_maxS_4229_, v_t_4228_, v_a_4230_, v_a_4231_, v_a_4232_, v_a_4233_, v_a_4234_, v_a_4235_);
return v___x_4237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe___boxed(lean_object* v_t_4238_, lean_object* v_maxS_4239_, lean_object* v_a_4240_, lean_object* v_a_4241_, lean_object* v_a_4242_, lean_object* v_a_4243_, lean_object* v_a_4244_, lean_object* v_a_4245_, lean_object* v_a_4246_){
_start:
{
lean_object* v_res_4247_; 
v_res_4247_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe(v_t_4238_, v_maxS_4239_, v_a_4240_, v_a_4241_, v_a_4242_, v_a_4243_, v_a_4244_, v_a_4245_);
lean_dec(v_a_4245_);
lean_dec_ref(v_a_4244_);
lean_dec(v_a_4243_);
lean_dec_ref(v_a_4242_);
lean_dec(v_a_4241_);
lean_dec_ref(v_a_4240_);
return v_res_4247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr_spec__0(lean_object* v_msg_4248_){
_start:
{
lean_object* v___x_4249_; lean_object* v___x_4250_; 
v___x_4249_ = ((lean_object*)(lp_mathlib_FBinopElab_instInhabitedSRec_default));
v___x_4250_ = lean_panic_fn_borrowed(v___x_4249_, v_msg_4248_);
return v___x_4250_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1(void){
_start:
{
lean_object* v___x_4252_; lean_object* v___x_4253_; 
v___x_4252_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__0));
v___x_4253_ = l_Lean_stringToMessageData(v___x_4252_);
return v___x_4253_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3(void){
_start:
{
lean_object* v___x_4255_; lean_object* v___x_4256_; 
v___x_4255_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__2));
v___x_4256_ = l_Lean_stringToMessageData(v___x_4255_);
return v___x_4256_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5(void){
_start:
{
lean_object* v___x_4258_; lean_object* v___x_4259_; 
v___x_4258_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__4));
v___x_4259_ = l_Lean_stringToMessageData(v___x_4258_);
return v___x_4259_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9(void){
_start:
{
lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4265_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10));
v___x_4266_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__8));
v___x_4267_ = l_Lean_mkConst(v___x_4266_, v___x_4265_);
return v___x_4267_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10(void){
_start:
{
lean_object* v___x_4268_; lean_object* v___x_4269_; lean_object* v___x_4270_; 
v___x_4268_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec___closed__2, &lp_mathlib_FBinopElab_instToExprSRec___closed__2_once, _init_lp_mathlib_FBinopElab_instToExprSRec___closed__2);
v___x_4269_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__9);
v___x_4270_ = l_Lean_Expr_app___override(v___x_4269_, v___x_4268_);
return v___x_4270_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13(void){
_start:
{
lean_object* v___x_4275_; lean_object* v___x_4276_; lean_object* v___x_4277_; 
v___x_4275_ = ((lean_object*)(lp_mathlib_FBinopElab_instToExprSRec_toExpr___closed__10));
v___x_4276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__12));
v___x_4277_ = l_Lean_mkConst(v___x_4276_, v___x_4275_);
return v___x_4277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr(lean_object* v_tree_4280_, lean_object* v_expectedType_x3f_4281_, lean_object* v_a_4282_, lean_object* v_a_4283_, lean_object* v_a_4284_, lean_object* v_a_4285_, lean_object* v_a_4286_, lean_object* v_a_4287_){
_start:
{
lean_object* v___y_4290_; lean_object* v___y_4291_; lean_object* v___y_4292_; lean_object* v___y_4293_; lean_object* v___y_4294_; lean_object* v___y_4295_; lean_object* v___y_4301_; lean_object* v___y_4302_; lean_object* v___y_4303_; lean_object* v___y_4304_; lean_object* v___y_4305_; lean_object* v___y_4306_; lean_object* v___y_4307_; lean_object* v___x_4310_; 
lean_inc(v_expectedType_x3f_4281_);
lean_inc_ref(v_tree_4280_);
v___x_4310_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_analyze(v_tree_4280_, v_expectedType_x3f_4281_, v_a_4282_, v_a_4283_, v_a_4284_, v_a_4285_, v_a_4286_, v_a_4287_);
if (lean_obj_tag(v___x_4310_) == 0)
{
lean_object* v_options_4311_; lean_object* v_a_4312_; lean_object* v_inheritedTraceOptions_4313_; uint8_t v_hasTrace_4314_; lean_object* v___x_4315_; lean_object* v___y_4317_; lean_object* v___y_4318_; lean_object* v___y_4319_; lean_object* v___y_4320_; lean_object* v___y_4321_; lean_object* v___y_4322_; lean_object* v___y_4357_; lean_object* v___y_4358_; 
v_options_4311_ = lean_ctor_get(v_a_4286_, 2);
v_a_4312_ = lean_ctor_get(v___x_4310_, 0);
lean_inc(v_a_4312_);
lean_dec_ref_known(v___x_4310_, 1);
v_inheritedTraceOptions_4313_ = lean_ctor_get(v_a_4286_, 13);
v_hasTrace_4314_ = lean_ctor_get_uint8(v_options_4311_, sizeof(void*)*1);
v___x_4315_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn___closed__2_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_));
if (v_hasTrace_4314_ == 0)
{
v___y_4317_ = v_a_4282_;
v___y_4318_ = v_a_4283_;
v___y_4319_ = v_a_4284_;
v___y_4320_ = v_a_4285_;
v___y_4321_ = v_a_4286_;
v___y_4322_ = v_a_4287_;
goto v___jp_4316_;
}
else
{
lean_object* v___x_4370_; uint8_t v___x_4371_; 
v___x_4370_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2);
v___x_4371_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4313_, v_options_4311_, v___x_4370_);
if (v___x_4371_ == 0)
{
v___y_4317_ = v_a_4282_;
v___y_4318_ = v_a_4283_;
v___y_4319_ = v_a_4284_;
v___y_4320_ = v_a_4285_;
v___y_4321_ = v_a_4286_;
v___y_4322_ = v_a_4287_;
goto v___jp_4316_;
}
else
{
lean_object* v_maxS_x3f_4372_; uint8_t v_hasUncomparable_4373_; lean_object* v___x_4374_; lean_object* v___y_4376_; 
v_maxS_x3f_4372_ = lean_ctor_get(v_a_4312_, 0);
v_hasUncomparable_4373_ = lean_ctor_get_uint8(v_a_4312_, sizeof(void*)*1);
v___x_4374_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__3);
if (v_hasUncomparable_4373_ == 0)
{
lean_object* v___x_4388_; 
v___x_4388_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__14));
v___y_4376_ = v___x_4388_;
goto v___jp_4375_;
}
else
{
lean_object* v___x_4389_; 
v___x_4389_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__15));
v___y_4376_ = v___x_4389_;
goto v___jp_4375_;
}
v___jp_4375_:
{
lean_object* v___x_4377_; lean_object* v___x_4378_; lean_object* v___x_4379_; lean_object* v___x_4380_; lean_object* v___x_4381_; lean_object* v___x_4382_; 
lean_inc_ref(v___y_4376_);
v___x_4377_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4377_, 0, v___y_4376_);
v___x_4378_ = l_Lean_MessageData_ofFormat(v___x_4377_);
v___x_4379_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4379_, 0, v___x_4374_);
lean_ctor_set(v___x_4379_, 1, v___x_4378_);
v___x_4380_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__5);
v___x_4381_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4381_, 0, v___x_4379_);
lean_ctor_set(v___x_4381_, 1, v___x_4380_);
v___x_4382_ = lean_obj_once(&lp_mathlib_FBinopElab_instToExprSRec___closed__2, &lp_mathlib_FBinopElab_instToExprSRec___closed__2_once, _init_lp_mathlib_FBinopElab_instToExprSRec___closed__2);
if (lean_obj_tag(v_maxS_x3f_4372_) == 0)
{
lean_object* v___x_4383_; 
v___x_4383_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__10);
v___y_4357_ = v___x_4381_;
v___y_4358_ = v___x_4383_;
goto v___jp_4356_;
}
else
{
lean_object* v_val_4384_; lean_object* v___x_4385_; lean_object* v___x_4386_; lean_object* v___x_4387_; 
v_val_4384_ = lean_ctor_get(v_maxS_x3f_4372_, 0);
v___x_4385_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__13);
lean_inc(v_val_4384_);
v___x_4386_ = lp_mathlib_FBinopElab_instToExprSRec_toExpr(v_val_4384_);
v___x_4387_ = l_Lean_mkAppB(v___x_4385_, v___x_4382_, v___x_4386_);
v___y_4357_ = v___x_4381_;
v___y_4358_ = v___x_4387_;
goto v___jp_4356_;
}
}
}
}
v___jp_4316_:
{
uint8_t v_hasUncomparable_4323_; 
v_hasUncomparable_4323_ = lean_ctor_get_uint8(v_a_4312_, sizeof(void*)*1);
if (v_hasUncomparable_4323_ == 0)
{
lean_object* v_maxS_x3f_4324_; 
v_maxS_x3f_4324_ = lean_ctor_get(v_a_4312_, 0);
lean_inc(v_maxS_x3f_4324_);
lean_dec(v_a_4312_);
if (lean_obj_tag(v_maxS_x3f_4324_) == 0)
{
v___y_4290_ = v___y_4322_;
v___y_4291_ = v___y_4320_;
v___y_4292_ = v___y_4318_;
v___y_4293_ = v___y_4319_;
v___y_4294_ = v___y_4317_;
v___y_4295_ = v___y_4321_;
goto v___jp_4289_;
}
else
{
lean_object* v_val_4325_; lean_object* v___x_4326_; 
v_val_4325_ = lean_ctor_get(v_maxS_x3f_4324_, 0);
lean_inc(v_val_4325_);
lean_dec_ref_known(v_maxS_x3f_4324_, 1);
v___x_4326_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_applyCoe_go___redArg(v_val_4325_, v_tree_4280_, v___y_4317_, v___y_4318_, v___y_4319_, v___y_4320_, v___y_4321_, v___y_4322_);
if (lean_obj_tag(v___x_4326_) == 0)
{
lean_object* v_a_4327_; lean_object* v___x_4328_; 
v_a_4327_ = lean_ctor_get(v___x_4326_, 0);
lean_inc(v_a_4327_);
lean_dec_ref_known(v___x_4326_, 1);
v___x_4328_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(v_a_4327_, v___y_4317_, v___y_4318_, v___y_4319_, v___y_4320_, v___y_4321_, v___y_4322_);
if (lean_obj_tag(v___x_4328_) == 0)
{
lean_object* v_options_4329_; uint8_t v_hasTrace_4330_; 
v_options_4329_ = lean_ctor_get(v___y_4321_, 2);
v_hasTrace_4330_ = lean_ctor_get_uint8(v_options_4329_, sizeof(void*)*1);
if (v_hasTrace_4330_ == 0)
{
lean_object* v_a_4331_; 
v_a_4331_ = lean_ctor_get(v___x_4328_, 0);
lean_inc(v_a_4331_);
lean_dec_ref_known(v___x_4328_, 1);
v___y_4301_ = v_a_4331_;
v___y_4302_ = v___y_4317_;
v___y_4303_ = v___y_4318_;
v___y_4304_ = v___y_4319_;
v___y_4305_ = v___y_4320_;
v___y_4306_ = v___y_4321_;
v___y_4307_ = v___y_4322_;
goto v___jp_4300_;
}
else
{
lean_object* v_a_4332_; lean_object* v_inheritedTraceOptions_4333_; lean_object* v___x_4334_; uint8_t v___x_4335_; 
v_a_4332_ = lean_ctor_get(v___x_4328_, 0);
lean_inc(v_a_4332_);
lean_dec_ref_known(v___x_4328_, 1);
v_inheritedTraceOptions_4333_ = lean_ctor_get(v___y_4321_, 13);
v___x_4334_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_hasCoeS___closed__2);
v___x_4335_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4333_, v_options_4329_, v___x_4334_);
if (v___x_4335_ == 0)
{
v___y_4301_ = v_a_4332_;
v___y_4302_ = v___y_4317_;
v___y_4303_ = v___y_4318_;
v___y_4304_ = v___y_4319_;
v___y_4305_ = v___y_4320_;
v___y_4306_ = v___y_4321_;
v___y_4307_ = v___y_4322_;
goto v___jp_4300_;
}
else
{
lean_object* v___x_4336_; lean_object* v___x_4337_; lean_object* v___x_4338_; lean_object* v___x_4339_; 
v___x_4336_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1, &lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___closed__1);
lean_inc(v_a_4332_);
v___x_4337_ = l_Lean_MessageData_ofExpr(v_a_4332_);
v___x_4338_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4338_, 0, v___x_4336_);
lean_ctor_set(v___x_4338_, 1, v___x_4337_);
v___x_4339_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_4315_, v___x_4338_, v___y_4319_, v___y_4320_, v___y_4321_, v___y_4322_);
if (lean_obj_tag(v___x_4339_) == 0)
{
lean_dec_ref_known(v___x_4339_, 1);
v___y_4301_ = v_a_4332_;
v___y_4302_ = v___y_4317_;
v___y_4303_ = v___y_4318_;
v___y_4304_ = v___y_4319_;
v___y_4305_ = v___y_4320_;
v___y_4306_ = v___y_4321_;
v___y_4307_ = v___y_4322_;
goto v___jp_4300_;
}
else
{
lean_object* v_a_4340_; lean_object* v___x_4342_; uint8_t v_isShared_4343_; uint8_t v_isSharedCheck_4347_; 
lean_dec(v_a_4332_);
lean_dec(v_expectedType_x3f_4281_);
v_a_4340_ = lean_ctor_get(v___x_4339_, 0);
v_isSharedCheck_4347_ = !lean_is_exclusive(v___x_4339_);
if (v_isSharedCheck_4347_ == 0)
{
v___x_4342_ = v___x_4339_;
v_isShared_4343_ = v_isSharedCheck_4347_;
goto v_resetjp_4341_;
}
else
{
lean_inc(v_a_4340_);
lean_dec(v___x_4339_);
v___x_4342_ = lean_box(0);
v_isShared_4343_ = v_isSharedCheck_4347_;
goto v_resetjp_4341_;
}
v_resetjp_4341_:
{
lean_object* v___x_4345_; 
if (v_isShared_4343_ == 0)
{
v___x_4345_ = v___x_4342_;
goto v_reusejp_4344_;
}
else
{
lean_object* v_reuseFailAlloc_4346_; 
v_reuseFailAlloc_4346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4346_, 0, v_a_4340_);
v___x_4345_ = v_reuseFailAlloc_4346_;
goto v_reusejp_4344_;
}
v_reusejp_4344_:
{
return v___x_4345_;
}
}
}
}
}
}
else
{
lean_dec(v_expectedType_x3f_4281_);
return v___x_4328_;
}
}
else
{
lean_object* v_a_4348_; lean_object* v___x_4350_; uint8_t v_isShared_4351_; uint8_t v_isSharedCheck_4355_; 
lean_dec(v_expectedType_x3f_4281_);
v_a_4348_ = lean_ctor_get(v___x_4326_, 0);
v_isSharedCheck_4355_ = !lean_is_exclusive(v___x_4326_);
if (v_isSharedCheck_4355_ == 0)
{
v___x_4350_ = v___x_4326_;
v_isShared_4351_ = v_isSharedCheck_4355_;
goto v_resetjp_4349_;
}
else
{
lean_inc(v_a_4348_);
lean_dec(v___x_4326_);
v___x_4350_ = lean_box(0);
v_isShared_4351_ = v_isSharedCheck_4355_;
goto v_resetjp_4349_;
}
v_resetjp_4349_:
{
lean_object* v___x_4353_; 
if (v_isShared_4351_ == 0)
{
v___x_4353_ = v___x_4350_;
goto v_reusejp_4352_;
}
else
{
lean_object* v_reuseFailAlloc_4354_; 
v_reuseFailAlloc_4354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4354_, 0, v_a_4348_);
v___x_4353_ = v_reuseFailAlloc_4354_;
goto v_reusejp_4352_;
}
v_reusejp_4352_:
{
return v___x_4353_;
}
}
}
}
}
else
{
lean_dec(v_a_4312_);
v___y_4290_ = v___y_4322_;
v___y_4291_ = v___y_4320_;
v___y_4292_ = v___y_4318_;
v___y_4293_ = v___y_4319_;
v___y_4294_ = v___y_4317_;
v___y_4295_ = v___y_4321_;
goto v___jp_4289_;
}
}
v___jp_4356_:
{
lean_object* v___x_4359_; lean_object* v___x_4360_; lean_object* v___x_4361_; 
v___x_4359_ = l_Lean_MessageData_ofExpr(v___y_4358_);
v___x_4360_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4360_, 0, v___y_4357_);
lean_ctor_set(v___x_4360_, 1, v___x_4359_);
v___x_4361_ = lp_mathlib_Lean_addTrace___at___00Lean_Elab_liftMacroM___at___00__private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree_go_spec__0_spec__0___redArg(v___x_4315_, v___x_4360_, v_a_4284_, v_a_4285_, v_a_4286_, v_a_4287_);
if (lean_obj_tag(v___x_4361_) == 0)
{
lean_dec_ref_known(v___x_4361_, 1);
v___y_4317_ = v_a_4282_;
v___y_4318_ = v_a_4283_;
v___y_4319_ = v_a_4284_;
v___y_4320_ = v_a_4285_;
v___y_4321_ = v_a_4286_;
v___y_4322_ = v_a_4287_;
goto v___jp_4316_;
}
else
{
lean_object* v_a_4362_; lean_object* v___x_4364_; uint8_t v_isShared_4365_; uint8_t v_isSharedCheck_4369_; 
lean_dec(v_a_4312_);
lean_dec(v_expectedType_x3f_4281_);
lean_dec_ref(v_tree_4280_);
v_a_4362_ = lean_ctor_get(v___x_4361_, 0);
v_isSharedCheck_4369_ = !lean_is_exclusive(v___x_4361_);
if (v_isSharedCheck_4369_ == 0)
{
v___x_4364_ = v___x_4361_;
v_isShared_4365_ = v_isSharedCheck_4369_;
goto v_resetjp_4363_;
}
else
{
lean_inc(v_a_4362_);
lean_dec(v___x_4361_);
v___x_4364_ = lean_box(0);
v_isShared_4365_ = v_isSharedCheck_4369_;
goto v_resetjp_4363_;
}
v_resetjp_4363_:
{
lean_object* v___x_4367_; 
if (v_isShared_4365_ == 0)
{
v___x_4367_ = v___x_4364_;
goto v_reusejp_4366_;
}
else
{
lean_object* v_reuseFailAlloc_4368_; 
v_reuseFailAlloc_4368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4368_, 0, v_a_4362_);
v___x_4367_ = v_reuseFailAlloc_4368_;
goto v_reusejp_4366_;
}
v_reusejp_4366_:
{
return v___x_4367_;
}
}
}
}
}
else
{
lean_object* v_a_4390_; lean_object* v___x_4392_; uint8_t v_isShared_4393_; uint8_t v_isSharedCheck_4397_; 
lean_dec(v_expectedType_x3f_4281_);
lean_dec_ref(v_tree_4280_);
v_a_4390_ = lean_ctor_get(v___x_4310_, 0);
v_isSharedCheck_4397_ = !lean_is_exclusive(v___x_4310_);
if (v_isSharedCheck_4397_ == 0)
{
v___x_4392_ = v___x_4310_;
v_isShared_4393_ = v_isSharedCheck_4397_;
goto v_resetjp_4391_;
}
else
{
lean_inc(v_a_4390_);
lean_dec(v___x_4310_);
v___x_4392_ = lean_box(0);
v_isShared_4393_ = v_isSharedCheck_4397_;
goto v_resetjp_4391_;
}
v_resetjp_4391_:
{
lean_object* v___x_4395_; 
if (v_isShared_4393_ == 0)
{
v___x_4395_ = v___x_4392_;
goto v_reusejp_4394_;
}
else
{
lean_object* v_reuseFailAlloc_4396_; 
v_reuseFailAlloc_4396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4396_, 0, v_a_4390_);
v___x_4395_ = v_reuseFailAlloc_4396_;
goto v_reusejp_4394_;
}
v_reusejp_4394_:
{
return v___x_4395_;
}
}
}
v___jp_4289_:
{
lean_object* v___x_4296_; 
v___x_4296_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExprCore(v_tree_4280_, v___y_4294_, v___y_4292_, v___y_4293_, v___y_4291_, v___y_4295_, v___y_4290_);
if (lean_obj_tag(v___x_4296_) == 0)
{
lean_object* v_a_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; 
v_a_4297_ = lean_ctor_get(v___x_4296_, 0);
lean_inc(v_a_4297_);
lean_dec_ref_known(v___x_4296_, 1);
v___x_4298_ = lean_box(0);
v___x_4299_ = l_Lean_Elab_Term_ensureHasType(v_expectedType_x3f_4281_, v_a_4297_, v___x_4298_, v___x_4298_, v___y_4294_, v___y_4292_, v___y_4293_, v___y_4291_, v___y_4295_, v___y_4290_);
return v___x_4299_;
}
else
{
lean_dec(v_expectedType_x3f_4281_);
return v___x_4296_;
}
}
v___jp_4300_:
{
lean_object* v___x_4308_; lean_object* v___x_4309_; 
v___x_4308_ = lean_box(0);
v___x_4309_ = l_Lean_Elab_Term_ensureHasType(v_expectedType_x3f_4281_, v___y_4301_, v___x_4308_, v___x_4308_, v___y_4302_, v___y_4303_, v___y_4304_, v___y_4305_, v___y_4306_, v___y_4307_);
return v___x_4309_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr___boxed(lean_object* v_tree_4398_, lean_object* v_expectedType_x3f_4399_, lean_object* v_a_4400_, lean_object* v_a_4401_, lean_object* v_a_4402_, lean_object* v_a_4403_, lean_object* v_a_4404_, lean_object* v_a_4405_, lean_object* v_a_4406_){
_start:
{
lean_object* v_res_4407_; 
v_res_4407_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr(v_tree_4398_, v_expectedType_x3f_4399_, v_a_4400_, v_a_4401_, v_a_4402_, v_a_4403_, v_a_4404_, v_a_4405_);
lean_dec(v_a_4405_);
lean_dec_ref(v_a_4404_);
lean_dec(v_a_4403_);
lean_dec_ref(v_a_4402_);
lean_dec(v_a_4401_);
lean_dec_ref(v_a_4400_);
return v_res_4407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_elabBinOp(lean_object* v_stx_4408_, lean_object* v_expectedType_x3f_4409_, lean_object* v_a_4410_, lean_object* v_a_4411_, lean_object* v_a_4412_, lean_object* v_a_4413_, lean_object* v_a_4414_, lean_object* v_a_4415_){
_start:
{
lean_object* v___x_4417_; 
v___x_4417_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toTree(v_stx_4408_, v_a_4410_, v_a_4411_, v_a_4412_, v_a_4413_, v_a_4414_, v_a_4415_);
if (lean_obj_tag(v___x_4417_) == 0)
{
lean_object* v_a_4418_; lean_object* v___x_4419_; 
v_a_4418_ = lean_ctor_get(v___x_4417_, 0);
lean_inc(v_a_4418_);
lean_dec_ref_known(v___x_4417_, 1);
v___x_4419_ = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_toExpr(v_a_4418_, v_expectedType_x3f_4409_, v_a_4410_, v_a_4411_, v_a_4412_, v_a_4413_, v_a_4414_, v_a_4415_);
return v___x_4419_;
}
else
{
lean_object* v_a_4420_; lean_object* v___x_4422_; uint8_t v_isShared_4423_; uint8_t v_isSharedCheck_4427_; 
lean_dec(v_expectedType_x3f_4409_);
v_a_4420_ = lean_ctor_get(v___x_4417_, 0);
v_isSharedCheck_4427_ = !lean_is_exclusive(v___x_4417_);
if (v_isSharedCheck_4427_ == 0)
{
v___x_4422_ = v___x_4417_;
v_isShared_4423_ = v_isSharedCheck_4427_;
goto v_resetjp_4421_;
}
else
{
lean_inc(v_a_4420_);
lean_dec(v___x_4417_);
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
LEAN_EXPORT lean_object* lp_mathlib_FBinopElab_elabBinOp___boxed(lean_object* v_stx_4428_, lean_object* v_expectedType_x3f_4429_, lean_object* v_a_4430_, lean_object* v_a_4431_, lean_object* v_a_4432_, lean_object* v_a_4433_, lean_object* v_a_4434_, lean_object* v_a_4435_, lean_object* v_a_4436_){
_start:
{
lean_object* v_res_4437_; 
v_res_4437_ = lp_mathlib_FBinopElab_elabBinOp(v_stx_4428_, v_expectedType_x3f_4429_, v_a_4430_, v_a_4431_, v_a_4432_, v_a_4433_, v_a_4434_, v_a_4435_);
lean_dec(v_a_4435_);
lean_dec_ref(v_a_4434_);
lean_dec(v_a_4433_);
lean_dec_ref(v_a_4432_);
lean_dec(v_a_4431_);
lean_dec_ref(v_a_4430_);
return v_res_4437_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FBinop(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_App(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_BuiltinNotation(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FBinop(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_App(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_BuiltinNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FBinop_0__FBinopElab_initFn_00___x40_Mathlib_Tactic_FBinop_1378038792____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_FBinopElab_instToExprSRec = _init_lp_mathlib_FBinopElab_instToExprSRec();
lean_mark_persistent(lp_mathlib_FBinopElab_instToExprSRec);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_App(uint8_t builtin);
lean_object* initialize_Lean_Elab_BuiltinNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FBinop(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_App(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_BuiltinNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FBinop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FBinop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FBinop(builtin);
}
#ifdef __cplusplus
}
#endif
