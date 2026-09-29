// Lean compiler output
// Module: Mathlib.Tactic.FunProp.FunctionData
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.FunProp.Mor public import Mathlib.Tactic.FunProp.Mor public import Mathlib.Tactic.FunProp.ToBatteries
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getStructureInfo_x3f(lean_object*, lean_object*);
lean_object* l_Lean_StructureInfo_getProjFn_x3f(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
uint8_t l_Lean_getReducibilityStatusCore(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_Meta_inferType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
size_t lean_array_size(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn_x27(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Expr_eta(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Expr_ctorName(lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_FVarId_getUserName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(lean_object*, lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_Expr_getNumHeadForalls(lean_object*);
lean_object* l_Lean_FVarId_getValue_x3f___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isIdentityFun(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isIdentityFun___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isConstantFun(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isConstantFun___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_domainType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_domainType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Mathlib.Tactic.FunProp.FunctionData"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Meta.FunProp.getFunctionData"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_letE_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_letE_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_lam_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_lam_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_data_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_data_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "fun_prop bug: function expected, got `"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = ", type ctor "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_unfoldHeadFVar_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_unfoldHeadFVar_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_comp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_comp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_uncurried_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_uncurried_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_failed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_failed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "y"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(72, 55, 55, 9, 143, 73, 230, 150)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed__const__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(lean_object* v_lctx_1_, lean_object* v_localInsts_2_, lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1_, v_localInsts_2_, v_x_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_);
if (lean_obj_tag(v___x_9_) == 0)
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_17_; 
v_a_10_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_17_ == 0)
{
v___x_12_ = v___x_9_;
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_15_; 
if (v_isShared_13_ == 0)
{
v___x_15_ = v___x_12_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v_a_10_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
else
{
lean_object* v_a_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_25_; 
v_a_18_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_25_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_25_ == 0)
{
v___x_20_ = v___x_9_;
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_a_18_);
lean_dec(v___x_9_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_a_18_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg___boxed(lean_object* v_lctx_26_, lean_object* v_localInsts_27_, lean_object* v_x_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_26_, v_localInsts_27_, v_x_28_, v___y_29_, v___y_30_, v___y_31_, v___y_32_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0(lean_object* v_00_u03b1_35_, lean_object* v_lctx_36_, lean_object* v_localInsts_37_, lean_object* v_x_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_36_, v_localInsts_37_, v_x_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___boxed(lean_object* v_00_u03b1_45_, lean_object* v_lctx_46_, lean_object* v_localInsts_47_, lean_object* v_x_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0(v_00_u03b1_45_, v_lctx_46_, v_localInsts_47_, v_x_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr(lean_object* v_f_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
lean_object* v_lctx_61_; lean_object* v_insts_62_; lean_object* v_fn_63_; lean_object* v_args_64_; lean_object* v_mainVar_65_; lean_object* v_body_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; uint8_t v___x_71_; uint8_t v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_lctx_61_ = lean_ctor_get(v_f_55_, 0);
lean_inc_ref(v_lctx_61_);
v_insts_62_ = lean_ctor_get(v_f_55_, 1);
lean_inc_ref(v_insts_62_);
v_fn_63_ = lean_ctor_get(v_f_55_, 2);
lean_inc_ref(v_fn_63_);
v_args_64_ = lean_ctor_get(v_f_55_, 3);
lean_inc_ref(v_args_64_);
v_mainVar_65_ = lean_ctor_get(v_f_55_, 4);
lean_inc_ref(v_mainVar_65_);
lean_dec_ref(v_f_55_);
v_body_66_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_fn_63_, v_args_64_);
lean_dec_ref(v_args_64_);
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_mk_empty_array_with_capacity(v___x_67_);
v___x_69_ = lean_array_push(v___x_68_, v_mainVar_65_);
v___x_70_ = 0;
v___x_71_ = 1;
v___x_72_ = 1;
v___x_73_ = lean_box(v___x_70_);
v___x_74_ = lean_box(v___x_71_);
v___x_75_ = lean_box(v___x_70_);
v___x_76_ = lean_box(v___x_71_);
v___x_77_ = lean_box(v___x_72_);
v___x_78_ = lean_alloc_closure((void*)(l_Lean_Meta_mkLambdaFVars___boxed), 12, 7);
lean_closure_set(v___x_78_, 0, v___x_69_);
lean_closure_set(v___x_78_, 1, v_body_66_);
lean_closure_set(v___x_78_, 2, v___x_73_);
lean_closure_set(v___x_78_, 3, v___x_74_);
lean_closure_set(v___x_78_, 4, v___x_75_);
lean_closure_set(v___x_78_, 5, v___x_76_);
lean_closure_set(v___x_78_, 6, v___x_77_);
v___x_79_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_61_, v_insts_62_, v___x_78_, v_a_56_, v_a_57_, v_a_58_, v_a_59_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr___boxed(lean_object* v_f_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr(v_f_80_, v_a_81_, v_a_82_, v_a_83_, v_a_84_);
lean_dec(v_a_84_);
lean_dec_ref(v_a_83_);
lean_dec(v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_86_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isIdentityFun(lean_object* v_f_87_){
_start:
{
lean_object* v_fn_88_; lean_object* v_args_89_; lean_object* v_mainVar_90_; lean_object* v___x_91_; lean_object* v___x_92_; uint8_t v___x_93_; 
v_fn_88_ = lean_ctor_get(v_f_87_, 2);
v_args_89_ = lean_ctor_get(v_f_87_, 3);
v_mainVar_90_ = lean_ctor_get(v_f_87_, 4);
v___x_91_ = lean_array_get_size(v_args_89_);
v___x_92_ = lean_unsigned_to_nat(0u);
v___x_93_ = lean_nat_dec_eq(v___x_91_, v___x_92_);
if (v___x_93_ == 0)
{
return v___x_93_;
}
else
{
uint8_t v___x_94_; 
v___x_94_ = lean_expr_eqv(v_fn_88_, v_mainVar_90_);
return v___x_94_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isIdentityFun___boxed(lean_object* v_f_95_){
_start:
{
uint8_t v_res_96_; lean_object* v_r_97_; 
v_res_96_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isIdentityFun(v_f_95_);
lean_dec_ref(v_f_95_);
v_r_97_ = lean_box(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isConstantFun(lean_object* v_f_98_){
_start:
{
lean_object* v_fn_99_; lean_object* v_mainVar_100_; lean_object* v_mainArgs_101_; lean_object* v___x_102_; lean_object* v___x_103_; uint8_t v___x_104_; 
v_fn_99_ = lean_ctor_get(v_f_98_, 2);
v_mainVar_100_ = lean_ctor_get(v_f_98_, 4);
v_mainArgs_101_ = lean_ctor_get(v_f_98_, 5);
v___x_102_ = lean_array_get_size(v_mainArgs_101_);
v___x_103_ = lean_unsigned_to_nat(0u);
v___x_104_ = lean_nat_dec_eq(v___x_102_, v___x_103_);
if (v___x_104_ == 0)
{
return v___x_104_;
}
else
{
lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_105_ = l_Lean_Expr_fvarId_x21(v_mainVar_100_);
v___x_106_ = l_Lean_Expr_containsFVar(v_fn_99_, v___x_105_);
lean_dec(v___x_105_);
if (v___x_106_ == 0)
{
return v___x_104_;
}
else
{
uint8_t v___x_107_; 
v___x_107_ = 0;
return v___x_107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isConstantFun___boxed(lean_object* v_f_108_){
_start:
{
uint8_t v_res_109_; lean_object* v_r_110_; 
v_res_109_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isConstantFun(v_f_108_);
lean_dec_ref(v_f_108_);
v_r_110_ = lean_box(v_res_109_);
return v_r_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_domainType(lean_object* v_f_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_lctx_117_; lean_object* v_insts_118_; lean_object* v_mainVar_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v_lctx_117_ = lean_ctor_get(v_f_111_, 0);
lean_inc_ref(v_lctx_117_);
v_insts_118_ = lean_ctor_get(v_f_111_, 1);
lean_inc_ref(v_insts_118_);
v_mainVar_119_ = lean_ctor_get(v_f_111_, 4);
lean_inc_ref(v_mainVar_119_);
lean_dec_ref(v_f_111_);
v___x_120_ = lean_alloc_closure((void*)(l_Lean_Meta_inferType___boxed), 6, 1);
lean_closure_set(v___x_120_, 0, v_mainVar_119_);
v___x_121_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_117_, v_insts_118_, v___x_120_, v_a_112_, v_a_113_, v_a_114_, v_a_115_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_domainType___boxed(lean_object* v_f_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_, lean_object* v_a_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_domainType(v_f_122_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
lean_dec(v_a_126_);
lean_dec_ref(v_a_125_);
lean_dec(v_a_124_);
lean_dec_ref(v_a_123_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg(lean_object* v_f_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_fn_132_; 
v_fn_132_ = lean_ctor_get(v_f_129_, 2);
lean_inc_ref(v_fn_132_);
lean_dec_ref(v_f_129_);
switch(lean_obj_tag(v_fn_132_))
{
case 4:
{
lean_object* v_declName_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_declName_133_ = lean_ctor_get(v_fn_132_, 0);
lean_inc(v_declName_133_);
lean_dec_ref_known(v_fn_132_, 2);
v___x_134_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_134_, 0, v_declName_133_);
v___x_135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
return v___x_135_;
}
case 11:
{
lean_object* v_typeName_136_; lean_object* v_idx_137_; lean_object* v___x_138_; lean_object* v_env_139_; lean_object* v___x_140_; 
v_typeName_136_ = lean_ctor_get(v_fn_132_, 0);
lean_inc(v_typeName_136_);
v_idx_137_ = lean_ctor_get(v_fn_132_, 1);
lean_inc(v_idx_137_);
lean_dec_ref_known(v_fn_132_, 3);
v___x_138_ = lean_st_ref_get(v_a_130_);
v_env_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc_ref(v_env_139_);
lean_dec(v___x_138_);
v___x_140_ = l_Lean_getStructureInfo_x3f(v_env_139_, v_typeName_136_);
if (lean_obj_tag(v___x_140_) == 1)
{
lean_object* v_val_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_153_; 
v_val_141_ = lean_ctor_get(v___x_140_, 0);
v_isSharedCheck_153_ = !lean_is_exclusive(v___x_140_);
if (v_isSharedCheck_153_ == 0)
{
v___x_143_ = v___x_140_;
v_isShared_144_ = v_isSharedCheck_153_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_val_141_);
lean_dec(v___x_140_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_153_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___x_145_; 
v___x_145_ = l_Lean_StructureInfo_getProjFn_x3f(v_val_141_, v_idx_137_);
lean_dec(v_idx_137_);
lean_dec(v_val_141_);
if (lean_obj_tag(v___x_145_) == 1)
{
lean_object* v___x_147_; 
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 0);
lean_ctor_set(v___x_143_, 0, v___x_145_);
v___x_147_ = v___x_143_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
else
{
lean_object* v___x_149_; lean_object* v___x_151_; 
lean_dec(v___x_145_);
v___x_149_ = lean_box(0);
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 0);
lean_ctor_set(v___x_143_, 0, v___x_149_);
v___x_151_ = v___x_143_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v___x_149_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
}
else
{
lean_object* v___x_154_; lean_object* v___x_155_; 
lean_dec(v___x_140_);
lean_dec(v_idx_137_);
v___x_154_ = lean_box(0);
v___x_155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
return v___x_155_;
}
}
default: 
{
lean_object* v___x_156_; lean_object* v___x_157_; 
lean_dec_ref(v_fn_132_);
v___x_156_ = lean_box(0);
v___x_157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
return v___x_157_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg___boxed(lean_object* v_f_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg(v_f_158_, v_a_159_);
lean_dec(v_a_159_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f(lean_object* v_f_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___redArg(v_f_162_, v_a_166_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f___boxed(lean_object* v_f_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnConstName_x3f(v_f_169_, v_a_170_, v_a_171_, v_a_172_, v_a_173_);
lean_dec(v_a_173_);
lean_dec_ref(v_a_172_);
lean_dec(v_a_171_);
lean_dec_ref(v_a_170_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3(lean_object* v_msg_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v___f_183_; lean_object* v___x_1709__overap_184_; lean_object* v___x_185_; 
v___f_183_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___closed__0));
v___x_1709__overap_184_ = lean_panic_fn_borrowed(v___f_183_, v_msg_177_);
lean_inc(v___y_181_);
lean_inc_ref(v___y_180_);
lean_inc(v___y_179_);
lean_inc_ref(v___y_178_);
v___x_185_ = lean_apply_5(v___x_1709__overap_184_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, lean_box(0));
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3___boxed(lean_object* v_msg_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3(v_msg_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0(lean_object* v_k_193_, lean_object* v_b_194_, lean_object* v_c_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_){
_start:
{
lean_object* v___x_201_; 
lean_inc(v___y_199_);
lean_inc_ref(v___y_198_);
lean_inc(v___y_197_);
lean_inc_ref(v___y_196_);
v___x_201_ = lean_apply_7(v_k_193_, v_b_194_, v_c_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_, lean_box(0));
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0___boxed(lean_object* v_k_202_, lean_object* v_b_203_, lean_object* v_c_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0(v_k_202_, v_b_203_, v_c_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg(lean_object* v_e_211_, lean_object* v_k_212_, uint8_t v_cleanupAnnotations_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
lean_object* v___f_219_; uint8_t v___x_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_219_, 0, v_k_212_);
v___x_220_ = 1;
v___x_221_ = 0;
v___x_222_ = lean_box(0);
v___x_223_ = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), v_e_211_, v___x_220_, v___x_221_, v___x_220_, v___x_221_, v___x_222_, v___f_219_, v_cleanupAnnotations_213_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_231_; 
v_a_224_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_231_ == 0)
{
v___x_226_ = v___x_223_;
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_223_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_231_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_229_; 
if (v_isShared_227_ == 0)
{
v___x_229_ = v___x_226_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_a_224_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
else
{
lean_object* v_a_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_239_; 
v_a_232_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_239_ == 0)
{
v___x_234_ = v___x_223_;
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_a_232_);
lean_dec(v___x_223_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v_a_232_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg___boxed(lean_object* v_e_240_, lean_object* v_k_241_, lean_object* v_cleanupAnnotations_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_248_; lean_object* v_res_249_; 
v_cleanupAnnotations_boxed_248_ = lean_unbox(v_cleanupAnnotations_242_);
v_res_249_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg(v_e_240_, v_k_241_, v_cleanupAnnotations_boxed_248_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4(lean_object* v_00_u03b1_250_, lean_object* v_e_251_, lean_object* v_k_252_, uint8_t v_cleanupAnnotations_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg(v_e_251_, v_k_252_, v_cleanupAnnotations_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___boxed(lean_object* v_00_u03b1_260_, lean_object* v_e_261_, lean_object* v_k_262_, lean_object* v_cleanupAnnotations_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_269_; lean_object* v_res_270_; 
v_cleanupAnnotations_boxed_269_ = lean_unbox(v_cleanupAnnotations_263_);
v_res_270_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4(v_00_u03b1_260_, v_e_261_, v_k_262_, v_cleanupAnnotations_boxed_269_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1(lean_object* v_as_271_, size_t v_i_272_, size_t v_stop_273_, lean_object* v_b_274_){
_start:
{
lean_object* v___y_276_; uint8_t v___x_280_; 
v___x_280_ = lean_usize_dec_eq(v_i_272_, v_stop_273_);
if (v___x_280_ == 0)
{
lean_object* v___x_281_; 
v___x_281_ = lean_array_uget_borrowed(v_as_271_, v_i_272_);
if (lean_obj_tag(v___x_281_) == 0)
{
v___y_276_ = v_b_274_;
goto v___jp_275_;
}
else
{
lean_object* v_val_282_; lean_object* v___x_283_; 
v_val_282_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_val_282_);
v___x_283_ = lean_array_push(v_b_274_, v_val_282_);
v___y_276_ = v___x_283_;
goto v___jp_275_;
}
}
else
{
return v_b_274_;
}
v___jp_275_:
{
size_t v___x_277_; size_t v___x_278_; 
v___x_277_ = ((size_t)1ULL);
v___x_278_ = lean_usize_add(v_i_272_, v___x_277_);
v_i_272_ = v___x_278_;
v_b_274_ = v___y_276_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1___boxed(lean_object* v_as_284_, lean_object* v_i_285_, lean_object* v_stop_286_, lean_object* v_b_287_){
_start:
{
size_t v_i_boxed_288_; size_t v_stop_boxed_289_; lean_object* v_res_290_; 
v_i_boxed_288_ = lean_unbox_usize(v_i_285_);
lean_dec(v_i_285_);
v_stop_boxed_289_ = lean_unbox_usize(v_stop_286_);
lean_dec(v_stop_286_);
v_res_290_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1(v_as_284_, v_i_boxed_288_, v_stop_boxed_289_, v_b_287_);
lean_dec_ref(v_as_284_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1(lean_object* v_as_293_, lean_object* v_start_294_, lean_object* v_stop_295_){
_start:
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___closed__0));
v___x_297_ = lean_nat_dec_lt(v_start_294_, v_stop_295_);
if (v___x_297_ == 0)
{
return v___x_296_;
}
else
{
lean_object* v___x_298_; uint8_t v___x_299_; 
v___x_298_ = lean_array_get_size(v_as_293_);
v___x_299_ = lean_nat_dec_le(v_stop_295_, v___x_298_);
if (v___x_299_ == 0)
{
uint8_t v___x_300_; 
v___x_300_ = lean_nat_dec_lt(v_start_294_, v___x_298_);
if (v___x_300_ == 0)
{
return v___x_296_;
}
else
{
size_t v___x_301_; size_t v___x_302_; lean_object* v___x_303_; 
v___x_301_ = lean_usize_of_nat(v_start_294_);
v___x_302_ = lean_usize_of_nat(v___x_298_);
v___x_303_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1(v_as_293_, v___x_301_, v___x_302_, v___x_296_);
return v___x_303_;
}
}
else
{
size_t v___x_304_; size_t v___x_305_; lean_object* v___x_306_; 
v___x_304_ = lean_usize_of_nat(v_start_294_);
v___x_305_ = lean_usize_of_nat(v_stop_295_);
v___x_306_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1_spec__1(v_as_293_, v___x_304_, v___x_305_, v___x_296_);
return v___x_306_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1___boxed(lean_object* v_as_307_, lean_object* v_start_308_, lean_object* v_stop_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1(v_as_307_, v_start_308_, v_stop_309_);
lean_dec(v_stop_309_);
lean_dec(v_start_308_);
lean_dec_ref(v_as_307_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg(lean_object* v_xId_311_, size_t v_sz_312_, size_t v_i_313_, lean_object* v_bs_314_){
_start:
{
uint8_t v___x_315_; 
v___x_315_ = lean_usize_dec_lt(v_i_313_, v_sz_312_);
if (v___x_315_ == 0)
{
return v_bs_314_;
}
else
{
lean_object* v_v_316_; lean_object* v_expr_317_; lean_object* v___x_318_; lean_object* v_bs_x27_319_; lean_object* v___y_321_; uint8_t v___x_326_; 
v_v_316_ = lean_array_uget_borrowed(v_bs_314_, v_i_313_);
v_expr_317_ = lean_ctor_get(v_v_316_, 0);
lean_inc_ref(v_expr_317_);
v___x_318_ = lean_unsigned_to_nat(0u);
v_bs_x27_319_ = lean_array_uset(v_bs_314_, v_i_313_, v___x_318_);
v___x_326_ = l_Lean_Expr_containsFVar(v_expr_317_, v_xId_311_);
lean_dec_ref(v_expr_317_);
if (v___x_326_ == 0)
{
lean_object* v___x_327_; 
v___x_327_ = lean_box(0);
v___y_321_ = v___x_327_;
goto v___jp_320_;
}
else
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = lean_usize_to_nat(v_i_313_);
v___x_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
v___y_321_ = v___x_329_;
goto v___jp_320_;
}
v___jp_320_:
{
size_t v___x_322_; size_t v___x_323_; lean_object* v___x_324_; 
v___x_322_ = ((size_t)1ULL);
v___x_323_ = lean_usize_add(v_i_313_, v___x_322_);
v___x_324_ = lean_array_uset(v_bs_x27_319_, v_i_313_, v___y_321_);
v_i_313_ = v___x_323_;
v_bs_314_ = v___x_324_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg___boxed(lean_object* v_xId_330_, lean_object* v_sz_331_, lean_object* v_i_332_, lean_object* v_bs_333_){
_start:
{
size_t v_sz_boxed_334_; size_t v_i_boxed_335_; lean_object* v_res_336_; 
v_sz_boxed_334_ = lean_unbox_usize(v_sz_331_);
lean_dec(v_sz_331_);
v_i_boxed_335_ = lean_unbox_usize(v_i_332_);
lean_dec(v_i_332_);
v_res_336_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg(v_xId_330_, v_sz_boxed_334_, v_i_boxed_335_, v_bs_333_);
lean_dec(v_xId_330_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2(size_t v_sz_337_, size_t v_i_338_, lean_object* v_bs_339_){
_start:
{
uint8_t v___x_340_; 
v___x_340_ = lean_usize_dec_lt(v_i_338_, v_sz_337_);
if (v___x_340_ == 0)
{
return v_bs_339_;
}
else
{
lean_object* v_v_341_; lean_object* v___x_342_; lean_object* v_bs_x27_343_; lean_object* v___x_344_; lean_object* v___x_345_; size_t v___x_346_; size_t v___x_347_; lean_object* v___x_348_; 
v_v_341_ = lean_array_uget(v_bs_339_, v_i_338_);
v___x_342_ = lean_unsigned_to_nat(0u);
v_bs_x27_343_ = lean_array_uset(v_bs_339_, v_i_338_, v___x_342_);
v___x_344_ = lean_box(0);
v___x_345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_345_, 0, v_v_341_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = ((size_t)1ULL);
v___x_347_ = lean_usize_add(v_i_338_, v___x_346_);
v___x_348_ = lean_array_uset(v_bs_x27_343_, v_i_338_, v___x_345_);
v_i_338_ = v___x_347_;
v_bs_339_ = v___x_348_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2___boxed(lean_object* v_sz_350_, lean_object* v_i_351_, lean_object* v_bs_352_){
_start:
{
size_t v_sz_boxed_353_; size_t v_i_boxed_354_; lean_object* v_res_355_; 
v_sz_boxed_353_ = lean_unbox_usize(v_sz_350_);
lean_dec(v_sz_350_);
v_i_boxed_354_ = lean_unbox_usize(v_i_351_);
lean_dec(v_i_351_);
v_res_355_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2(v_sz_boxed_353_, v_i_boxed_354_, v_bs_352_);
return v_res_355_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0(void){
_start:
{
lean_object* v___x_356_; lean_object* v_dummy_357_; 
v___x_356_ = lean_box(0);
v_dummy_357_ = l_Lean_Expr_sort___override(v___x_356_);
return v_dummy_357_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_361_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__3));
v___x_362_ = lean_unsigned_to_nat(49u);
v___x_363_ = lean_unsigned_to_nat(89u);
v___x_364_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__2));
v___x_365_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__1));
v___x_366_ = l_mkPanicMessageWithDecl(v___x_365_, v___x_364_, v___x_363_, v___x_362_, v___x_361_);
return v___x_366_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5(void){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_367_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__3));
v___x_368_ = lean_unsigned_to_nat(58u);
v___x_369_ = lean_unsigned_to_nat(88u);
v___x_370_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__2));
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__1));
v___x_372_ = l_mkPanicMessageWithDecl(v___x_371_, v___x_370_, v___x_369_, v___x_368_, v___x_367_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0(lean_object* v_xId_373_, lean_object* v___x_374_, lean_object* v___x_375_, lean_object* v_fn_376_, lean_object* v_args_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v_fn_384_; lean_object* v_args_385_; 
if (lean_obj_tag(v_fn_376_) == 11)
{
lean_object* v_typeName_395_; lean_object* v_idx_396_; lean_object* v_struct_397_; lean_object* v___x_398_; lean_object* v_env_399_; lean_object* v___x_400_; 
v_typeName_395_ = lean_ctor_get(v_fn_376_, 0);
v_idx_396_ = lean_ctor_get(v_fn_376_, 1);
v_struct_397_ = lean_ctor_get(v_fn_376_, 2);
v___x_398_ = lean_st_ref_get(v___y_381_);
v_env_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc_ref(v_env_399_);
lean_dec(v___x_398_);
lean_inc(v_typeName_395_);
v___x_400_ = l_Lean_getStructureInfo_x3f(v_env_399_, v_typeName_395_);
if (lean_obj_tag(v___x_400_) == 1)
{
lean_object* v_val_401_; lean_object* v___x_402_; 
v_val_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_val_401_);
lean_dec_ref_known(v___x_400_, 1);
v___x_402_ = l_Lean_StructureInfo_getProjFn_x3f(v_val_401_, v_idx_396_);
lean_dec(v_val_401_);
if (lean_obj_tag(v___x_402_) == 1)
{
lean_object* v_val_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
lean_inc_ref(v_struct_397_);
lean_dec_ref_known(v_fn_376_, 3);
v_val_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_val_403_);
lean_dec_ref_known(v___x_402_, 1);
v___x_404_ = lean_unsigned_to_nat(1u);
v___x_405_ = lean_mk_empty_array_with_capacity(v___x_404_);
v___x_406_ = lean_array_push(v___x_405_, v_struct_397_);
v___x_407_ = l_Lean_Meta_mkAppM(v_val_403_, v___x_406_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
if (lean_obj_tag(v___x_407_) == 0)
{
lean_object* v_a_408_; lean_object* v___x_409_; lean_object* v_dummy_410_; lean_object* v_nargs_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; size_t v_sz_415_; size_t v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_a_408_ = lean_ctor_get(v___x_407_, 0);
lean_inc(v_a_408_);
lean_dec_ref_known(v___x_407_, 1);
v___x_409_ = l_Lean_Expr_getAppFn(v_a_408_);
v_dummy_410_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__0);
v_nargs_411_ = l_Lean_Expr_getAppNumArgs(v_a_408_);
lean_inc(v_nargs_411_);
v___x_412_ = lean_mk_array(v_nargs_411_, v_dummy_410_);
v___x_413_ = lean_nat_sub(v_nargs_411_, v___x_404_);
lean_dec(v_nargs_411_);
v___x_414_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_408_, v___x_412_, v___x_413_);
v_sz_415_ = lean_array_size(v___x_414_);
v___x_416_ = ((size_t)0ULL);
v___x_417_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__2(v_sz_415_, v___x_416_, v___x_414_);
v___x_418_ = l_Array_append___redArg(v___x_417_, v_args_377_);
lean_dec_ref(v_args_377_);
v_fn_384_ = v___x_409_;
v_args_385_ = v___x_418_;
goto v___jp_383_;
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref(v_args_377_);
lean_dec_ref(v___x_375_);
v_a_419_ = lean_ctor_get(v___x_407_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_407_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_407_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_407_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
else
{
lean_object* v___x_427_; lean_object* v___x_428_; 
lean_dec(v___x_402_);
v___x_427_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__4);
v___x_428_ = lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3(v___x_427_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
if (lean_obj_tag(v___x_428_) == 0)
{
lean_dec_ref_known(v___x_428_, 1);
v_fn_384_ = v_fn_376_;
v_args_385_ = v_args_377_;
goto v___jp_383_;
}
else
{
lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_436_; 
lean_dec_ref_known(v_fn_376_, 3);
lean_dec_ref(v_args_377_);
lean_dec_ref(v___x_375_);
v_a_429_ = lean_ctor_get(v___x_428_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v___x_428_);
if (v_isSharedCheck_436_ == 0)
{
v___x_431_ = v___x_428_;
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_428_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_434_; 
if (v_isShared_432_ == 0)
{
v___x_434_ = v___x_431_;
goto v_reusejp_433_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v_a_429_);
v___x_434_ = v_reuseFailAlloc_435_;
goto v_reusejp_433_;
}
v_reusejp_433_:
{
return v___x_434_;
}
}
}
}
}
else
{
lean_object* v___x_437_; lean_object* v___x_438_; 
lean_dec(v___x_400_);
v___x_437_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___closed__5);
v___x_438_ = lp_mathlib_panic___at___00Mathlib_Meta_FunProp_getFunctionData_spec__3(v___x_437_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
if (lean_obj_tag(v___x_438_) == 0)
{
lean_dec_ref_known(v___x_438_, 1);
v_fn_384_ = v_fn_376_;
v_args_385_ = v_args_377_;
goto v___jp_383_;
}
else
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_446_; 
lean_dec_ref_known(v_fn_376_, 3);
lean_dec_ref(v_args_377_);
lean_dec_ref(v___x_375_);
v_a_439_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_446_ == 0)
{
v___x_441_ = v___x_438_;
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_438_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___x_444_; 
if (v_isShared_442_ == 0)
{
v___x_444_ = v___x_441_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_a_439_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
}
}
else
{
v_fn_384_ = v_fn_376_;
v_args_385_ = v_args_377_;
goto v___jp_383_;
}
v___jp_383_:
{
size_t v_sz_386_; lean_object* v_lctx_387_; lean_object* v_localInstances_388_; size_t v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v_mainArgs_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v_sz_386_ = lean_array_size(v_args_385_);
v_lctx_387_ = lean_ctor_get(v___y_378_, 2);
v_localInstances_388_ = lean_ctor_get(v___y_378_, 3);
v___x_389_ = ((size_t)0ULL);
lean_inc_ref(v_args_385_);
v___x_390_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg(v_xId_373_, v_sz_386_, v___x_389_, v_args_385_);
v___x_391_ = lean_array_get_size(v___x_390_);
v_mainArgs_392_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Meta_FunProp_getFunctionData_spec__1(v___x_390_, v___x_374_, v___x_391_);
lean_dec_ref(v___x_390_);
lean_inc_ref(v_localInstances_388_);
lean_inc_ref(v_lctx_387_);
v___x_393_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_393_, 0, v_lctx_387_);
lean_ctor_set(v___x_393_, 1, v_localInstances_388_);
lean_ctor_set(v___x_393_, 2, v_fn_384_);
lean_ctor_set(v___x_393_, 3, v_args_385_);
lean_ctor_set(v___x_393_, 4, v___x_375_);
lean_ctor_set(v___x_393_, 5, v_mainArgs_392_);
v___x_394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
return v___x_394_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___boxed(lean_object* v_xId_447_, lean_object* v___x_448_, lean_object* v___x_449_, lean_object* v_fn_450_, lean_object* v_args_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0(v_xId_447_, v___x_448_, v___x_449_, v_fn_450_, v_args_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_);
lean_dec(v___y_455_);
lean_dec_ref(v___y_454_);
lean_dec(v___y_453_);
lean_dec_ref(v___y_452_);
lean_dec(v___x_448_);
lean_dec(v_xId_447_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1(lean_object* v___x_458_, lean_object* v_xs_459_, lean_object* v_b_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v_xId_468_; lean_object* v___f_469_; lean_object* v___x_470_; 
v___x_466_ = lean_unsigned_to_nat(0u);
v___x_467_ = lean_array_get_borrowed(v___x_458_, v_xs_459_, v___x_466_);
v_xId_468_ = l_Lean_Expr_fvarId_x21(v___x_467_);
lean_inc(v___x_467_);
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__0___boxed), 10, 3);
lean_closure_set(v___f_469_, 0, v_xId_468_);
lean_closure_set(v___f_469_, 1, v___x_466_);
lean_closure_set(v___f_469_, 2, v___x_467_);
v___x_470_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(v_b_460_, v___f_469_, v___y_461_, v___y_462_, v___y_463_, v___y_464_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1___boxed(lean_object* v___x_471_, lean_object* v_xs_472_, lean_object* v_b_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1(v___x_471_, v_xs_472_, v_b_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_);
lean_dec(v___y_477_);
lean_dec_ref(v___y_476_);
lean_dec(v___y_475_);
lean_dec_ref(v___y_474_);
lean_dec_ref(v_xs_472_);
lean_dec_ref(v___x_471_);
return v_res_479_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0(void){
_start:
{
lean_object* v___x_480_; lean_object* v___f_481_; 
v___x_480_ = l_Lean_instInhabitedExpr;
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___lam__1___boxed), 8, 1);
lean_closure_set(v___f_481_, 0, v___x_480_);
return v___f_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData(lean_object* v_f_482_, lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_){
_start:
{
lean_object* v___f_488_; uint8_t v___x_489_; lean_object* v___x_490_; 
v___f_488_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___closed__0);
v___x_489_ = 0;
v___x_490_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Mathlib_Meta_FunProp_getFunctionData_spec__4___redArg(v_f_482_, v___f_488_, v___x_489_, v_a_483_, v_a_484_, v_a_485_, v_a_486_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData___boxed(lean_object* v_f_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData(v_f_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
lean_dec(v_a_495_);
lean_dec_ref(v_a_494_);
lean_dec(v_a_493_);
lean_dec_ref(v_a_492_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0(lean_object* v_xId_498_, lean_object* v_as_499_, size_t v_sz_500_, size_t v_i_501_, lean_object* v_bs_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___redArg(v_xId_498_, v_sz_500_, v_i_501_, v_bs_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0___boxed(lean_object* v_xId_504_, lean_object* v_as_505_, lean_object* v_sz_506_, lean_object* v_i_507_, lean_object* v_bs_508_){
_start:
{
size_t v_sz_boxed_509_; size_t v_i_boxed_510_; lean_object* v_res_511_; 
v_sz_boxed_509_ = lean_unbox_usize(v_sz_506_);
lean_dec(v_sz_506_);
v_i_boxed_510_ = lean_unbox_usize(v_i_507_);
lean_dec(v_i_507_);
v_res_511_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Meta_FunProp_getFunctionData_spec__0(v_xId_504_, v_as_505_, v_sz_boxed_509_, v_i_boxed_510_, v_bs_508_);
lean_dec_ref(v_as_505_);
lean_dec(v_xId_504_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorIdx(lean_object* v_x_512_){
_start:
{
switch(lean_obj_tag(v_x_512_))
{
case 0:
{
lean_object* v___x_513_; 
v___x_513_ = lean_unsigned_to_nat(0u);
return v___x_513_;
}
case 1:
{
lean_object* v___x_514_; 
v___x_514_ = lean_unsigned_to_nat(1u);
return v___x_514_;
}
default: 
{
lean_object* v___x_515_; 
v___x_515_ = lean_unsigned_to_nat(2u);
return v___x_515_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorIdx___boxed(lean_object* v_x_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorIdx(v_x_516_);
lean_dec_ref(v_x_516_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(lean_object* v_t_518_, lean_object* v_k_519_){
_start:
{
lean_object* v_f_520_; lean_object* v___x_521_; 
v_f_520_ = lean_ctor_get(v_t_518_, 0);
lean_inc_ref(v_f_520_);
lean_dec_ref(v_t_518_);
v___x_521_ = lean_apply_1(v_k_519_, v_f_520_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim(lean_object* v_motive_522_, lean_object* v_ctorIdx_523_, lean_object* v_t_524_, lean_object* v_h_525_, lean_object* v_k_526_){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_524_, v_k_526_);
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___boxed(lean_object* v_motive_528_, lean_object* v_ctorIdx_529_, lean_object* v_t_530_, lean_object* v_h_531_, lean_object* v_k_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim(v_motive_528_, v_ctorIdx_529_, v_t_530_, v_h_531_, v_k_532_);
lean_dec(v_ctorIdx_529_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_letE_elim___redArg(lean_object* v_t_534_, lean_object* v_letE_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_534_, v_letE_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_letE_elim(lean_object* v_motive_537_, lean_object* v_t_538_, lean_object* v_h_539_, lean_object* v_letE_540_){
_start:
{
lean_object* v___x_541_; 
v___x_541_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_538_, v_letE_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_lam_elim___redArg(lean_object* v_t_542_, lean_object* v_lam_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_542_, v_lam_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_lam_elim(lean_object* v_motive_545_, lean_object* v_t_546_, lean_object* v_h_547_, lean_object* v_lam_548_){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_546_, v_lam_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_data_elim___redArg(lean_object* v_t_550_, lean_object* v_data_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_550_, v_data_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_data_elim(lean_object* v_motive_553_, lean_object* v_t_554_, lean_object* v_h_555_, lean_object* v_data_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_ctorElim___redArg(v_t_554_, v_data_556_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get(lean_object* v_fData_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_){
_start:
{
switch(lean_obj_tag(v_fData_558_))
{
case 0:
{
lean_object* v_f_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_571_; 
v_f_564_ = lean_ctor_get(v_fData_558_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v_fData_558_);
if (v_isSharedCheck_571_ == 0)
{
v___x_566_ = v_fData_558_;
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_f_564_);
lean_dec(v_fData_558_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_569_; 
if (v_isShared_567_ == 0)
{
v___x_569_ = v___x_566_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_f_564_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
case 1:
{
lean_object* v_f_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_579_; 
v_f_572_ = lean_ctor_get(v_fData_558_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v_fData_558_);
if (v_isSharedCheck_579_ == 0)
{
v___x_574_ = v_fData_558_;
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_f_572_);
lean_dec(v_fData_558_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_577_; 
if (v_isShared_575_ == 0)
{
lean_ctor_set_tag(v___x_574_, 0);
v___x_577_ = v___x_574_;
goto v_reusejp_576_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_f_572_);
v___x_577_ = v_reuseFailAlloc_578_;
goto v_reusejp_576_;
}
v_reusejp_576_:
{
return v___x_577_;
}
}
}
default: 
{
lean_object* v_fData_580_; lean_object* v___x_581_; 
v_fData_580_ = lean_ctor_get(v_fData_558_, 0);
lean_inc_ref(v_fData_580_);
lean_dec_ref_known(v_fData_558_, 1);
v___x_581_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr(v_fData_580_, v_a_559_, v_a_560_, v_a_561_, v_a_562_);
return v___x_581_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get___boxed(lean_object* v_fData_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_Mathlib_Meta_FunProp_MaybeFunctionData_get(v_fData_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_);
lean_dec(v_a_586_);
lean_dec_ref(v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_a_583_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg(lean_object* v_e_589_, lean_object* v___y_590_){
_start:
{
uint8_t v___x_592_; 
v___x_592_ = l_Lean_Expr_hasMVar(v_e_589_);
if (v___x_592_ == 0)
{
lean_object* v___x_593_; 
v___x_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_593_, 0, v_e_589_);
return v___x_593_;
}
else
{
lean_object* v___x_594_; lean_object* v_mctx_595_; lean_object* v___x_596_; lean_object* v_fst_597_; lean_object* v_snd_598_; lean_object* v___x_599_; lean_object* v_cache_600_; lean_object* v_zetaDeltaFVarIds_601_; lean_object* v_postponed_602_; lean_object* v_diag_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_612_; 
v___x_594_ = lean_st_ref_get(v___y_590_);
v_mctx_595_ = lean_ctor_get(v___x_594_, 0);
lean_inc_ref(v_mctx_595_);
lean_dec(v___x_594_);
v___x_596_ = l_Lean_instantiateMVarsCore(v_mctx_595_, v_e_589_);
v_fst_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_fst_597_);
v_snd_598_ = lean_ctor_get(v___x_596_, 1);
lean_inc(v_snd_598_);
lean_dec_ref(v___x_596_);
v___x_599_ = lean_st_ref_take(v___y_590_);
v_cache_600_ = lean_ctor_get(v___x_599_, 1);
v_zetaDeltaFVarIds_601_ = lean_ctor_get(v___x_599_, 2);
v_postponed_602_ = lean_ctor_get(v___x_599_, 3);
v_diag_603_ = lean_ctor_get(v___x_599_, 4);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_599_);
if (v_isSharedCheck_612_ == 0)
{
lean_object* v_unused_613_; 
v_unused_613_ = lean_ctor_get(v___x_599_, 0);
lean_dec(v_unused_613_);
v___x_605_ = v___x_599_;
v_isShared_606_ = v_isSharedCheck_612_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_diag_603_);
lean_inc(v_postponed_602_);
lean_inc(v_zetaDeltaFVarIds_601_);
lean_inc(v_cache_600_);
lean_dec(v___x_599_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_612_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v_snd_598_);
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_snd_598_);
lean_ctor_set(v_reuseFailAlloc_611_, 1, v_cache_600_);
lean_ctor_set(v_reuseFailAlloc_611_, 2, v_zetaDeltaFVarIds_601_);
lean_ctor_set(v_reuseFailAlloc_611_, 3, v_postponed_602_);
lean_ctor_set(v_reuseFailAlloc_611_, 4, v_diag_603_);
v___x_608_ = v_reuseFailAlloc_611_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
lean_object* v___x_609_; lean_object* v___x_610_; 
v___x_609_ = lean_st_ref_set(v___y_590_, v___x_608_);
v___x_610_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_610_, 0, v_fst_597_);
return v___x_610_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg___boxed(lean_object* v_e_614_, lean_object* v___y_615_, lean_object* v___y_616_){
_start:
{
lean_object* v_res_617_; 
v_res_617_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg(v_e_614_, v___y_615_);
lean_dec(v___y_615_);
return v_res_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1(lean_object* v_e_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg(v_e_618_, v___y_620_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___boxed(lean_object* v_e_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1(v_e_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg(lean_object* v_declName_632_, lean_object* v___y_633_){
_start:
{
lean_object* v___x_635_; lean_object* v_env_636_; uint8_t v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v___x_635_ = lean_st_ref_get(v___y_633_);
v_env_636_ = lean_ctor_get(v___x_635_, 0);
lean_inc_ref(v_env_636_);
lean_dec(v___x_635_);
v___x_637_ = l_Lean_getReducibilityStatusCore(v_env_636_, v_declName_632_);
v___x_638_ = lean_box(v___x_637_);
v___x_639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_639_, 0, v___x_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_declName_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg(v_declName_640_, v___y_641_);
lean_dec(v___y_641_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0(lean_object* v_declName_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
lean_object* v___x_650_; lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_666_; 
v___x_650_ = lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg(v_declName_644_, v___y_648_);
v_a_651_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_666_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_666_ == 0)
{
v___x_653_ = v___x_650_;
v_isShared_654_ = v_isSharedCheck_666_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_650_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_666_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
uint8_t v___x_655_; 
v___x_655_ = lean_unbox(v_a_651_);
lean_dec(v_a_651_);
if (v___x_655_ == 0)
{
uint8_t v___x_656_; lean_object* v___x_657_; lean_object* v___x_659_; 
v___x_656_ = 1;
v___x_657_ = lean_box(v___x_656_);
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_657_);
v___x_659_ = v___x_653_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v___x_657_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
else
{
uint8_t v___x_661_; lean_object* v___x_662_; lean_object* v___x_664_; 
v___x_661_ = 0;
v___x_662_ = lean_box(v___x_661_);
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_662_);
v___x_664_ = v___x_653_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v___x_662_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0___boxed(lean_object* v_declName_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0(v_declName_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
lean_dec(v___y_671_);
lean_dec_ref(v___y_670_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0(lean_object* v_unfoldPred_674_, lean_object* v_e_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_){
_start:
{
lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_681_ = l_Lean_Expr_getAppFn_x27(v_e_675_);
v___x_682_ = l_Lean_Expr_constName_x3f(v___x_681_);
lean_dec_ref(v___x_681_);
if (lean_obj_tag(v___x_682_) == 1)
{
lean_object* v_val_683_; lean_object* v___x_684_; 
v_val_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc_n(v_val_683_, 2);
lean_dec_ref_known(v___x_682_, 1);
v___x_684_ = lp_mathlib_Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0(v_val_683_, v___y_676_, v___y_677_, v___y_678_, v___y_679_);
if (lean_obj_tag(v___x_684_) == 0)
{
lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_685_ = lean_apply_1(v_unfoldPred_674_, v_val_683_);
v___x_686_ = lean_unbox(v___x_685_);
if (v___x_686_ == 0)
{
return v___x_684_;
}
else
{
lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_693_; 
v_isSharedCheck_693_ = !lean_is_exclusive(v___x_684_);
if (v_isSharedCheck_693_ == 0)
{
lean_object* v_unused_694_; 
v_unused_694_ = lean_ctor_get(v___x_684_, 0);
lean_dec(v_unused_694_);
v___x_688_ = v___x_684_;
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
else
{
lean_dec(v___x_684_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_691_; 
if (v_isShared_689_ == 0)
{
lean_ctor_set(v___x_688_, 0, v___x_685_);
v___x_691_ = v___x_688_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v___x_685_);
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
else
{
lean_dec(v_val_683_);
lean_dec_ref(v_unfoldPred_674_);
return v___x_684_;
}
}
else
{
uint8_t v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
lean_dec(v___x_682_);
lean_dec_ref(v_unfoldPred_674_);
v___x_695_ = 0;
v___x_696_ = lean_box(v___x_695_);
v___x_697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_697_, 0, v___x_696_);
return v___x_697_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0___boxed(lean_object* v_unfoldPred_698_, lean_object* v_e_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0(v_unfoldPred_698_, v_e_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_);
lean_dec(v___y_703_);
lean_dec_ref(v___y_702_);
lean_dec(v___y_701_);
lean_dec_ref(v___y_700_);
lean_dec_ref(v_e_699_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1(lean_object* v_f_706_, lean_object* v_unfold_707_, lean_object* v_x_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_714_ = lean_unsigned_to_nat(1u);
v___x_715_ = lean_mk_empty_array_with_capacity(v___x_714_);
v___x_716_ = lean_array_push(v___x_715_, v_x_708_);
lean_inc_ref(v___x_716_);
v___x_717_ = l_Lean_Expr_beta(v_f_706_, v___x_716_);
v___x_718_ = l_Lean_Expr_eta(v___x_717_);
v___x_719_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(v___x_718_, v_unfold_707_, v___y_709_, v___y_710_, v___y_711_, v___y_712_);
if (lean_obj_tag(v___x_719_) == 0)
{
lean_object* v_a_720_; lean_object* v___x_721_; uint8_t v___x_722_; uint8_t v___x_723_; uint8_t v___x_724_; lean_object* v___x_725_; 
v_a_720_ = lean_ctor_get(v___x_719_, 0);
lean_inc(v_a_720_);
lean_dec_ref_known(v___x_719_, 1);
v___x_721_ = lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet(v_a_720_);
v___x_722_ = 0;
v___x_723_ = 1;
v___x_724_ = 1;
lean_inc_ref(v___x_721_);
v___x_725_ = l_Lean_Meta_mkLambdaFVars(v___x_716_, v___x_721_, v___x_722_, v___x_723_, v___x_722_, v___x_723_, v___x_724_, v___y_709_, v___y_710_, v___y_711_, v___y_712_);
lean_dec_ref(v___x_716_);
if (lean_obj_tag(v___x_725_) == 0)
{
switch(lean_obj_tag(v___x_721_))
{
case 8:
{
lean_object* v_a_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_734_; 
lean_dec_ref_known(v___x_721_, 4);
v_a_726_ = lean_ctor_get(v___x_725_, 0);
v_isSharedCheck_734_ = !lean_is_exclusive(v___x_725_);
if (v_isSharedCheck_734_ == 0)
{
v___x_728_ = v___x_725_;
v_isShared_729_ = v_isSharedCheck_734_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_a_726_);
lean_dec(v___x_725_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_734_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v___x_730_; lean_object* v___x_732_; 
v___x_730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_730_, 0, v_a_726_);
if (v_isShared_729_ == 0)
{
lean_ctor_set(v___x_728_, 0, v___x_730_);
v___x_732_ = v___x_728_;
goto v_reusejp_731_;
}
else
{
lean_object* v_reuseFailAlloc_733_; 
v_reuseFailAlloc_733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_733_, 0, v___x_730_);
v___x_732_ = v_reuseFailAlloc_733_;
goto v_reusejp_731_;
}
v_reusejp_731_:
{
return v___x_732_;
}
}
}
case 6:
{
lean_object* v_a_735_; lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_743_; 
lean_dec_ref_known(v___x_721_, 3);
v_a_735_ = lean_ctor_get(v___x_725_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_725_);
if (v_isSharedCheck_743_ == 0)
{
v___x_737_ = v___x_725_;
v_isShared_738_ = v_isSharedCheck_743_;
goto v_resetjp_736_;
}
else
{
lean_inc(v_a_735_);
lean_dec(v___x_725_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_743_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
lean_object* v___x_739_; lean_object* v___x_741_; 
v___x_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_739_, 0, v_a_735_);
if (v_isShared_738_ == 0)
{
lean_ctor_set(v___x_737_, 0, v___x_739_);
v___x_741_ = v___x_737_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v___x_739_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
default: 
{
lean_object* v_a_744_; lean_object* v___x_745_; 
lean_dec_ref(v___x_721_);
v_a_744_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_a_744_);
lean_dec_ref_known(v___x_725_, 1);
v___x_745_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData(v_a_744_, v___y_709_, v___y_710_, v___y_711_, v___y_712_);
if (lean_obj_tag(v___x_745_) == 0)
{
lean_object* v_a_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_754_; 
v_a_746_ = lean_ctor_get(v___x_745_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_745_);
if (v_isSharedCheck_754_ == 0)
{
v___x_748_ = v___x_745_;
v_isShared_749_ = v_isSharedCheck_754_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_a_746_);
lean_dec(v___x_745_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_754_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_750_; lean_object* v___x_752_; 
v___x_750_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_750_, 0, v_a_746_);
if (v_isShared_749_ == 0)
{
lean_ctor_set(v___x_748_, 0, v___x_750_);
v___x_752_ = v___x_748_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v___x_750_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
else
{
lean_object* v_a_755_; lean_object* v___x_757_; uint8_t v_isShared_758_; uint8_t v_isSharedCheck_762_; 
v_a_755_ = lean_ctor_get(v___x_745_, 0);
v_isSharedCheck_762_ = !lean_is_exclusive(v___x_745_);
if (v_isSharedCheck_762_ == 0)
{
v___x_757_ = v___x_745_;
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
else
{
lean_inc(v_a_755_);
lean_dec(v___x_745_);
v___x_757_ = lean_box(0);
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
v_resetjp_756_:
{
lean_object* v___x_760_; 
if (v_isShared_758_ == 0)
{
v___x_760_ = v___x_757_;
goto v_reusejp_759_;
}
else
{
lean_object* v_reuseFailAlloc_761_; 
v_reuseFailAlloc_761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_761_, 0, v_a_755_);
v___x_760_ = v_reuseFailAlloc_761_;
goto v_reusejp_759_;
}
v_reusejp_759_:
{
return v___x_760_;
}
}
}
}
}
}
else
{
lean_object* v_a_763_; lean_object* v___x_765_; uint8_t v_isShared_766_; uint8_t v_isSharedCheck_770_; 
lean_dec_ref(v___x_721_);
v_a_763_ = lean_ctor_get(v___x_725_, 0);
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_725_);
if (v_isSharedCheck_770_ == 0)
{
v___x_765_ = v___x_725_;
v_isShared_766_ = v_isSharedCheck_770_;
goto v_resetjp_764_;
}
else
{
lean_inc(v_a_763_);
lean_dec(v___x_725_);
v___x_765_ = lean_box(0);
v_isShared_766_ = v_isSharedCheck_770_;
goto v_resetjp_764_;
}
v_resetjp_764_:
{
lean_object* v___x_768_; 
if (v_isShared_766_ == 0)
{
v___x_768_ = v___x_765_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_a_763_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
}
}
else
{
lean_object* v_a_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_778_; 
lean_dec_ref(v___x_716_);
v_a_771_ = lean_ctor_get(v___x_719_, 0);
v_isSharedCheck_778_ = !lean_is_exclusive(v___x_719_);
if (v_isSharedCheck_778_ == 0)
{
v___x_773_ = v___x_719_;
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_a_771_);
lean_dec(v___x_719_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_776_; 
if (v_isShared_774_ == 0)
{
v___x_776_ = v___x_773_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v_a_771_);
v___x_776_ = v_reuseFailAlloc_777_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
return v___x_776_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1___boxed(lean_object* v_f_779_, lean_object* v_unfold_780_, lean_object* v_x_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1(v_f_779_, v_unfold_780_, v_x_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
lean_dec(v___y_783_);
lean_dec_ref(v___y_782_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5(lean_object* v_msgData_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_){
_start:
{
lean_object* v___x_794_; lean_object* v_env_795_; lean_object* v___x_796_; lean_object* v_mctx_797_; lean_object* v_lctx_798_; lean_object* v_options_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_794_ = lean_st_ref_get(v___y_792_);
v_env_795_ = lean_ctor_get(v___x_794_, 0);
lean_inc_ref(v_env_795_);
lean_dec(v___x_794_);
v___x_796_ = lean_st_ref_get(v___y_790_);
v_mctx_797_ = lean_ctor_get(v___x_796_, 0);
lean_inc_ref(v_mctx_797_);
lean_dec(v___x_796_);
v_lctx_798_ = lean_ctor_get(v___y_789_, 2);
v_options_799_ = lean_ctor_get(v___y_791_, 2);
lean_inc_ref(v_options_799_);
lean_inc_ref(v_lctx_798_);
v___x_800_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_800_, 0, v_env_795_);
lean_ctor_set(v___x_800_, 1, v_mctx_797_);
lean_ctor_set(v___x_800_, 2, v_lctx_798_);
lean_ctor_set(v___x_800_, 3, v_options_799_);
v___x_801_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
lean_ctor_set(v___x_801_, 1, v_msgData_788_);
v___x_802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_802_, 0, v___x_801_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5___boxed(lean_object* v_msgData_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5(v_msgData_803_, v___y_804_, v___y_805_, v___y_806_, v___y_807_);
lean_dec(v___y_807_);
lean_dec_ref(v___y_806_);
lean_dec(v___y_805_);
lean_dec_ref(v___y_804_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(lean_object* v_msg_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
lean_object* v_ref_816_; lean_object* v___x_817_; lean_object* v_a_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_826_; 
v_ref_816_ = lean_ctor_get(v___y_813_, 5);
v___x_817_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3_spec__5(v_msg_810_, v___y_811_, v___y_812_, v___y_813_, v___y_814_);
v_a_818_ = lean_ctor_get(v___x_817_, 0);
v_isSharedCheck_826_ = !lean_is_exclusive(v___x_817_);
if (v_isSharedCheck_826_ == 0)
{
v___x_820_ = v___x_817_;
v_isShared_821_ = v_isSharedCheck_826_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_a_818_);
lean_dec(v___x_817_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_826_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_822_; lean_object* v___x_824_; 
lean_inc(v_ref_816_);
v___x_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_822_, 0, v_ref_816_);
lean_ctor_set(v___x_822_, 1, v_a_818_);
if (v_isShared_821_ == 0)
{
lean_ctor_set_tag(v___x_820_, 1);
lean_ctor_set(v___x_820_, 0, v___x_822_);
v___x_824_ = v___x_820_;
goto v_reusejp_823_;
}
else
{
lean_object* v_reuseFailAlloc_825_; 
v_reuseFailAlloc_825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_825_, 0, v___x_822_);
v___x_824_ = v_reuseFailAlloc_825_;
goto v_reusejp_823_;
}
v_reusejp_823_:
{
return v___x_824_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg___boxed(lean_object* v_msg_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_){
_start:
{
lean_object* v_res_833_; 
v_res_833_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(v_msg_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec(v___y_829_);
lean_dec_ref(v___y_828_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0(lean_object* v_k_834_, lean_object* v_b_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v___x_841_; 
lean_inc(v___y_839_);
lean_inc_ref(v___y_838_);
lean_inc(v___y_837_);
lean_inc_ref(v___y_836_);
v___x_841_ = lean_apply_6(v_k_834_, v_b_835_, v___y_836_, v___y_837_, v___y_838_, v___y_839_, lean_box(0));
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v_k_842_, lean_object* v_b_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0(v_k_842_, v_b_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___y_845_);
lean_dec_ref(v___y_844_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg(lean_object* v_name_850_, uint8_t v_bi_851_, lean_object* v_type_852_, lean_object* v_k_853_, uint8_t v_kind_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_){
_start:
{
lean_object* v___f_860_; lean_object* v___x_861_; 
v___f_860_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_860_, 0, v_k_853_);
v___x_861_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_850_, v_bi_851_, v_type_852_, v___f_860_, v_kind_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
if (lean_obj_tag(v___x_861_) == 0)
{
lean_object* v_a_862_; lean_object* v___x_864_; uint8_t v_isShared_865_; uint8_t v_isSharedCheck_869_; 
v_a_862_ = lean_ctor_get(v___x_861_, 0);
v_isSharedCheck_869_ = !lean_is_exclusive(v___x_861_);
if (v_isSharedCheck_869_ == 0)
{
v___x_864_ = v___x_861_;
v_isShared_865_ = v_isSharedCheck_869_;
goto v_resetjp_863_;
}
else
{
lean_inc(v_a_862_);
lean_dec(v___x_861_);
v___x_864_ = lean_box(0);
v_isShared_865_ = v_isSharedCheck_869_;
goto v_resetjp_863_;
}
v_resetjp_863_:
{
lean_object* v___x_867_; 
if (v_isShared_865_ == 0)
{
v___x_867_ = v___x_864_;
goto v_reusejp_866_;
}
else
{
lean_object* v_reuseFailAlloc_868_; 
v_reuseFailAlloc_868_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_868_, 0, v_a_862_);
v___x_867_ = v_reuseFailAlloc_868_;
goto v_reusejp_866_;
}
v_reusejp_866_:
{
return v___x_867_;
}
}
}
else
{
lean_object* v_a_870_; lean_object* v___x_872_; uint8_t v_isShared_873_; uint8_t v_isSharedCheck_877_; 
v_a_870_ = lean_ctor_get(v___x_861_, 0);
v_isSharedCheck_877_ = !lean_is_exclusive(v___x_861_);
if (v_isSharedCheck_877_ == 0)
{
v___x_872_ = v___x_861_;
v_isShared_873_ = v_isSharedCheck_877_;
goto v_resetjp_871_;
}
else
{
lean_inc(v_a_870_);
lean_dec(v___x_861_);
v___x_872_ = lean_box(0);
v_isShared_873_ = v_isSharedCheck_877_;
goto v_resetjp_871_;
}
v_resetjp_871_:
{
lean_object* v___x_875_; 
if (v_isShared_873_ == 0)
{
v___x_875_ = v___x_872_;
goto v_reusejp_874_;
}
else
{
lean_object* v_reuseFailAlloc_876_; 
v_reuseFailAlloc_876_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_876_, 0, v_a_870_);
v___x_875_ = v_reuseFailAlloc_876_;
goto v_reusejp_874_;
}
v_reusejp_874_:
{
return v___x_875_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg___boxed(lean_object* v_name_878_, lean_object* v_bi_879_, lean_object* v_type_880_, lean_object* v_k_881_, lean_object* v_kind_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_){
_start:
{
uint8_t v_bi_boxed_888_; uint8_t v_kind_boxed_889_; lean_object* v_res_890_; 
v_bi_boxed_888_ = lean_unbox(v_bi_879_);
v_kind_boxed_889_ = lean_unbox(v_kind_882_);
v_res_890_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg(v_name_878_, v_bi_boxed_888_, v_type_880_, v_k_881_, v_kind_boxed_889_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
lean_dec(v___y_886_);
lean_dec_ref(v___y_885_);
lean_dec(v___y_884_);
lean_dec_ref(v___y_883_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(lean_object* v_name_891_, lean_object* v_type_892_, lean_object* v_k_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
uint8_t v___x_899_; uint8_t v___x_900_; lean_object* v___x_901_; 
v___x_899_ = 0;
v___x_900_ = 0;
v___x_901_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg(v_name_891_, v___x_899_, v_type_892_, v_k_893_, v___x_900_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg___boxed(lean_object* v_name_902_, lean_object* v_type_903_, lean_object* v_k_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(v_name_902_, v_type_903_, v_k_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
return v_res_910_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1(void){
_start:
{
lean_object* v___x_912_; lean_object* v___x_913_; 
v___x_912_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__0));
v___x_913_ = l_Lean_stringToMessageData(v___x_912_);
return v___x_913_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3(void){
_start:
{
lean_object* v___x_915_; lean_object* v___x_916_; 
v___x_915_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__2));
v___x_916_ = l_Lean_stringToMessageData(v___x_915_);
return v___x_916_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5(void){
_start:
{
lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__4));
v___x_919_ = l_Lean_stringToMessageData(v___x_918_);
return v___x_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f(lean_object* v_f_920_, lean_object* v_unfoldPred_921_, lean_object* v_a_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_){
_start:
{
lean_object* v___x_927_; uint8_t v_foApprox_928_; uint8_t v_ctxApprox_929_; uint8_t v_quasiPatternApprox_930_; uint8_t v_constApprox_931_; uint8_t v_isDefEqStuckEx_932_; uint8_t v_unificationHints_933_; uint8_t v_proofIrrelevance_934_; uint8_t v_assignSyntheticOpaque_935_; uint8_t v_offsetCnstrs_936_; uint8_t v_transparency_937_; uint8_t v_etaStruct_938_; uint8_t v_univApprox_939_; uint8_t v_iota_940_; uint8_t v_beta_941_; uint8_t v_proj_942_; uint8_t v_zetaUnused_943_; uint8_t v_zetaHave_944_; uint8_t v_canUnfoldPredicateConfig_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_1016_; 
v___x_927_ = l_Lean_Meta_Context_config(v_a_922_);
v_foApprox_928_ = lean_ctor_get_uint8(v___x_927_, 0);
v_ctxApprox_929_ = lean_ctor_get_uint8(v___x_927_, 1);
v_quasiPatternApprox_930_ = lean_ctor_get_uint8(v___x_927_, 2);
v_constApprox_931_ = lean_ctor_get_uint8(v___x_927_, 3);
v_isDefEqStuckEx_932_ = lean_ctor_get_uint8(v___x_927_, 4);
v_unificationHints_933_ = lean_ctor_get_uint8(v___x_927_, 5);
v_proofIrrelevance_934_ = lean_ctor_get_uint8(v___x_927_, 6);
v_assignSyntheticOpaque_935_ = lean_ctor_get_uint8(v___x_927_, 7);
v_offsetCnstrs_936_ = lean_ctor_get_uint8(v___x_927_, 8);
v_transparency_937_ = lean_ctor_get_uint8(v___x_927_, 9);
v_etaStruct_938_ = lean_ctor_get_uint8(v___x_927_, 10);
v_univApprox_939_ = lean_ctor_get_uint8(v___x_927_, 11);
v_iota_940_ = lean_ctor_get_uint8(v___x_927_, 12);
v_beta_941_ = lean_ctor_get_uint8(v___x_927_, 13);
v_proj_942_ = lean_ctor_get_uint8(v___x_927_, 14);
v_zetaUnused_943_ = lean_ctor_get_uint8(v___x_927_, 17);
v_zetaHave_944_ = lean_ctor_get_uint8(v___x_927_, 18);
v_canUnfoldPredicateConfig_945_ = lean_ctor_get_uint8(v___x_927_, 19);
v_isSharedCheck_1016_ = !lean_is_exclusive(v___x_927_);
if (v_isSharedCheck_1016_ == 0)
{
v___x_947_ = v___x_927_;
v_isShared_948_ = v_isSharedCheck_1016_;
goto v_resetjp_946_;
}
else
{
lean_dec(v___x_927_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_1016_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
uint8_t v___x_949_; lean_object* v___x_951_; 
v___x_949_ = 0;
if (v_isShared_948_ == 0)
{
v___x_951_ = v___x_947_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 0, v_foApprox_928_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 1, v_ctxApprox_929_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 2, v_quasiPatternApprox_930_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 3, v_constApprox_931_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 4, v_isDefEqStuckEx_932_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 5, v_unificationHints_933_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 6, v_proofIrrelevance_934_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 7, v_assignSyntheticOpaque_935_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 8, v_offsetCnstrs_936_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 9, v_transparency_937_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 10, v_etaStruct_938_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 11, v_univApprox_939_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 12, v_iota_940_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 13, v_beta_941_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 14, v_proj_942_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 17, v_zetaUnused_943_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 18, v_zetaHave_944_);
lean_ctor_set_uint8(v_reuseFailAlloc_1015_, 19, v_canUnfoldPredicateConfig_945_);
v___x_951_ = v_reuseFailAlloc_1015_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
uint8_t v_trackZetaDelta_952_; lean_object* v_zetaDeltaSet_953_; lean_object* v_lctx_954_; lean_object* v_localInstances_955_; lean_object* v_defEqCtx_x3f_956_; lean_object* v_synthPendingDepth_957_; lean_object* v_customCanUnfoldPredicate_x3f_958_; uint8_t v_univApprox_959_; uint8_t v_inTypeClassResolution_960_; uint8_t v_cacheInferType_961_; uint64_t v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
lean_ctor_set_uint8(v___x_951_, 15, v___x_949_);
lean_ctor_set_uint8(v___x_951_, 16, v___x_949_);
v_trackZetaDelta_952_ = lean_ctor_get_uint8(v_a_922_, sizeof(void*)*7);
v_zetaDeltaSet_953_ = lean_ctor_get(v_a_922_, 1);
v_lctx_954_ = lean_ctor_get(v_a_922_, 2);
v_localInstances_955_ = lean_ctor_get(v_a_922_, 3);
v_defEqCtx_x3f_956_ = lean_ctor_get(v_a_922_, 4);
v_synthPendingDepth_957_ = lean_ctor_get(v_a_922_, 5);
v_customCanUnfoldPredicate_x3f_958_ = lean_ctor_get(v_a_922_, 6);
v_univApprox_959_ = lean_ctor_get_uint8(v_a_922_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_960_ = lean_ctor_get_uint8(v_a_922_, sizeof(void*)*7 + 2);
v_cacheInferType_961_ = lean_ctor_get_uint8(v_a_922_, sizeof(void*)*7 + 3);
v___x_962_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_951_);
v___x_963_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_963_, 0, v___x_951_);
lean_ctor_set_uint64(v___x_963_, sizeof(void*)*1, v___x_962_);
lean_inc(v_customCanUnfoldPredicate_x3f_958_);
lean_inc(v_synthPendingDepth_957_);
lean_inc(v_defEqCtx_x3f_956_);
lean_inc_ref(v_localInstances_955_);
lean_inc_ref(v_lctx_954_);
lean_inc(v_zetaDeltaSet_953_);
v___x_964_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_964_, 0, v___x_963_);
lean_ctor_set(v___x_964_, 1, v_zetaDeltaSet_953_);
lean_ctor_set(v___x_964_, 2, v_lctx_954_);
lean_ctor_set(v___x_964_, 3, v_localInstances_955_);
lean_ctor_set(v___x_964_, 4, v_defEqCtx_x3f_956_);
lean_ctor_set(v___x_964_, 5, v_synthPendingDepth_957_);
lean_ctor_set(v___x_964_, 6, v_customCanUnfoldPredicate_x3f_958_);
lean_ctor_set_uint8(v___x_964_, sizeof(void*)*7, v_trackZetaDelta_952_);
lean_ctor_set_uint8(v___x_964_, sizeof(void*)*7 + 1, v_univApprox_959_);
lean_ctor_set_uint8(v___x_964_, sizeof(void*)*7 + 2, v_inTypeClassResolution_960_);
lean_ctor_set_uint8(v___x_964_, sizeof(void*)*7 + 3, v_cacheInferType_961_);
lean_inc(v_a_925_);
lean_inc_ref(v_a_924_);
lean_inc(v_a_923_);
lean_inc_ref(v___x_964_);
lean_inc_ref(v_f_920_);
v___x_965_ = lean_infer_type(v_f_920_, v___x_964_, v_a_923_, v_a_924_, v_a_925_);
if (lean_obj_tag(v___x_965_) == 0)
{
lean_object* v_a_966_; lean_object* v___x_967_; lean_object* v_a_968_; 
v_a_966_ = lean_ctor_get(v___x_965_, 0);
lean_inc(v_a_966_);
lean_dec_ref_known(v___x_965_, 1);
v___x_967_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__1___redArg(v_a_966_, v_a_923_);
v_a_968_ = lean_ctor_get(v___x_967_, 0);
lean_inc(v_a_968_);
lean_dec_ref(v___x_967_);
if (lean_obj_tag(v_a_968_) == 7)
{
lean_object* v_binderName_969_; lean_object* v_binderType_970_; lean_object* v_unfold_971_; lean_object* v___f_972_; lean_object* v___x_973_; 
v_binderName_969_ = lean_ctor_get(v_a_968_, 0);
lean_inc(v_binderName_969_);
v_binderType_970_ = lean_ctor_get(v_a_968_, 1);
lean_inc_ref(v_binderType_970_);
lean_dec_ref_known(v_a_968_, 3);
v_unfold_971_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__0___boxed), 7, 1);
lean_closure_set(v_unfold_971_, 0, v_unfoldPred_921_);
v___f_972_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___lam__1___boxed), 8, 2);
lean_closure_set(v___f_972_, 0, v_f_920_);
lean_closure_set(v___f_972_, 1, v_unfold_971_);
v___x_973_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(v_binderName_969_, v_binderType_970_, v___f_972_, v___x_964_, v_a_923_, v_a_924_, v_a_925_);
lean_dec_ref_known(v___x_964_, 7);
return v___x_973_;
}
else
{
lean_object* v___x_974_; 
lean_dec(v_a_968_);
lean_dec_ref(v_unfoldPred_921_);
lean_inc(v_a_925_);
lean_inc_ref(v_a_924_);
lean_inc(v_a_923_);
lean_inc_ref(v___x_964_);
lean_inc_ref(v_f_920_);
v___x_974_ = lean_infer_type(v_f_920_, v___x_964_, v_a_923_, v_a_924_, v_a_925_);
if (lean_obj_tag(v___x_974_) == 0)
{
lean_object* v_a_975_; lean_object* v___x_976_; 
v_a_975_ = lean_ctor_get(v___x_974_, 0);
lean_inc(v_a_975_);
lean_dec_ref_known(v___x_974_, 1);
lean_inc(v_a_925_);
lean_inc_ref(v_a_924_);
lean_inc(v_a_923_);
lean_inc_ref(v___x_964_);
lean_inc_ref(v_f_920_);
v___x_976_ = lean_infer_type(v_f_920_, v___x_964_, v_a_923_, v_a_924_, v_a_925_);
if (lean_obj_tag(v___x_976_) == 0)
{
lean_object* v_a_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; 
v_a_977_ = lean_ctor_get(v___x_976_, 0);
lean_inc(v_a_977_);
lean_dec_ref_known(v___x_976_, 1);
v___x_978_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__1);
v___x_979_ = l_Lean_MessageData_ofExpr(v_f_920_);
v___x_980_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_980_, 0, v___x_978_);
lean_ctor_set(v___x_980_, 1, v___x_979_);
v___x_981_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__3);
v___x_982_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_982_, 0, v___x_980_);
lean_ctor_set(v___x_982_, 1, v___x_981_);
v___x_983_ = l_Lean_MessageData_ofExpr(v_a_975_);
v___x_984_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_984_, 0, v___x_982_);
lean_ctor_set(v___x_984_, 1, v___x_983_);
v___x_985_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5, &lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5_once, _init_lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___closed__5);
v___x_986_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_986_, 0, v___x_984_);
lean_ctor_set(v___x_986_, 1, v___x_985_);
v___x_987_ = l_Lean_Expr_ctorName(v_a_977_);
lean_dec(v_a_977_);
v___x_988_ = l_Lean_stringToMessageData(v___x_987_);
v___x_989_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_989_, 0, v___x_986_);
lean_ctor_set(v___x_989_, 1, v___x_988_);
v___x_990_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(v___x_989_, v___x_964_, v_a_923_, v_a_924_, v_a_925_);
lean_dec_ref_known(v___x_964_, 7);
return v___x_990_;
}
else
{
lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
lean_dec(v_a_975_);
lean_dec_ref_known(v___x_964_, 7);
lean_dec_ref(v_f_920_);
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
else
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
lean_dec_ref_known(v___x_964_, 7);
lean_dec_ref(v_f_920_);
v_a_999_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_1001_ = v___x_974_;
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v___x_974_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
lean_object* v___x_1004_; 
if (v_isShared_1002_ == 0)
{
v___x_1004_ = v___x_1001_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v_a_999_);
v___x_1004_ = v_reuseFailAlloc_1005_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
return v___x_1004_;
}
}
}
}
}
else
{
lean_object* v_a_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1014_; 
lean_dec_ref_known(v___x_964_, 7);
lean_dec_ref(v_unfoldPred_921_);
lean_dec_ref(v_f_920_);
v_a_1007_ = lean_ctor_get(v___x_965_, 0);
v_isSharedCheck_1014_ = !lean_is_exclusive(v___x_965_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_1009_ = v___x_965_;
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_a_1007_);
lean_dec(v___x_965_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1012_; 
if (v_isShared_1010_ == 0)
{
v___x_1012_ = v___x_1009_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1013_; 
v_reuseFailAlloc_1013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1013_, 0, v_a_1007_);
v___x_1012_ = v_reuseFailAlloc_1013_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
return v___x_1012_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f___boxed(lean_object* v_f_1017_, lean_object* v_unfoldPred_1018_, lean_object* v_a_1019_, lean_object* v_a_1020_, lean_object* v_a_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_){
_start:
{
lean_object* v_res_1024_; 
v_res_1024_ = lp_mathlib_Mathlib_Meta_FunProp_getFunctionData_x3f(v_f_1017_, v_unfoldPred_1018_, v_a_1019_, v_a_1020_, v_a_1021_, v_a_1022_);
lean_dec(v_a_1022_);
lean_dec_ref(v_a_1021_);
lean_dec(v_a_1020_);
lean_dec_ref(v_a_1019_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0(lean_object* v_declName_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_){
_start:
{
lean_object* v___x_1031_; 
v___x_1031_ = lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___redArg(v_declName_1025_, v___y_1029_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0___boxed(lean_object* v_declName_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_){
_start:
{
lean_object* v_res_1038_; 
v_res_1038_ = lp_mathlib_Lean_getReducibilityStatus___at___00Lean_isReducible___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__0_spec__0(v_declName_1032_, v___y_1033_, v___y_1034_, v___y_1035_, v___y_1036_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
lean_dec(v___y_1034_);
lean_dec_ref(v___y_1033_);
return v_res_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3(lean_object* v_00_u03b1_1039_, lean_object* v_name_1040_, uint8_t v_bi_1041_, lean_object* v_type_1042_, lean_object* v_k_1043_, uint8_t v_kind_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_){
_start:
{
lean_object* v___x_1050_; 
v___x_1050_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___redArg(v_name_1040_, v_bi_1041_, v_type_1042_, v_k_1043_, v_kind_1044_, v___y_1045_, v___y_1046_, v___y_1047_, v___y_1048_);
return v___x_1050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1051_, lean_object* v_name_1052_, lean_object* v_bi_1053_, lean_object* v_type_1054_, lean_object* v_k_1055_, lean_object* v_kind_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
uint8_t v_bi_boxed_1062_; uint8_t v_kind_boxed_1063_; lean_object* v_res_1064_; 
v_bi_boxed_1062_ = lean_unbox(v_bi_1053_);
v_kind_boxed_1063_ = lean_unbox(v_kind_1056_);
v_res_1064_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2_spec__3(v_00_u03b1_1051_, v_name_1052_, v_bi_boxed_1062_, v_type_1054_, v_k_1055_, v_kind_boxed_1063_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
lean_dec(v___y_1060_);
lean_dec_ref(v___y_1059_);
lean_dec(v___y_1058_);
lean_dec_ref(v___y_1057_);
return v_res_1064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2(lean_object* v_00_u03b1_1065_, lean_object* v_name_1066_, lean_object* v_type_1067_, lean_object* v_k_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v___x_1074_; 
v___x_1074_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(v_name_1066_, v_type_1067_, v_k_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___boxed(lean_object* v_00_u03b1_1075_, lean_object* v_name_1076_, lean_object* v_type_1077_, lean_object* v_k_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_){
_start:
{
lean_object* v_res_1084_; 
v_res_1084_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2(v_00_u03b1_1075_, v_name_1076_, v_type_1077_, v_k_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
lean_dec(v___y_1082_);
lean_dec_ref(v___y_1081_);
lean_dec(v___y_1080_);
lean_dec_ref(v___y_1079_);
return v_res_1084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3(lean_object* v_00_u03b1_1085_, lean_object* v_msg_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(v_msg_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___boxed(lean_object* v_00_u03b1_1093_, lean_object* v_msg_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3(v_00_u03b1_1093_, v_msg_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
lean_dec(v___y_1098_);
lean_dec_ref(v___y_1097_);
lean_dec(v___y_1096_);
lean_dec_ref(v___y_1095_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_unfoldHeadFVar_x3f(lean_object* v_fData_1101_, lean_object* v_a_1102_, lean_object* v_a_1103_, lean_object* v_a_1104_, lean_object* v_a_1105_){
_start:
{
lean_object* v_fn_1107_; 
v_fn_1107_ = lean_ctor_get(v_fData_1101_, 2);
lean_inc_ref(v_fn_1107_);
if (lean_obj_tag(v_fn_1107_) == 1)
{
lean_object* v_lctx_1108_; lean_object* v_insts_1109_; lean_object* v_args_1110_; lean_object* v_mainVar_1111_; lean_object* v_fvarId_1112_; uint8_t v___x_1113_; lean_object* v___x_1114_; 
v_lctx_1108_ = lean_ctor_get(v_fData_1101_, 0);
lean_inc_ref(v_lctx_1108_);
v_insts_1109_ = lean_ctor_get(v_fData_1101_, 1);
lean_inc_ref(v_insts_1109_);
v_args_1110_ = lean_ctor_get(v_fData_1101_, 3);
lean_inc_ref(v_args_1110_);
v_mainVar_1111_ = lean_ctor_get(v_fData_1101_, 4);
lean_inc_ref(v_mainVar_1111_);
lean_dec_ref(v_fData_1101_);
v_fvarId_1112_ = lean_ctor_get(v_fn_1107_, 0);
lean_inc(v_fvarId_1112_);
lean_dec_ref_known(v_fn_1107_, 1);
v___x_1113_ = 0;
v___x_1114_ = l_Lean_FVarId_getValue_x3f___redArg(v_fvarId_1112_, v___x_1113_, v_a_1102_, v_a_1104_, v_a_1105_);
if (lean_obj_tag(v___x_1114_) == 0)
{
lean_object* v_a_1115_; lean_object* v___x_1117_; uint8_t v_isShared_1118_; uint8_t v_isSharedCheck_1161_; 
v_a_1115_ = lean_ctor_get(v___x_1114_, 0);
v_isSharedCheck_1161_ = !lean_is_exclusive(v___x_1114_);
if (v_isSharedCheck_1161_ == 0)
{
v___x_1117_ = v___x_1114_;
v_isShared_1118_ = v_isSharedCheck_1161_;
goto v_resetjp_1116_;
}
else
{
lean_inc(v_a_1115_);
lean_dec(v___x_1114_);
v___x_1117_ = lean_box(0);
v_isShared_1118_ = v_isSharedCheck_1161_;
goto v_resetjp_1116_;
}
v_resetjp_1116_:
{
if (lean_obj_tag(v_a_1115_) == 1)
{
lean_object* v_val_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1156_; 
lean_del_object(v___x_1117_);
v_val_1119_ = lean_ctor_get(v_a_1115_, 0);
v_isSharedCheck_1156_ = !lean_is_exclusive(v_a_1115_);
if (v_isSharedCheck_1156_ == 0)
{
v___x_1121_ = v_a_1115_;
v_isShared_1122_ = v_isSharedCheck_1156_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_val_1119_);
lean_dec(v_a_1115_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1156_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; uint8_t v___x_1128_; uint8_t v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1123_ = lean_unsigned_to_nat(1u);
v___x_1124_ = lean_mk_empty_array_with_capacity(v___x_1123_);
v___x_1125_ = lean_array_push(v___x_1124_, v_mainVar_1111_);
v___x_1126_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_val_1119_, v_args_1110_);
lean_dec_ref(v_args_1110_);
v___x_1127_ = lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet(v___x_1126_);
v___x_1128_ = 1;
v___x_1129_ = 1;
v___x_1130_ = lean_box(v___x_1113_);
v___x_1131_ = lean_box(v___x_1128_);
v___x_1132_ = lean_box(v___x_1113_);
v___x_1133_ = lean_box(v___x_1128_);
v___x_1134_ = lean_box(v___x_1129_);
v___x_1135_ = lean_alloc_closure((void*)(l_Lean_Meta_mkLambdaFVars___boxed), 12, 7);
lean_closure_set(v___x_1135_, 0, v___x_1125_);
lean_closure_set(v___x_1135_, 1, v___x_1127_);
lean_closure_set(v___x_1135_, 2, v___x_1130_);
lean_closure_set(v___x_1135_, 3, v___x_1131_);
lean_closure_set(v___x_1135_, 4, v___x_1132_);
lean_closure_set(v___x_1135_, 5, v___x_1133_);
lean_closure_set(v___x_1135_, 6, v___x_1134_);
v___x_1136_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_1108_, v_insts_1109_, v___x_1135_, v_a_1102_, v_a_1103_, v_a_1104_, v_a_1105_);
if (lean_obj_tag(v___x_1136_) == 0)
{
lean_object* v_a_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1147_; 
v_a_1137_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1139_ = v___x_1136_;
v_isShared_1140_ = v_isSharedCheck_1147_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_a_1137_);
lean_dec(v___x_1136_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1147_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1142_; 
if (v_isShared_1122_ == 0)
{
lean_ctor_set(v___x_1121_, 0, v_a_1137_);
v___x_1142_ = v___x_1121_;
goto v_reusejp_1141_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_a_1137_);
v___x_1142_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1141_;
}
v_reusejp_1141_:
{
lean_object* v___x_1144_; 
if (v_isShared_1140_ == 0)
{
lean_ctor_set(v___x_1139_, 0, v___x_1142_);
v___x_1144_ = v___x_1139_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v___x_1142_);
v___x_1144_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
return v___x_1144_;
}
}
}
}
else
{
lean_object* v_a_1148_; lean_object* v___x_1150_; uint8_t v_isShared_1151_; uint8_t v_isSharedCheck_1155_; 
lean_del_object(v___x_1121_);
v_a_1148_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1155_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1155_ == 0)
{
v___x_1150_ = v___x_1136_;
v_isShared_1151_ = v_isSharedCheck_1155_;
goto v_resetjp_1149_;
}
else
{
lean_inc(v_a_1148_);
lean_dec(v___x_1136_);
v___x_1150_ = lean_box(0);
v_isShared_1151_ = v_isSharedCheck_1155_;
goto v_resetjp_1149_;
}
v_resetjp_1149_:
{
lean_object* v___x_1153_; 
if (v_isShared_1151_ == 0)
{
v___x_1153_ = v___x_1150_;
goto v_reusejp_1152_;
}
else
{
lean_object* v_reuseFailAlloc_1154_; 
v_reuseFailAlloc_1154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1154_, 0, v_a_1148_);
v___x_1153_ = v_reuseFailAlloc_1154_;
goto v_reusejp_1152_;
}
v_reusejp_1152_:
{
return v___x_1153_;
}
}
}
}
}
else
{
lean_object* v___x_1157_; lean_object* v___x_1159_; 
lean_dec(v_a_1115_);
lean_dec_ref(v_mainVar_1111_);
lean_dec_ref(v_args_1110_);
lean_dec_ref(v_insts_1109_);
lean_dec_ref(v_lctx_1108_);
v___x_1157_ = lean_box(0);
if (v_isShared_1118_ == 0)
{
lean_ctor_set(v___x_1117_, 0, v___x_1157_);
v___x_1159_ = v___x_1117_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1160_; 
v_reuseFailAlloc_1160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1160_, 0, v___x_1157_);
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
else
{
lean_dec_ref(v_mainVar_1111_);
lean_dec_ref(v_args_1110_);
lean_dec_ref(v_insts_1109_);
lean_dec_ref(v_lctx_1108_);
return v___x_1114_;
}
}
else
{
lean_object* v___x_1162_; lean_object* v___x_1163_; 
lean_dec_ref(v_fn_1107_);
lean_dec_ref(v_fData_1101_);
v___x_1162_ = lean_box(0);
v___x_1163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1162_);
return v___x_1163_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_unfoldHeadFVar_x3f___boxed(lean_object* v_fData_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_){
_start:
{
lean_object* v_res_1170_; 
v_res_1170_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_unfoldHeadFVar_x3f(v_fData_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_);
lean_dec(v_a_1168_);
lean_dec_ref(v_a_1167_);
lean_dec(v_a_1166_);
lean_dec_ref(v_a_1165_);
return v_res_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx(uint8_t v_x_1171_){
_start:
{
switch(v_x_1171_)
{
case 0:
{
lean_object* v___x_1172_; 
v___x_1172_ = lean_unsigned_to_nat(0u);
return v___x_1172_;
}
case 1:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_unsigned_to_nat(1u);
return v___x_1173_;
}
case 2:
{
lean_object* v___x_1174_; 
v___x_1174_ = lean_unsigned_to_nat(2u);
return v___x_1174_;
}
default: 
{
lean_object* v___x_1175_; 
v___x_1175_ = lean_unsigned_to_nat(3u);
return v___x_1175_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx___boxed(lean_object* v_x_1176_){
_start:
{
uint8_t v_x_boxed_1177_; lean_object* v_res_1178_; 
v_x_boxed_1177_ = lean_unbox(v_x_1176_);
v_res_1178_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx(v_x_boxed_1177_);
return v_res_1178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___redArg(lean_object* v_k_1179_){
_start:
{
lean_inc(v_k_1179_);
return v_k_1179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___redArg___boxed(lean_object* v_k_1180_){
_start:
{
lean_object* v_res_1181_; 
v_res_1181_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___redArg(v_k_1180_);
lean_dec(v_k_1180_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim(lean_object* v_motive_1182_, lean_object* v_ctorIdx_1183_, uint8_t v_t_1184_, lean_object* v_h_1185_, lean_object* v_k_1186_){
_start:
{
lean_inc(v_k_1186_);
return v_k_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim___boxed(lean_object* v_motive_1187_, lean_object* v_ctorIdx_1188_, lean_object* v_t_1189_, lean_object* v_h_1190_, lean_object* v_k_1191_){
_start:
{
uint8_t v_t_boxed_1192_; lean_object* v_res_1193_; 
v_t_boxed_1192_ = lean_unbox(v_t_1189_);
v_res_1193_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorElim(v_motive_1187_, v_ctorIdx_1188_, v_t_boxed_1192_, v_h_1190_, v_k_1191_);
lean_dec(v_k_1191_);
lean_dec(v_ctorIdx_1188_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___redArg(lean_object* v_underApplied_1194_){
_start:
{
lean_inc(v_underApplied_1194_);
return v_underApplied_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___redArg___boxed(lean_object* v_underApplied_1195_){
_start:
{
lean_object* v_res_1196_; 
v_res_1196_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___redArg(v_underApplied_1195_);
lean_dec(v_underApplied_1195_);
return v_res_1196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim(lean_object* v_motive_1197_, uint8_t v_t_1198_, lean_object* v_h_1199_, lean_object* v_underApplied_1200_){
_start:
{
lean_inc(v_underApplied_1200_);
return v_underApplied_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim___boxed(lean_object* v_motive_1201_, lean_object* v_t_1202_, lean_object* v_h_1203_, lean_object* v_underApplied_1204_){
_start:
{
uint8_t v_t_boxed_1205_; lean_object* v_res_1206_; 
v_t_boxed_1205_ = lean_unbox(v_t_1202_);
v_res_1206_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_underApplied_elim(v_motive_1201_, v_t_boxed_1205_, v_h_1203_, v_underApplied_1204_);
lean_dec(v_underApplied_1204_);
return v_res_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___redArg(lean_object* v_exact_1207_){
_start:
{
lean_inc(v_exact_1207_);
return v_exact_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___redArg___boxed(lean_object* v_exact_1208_){
_start:
{
lean_object* v_res_1209_; 
v_res_1209_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___redArg(v_exact_1208_);
lean_dec(v_exact_1208_);
return v_res_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim(lean_object* v_motive_1210_, uint8_t v_t_1211_, lean_object* v_h_1212_, lean_object* v_exact_1213_){
_start:
{
lean_inc(v_exact_1213_);
return v_exact_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim___boxed(lean_object* v_motive_1214_, lean_object* v_t_1215_, lean_object* v_h_1216_, lean_object* v_exact_1217_){
_start:
{
uint8_t v_t_boxed_1218_; lean_object* v_res_1219_; 
v_t_boxed_1218_ = lean_unbox(v_t_1215_);
v_res_1219_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_exact_elim(v_motive_1214_, v_t_boxed_1218_, v_h_1216_, v_exact_1217_);
lean_dec(v_exact_1217_);
return v_res_1219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___redArg(lean_object* v_overApplied_1220_){
_start:
{
lean_inc(v_overApplied_1220_);
return v_overApplied_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___redArg___boxed(lean_object* v_overApplied_1221_){
_start:
{
lean_object* v_res_1222_; 
v_res_1222_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___redArg(v_overApplied_1221_);
lean_dec(v_overApplied_1221_);
return v_res_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim(lean_object* v_motive_1223_, uint8_t v_t_1224_, lean_object* v_h_1225_, lean_object* v_overApplied_1226_){
_start:
{
lean_inc(v_overApplied_1226_);
return v_overApplied_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim___boxed(lean_object* v_motive_1227_, lean_object* v_t_1228_, lean_object* v_h_1229_, lean_object* v_overApplied_1230_){
_start:
{
uint8_t v_t_boxed_1231_; lean_object* v_res_1232_; 
v_t_boxed_1231_ = lean_unbox(v_t_1228_);
v_res_1232_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_overApplied_elim(v_motive_1227_, v_t_boxed_1231_, v_h_1229_, v_overApplied_1230_);
lean_dec(v_overApplied_1230_);
return v_res_1232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___redArg(lean_object* v_none_1233_){
_start:
{
lean_inc(v_none_1233_);
return v_none_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___redArg___boxed(lean_object* v_none_1234_){
_start:
{
lean_object* v_res_1235_; 
v_res_1235_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___redArg(v_none_1234_);
lean_dec(v_none_1234_);
return v_res_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim(lean_object* v_motive_1236_, uint8_t v_t_1237_, lean_object* v_h_1238_, lean_object* v_none_1239_){
_start:
{
lean_inc(v_none_1239_);
return v_none_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim___boxed(lean_object* v_motive_1240_, lean_object* v_t_1241_, lean_object* v_h_1242_, lean_object* v_none_1243_){
_start:
{
uint8_t v_t_boxed_1244_; lean_object* v_res_1245_; 
v_t_boxed_1244_ = lean_unbox(v_t_1241_);
v_res_1245_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_none_elim(v_motive_1240_, v_t_boxed_1244_, v_h_1242_, v_none_1243_);
lean_dec(v_none_1243_);
return v_res_1245_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication_default(void){
_start:
{
uint8_t v___x_1246_; 
v___x_1246_ = 0;
return v___x_1246_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication(void){
_start:
{
uint8_t v___x_1247_; 
v___x_1247_ = 0;
return v___x_1247_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq(uint8_t v_x_1248_, uint8_t v_y_1249_){
_start:
{
lean_object* v___x_1250_; lean_object* v___x_1251_; uint8_t v___x_1252_; 
v___x_1250_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx(v_x_1248_);
v___x_1251_ = lp_mathlib_Mathlib_Meta_FunProp_MorApplication_ctorIdx(v_y_1249_);
v___x_1252_ = lean_nat_dec_eq(v___x_1250_, v___x_1251_);
lean_dec(v___x_1251_);
lean_dec(v___x_1250_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq___boxed(lean_object* v_x_1253_, lean_object* v_y_1254_){
_start:
{
uint8_t v_x_17__boxed_1255_; uint8_t v_y_18__boxed_1256_; uint8_t v_res_1257_; lean_object* v_r_1258_; 
v_x_17__boxed_1255_ = lean_unbox(v_x_1253_);
v_y_18__boxed_1256_ = lean_unbox(v_y_1254_);
v_res_1257_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqMorApplication_beq(v_x_17__boxed_1255_, v_y_18__boxed_1256_);
v_r_1258_ = lean_box(v_res_1257_);
return v_r_1258_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0(lean_object* v_as_1261_, size_t v_i_1262_, size_t v_stop_1263_){
_start:
{
uint8_t v___x_1264_; 
v___x_1264_ = lean_usize_dec_eq(v_i_1262_, v_stop_1263_);
if (v___x_1264_ == 0)
{
lean_object* v___x_1265_; lean_object* v_coe_1266_; 
v___x_1265_ = lean_array_uget_borrowed(v_as_1261_, v_i_1262_);
v_coe_1266_ = lean_ctor_get(v___x_1265_, 1);
if (lean_obj_tag(v_coe_1266_) == 0)
{
size_t v___x_1267_; size_t v___x_1268_; 
v___x_1267_ = ((size_t)1ULL);
v___x_1268_ = lean_usize_add(v_i_1262_, v___x_1267_);
v_i_1262_ = v___x_1268_;
goto _start;
}
else
{
uint8_t v___x_1270_; 
v___x_1270_ = 1;
return v___x_1270_;
}
}
else
{
uint8_t v___x_1271_; 
v___x_1271_ = 0;
return v___x_1271_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0___boxed(lean_object* v_as_1272_, lean_object* v_i_1273_, lean_object* v_stop_1274_){
_start:
{
size_t v_i_boxed_1275_; size_t v_stop_boxed_1276_; uint8_t v_res_1277_; lean_object* v_r_1278_; 
v_i_boxed_1275_ = lean_unbox_usize(v_i_1273_);
lean_dec(v_i_1273_);
v_stop_boxed_1276_ = lean_unbox_usize(v_stop_1274_);
lean_dec(v_stop_1274_);
v_res_1277_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0(v_as_1272_, v_i_boxed_1275_, v_stop_boxed_1276_);
lean_dec_ref(v_as_1272_);
v_r_1278_ = lean_box(v_res_1277_);
return v_r_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg(lean_object* v_ref_1279_, lean_object* v_msg_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v_fileName_1286_; lean_object* v_fileMap_1287_; lean_object* v_options_1288_; lean_object* v_currRecDepth_1289_; lean_object* v_maxRecDepth_1290_; lean_object* v_ref_1291_; lean_object* v_currNamespace_1292_; lean_object* v_openDecls_1293_; lean_object* v_initHeartbeats_1294_; lean_object* v_maxHeartbeats_1295_; lean_object* v_quotContext_1296_; lean_object* v_currMacroScope_1297_; uint8_t v_diag_1298_; lean_object* v_cancelTk_x3f_1299_; uint8_t v_suppressElabErrors_1300_; lean_object* v_inheritedTraceOptions_1301_; lean_object* v_ref_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; 
v_fileName_1286_ = lean_ctor_get(v___y_1283_, 0);
v_fileMap_1287_ = lean_ctor_get(v___y_1283_, 1);
v_options_1288_ = lean_ctor_get(v___y_1283_, 2);
v_currRecDepth_1289_ = lean_ctor_get(v___y_1283_, 3);
v_maxRecDepth_1290_ = lean_ctor_get(v___y_1283_, 4);
v_ref_1291_ = lean_ctor_get(v___y_1283_, 5);
v_currNamespace_1292_ = lean_ctor_get(v___y_1283_, 6);
v_openDecls_1293_ = lean_ctor_get(v___y_1283_, 7);
v_initHeartbeats_1294_ = lean_ctor_get(v___y_1283_, 8);
v_maxHeartbeats_1295_ = lean_ctor_get(v___y_1283_, 9);
v_quotContext_1296_ = lean_ctor_get(v___y_1283_, 10);
v_currMacroScope_1297_ = lean_ctor_get(v___y_1283_, 11);
v_diag_1298_ = lean_ctor_get_uint8(v___y_1283_, sizeof(void*)*14);
v_cancelTk_x3f_1299_ = lean_ctor_get(v___y_1283_, 12);
v_suppressElabErrors_1300_ = lean_ctor_get_uint8(v___y_1283_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1301_ = lean_ctor_get(v___y_1283_, 13);
v_ref_1302_ = l_Lean_replaceRef(v_ref_1279_, v_ref_1291_);
lean_inc_ref(v_inheritedTraceOptions_1301_);
lean_inc(v_cancelTk_x3f_1299_);
lean_inc(v_currMacroScope_1297_);
lean_inc(v_quotContext_1296_);
lean_inc(v_maxHeartbeats_1295_);
lean_inc(v_initHeartbeats_1294_);
lean_inc(v_openDecls_1293_);
lean_inc(v_currNamespace_1292_);
lean_inc(v_maxRecDepth_1290_);
lean_inc(v_currRecDepth_1289_);
lean_inc_ref(v_options_1288_);
lean_inc_ref(v_fileMap_1287_);
lean_inc_ref(v_fileName_1286_);
v___x_1303_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1303_, 0, v_fileName_1286_);
lean_ctor_set(v___x_1303_, 1, v_fileMap_1287_);
lean_ctor_set(v___x_1303_, 2, v_options_1288_);
lean_ctor_set(v___x_1303_, 3, v_currRecDepth_1289_);
lean_ctor_set(v___x_1303_, 4, v_maxRecDepth_1290_);
lean_ctor_set(v___x_1303_, 5, v_ref_1302_);
lean_ctor_set(v___x_1303_, 6, v_currNamespace_1292_);
lean_ctor_set(v___x_1303_, 7, v_openDecls_1293_);
lean_ctor_set(v___x_1303_, 8, v_initHeartbeats_1294_);
lean_ctor_set(v___x_1303_, 9, v_maxHeartbeats_1295_);
lean_ctor_set(v___x_1303_, 10, v_quotContext_1296_);
lean_ctor_set(v___x_1303_, 11, v_currMacroScope_1297_);
lean_ctor_set(v___x_1303_, 12, v_cancelTk_x3f_1299_);
lean_ctor_set(v___x_1303_, 13, v_inheritedTraceOptions_1301_);
lean_ctor_set_uint8(v___x_1303_, sizeof(void*)*14, v_diag_1298_);
lean_ctor_set_uint8(v___x_1303_, sizeof(void*)*14 + 1, v_suppressElabErrors_1300_);
v___x_1304_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__3___redArg(v_msg_1280_, v___y_1281_, v___y_1282_, v___x_1303_, v___y_1284_);
lean_dec_ref_known(v___x_1303_, 14);
return v___x_1304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_ref_1305_, lean_object* v_msg_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg(v_ref_1305_, v_msg_1306_, v___y_1307_, v___y_1308_, v___y_1309_, v___y_1310_);
lean_dec(v___y_1310_);
lean_dec_ref(v___y_1309_);
lean_dec(v___y_1308_);
lean_dec_ref(v___y_1307_);
lean_dec(v_ref_1305_);
return v_res_1312_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_1313_; 
v___x_1313_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1313_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_1314_; lean_object* v___x_1315_; 
v___x_1314_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__0);
v___x_1315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1315_, 0, v___x_1314_);
return v___x_1315_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1316_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_1317_ = lean_unsigned_to_nat(0u);
v___x_1318_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1317_);
lean_ctor_set(v___x_1318_, 1, v___x_1317_);
lean_ctor_set(v___x_1318_, 2, v___x_1317_);
lean_ctor_set(v___x_1318_, 3, v___x_1317_);
lean_ctor_set(v___x_1318_, 4, v___x_1316_);
lean_ctor_set(v___x_1318_, 5, v___x_1316_);
lean_ctor_set(v___x_1318_, 6, v___x_1316_);
lean_ctor_set(v___x_1318_, 7, v___x_1316_);
lean_ctor_set(v___x_1318_, 8, v___x_1316_);
lean_ctor_set(v___x_1318_, 9, v___x_1316_);
return v___x_1318_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; 
v___x_1319_ = lean_unsigned_to_nat(32u);
v___x_1320_ = lean_mk_empty_array_with_capacity(v___x_1319_);
v___x_1321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
return v___x_1321_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4(void){
_start:
{
size_t v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1322_ = ((size_t)5ULL);
v___x_1323_ = lean_unsigned_to_nat(0u);
v___x_1324_ = lean_unsigned_to_nat(32u);
v___x_1325_ = lean_mk_empty_array_with_capacity(v___x_1324_);
v___x_1326_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__3);
v___x_1327_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
lean_ctor_set(v___x_1327_, 1, v___x_1325_);
lean_ctor_set(v___x_1327_, 2, v___x_1323_);
lean_ctor_set(v___x_1327_, 3, v___x_1323_);
lean_ctor_set_usize(v___x_1327_, 4, v___x_1322_);
return v___x_1327_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; 
v___x_1328_ = lean_box(1);
v___x_1329_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__4);
v___x_1330_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_1331_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1331_, 0, v___x_1330_);
lean_ctor_set(v___x_1331_, 1, v___x_1329_);
lean_ctor_set(v___x_1331_, 2, v___x_1328_);
return v___x_1331_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7(void){
_start:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1333_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__6));
v___x_1334_ = l_Lean_stringToMessageData(v___x_1333_);
return v___x_1334_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9(void){
_start:
{
lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1336_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__8));
v___x_1337_ = l_Lean_stringToMessageData(v___x_1336_);
return v___x_1337_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11(void){
_start:
{
lean_object* v___x_1339_; lean_object* v___x_1340_; 
v___x_1339_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__10));
v___x_1340_ = l_Lean_stringToMessageData(v___x_1339_);
return v___x_1340_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13(void){
_start:
{
lean_object* v___x_1342_; lean_object* v___x_1343_; 
v___x_1342_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__12));
v___x_1343_ = l_Lean_stringToMessageData(v___x_1342_);
return v___x_1343_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15(void){
_start:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; 
v___x_1345_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__14));
v___x_1346_ = l_Lean_stringToMessageData(v___x_1345_);
return v___x_1346_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17(void){
_start:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1348_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__16));
v___x_1349_ = l_Lean_stringToMessageData(v___x_1348_);
return v___x_1349_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19(void){
_start:
{
lean_object* v___x_1351_; lean_object* v___x_1352_; 
v___x_1351_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__18));
v___x_1352_ = l_Lean_stringToMessageData(v___x_1351_);
return v___x_1352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg(lean_object* v_msg_1353_, lean_object* v_declHint_1354_, lean_object* v___y_1355_){
_start:
{
lean_object* v___x_1357_; lean_object* v_env_1358_; uint8_t v___x_1359_; 
v___x_1357_ = lean_st_ref_get(v___y_1355_);
v_env_1358_ = lean_ctor_get(v___x_1357_, 0);
lean_inc_ref(v_env_1358_);
lean_dec(v___x_1357_);
v___x_1359_ = l_Lean_Name_isAnonymous(v_declHint_1354_);
if (v___x_1359_ == 0)
{
uint8_t v_isExporting_1360_; 
v_isExporting_1360_ = lean_ctor_get_uint8(v_env_1358_, sizeof(void*)*8);
if (v_isExporting_1360_ == 0)
{
lean_object* v___x_1361_; 
lean_dec_ref(v_env_1358_);
lean_dec(v_declHint_1354_);
v___x_1361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1361_, 0, v_msg_1353_);
return v___x_1361_;
}
else
{
lean_object* v___x_1362_; uint8_t v___x_1363_; 
lean_inc_ref(v_env_1358_);
v___x_1362_ = l_Lean_Environment_setExporting(v_env_1358_, v___x_1359_);
lean_inc(v_declHint_1354_);
lean_inc_ref(v___x_1362_);
v___x_1363_ = l_Lean_Environment_contains(v___x_1362_, v_declHint_1354_, v_isExporting_1360_);
if (v___x_1363_ == 0)
{
lean_object* v___x_1364_; 
lean_dec_ref(v___x_1362_);
lean_dec_ref(v_env_1358_);
lean_dec(v_declHint_1354_);
v___x_1364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1364_, 0, v_msg_1353_);
return v___x_1364_;
}
else
{
lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v_c_1370_; lean_object* v___x_1371_; 
v___x_1365_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__2);
v___x_1366_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__5);
v___x_1367_ = l_Lean_Options_empty;
v___x_1368_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1368_, 0, v___x_1362_);
lean_ctor_set(v___x_1368_, 1, v___x_1365_);
lean_ctor_set(v___x_1368_, 2, v___x_1366_);
lean_ctor_set(v___x_1368_, 3, v___x_1367_);
lean_inc(v_declHint_1354_);
v___x_1369_ = l_Lean_MessageData_ofConstName(v_declHint_1354_, v___x_1359_);
v_c_1370_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_1370_, 0, v___x_1368_);
lean_ctor_set(v_c_1370_, 1, v___x_1369_);
v___x_1371_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1358_, v_declHint_1354_);
if (lean_obj_tag(v___x_1371_) == 0)
{
lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
lean_dec_ref(v_env_1358_);
lean_dec(v_declHint_1354_);
v___x_1372_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7);
v___x_1373_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
lean_ctor_set(v___x_1373_, 1, v_c_1370_);
v___x_1374_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__9);
v___x_1375_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1375_, 0, v___x_1373_);
lean_ctor_set(v___x_1375_, 1, v___x_1374_);
v___x_1376_ = l_Lean_MessageData_note(v___x_1375_);
v___x_1377_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1377_, 0, v_msg_1353_);
lean_ctor_set(v___x_1377_, 1, v___x_1376_);
v___x_1378_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
return v___x_1378_;
}
else
{
lean_object* v_val_1379_; lean_object* v___x_1381_; uint8_t v_isShared_1382_; uint8_t v_isSharedCheck_1414_; 
v_val_1379_ = lean_ctor_get(v___x_1371_, 0);
v_isSharedCheck_1414_ = !lean_is_exclusive(v___x_1371_);
if (v_isSharedCheck_1414_ == 0)
{
v___x_1381_ = v___x_1371_;
v_isShared_1382_ = v_isSharedCheck_1414_;
goto v_resetjp_1380_;
}
else
{
lean_inc(v_val_1379_);
lean_dec(v___x_1371_);
v___x_1381_ = lean_box(0);
v_isShared_1382_ = v_isSharedCheck_1414_;
goto v_resetjp_1380_;
}
v_resetjp_1380_:
{
lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v_mod_1386_; uint8_t v___x_1387_; 
v___x_1383_ = lean_box(0);
v___x_1384_ = l_Lean_Environment_header(v_env_1358_);
lean_dec_ref(v_env_1358_);
v___x_1385_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1384_);
v_mod_1386_ = lean_array_get(v___x_1383_, v___x_1385_, v_val_1379_);
lean_dec(v_val_1379_);
lean_dec_ref(v___x_1385_);
v___x_1387_ = l_Lean_isPrivateName(v_declHint_1354_);
lean_dec(v_declHint_1354_);
if (v___x_1387_ == 0)
{
lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1399_; 
v___x_1388_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__11);
v___x_1389_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1389_, 0, v___x_1388_);
lean_ctor_set(v___x_1389_, 1, v_c_1370_);
v___x_1390_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__13);
v___x_1391_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1391_, 0, v___x_1389_);
lean_ctor_set(v___x_1391_, 1, v___x_1390_);
v___x_1392_ = l_Lean_MessageData_ofName(v_mod_1386_);
v___x_1393_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1391_);
lean_ctor_set(v___x_1393_, 1, v___x_1392_);
v___x_1394_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__15);
v___x_1395_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1395_, 0, v___x_1393_);
lean_ctor_set(v___x_1395_, 1, v___x_1394_);
v___x_1396_ = l_Lean_MessageData_note(v___x_1395_);
v___x_1397_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1397_, 0, v_msg_1353_);
lean_ctor_set(v___x_1397_, 1, v___x_1396_);
if (v_isShared_1382_ == 0)
{
lean_ctor_set_tag(v___x_1381_, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1397_);
v___x_1399_ = v___x_1381_;
goto v_reusejp_1398_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v___x_1397_);
v___x_1399_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1398_;
}
v_reusejp_1398_:
{
return v___x_1399_;
}
}
else
{
lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1412_; 
v___x_1401_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__7);
v___x_1402_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1402_, 0, v___x_1401_);
lean_ctor_set(v___x_1402_, 1, v_c_1370_);
v___x_1403_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__17);
v___x_1404_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1404_, 0, v___x_1402_);
lean_ctor_set(v___x_1404_, 1, v___x_1403_);
v___x_1405_ = l_Lean_MessageData_ofName(v_mod_1386_);
v___x_1406_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1406_, 0, v___x_1404_);
lean_ctor_set(v___x_1406_, 1, v___x_1405_);
v___x_1407_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___closed__19);
v___x_1408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1408_, 0, v___x_1406_);
lean_ctor_set(v___x_1408_, 1, v___x_1407_);
v___x_1409_ = l_Lean_MessageData_note(v___x_1408_);
v___x_1410_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1410_, 0, v_msg_1353_);
lean_ctor_set(v___x_1410_, 1, v___x_1409_);
if (v_isShared_1382_ == 0)
{
lean_ctor_set_tag(v___x_1381_, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1410_);
v___x_1412_ = v___x_1381_;
goto v_reusejp_1411_;
}
else
{
lean_object* v_reuseFailAlloc_1413_; 
v_reuseFailAlloc_1413_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1413_, 0, v___x_1410_);
v___x_1412_ = v_reuseFailAlloc_1413_;
goto v_reusejp_1411_;
}
v_reusejp_1411_:
{
return v___x_1412_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1415_; 
lean_dec_ref(v_env_1358_);
lean_dec(v_declHint_1354_);
v___x_1415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1415_, 0, v_msg_1353_);
return v___x_1415_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg___boxed(lean_object* v_msg_1416_, lean_object* v_declHint_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_){
_start:
{
lean_object* v_res_1420_; 
v_res_1420_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg(v_msg_1416_, v_declHint_1417_, v___y_1418_);
lean_dec(v___y_1418_);
return v_res_1420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object* v_msg_1421_, lean_object* v_declHint_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_){
_start:
{
lean_object* v___x_1428_; lean_object* v_a_1429_; lean_object* v___x_1431_; uint8_t v_isShared_1432_; uint8_t v_isSharedCheck_1438_; 
v___x_1428_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg(v_msg_1421_, v_declHint_1422_, v___y_1426_);
v_a_1429_ = lean_ctor_get(v___x_1428_, 0);
v_isSharedCheck_1438_ = !lean_is_exclusive(v___x_1428_);
if (v_isSharedCheck_1438_ == 0)
{
v___x_1431_ = v___x_1428_;
v_isShared_1432_ = v_isSharedCheck_1438_;
goto v_resetjp_1430_;
}
else
{
lean_inc(v_a_1429_);
lean_dec(v___x_1428_);
v___x_1431_ = lean_box(0);
v_isShared_1432_ = v_isSharedCheck_1438_;
goto v_resetjp_1430_;
}
v_resetjp_1430_:
{
lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1436_; 
v___x_1433_ = l_Lean_unknownIdentifierMessageTag;
v___x_1434_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1434_, 0, v___x_1433_);
lean_ctor_set(v___x_1434_, 1, v_a_1429_);
if (v_isShared_1432_ == 0)
{
lean_ctor_set(v___x_1431_, 0, v___x_1434_);
v___x_1436_ = v___x_1431_;
goto v_reusejp_1435_;
}
else
{
lean_object* v_reuseFailAlloc_1437_; 
v_reuseFailAlloc_1437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1437_, 0, v___x_1434_);
v___x_1436_ = v_reuseFailAlloc_1437_;
goto v_reusejp_1435_;
}
v_reusejp_1435_:
{
return v___x_1436_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4___boxed(lean_object* v_msg_1439_, lean_object* v_declHint_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_){
_start:
{
lean_object* v_res_1446_; 
v_res_1446_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4(v_msg_1439_, v_declHint_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_);
lean_dec(v___y_1444_);
lean_dec_ref(v___y_1443_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
return v_res_1446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_ref_1447_, lean_object* v_msg_1448_, lean_object* v_declHint_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_){
_start:
{
lean_object* v___x_1455_; lean_object* v_a_1456_; lean_object* v___x_1457_; 
v___x_1455_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4(v_msg_1448_, v_declHint_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
v_a_1456_ = lean_ctor_get(v___x_1455_, 0);
lean_inc(v_a_1456_);
lean_dec_ref(v___x_1455_);
v___x_1457_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg(v_ref_1447_, v_a_1456_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
return v___x_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object* v_ref_1458_, lean_object* v_msg_1459_, lean_object* v_declHint_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_){
_start:
{
lean_object* v_res_1466_; 
v_res_1466_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg(v_ref_1458_, v_msg_1459_, v_declHint_1460_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
lean_dec(v___y_1464_);
lean_dec_ref(v___y_1463_);
lean_dec(v___y_1462_);
lean_dec_ref(v___y_1461_);
lean_dec(v_ref_1458_);
return v_res_1466_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; 
v___x_1468_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__0));
v___x_1469_ = l_Lean_stringToMessageData(v___x_1468_);
return v___x_1469_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1471_; lean_object* v___x_1472_; 
v___x_1471_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__2));
v___x_1472_ = l_Lean_stringToMessageData(v___x_1471_);
return v___x_1472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg(lean_object* v_ref_1473_, lean_object* v_constName_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_){
_start:
{
lean_object* v___x_1480_; uint8_t v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; 
v___x_1480_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__1);
v___x_1481_ = 0;
lean_inc(v_constName_1474_);
v___x_1482_ = l_Lean_MessageData_ofConstName(v_constName_1474_, v___x_1481_);
v___x_1483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1483_, 0, v___x_1480_);
lean_ctor_set(v___x_1483_, 1, v___x_1482_);
v___x_1484_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___closed__3);
v___x_1485_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1483_);
lean_ctor_set(v___x_1485_, 1, v___x_1484_);
v___x_1486_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg(v_ref_1473_, v___x_1485_, v_constName_1474_, v___y_1475_, v___y_1476_, v___y_1477_, v___y_1478_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_ref_1487_, lean_object* v_constName_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_){
_start:
{
lean_object* v_res_1494_; 
v_res_1494_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg(v_ref_1487_, v_constName_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_);
lean_dec(v___y_1492_);
lean_dec_ref(v___y_1491_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
lean_dec(v_ref_1487_);
return v_res_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg(lean_object* v_constName_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_){
_start:
{
lean_object* v_ref_1501_; lean_object* v___x_1502_; 
v_ref_1501_ = lean_ctor_get(v___y_1498_, 5);
v___x_1502_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg(v_ref_1501_, v_constName_1495_, v___y_1496_, v___y_1497_, v___y_1498_, v___y_1499_);
return v___x_1502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg___boxed(lean_object* v_constName_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_){
_start:
{
lean_object* v_res_1509_; 
v_res_1509_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg(v_constName_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_);
lean_dec(v___y_1507_);
lean_dec_ref(v___y_1506_);
lean_dec(v___y_1505_);
lean_dec_ref(v___y_1504_);
return v_res_1509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1(lean_object* v_constName_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_){
_start:
{
lean_object* v___x_1516_; lean_object* v_env_1517_; uint8_t v___x_1518_; lean_object* v___x_1519_; 
v___x_1516_ = lean_st_ref_get(v___y_1514_);
v_env_1517_ = lean_ctor_get(v___x_1516_, 0);
lean_inc_ref(v_env_1517_);
lean_dec(v___x_1516_);
v___x_1518_ = 0;
lean_inc(v_constName_1510_);
v___x_1519_ = l_Lean_Environment_find_x3f(v_env_1517_, v_constName_1510_, v___x_1518_);
if (lean_obj_tag(v___x_1519_) == 0)
{
lean_object* v___x_1520_; 
v___x_1520_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg(v_constName_1510_, v___y_1511_, v___y_1512_, v___y_1513_, v___y_1514_);
return v___x_1520_;
}
else
{
lean_object* v_val_1521_; lean_object* v___x_1523_; uint8_t v_isShared_1524_; uint8_t v_isSharedCheck_1528_; 
lean_dec(v_constName_1510_);
v_val_1521_ = lean_ctor_get(v___x_1519_, 0);
v_isSharedCheck_1528_ = !lean_is_exclusive(v___x_1519_);
if (v_isSharedCheck_1528_ == 0)
{
v___x_1523_ = v___x_1519_;
v_isShared_1524_ = v_isSharedCheck_1528_;
goto v_resetjp_1522_;
}
else
{
lean_inc(v_val_1521_);
lean_dec(v___x_1519_);
v___x_1523_ = lean_box(0);
v_isShared_1524_ = v_isSharedCheck_1528_;
goto v_resetjp_1522_;
}
v_resetjp_1522_:
{
lean_object* v___x_1526_; 
if (v_isShared_1524_ == 0)
{
lean_ctor_set_tag(v___x_1523_, 0);
v___x_1526_ = v___x_1523_;
goto v_reusejp_1525_;
}
else
{
lean_object* v_reuseFailAlloc_1527_; 
v_reuseFailAlloc_1527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1527_, 0, v_val_1521_);
v___x_1526_ = v_reuseFailAlloc_1527_;
goto v_reusejp_1525_;
}
v_reusejp_1525_:
{
return v___x_1526_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1___boxed(lean_object* v_constName_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
lean_object* v_res_1535_; 
v_res_1535_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1(v_constName_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec(v___y_1531_);
lean_dec_ref(v___y_1530_);
return v_res_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication(lean_object* v_f_1536_, lean_object* v_a_1537_, lean_object* v_a_1538_, lean_object* v_a_1539_, lean_object* v_a_1540_){
_start:
{
lean_object* v_fn_1546_; lean_object* v_args_1547_; lean_object* v___x_1576_; 
v_fn_1546_ = lean_ctor_get(v_f_1536_, 2);
v_args_1547_ = lean_ctor_get(v_f_1536_, 3);
v___x_1576_ = l_Lean_Expr_constName_x3f(v_fn_1546_);
if (lean_obj_tag(v___x_1576_) == 1)
{
lean_object* v_val_1577_; lean_object* v___x_1578_; 
v_val_1577_ = lean_ctor_get(v___x_1576_, 0);
lean_inc(v_val_1577_);
lean_dec_ref_known(v___x_1576_, 1);
v___x_1578_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(v_val_1577_, v_a_1540_);
if (lean_obj_tag(v___x_1578_) == 0)
{
lean_object* v_a_1579_; uint8_t v___x_1580_; 
v_a_1579_ = lean_ctor_get(v___x_1578_, 0);
lean_inc(v_a_1579_);
lean_dec_ref_known(v___x_1578_, 1);
v___x_1580_ = lean_unbox(v_a_1579_);
lean_dec(v_a_1579_);
if (v___x_1580_ == 0)
{
lean_dec(v_val_1577_);
goto v___jp_1548_;
}
else
{
lean_object* v___x_1581_; 
v___x_1581_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1(v_val_1577_, v_a_1537_, v_a_1538_, v_a_1539_, v_a_1540_);
if (lean_obj_tag(v___x_1581_) == 0)
{
lean_object* v_a_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1606_; 
v_a_1582_ = lean_ctor_get(v___x_1581_, 0);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1581_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1584_ = v___x_1581_;
v_isShared_1585_ = v_isSharedCheck_1606_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_a_1582_);
lean_dec(v___x_1581_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1606_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; uint8_t v___x_1589_; 
v___x_1586_ = l_Lean_ConstantInfo_type(v_a_1582_);
lean_dec(v_a_1582_);
v___x_1587_ = l_Lean_Expr_getNumHeadForalls(v___x_1586_);
lean_dec_ref(v___x_1586_);
v___x_1588_ = lean_array_get_size(v_args_1547_);
v___x_1589_ = lean_nat_dec_lt(v___x_1587_, v___x_1588_);
if (v___x_1589_ == 0)
{
uint8_t v___x_1590_; 
v___x_1590_ = lean_nat_dec_eq(v___x_1587_, v___x_1588_);
lean_dec(v___x_1587_);
if (v___x_1590_ == 0)
{
uint8_t v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1594_; 
v___x_1591_ = 0;
v___x_1592_ = lean_box(v___x_1591_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1592_);
v___x_1594_ = v___x_1584_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v___x_1592_);
v___x_1594_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
return v___x_1594_;
}
}
else
{
uint8_t v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1599_; 
v___x_1596_ = 1;
v___x_1597_ = lean_box(v___x_1596_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1597_);
v___x_1599_ = v___x_1584_;
goto v_reusejp_1598_;
}
else
{
lean_object* v_reuseFailAlloc_1600_; 
v_reuseFailAlloc_1600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1600_, 0, v___x_1597_);
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
uint8_t v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1604_; 
lean_dec(v___x_1587_);
v___x_1601_ = 2;
v___x_1602_ = lean_box(v___x_1601_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1602_);
v___x_1604_ = v___x_1584_;
goto v_reusejp_1603_;
}
else
{
lean_object* v_reuseFailAlloc_1605_; 
v_reuseFailAlloc_1605_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1605_, 0, v___x_1602_);
v___x_1604_ = v_reuseFailAlloc_1605_;
goto v_reusejp_1603_;
}
v_reusejp_1603_:
{
return v___x_1604_;
}
}
}
}
else
{
lean_object* v_a_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1614_; 
v_a_1607_ = lean_ctor_get(v___x_1581_, 0);
v_isSharedCheck_1614_ = !lean_is_exclusive(v___x_1581_);
if (v_isSharedCheck_1614_ == 0)
{
v___x_1609_ = v___x_1581_;
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_a_1607_);
lean_dec(v___x_1581_);
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
}
else
{
lean_object* v_a_1615_; lean_object* v___x_1617_; uint8_t v_isShared_1618_; uint8_t v_isSharedCheck_1622_; 
lean_dec(v_val_1577_);
v_a_1615_ = lean_ctor_get(v___x_1578_, 0);
v_isSharedCheck_1622_ = !lean_is_exclusive(v___x_1578_);
if (v_isSharedCheck_1622_ == 0)
{
v___x_1617_ = v___x_1578_;
v_isShared_1618_ = v_isSharedCheck_1622_;
goto v_resetjp_1616_;
}
else
{
lean_inc(v_a_1615_);
lean_dec(v___x_1578_);
v___x_1617_ = lean_box(0);
v_isShared_1618_ = v_isSharedCheck_1622_;
goto v_resetjp_1616_;
}
v_resetjp_1616_:
{
lean_object* v___x_1620_; 
if (v_isShared_1618_ == 0)
{
v___x_1620_ = v___x_1617_;
goto v_reusejp_1619_;
}
else
{
lean_object* v_reuseFailAlloc_1621_; 
v_reuseFailAlloc_1621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1621_, 0, v_a_1615_);
v___x_1620_ = v_reuseFailAlloc_1621_;
goto v_reusejp_1619_;
}
v_reusejp_1619_:
{
return v___x_1620_;
}
}
}
}
else
{
lean_dec(v___x_1576_);
goto v___jp_1548_;
}
v___jp_1542_:
{
uint8_t v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; 
v___x_1543_ = 3;
v___x_1544_ = lean_box(v___x_1543_);
v___x_1545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1545_, 0, v___x_1544_);
return v___x_1545_;
}
v___jp_1548_:
{
lean_object* v___x_1549_; lean_object* v_zero_1550_; uint8_t v_isZero_1551_; 
v___x_1549_ = lean_array_get_size(v_args_1547_);
v_zero_1550_ = lean_unsigned_to_nat(0u);
v_isZero_1551_ = lean_nat_dec_eq(v___x_1549_, v_zero_1550_);
if (v_isZero_1551_ == 1)
{
uint8_t v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; 
v___x_1552_ = 3;
v___x_1553_ = lean_box(v___x_1552_);
v___x_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1554_, 0, v___x_1553_);
return v___x_1554_;
}
else
{
lean_object* v_one_1555_; lean_object* v_n_1556_; lean_object* v___x_1557_; lean_object* v_coe_1558_; 
v_one_1555_ = lean_unsigned_to_nat(1u);
v_n_1556_ = lean_nat_sub(v___x_1549_, v_one_1555_);
v___x_1557_ = lean_array_fget_borrowed(v_args_1547_, v_n_1556_);
lean_dec(v_n_1556_);
v_coe_1558_ = lean_ctor_get(v___x_1557_, 1);
lean_inc(v_coe_1558_);
if (lean_obj_tag(v_coe_1558_) == 0)
{
uint8_t v___x_1559_; 
v___x_1559_ = lean_nat_dec_lt(v_zero_1550_, v___x_1549_);
if (v___x_1559_ == 0)
{
goto v___jp_1542_;
}
else
{
if (v___x_1559_ == 0)
{
goto v___jp_1542_;
}
else
{
size_t v___x_1560_; size_t v___x_1561_; uint8_t v___x_1562_; 
v___x_1560_ = ((size_t)0ULL);
v___x_1561_ = lean_usize_of_nat(v___x_1549_);
v___x_1562_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__0(v_args_1547_, v___x_1560_, v___x_1561_);
if (v___x_1562_ == 0)
{
goto v___jp_1542_;
}
else
{
uint8_t v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1563_ = 2;
v___x_1564_ = lean_box(v___x_1563_);
v___x_1565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1565_, 0, v___x_1564_);
return v___x_1565_;
}
}
}
}
else
{
lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1574_; 
v_isSharedCheck_1574_ = !lean_is_exclusive(v_coe_1558_);
if (v_isSharedCheck_1574_ == 0)
{
lean_object* v_unused_1575_; 
v_unused_1575_ = lean_ctor_get(v_coe_1558_, 0);
lean_dec(v_unused_1575_);
v___x_1567_ = v_coe_1558_;
v_isShared_1568_ = v_isSharedCheck_1574_;
goto v_resetjp_1566_;
}
else
{
lean_dec(v_coe_1558_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1574_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
uint8_t v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1572_; 
v___x_1569_ = 1;
v___x_1570_ = lean_box(v___x_1569_);
if (v_isShared_1568_ == 0)
{
lean_ctor_set_tag(v___x_1567_, 0);
lean_ctor_set(v___x_1567_, 0, v___x_1570_);
v___x_1572_ = v___x_1567_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v___x_1570_);
v___x_1572_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
return v___x_1572_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication___boxed(lean_object* v_f_1623_, lean_object* v_a_1624_, lean_object* v_a_1625_, lean_object* v_a_1626_, lean_object* v_a_1627_, lean_object* v_a_1628_){
_start:
{
lean_object* v_res_1629_; 
v_res_1629_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_isMorApplication(v_f_1623_, v_a_1624_, v_a_1625_, v_a_1626_, v_a_1627_);
lean_dec(v_a_1627_);
lean_dec_ref(v_a_1626_);
lean_dec(v_a_1625_);
lean_dec_ref(v_a_1624_);
lean_dec_ref(v_f_1623_);
return v_res_1629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1(lean_object* v_00_u03b1_1630_, lean_object* v_constName_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
lean_object* v___x_1637_; 
v___x_1637_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___redArg(v_constName_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
return v___x_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1638_, lean_object* v_constName_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_){
_start:
{
lean_object* v_res_1645_; 
v_res_1645_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1(v_00_u03b1_1638_, v_constName_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
lean_dec(v___y_1643_);
lean_dec_ref(v___y_1642_);
lean_dec(v___y_1641_);
lean_dec_ref(v___y_1640_);
return v_res_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2(lean_object* v_00_u03b1_1646_, lean_object* v_ref_1647_, lean_object* v_constName_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_){
_start:
{
lean_object* v___x_1654_; 
v___x_1654_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___redArg(v_ref_1647_, v_constName_1648_, v___y_1649_, v___y_1650_, v___y_1651_, v___y_1652_);
return v___x_1654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b1_1655_, lean_object* v_ref_1656_, lean_object* v_constName_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_){
_start:
{
lean_object* v_res_1663_; 
v_res_1663_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2(v_00_u03b1_1655_, v_ref_1656_, v_constName_1657_, v___y_1658_, v___y_1659_, v___y_1660_, v___y_1661_);
lean_dec(v___y_1661_);
lean_dec_ref(v___y_1660_);
lean_dec(v___y_1659_);
lean_dec_ref(v___y_1658_);
lean_dec(v_ref_1656_);
return v_res_1663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3(lean_object* v_00_u03b1_1664_, lean_object* v_ref_1665_, lean_object* v_msg_1666_, lean_object* v_declHint_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_){
_start:
{
lean_object* v___x_1673_; 
v___x_1673_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___redArg(v_ref_1665_, v_msg_1666_, v_declHint_1667_, v___y_1668_, v___y_1669_, v___y_1670_, v___y_1671_);
return v___x_1673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1674_, lean_object* v_ref_1675_, lean_object* v_msg_1676_, lean_object* v_declHint_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_){
_start:
{
lean_object* v_res_1683_; 
v_res_1683_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3(v_00_u03b1_1674_, v_ref_1675_, v_msg_1676_, v_declHint_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
lean_dec(v___y_1679_);
lean_dec_ref(v___y_1678_);
lean_dec(v_ref_1675_);
return v_res_1683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5(lean_object* v_msg_1684_, lean_object* v_declHint_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_){
_start:
{
lean_object* v___x_1691_; 
v___x_1691_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___redArg(v_msg_1684_, v_declHint_1685_, v___y_1689_);
return v___x_1691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5___boxed(lean_object* v_msg_1692_, lean_object* v_declHint_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_){
_start:
{
lean_object* v_res_1699_; 
v_res_1699_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__4_spec__5(v_msg_1692_, v_declHint_1693_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_);
lean_dec(v___y_1697_);
lean_dec_ref(v___y_1696_);
lean_dec(v___y_1695_);
lean_dec_ref(v___y_1694_);
return v_res_1699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5(lean_object* v_00_u03b1_1700_, lean_object* v_ref_1701_, lean_object* v_msg_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
lean_object* v___x_1708_; 
v___x_1708_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___redArg(v_ref_1701_, v_msg_1702_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
return v___x_1708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b1_1709_, lean_object* v_ref_1710_, lean_object* v_msg_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_){
_start:
{
lean_object* v_res_1717_; 
v_res_1717_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Meta_FunProp_FunctionData_isMorApplication_spec__1_spec__1_spec__2_spec__3_spec__5(v_00_u03b1_1709_, v_ref_1710_, v_msg_1711_, v___y_1712_, v___y_1713_, v___y_1714_, v___y_1715_);
lean_dec(v___y_1715_);
lean_dec_ref(v___y_1714_);
lean_dec(v___y_1713_);
lean_dec_ref(v___y_1712_);
lean_dec(v_ref_1710_);
return v_res_1717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorIdx(lean_object* v_x_1718_){
_start:
{
switch(lean_obj_tag(v_x_1718_))
{
case 0:
{
lean_object* v___x_1719_; 
v___x_1719_ = lean_unsigned_to_nat(0u);
return v___x_1719_;
}
case 1:
{
lean_object* v___x_1720_; 
v___x_1720_ = lean_unsigned_to_nat(1u);
return v___x_1720_;
}
default: 
{
lean_object* v___x_1721_; 
v___x_1721_ = lean_unsigned_to_nat(2u);
return v___x_1721_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorIdx___boxed(lean_object* v_x_1722_){
_start:
{
lean_object* v_res_1723_; 
v_res_1723_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorIdx(v_x_1722_);
lean_dec(v_x_1722_);
return v_res_1723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(lean_object* v_t_1724_, lean_object* v_k_1725_){
_start:
{
if (lean_obj_tag(v_t_1724_) == 0)
{
lean_object* v_f_1726_; lean_object* v_g_1727_; lean_object* v___x_1728_; 
v_f_1726_ = lean_ctor_get(v_t_1724_, 0);
lean_inc_ref(v_f_1726_);
v_g_1727_ = lean_ctor_get(v_t_1724_, 1);
lean_inc_ref(v_g_1727_);
lean_dec_ref_known(v_t_1724_, 2);
v___x_1728_ = lean_apply_2(v_k_1725_, v_f_1726_, v_g_1727_);
return v___x_1728_;
}
else
{
lean_dec(v_t_1724_);
return v_k_1725_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim(lean_object* v_motive_1729_, lean_object* v_ctorIdx_1730_, lean_object* v_t_1731_, lean_object* v_h_1732_, lean_object* v_k_1733_){
_start:
{
lean_object* v___x_1734_; 
v___x_1734_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1731_, v_k_1733_);
return v___x_1734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___boxed(lean_object* v_motive_1735_, lean_object* v_ctorIdx_1736_, lean_object* v_t_1737_, lean_object* v_h_1738_, lean_object* v_k_1739_){
_start:
{
lean_object* v_res_1740_; 
v_res_1740_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim(v_motive_1735_, v_ctorIdx_1736_, v_t_1737_, v_h_1738_, v_k_1739_);
lean_dec(v_ctorIdx_1736_);
return v_res_1740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_comp_elim___redArg(lean_object* v_t_1741_, lean_object* v_comp_1742_){
_start:
{
lean_object* v___x_1743_; 
v___x_1743_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1741_, v_comp_1742_);
return v___x_1743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_comp_elim(lean_object* v_motive_1744_, lean_object* v_t_1745_, lean_object* v_h_1746_, lean_object* v_comp_1747_){
_start:
{
lean_object* v___x_1748_; 
v___x_1748_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1745_, v_comp_1747_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_uncurried_elim___redArg(lean_object* v_t_1749_, lean_object* v_uncurried_1750_){
_start:
{
lean_object* v___x_1751_; 
v___x_1751_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1749_, v_uncurried_1750_);
return v___x_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_uncurried_elim(lean_object* v_motive_1752_, lean_object* v_t_1753_, lean_object* v_h_1754_, lean_object* v_uncurried_1755_){
_start:
{
lean_object* v___x_1756_; 
v___x_1756_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1753_, v_uncurried_1755_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_failed_elim___redArg(lean_object* v_t_1757_, lean_object* v_failed_1758_){
_start:
{
lean_object* v___x_1759_; 
v___x_1759_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1757_, v_failed_1758_);
return v___x_1759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_failed_elim(lean_object* v_motive_1760_, lean_object* v_t_1761_, lean_object* v_h_1762_, lean_object* v_failed_1763_){
_start:
{
lean_object* v___x_1764_; 
v___x_1764_ = lp_mathlib_Mathlib_Meta_FunProp_DecompositionResult_ctorElim___redArg(v_t_1761_, v_failed_1763_);
return v___x_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0(uint8_t v___x_1768_, lean_object* v___x_1769_, lean_object* v_mainVar_1770_, uint8_t v___x_1771_, lean_object* v___x_1772_, lean_object* v_expr_1773_, lean_object* v_args_1774_, lean_object* v___x_1775_, lean_object* v_fn_1776_, lean_object* v_coe_1777_, lean_object* v_n_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_){
_start:
{
uint8_t v___y_1785_; lean_object* v___y_1786_; uint8_t v___y_1824_; 
if (v___x_1768_ == 0)
{
uint8_t v___x_1832_; 
v___x_1832_ = lean_nat_dec_eq(v_n_1778_, v___x_1769_);
if (v___x_1832_ == 0)
{
v___y_1824_ = v___x_1832_;
goto v___jp_1823_;
}
else
{
uint8_t v___x_1833_; 
v___x_1833_ = lean_expr_eqv(v_mainVar_1770_, v_fn_1776_);
v___y_1824_ = v___x_1833_;
goto v___jp_1823_;
}
}
else
{
lean_object* v___x_1834_; lean_object* v___x_1835_; 
lean_dec(v_coe_1777_);
lean_dec_ref(v_fn_1776_);
lean_dec(v___x_1775_);
lean_dec_ref(v_args_1774_);
lean_dec_ref(v_expr_1773_);
lean_dec(v___x_1772_);
lean_dec_ref(v_mainVar_1770_);
v___x_1834_ = lean_box(2);
v___x_1835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1834_);
return v___x_1835_;
}
v___jp_1784_:
{
lean_object* v___x_1787_; lean_object* v___x_1788_; uint8_t v___x_1789_; lean_object* v___x_1790_; 
v___x_1787_ = lean_mk_empty_array_with_capacity(v___x_1769_);
v___x_1788_ = lean_array_push(v___x_1787_, v_mainVar_1770_);
v___x_1789_ = 1;
lean_inc_ref(v___y_1786_);
v___x_1790_ = l_Lean_Meta_mkLambdaFVars(v___x_1788_, v___y_1786_, v___y_1785_, v___x_1771_, v___y_1785_, v___x_1771_, v___x_1789_, v___y_1779_, v___y_1780_, v___y_1781_, v___y_1782_);
lean_dec_ref(v___x_1788_);
if (lean_obj_tag(v___x_1790_) == 0)
{
lean_object* v_a_1791_; lean_object* v___x_1792_; 
v_a_1791_ = lean_ctor_get(v___x_1790_, 0);
lean_inc(v_a_1791_);
lean_dec_ref_known(v___x_1790_, 1);
lean_inc(v___y_1782_);
lean_inc_ref(v___y_1781_);
lean_inc(v___y_1780_);
lean_inc_ref(v___y_1779_);
v___x_1792_ = lean_infer_type(v___y_1786_, v___y_1779_, v___y_1780_, v___y_1781_, v___y_1782_);
if (lean_obj_tag(v___x_1792_) == 0)
{
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1806_; 
v_a_1793_ = lean_ctor_get(v___x_1792_, 0);
v_isSharedCheck_1806_ = !lean_is_exclusive(v___x_1792_);
if (v_isSharedCheck_1806_ == 0)
{
v___x_1795_ = v___x_1792_;
v_isShared_1796_ = v_isSharedCheck_1806_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1792_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1806_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; uint8_t v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1804_; 
v___x_1797_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___closed__1));
v___x_1798_ = l_Lean_Expr_bvar___override(v___x_1772_);
v___x_1799_ = l_Lean_Expr_app___override(v___x_1798_, v_expr_1773_);
v___x_1800_ = 0;
v___x_1801_ = l_Lean_Expr_lam___override(v___x_1797_, v_a_1793_, v___x_1799_, v___x_1800_);
v___x_1802_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1802_, 0, v___x_1801_);
lean_ctor_set(v___x_1802_, 1, v_a_1791_);
if (v_isShared_1796_ == 0)
{
lean_ctor_set(v___x_1795_, 0, v___x_1802_);
v___x_1804_ = v___x_1795_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1805_; 
v_reuseFailAlloc_1805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1805_, 0, v___x_1802_);
v___x_1804_ = v_reuseFailAlloc_1805_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
return v___x_1804_;
}
}
}
else
{
lean_object* v_a_1807_; lean_object* v___x_1809_; uint8_t v_isShared_1810_; uint8_t v_isSharedCheck_1814_; 
lean_dec(v_a_1791_);
lean_dec_ref(v_expr_1773_);
lean_dec(v___x_1772_);
v_a_1807_ = lean_ctor_get(v___x_1792_, 0);
v_isSharedCheck_1814_ = !lean_is_exclusive(v___x_1792_);
if (v_isSharedCheck_1814_ == 0)
{
v___x_1809_ = v___x_1792_;
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
else
{
lean_inc(v_a_1807_);
lean_dec(v___x_1792_);
v___x_1809_ = lean_box(0);
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
v_resetjp_1808_:
{
lean_object* v___x_1812_; 
if (v_isShared_1810_ == 0)
{
v___x_1812_ = v___x_1809_;
goto v_reusejp_1811_;
}
else
{
lean_object* v_reuseFailAlloc_1813_; 
v_reuseFailAlloc_1813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1813_, 0, v_a_1807_);
v___x_1812_ = v_reuseFailAlloc_1813_;
goto v_reusejp_1811_;
}
v_reusejp_1811_:
{
return v___x_1812_;
}
}
}
}
else
{
lean_object* v_a_1815_; lean_object* v___x_1817_; uint8_t v_isShared_1818_; uint8_t v_isSharedCheck_1822_; 
lean_dec_ref(v___y_1786_);
lean_dec_ref(v_expr_1773_);
lean_dec(v___x_1772_);
v_a_1815_ = lean_ctor_get(v___x_1790_, 0);
v_isSharedCheck_1822_ = !lean_is_exclusive(v___x_1790_);
if (v_isSharedCheck_1822_ == 0)
{
v___x_1817_ = v___x_1790_;
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
else
{
lean_inc(v_a_1815_);
lean_dec(v___x_1790_);
v___x_1817_ = lean_box(0);
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
v_resetjp_1816_:
{
lean_object* v___x_1820_; 
if (v_isShared_1818_ == 0)
{
v___x_1820_ = v___x_1817_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v_a_1815_);
v___x_1820_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
return v___x_1820_;
}
}
}
}
v___jp_1823_:
{
if (v___y_1824_ == 0)
{
lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v_gBody_x27_1827_; 
lean_inc(v___x_1772_);
v___x_1825_ = l_Array_toSubarray___redArg(v_args_1774_, v___x_1772_, v___x_1775_);
v___x_1826_ = l_Subarray_copy___redArg(v___x_1825_);
v_gBody_x27_1827_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_fn_1776_, v___x_1826_);
lean_dec_ref(v___x_1826_);
if (lean_obj_tag(v_coe_1777_) == 1)
{
lean_object* v_val_1828_; lean_object* v___x_1829_; 
v_val_1828_ = lean_ctor_get(v_coe_1777_, 0);
lean_inc(v_val_1828_);
lean_dec_ref_known(v_coe_1777_, 1);
v___x_1829_ = l_Lean_Expr_app___override(v_val_1828_, v_gBody_x27_1827_);
v___y_1785_ = v___y_1824_;
v___y_1786_ = v___x_1829_;
goto v___jp_1784_;
}
else
{
lean_dec(v_coe_1777_);
v___y_1785_ = v___y_1824_;
v___y_1786_ = v_gBody_x27_1827_;
goto v___jp_1784_;
}
}
else
{
lean_object* v___x_1830_; lean_object* v___x_1831_; 
lean_dec(v_coe_1777_);
lean_dec_ref(v_fn_1776_);
lean_dec(v___x_1775_);
lean_dec_ref(v_args_1774_);
lean_dec_ref(v_expr_1773_);
lean_dec(v___x_1772_);
lean_dec_ref(v_mainVar_1770_);
v___x_1830_ = lean_box(2);
v___x_1831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1831_, 0, v___x_1830_);
return v___x_1831_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___boxed(lean_object* v___x_1836_, lean_object* v___x_1837_, lean_object* v_mainVar_1838_, lean_object* v___x_1839_, lean_object* v___x_1840_, lean_object* v_expr_1841_, lean_object* v_args_1842_, lean_object* v___x_1843_, lean_object* v_fn_1844_, lean_object* v_coe_1845_, lean_object* v_n_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_){
_start:
{
uint8_t v___x_1075__boxed_1852_; uint8_t v___x_1077__boxed_1853_; lean_object* v_res_1854_; 
v___x_1075__boxed_1852_ = lean_unbox(v___x_1836_);
v___x_1077__boxed_1853_ = lean_unbox(v___x_1839_);
v_res_1854_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0(v___x_1075__boxed_1852_, v___x_1837_, v_mainVar_1838_, v___x_1077__boxed_1853_, v___x_1840_, v_expr_1841_, v_args_1842_, v___x_1843_, v_fn_1844_, v_coe_1845_, v_n_1846_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_);
lean_dec(v___y_1850_);
lean_dec_ref(v___y_1849_);
lean_dec(v___y_1848_);
lean_dec_ref(v___y_1847_);
lean_dec(v_n_1846_);
lean_dec(v___x_1837_);
return v_res_1854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition(lean_object* v_fData_1855_, lean_object* v_a_1856_, lean_object* v_a_1857_, lean_object* v_a_1858_, lean_object* v_a_1859_){
_start:
{
lean_object* v_lctx_1861_; lean_object* v_insts_1862_; lean_object* v_fn_1863_; lean_object* v_args_1864_; lean_object* v_mainVar_1865_; lean_object* v___x_1866_; lean_object* v_n_1867_; uint8_t v___x_1868_; 
v_lctx_1861_ = lean_ctor_get(v_fData_1855_, 0);
lean_inc_ref(v_lctx_1861_);
v_insts_1862_ = lean_ctor_get(v_fData_1855_, 1);
lean_inc_ref(v_insts_1862_);
v_fn_1863_ = lean_ctor_get(v_fData_1855_, 2);
lean_inc_ref(v_fn_1863_);
v_args_1864_ = lean_ctor_get(v_fData_1855_, 3);
lean_inc_ref(v_args_1864_);
v_mainVar_1865_ = lean_ctor_get(v_fData_1855_, 4);
lean_inc_ref(v_mainVar_1865_);
lean_dec_ref(v_fData_1855_);
v___x_1866_ = lean_unsigned_to_nat(0u);
v_n_1867_ = lean_array_get_size(v_args_1864_);
v___x_1868_ = lean_nat_dec_lt(v___x_1866_, v_n_1867_);
if (v___x_1868_ == 0)
{
lean_object* v___x_1869_; lean_object* v___x_1870_; 
lean_dec_ref(v_mainVar_1865_);
lean_dec_ref(v_args_1864_);
lean_dec_ref(v_fn_1863_);
lean_dec_ref(v_insts_1862_);
lean_dec_ref(v_lctx_1861_);
v___x_1869_ = lean_box(2);
v___x_1870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1870_, 0, v___x_1869_);
return v___x_1870_;
}
else
{
lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v_y_u2099_1874_; lean_object* v_expr_1875_; lean_object* v_coe_1876_; lean_object* v___x_1877_; uint8_t v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___y_1881_; lean_object* v___x_1882_; 
v___x_1871_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
v___x_1872_ = lean_unsigned_to_nat(1u);
v___x_1873_ = lean_nat_sub(v_n_1867_, v___x_1872_);
v_y_u2099_1874_ = lean_array_get_borrowed(v___x_1871_, v_args_1864_, v___x_1873_);
v_expr_1875_ = lean_ctor_get(v_y_u2099_1874_, 0);
lean_inc_ref(v_expr_1875_);
v_coe_1876_ = lean_ctor_get(v_y_u2099_1874_, 1);
lean_inc(v_coe_1876_);
v___x_1877_ = l_Lean_Expr_fvarId_x21(v_mainVar_1865_);
v___x_1878_ = l_Lean_Expr_containsFVar(v_expr_1875_, v___x_1877_);
lean_dec(v___x_1877_);
v___x_1879_ = lean_box(v___x_1878_);
v___x_1880_ = lean_box(v___x_1868_);
v___y_1881_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___lam__0___boxed), 16, 11);
lean_closure_set(v___y_1881_, 0, v___x_1879_);
lean_closure_set(v___y_1881_, 1, v___x_1872_);
lean_closure_set(v___y_1881_, 2, v_mainVar_1865_);
lean_closure_set(v___y_1881_, 3, v___x_1880_);
lean_closure_set(v___y_1881_, 4, v___x_1866_);
lean_closure_set(v___y_1881_, 5, v_expr_1875_);
lean_closure_set(v___y_1881_, 6, v_args_1864_);
lean_closure_set(v___y_1881_, 7, v___x_1873_);
lean_closure_set(v___y_1881_, 8, v_fn_1863_);
lean_closure_set(v___y_1881_, 9, v_coe_1876_);
lean_closure_set(v___y_1881_, 10, v_n_1867_);
v___x_1882_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_1861_, v_insts_1862_, v___y_1881_, v_a_1856_, v_a_1857_, v_a_1858_, v_a_1859_);
return v___x_1882_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition___boxed(lean_object* v_fData_1883_, lean_object* v_a_1884_, lean_object* v_a_1885_, lean_object* v_a_1886_, lean_object* v_a_1887_, lean_object* v_a_1888_){
_start:
{
lean_object* v_res_1889_; 
v_res_1889_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition(v_fData_1883_, v_a_1884_, v_a_1885_, v_a_1886_, v_a_1887_);
lean_dec(v_a_1887_);
lean_dec_ref(v_a_1886_);
lean_dec(v_a_1885_);
lean_dec_ref(v_a_1884_);
return v_res_1889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0(lean_object* v_fst_1890_, lean_object* v_mainVar_1891_, uint8_t v___x_1892_, uint8_t v___x_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_){
_start:
{
lean_object* v___x_1899_; 
v___x_1899_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(v_fst_1890_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
if (lean_obj_tag(v___x_1899_) == 0)
{
lean_object* v_a_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; uint8_t v___x_1904_; lean_object* v___x_1905_; 
v_a_1900_ = lean_ctor_get(v___x_1899_, 0);
lean_inc(v_a_1900_);
lean_dec_ref_known(v___x_1899_, 1);
v___x_1901_ = lean_unsigned_to_nat(1u);
v___x_1902_ = lean_mk_empty_array_with_capacity(v___x_1901_);
v___x_1903_ = lean_array_push(v___x_1902_, v_mainVar_1891_);
v___x_1904_ = 1;
v___x_1905_ = l_Lean_Meta_mkLambdaFVars(v___x_1903_, v_a_1900_, v___x_1892_, v___x_1893_, v___x_1892_, v___x_1893_, v___x_1904_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
lean_dec_ref(v___x_1903_);
return v___x_1905_;
}
else
{
lean_dec_ref(v_mainVar_1891_);
return v___x_1899_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0___boxed(lean_object* v_fst_1906_, lean_object* v_mainVar_1907_, lean_object* v___x_1908_, lean_object* v___x_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_){
_start:
{
uint8_t v___x_5755__boxed_1915_; uint8_t v___x_5756__boxed_1916_; lean_object* v_res_1917_; 
v___x_5755__boxed_1915_ = lean_unbox(v___x_1908_);
v___x_5756__boxed_1916_ = lean_unbox(v___x_1909_);
v_res_1917_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0(v_fst_1906_, v_mainVar_1907_, v___x_5755__boxed_1915_, v___x_5756__boxed_1916_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_);
lean_dec(v___y_1913_);
lean_dec_ref(v___y_1912_);
lean_dec(v___y_1911_);
lean_dec_ref(v___y_1910_);
return v_res_1917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1(lean_object* v_snd_1918_, lean_object* v___x_1919_, uint8_t v___x_1920_, uint8_t v___x_1921_, uint8_t v___x_1922_, lean_object* v___x_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_){
_start:
{
lean_object* v___x_1929_; 
v___x_1929_ = l_Lean_Meta_mkLambdaFVars(v_snd_1918_, v___x_1919_, v___x_1920_, v___x_1921_, v___x_1920_, v___x_1921_, v___x_1922_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_);
if (lean_obj_tag(v___x_1929_) == 0)
{
lean_object* v_a_1930_; lean_object* v___x_1931_; 
v_a_1930_ = lean_ctor_get(v___x_1929_, 0);
lean_inc(v_a_1930_);
lean_dec_ref_known(v___x_1929_, 1);
v___x_1931_ = lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun(v___x_1923_, v_a_1930_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_);
return v___x_1931_;
}
else
{
lean_dec(v___x_1923_);
return v___x_1929_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1___boxed(lean_object* v_snd_1932_, lean_object* v___x_1933_, lean_object* v___x_1934_, lean_object* v___x_1935_, lean_object* v___x_1936_, lean_object* v___x_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_){
_start:
{
uint8_t v___x_5792__boxed_1943_; uint8_t v___x_5793__boxed_1944_; uint8_t v___x_5794__boxed_1945_; lean_object* v_res_1946_; 
v___x_5792__boxed_1943_ = lean_unbox(v___x_1934_);
v___x_5793__boxed_1944_ = lean_unbox(v___x_1935_);
v___x_5794__boxed_1945_ = lean_unbox(v___x_1936_);
v_res_1946_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1(v_snd_1932_, v___x_1933_, v___x_5792__boxed_1943_, v___x_5793__boxed_1944_, v___x_5794__boxed_1945_, v___x_1937_, v___y_1938_, v___y_1939_, v___y_1940_, v___y_1941_);
lean_dec(v___y_1941_);
lean_dec_ref(v___y_1940_);
lean_dec(v___y_1939_);
lean_dec_ref(v___y_1938_);
lean_dec(v_snd_1932_);
return v_res_1946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg(lean_object* v___y_1947_){
_start:
{
lean_object* v___x_1949_; lean_object* v_ngen_1950_; lean_object* v_namePrefix_1951_; lean_object* v_idx_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_1981_; 
v___x_1949_ = lean_st_ref_get(v___y_1947_);
v_ngen_1950_ = lean_ctor_get(v___x_1949_, 2);
lean_inc_ref(v_ngen_1950_);
lean_dec(v___x_1949_);
v_namePrefix_1951_ = lean_ctor_get(v_ngen_1950_, 0);
v_idx_1952_ = lean_ctor_get(v_ngen_1950_, 1);
v_isSharedCheck_1981_ = !lean_is_exclusive(v_ngen_1950_);
if (v_isSharedCheck_1981_ == 0)
{
v___x_1954_ = v_ngen_1950_;
v_isShared_1955_ = v_isSharedCheck_1981_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_idx_1952_);
lean_inc(v_namePrefix_1951_);
lean_dec(v_ngen_1950_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_1981_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v___x_1956_; lean_object* v_env_1957_; lean_object* v_nextMacroScope_1958_; lean_object* v_auxDeclNGen_1959_; lean_object* v_traceState_1960_; lean_object* v_cache_1961_; lean_object* v_messages_1962_; lean_object* v_infoState_1963_; lean_object* v_snapshotTasks_1964_; lean_object* v___x_1966_; uint8_t v_isShared_1967_; uint8_t v_isSharedCheck_1979_; 
v___x_1956_ = lean_st_ref_take(v___y_1947_);
v_env_1957_ = lean_ctor_get(v___x_1956_, 0);
v_nextMacroScope_1958_ = lean_ctor_get(v___x_1956_, 1);
v_auxDeclNGen_1959_ = lean_ctor_get(v___x_1956_, 3);
v_traceState_1960_ = lean_ctor_get(v___x_1956_, 4);
v_cache_1961_ = lean_ctor_get(v___x_1956_, 5);
v_messages_1962_ = lean_ctor_get(v___x_1956_, 6);
v_infoState_1963_ = lean_ctor_get(v___x_1956_, 7);
v_snapshotTasks_1964_ = lean_ctor_get(v___x_1956_, 8);
v_isSharedCheck_1979_ = !lean_is_exclusive(v___x_1956_);
if (v_isSharedCheck_1979_ == 0)
{
lean_object* v_unused_1980_; 
v_unused_1980_ = lean_ctor_get(v___x_1956_, 2);
lean_dec(v_unused_1980_);
v___x_1966_ = v___x_1956_;
v_isShared_1967_ = v_isSharedCheck_1979_;
goto v_resetjp_1965_;
}
else
{
lean_inc(v_snapshotTasks_1964_);
lean_inc(v_infoState_1963_);
lean_inc(v_messages_1962_);
lean_inc(v_cache_1961_);
lean_inc(v_traceState_1960_);
lean_inc(v_auxDeclNGen_1959_);
lean_inc(v_nextMacroScope_1958_);
lean_inc(v_env_1957_);
lean_dec(v___x_1956_);
v___x_1966_ = lean_box(0);
v_isShared_1967_ = v_isSharedCheck_1979_;
goto v_resetjp_1965_;
}
v_resetjp_1965_:
{
lean_object* v_r_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1972_; 
lean_inc(v_idx_1952_);
lean_inc(v_namePrefix_1951_);
v_r_1968_ = l_Lean_Name_num___override(v_namePrefix_1951_, v_idx_1952_);
v___x_1969_ = lean_unsigned_to_nat(1u);
v___x_1970_ = lean_nat_add(v_idx_1952_, v___x_1969_);
lean_dec(v_idx_1952_);
if (v_isShared_1955_ == 0)
{
lean_ctor_set(v___x_1954_, 1, v___x_1970_);
v___x_1972_ = v___x_1954_;
goto v_reusejp_1971_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v_namePrefix_1951_);
lean_ctor_set(v_reuseFailAlloc_1978_, 1, v___x_1970_);
v___x_1972_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1971_;
}
v_reusejp_1971_:
{
lean_object* v___x_1974_; 
if (v_isShared_1967_ == 0)
{
lean_ctor_set(v___x_1966_, 2, v___x_1972_);
v___x_1974_ = v___x_1966_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_1977_; 
v_reuseFailAlloc_1977_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1977_, 0, v_env_1957_);
lean_ctor_set(v_reuseFailAlloc_1977_, 1, v_nextMacroScope_1958_);
lean_ctor_set(v_reuseFailAlloc_1977_, 2, v___x_1972_);
lean_ctor_set(v_reuseFailAlloc_1977_, 3, v_auxDeclNGen_1959_);
lean_ctor_set(v_reuseFailAlloc_1977_, 4, v_traceState_1960_);
lean_ctor_set(v_reuseFailAlloc_1977_, 5, v_cache_1961_);
lean_ctor_set(v_reuseFailAlloc_1977_, 6, v_messages_1962_);
lean_ctor_set(v_reuseFailAlloc_1977_, 7, v_infoState_1963_);
lean_ctor_set(v_reuseFailAlloc_1977_, 8, v_snapshotTasks_1964_);
v___x_1974_ = v_reuseFailAlloc_1977_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
lean_object* v___x_1975_; lean_object* v___x_1976_; 
v___x_1975_ = lean_st_ref_set(v___y_1947_, v___x_1974_);
v___x_1976_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1976_, 0, v_r_1968_);
return v___x_1976_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg___boxed(lean_object* v___y_1982_, lean_object* v___y_1983_){
_start:
{
lean_object* v_res_1984_; 
v_res_1984_ = lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg(v___y_1982_);
lean_dec(v___y_1982_);
return v_res_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0(lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_){
_start:
{
lean_object* v___x_1990_; lean_object* v_a_1991_; lean_object* v___x_1993_; uint8_t v_isShared_1994_; uint8_t v_isSharedCheck_1998_; 
v___x_1990_ = lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg(v___y_1988_);
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
v_isSharedCheck_1998_ = !lean_is_exclusive(v___x_1990_);
if (v_isSharedCheck_1998_ == 0)
{
v___x_1993_ = v___x_1990_;
v_isShared_1994_ = v_isSharedCheck_1998_;
goto v_resetjp_1992_;
}
else
{
lean_inc(v_a_1991_);
lean_dec(v___x_1990_);
v___x_1993_ = lean_box(0);
v_isShared_1994_ = v_isSharedCheck_1998_;
goto v_resetjp_1992_;
}
v_resetjp_1992_:
{
lean_object* v___x_1996_; 
if (v_isShared_1994_ == 0)
{
v___x_1996_ = v___x_1993_;
goto v_reusejp_1995_;
}
else
{
lean_object* v_reuseFailAlloc_1997_; 
v_reuseFailAlloc_1997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1997_, 0, v_a_1991_);
v___x_1996_ = v_reuseFailAlloc_1997_;
goto v_reusejp_1995_;
}
v_reusejp_1995_:
{
return v___x_1996_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0___boxed(lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_){
_start:
{
lean_object* v_res_2004_; 
v_res_2004_ = lp_mathlib_Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0(v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_);
lean_dec(v___y_2002_);
lean_dec_ref(v___y_2001_);
lean_dec(v___y_2000_);
lean_dec_ref(v___y_1999_);
return v_res_2004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1(lean_object* v_insts_2008_, lean_object* v_x_2009_, lean_object* v_a_2010_, lean_object* v_as_2011_, size_t v_sz_2012_, size_t v_i_2013_, lean_object* v_b_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_){
_start:
{
uint8_t v___x_2020_; 
v___x_2020_ = lean_usize_dec_lt(v_i_2013_, v_sz_2012_);
if (v___x_2020_ == 0)
{
lean_object* v___x_2021_; 
lean_dec(v_a_2010_);
lean_dec_ref(v_insts_2008_);
v___x_2021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2021_, 0, v_b_2014_);
return v___x_2021_;
}
else
{
lean_object* v_snd_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2125_; 
v_snd_2022_ = lean_ctor_get(v_b_2014_, 1);
v_isSharedCheck_2125_ = !lean_is_exclusive(v_b_2014_);
if (v_isSharedCheck_2125_ == 0)
{
lean_object* v_unused_2126_; 
v_unused_2126_ = lean_ctor_get(v_b_2014_, 0);
lean_dec(v_unused_2126_);
v___x_2024_ = v_b_2014_;
v_isShared_2025_ = v_isSharedCheck_2125_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_snd_2022_);
lean_dec(v_b_2014_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2125_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v_fst_2026_; lean_object* v_snd_2027_; lean_object* v___x_2029_; uint8_t v_isShared_2030_; uint8_t v_isSharedCheck_2124_; 
v_fst_2026_ = lean_ctor_get(v_snd_2022_, 0);
v_snd_2027_ = lean_ctor_get(v_snd_2022_, 1);
v_isSharedCheck_2124_ = !lean_is_exclusive(v_snd_2022_);
if (v_isSharedCheck_2124_ == 0)
{
v___x_2029_ = v_snd_2022_;
v_isShared_2030_ = v_isSharedCheck_2124_;
goto v_resetjp_2028_;
}
else
{
lean_inc(v_snd_2027_);
lean_inc(v_fst_2026_);
lean_dec(v_snd_2022_);
v___x_2029_ = lean_box(0);
v_isShared_2030_ = v_isSharedCheck_2124_;
goto v_resetjp_2028_;
}
v_resetjp_2028_:
{
lean_object* v___x_2031_; lean_object* v___x_2032_; 
v___x_2031_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__0));
lean_inc_ref(v_insts_2008_);
lean_inc(v_fst_2026_);
v___x_2032_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_fst_2026_, v_insts_2008_, v___x_2031_, v___y_2015_, v___y_2016_, v___y_2017_, v___y_2018_);
if (lean_obj_tag(v___x_2032_) == 0)
{
lean_object* v_a_2033_; lean_object* v_fst_2034_; lean_object* v_snd_2035_; lean_object* v___x_2037_; uint8_t v_isShared_2038_; uint8_t v_isSharedCheck_2115_; 
v_a_2033_ = lean_ctor_get(v___x_2032_, 0);
lean_inc(v_a_2033_);
lean_dec_ref_known(v___x_2032_, 1);
v_fst_2034_ = lean_ctor_get(v_snd_2027_, 0);
v_snd_2035_ = lean_ctor_get(v_snd_2027_, 1);
v_isSharedCheck_2115_ = !lean_is_exclusive(v_snd_2027_);
if (v_isSharedCheck_2115_ == 0)
{
v___x_2037_ = v_snd_2027_;
v_isShared_2038_ = v_isSharedCheck_2115_;
goto v_resetjp_2036_;
}
else
{
lean_inc(v_snd_2035_);
lean_inc(v_fst_2034_);
lean_dec(v_snd_2027_);
v___x_2037_ = lean_box(0);
v_isShared_2038_ = v_isSharedCheck_2115_;
goto v_resetjp_2036_;
}
v_resetjp_2036_:
{
lean_object* v_a_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v_expr_2042_; lean_object* v_coe_2043_; lean_object* v___x_2045_; uint8_t v_isShared_2046_; uint8_t v_isSharedCheck_2114_; 
v_a_2039_ = lean_array_uget_borrowed(v_as_2011_, v_i_2013_);
v___x_2040_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
v___x_2041_ = lean_array_get(v___x_2040_, v_fst_2034_, v_a_2039_);
v_expr_2042_ = lean_ctor_get(v___x_2041_, 0);
v_coe_2043_ = lean_ctor_get(v___x_2041_, 1);
v_isSharedCheck_2114_ = !lean_is_exclusive(v___x_2041_);
if (v_isSharedCheck_2114_ == 0)
{
v___x_2045_ = v___x_2041_;
v_isShared_2046_ = v_isSharedCheck_2114_;
goto v_resetjp_2044_;
}
else
{
lean_inc(v_coe_2043_);
lean_inc(v_expr_2042_);
lean_dec(v___x_2041_);
v___x_2045_ = lean_box(0);
v_isShared_2046_ = v_isSharedCheck_2114_;
goto v_resetjp_2044_;
}
v_resetjp_2044_:
{
lean_object* v___x_2047_; lean_object* v___x_2048_; 
lean_inc_ref(v_expr_2042_);
v___x_2047_ = lean_alloc_closure((void*)(l_Lean_Meta_inferType___boxed), 6, 1);
lean_closure_set(v___x_2047_, 0, v_expr_2042_);
lean_inc_ref(v_insts_2008_);
lean_inc(v_fst_2026_);
v___x_2048_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_fst_2026_, v_insts_2008_, v___x_2047_, v___y_2015_, v___y_2016_, v___y_2017_, v___y_2018_);
if (lean_obj_tag(v___x_2048_) == 0)
{
lean_object* v_a_2049_; lean_object* v___x_2051_; uint8_t v_isShared_2052_; uint8_t v_isSharedCheck_2105_; 
v_a_2049_ = lean_ctor_get(v___x_2048_, 0);
v_isSharedCheck_2105_ = !lean_is_exclusive(v___x_2048_);
if (v_isSharedCheck_2105_ == 0)
{
v___x_2051_ = v___x_2048_;
v_isShared_2052_ = v_isSharedCheck_2105_;
goto v_resetjp_2050_;
}
else
{
lean_inc(v_a_2049_);
lean_dec(v___x_2048_);
v___x_2051_ = lean_box(0);
v_isShared_2052_ = v_isSharedCheck_2105_;
goto v_resetjp_2050_;
}
v_resetjp_2050_:
{
lean_object* v_fst_2053_; lean_object* v_snd_2054_; lean_object* v___x_2056_; uint8_t v_isShared_2057_; uint8_t v_isSharedCheck_2104_; 
v_fst_2053_ = lean_ctor_get(v_snd_2035_, 0);
v_snd_2054_ = lean_ctor_get(v_snd_2035_, 1);
v_isSharedCheck_2104_ = !lean_is_exclusive(v_snd_2035_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2056_ = v_snd_2035_;
v_isShared_2057_ = v_isSharedCheck_2104_;
goto v_resetjp_2055_;
}
else
{
lean_inc(v_snd_2054_);
lean_inc(v_fst_2053_);
lean_dec(v_snd_2035_);
v___x_2056_ = lean_box(0);
v_isShared_2057_ = v_isSharedCheck_2104_;
goto v_resetjp_2055_;
}
v_resetjp_2055_:
{
lean_object* v___x_2058_; uint8_t v___x_2059_; 
v___x_2058_ = l_Lean_Expr_fvarId_x21(v_x_2009_);
v___x_2059_ = l_Lean_Expr_containsFVar(v_a_2049_, v___x_2058_);
lean_dec(v___x_2058_);
if (v___x_2059_ == 0)
{
lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; uint8_t v___x_2063_; uint8_t v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2070_; 
lean_del_object(v___x_2051_);
v___x_2060_ = lean_box(0);
lean_inc(v_a_2039_);
v___x_2061_ = l_Nat_reprFast(v_a_2039_);
lean_inc(v_a_2010_);
v___x_2062_ = lean_name_append_after(v_a_2010_, v___x_2061_);
v___x_2063_ = 0;
v___x_2064_ = 0;
lean_inc(v_a_2033_);
v___x_2065_ = l_Lean_LocalContext_mkLocalDecl(v_fst_2026_, v_a_2033_, v___x_2062_, v_a_2049_, v___x_2063_, v___x_2064_);
v___x_2066_ = l_Lean_Expr_fvar___override(v_a_2033_);
lean_inc_ref(v___x_2066_);
v___x_2067_ = lean_array_push(v_snd_2054_, v___x_2066_);
v___x_2068_ = lean_array_push(v_fst_2053_, v_expr_2042_);
if (v_isShared_2046_ == 0)
{
lean_ctor_set(v___x_2045_, 0, v___x_2066_);
v___x_2070_ = v___x_2045_;
goto v_reusejp_2069_;
}
else
{
lean_object* v_reuseFailAlloc_2087_; 
v_reuseFailAlloc_2087_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2087_, 0, v___x_2066_);
lean_ctor_set(v_reuseFailAlloc_2087_, 1, v_coe_2043_);
v___x_2070_ = v_reuseFailAlloc_2087_;
goto v_reusejp_2069_;
}
v_reusejp_2069_:
{
lean_object* v___x_2071_; lean_object* v___x_2073_; 
v___x_2071_ = lean_array_set(v_fst_2034_, v_a_2039_, v___x_2070_);
if (v_isShared_2057_ == 0)
{
lean_ctor_set(v___x_2056_, 1, v___x_2067_);
lean_ctor_set(v___x_2056_, 0, v___x_2068_);
v___x_2073_ = v___x_2056_;
goto v_reusejp_2072_;
}
else
{
lean_object* v_reuseFailAlloc_2086_; 
v_reuseFailAlloc_2086_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2086_, 0, v___x_2068_);
lean_ctor_set(v_reuseFailAlloc_2086_, 1, v___x_2067_);
v___x_2073_ = v_reuseFailAlloc_2086_;
goto v_reusejp_2072_;
}
v_reusejp_2072_:
{
lean_object* v___x_2075_; 
if (v_isShared_2038_ == 0)
{
lean_ctor_set(v___x_2037_, 1, v___x_2073_);
lean_ctor_set(v___x_2037_, 0, v___x_2071_);
v___x_2075_ = v___x_2037_;
goto v_reusejp_2074_;
}
else
{
lean_object* v_reuseFailAlloc_2085_; 
v_reuseFailAlloc_2085_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2085_, 0, v___x_2071_);
lean_ctor_set(v_reuseFailAlloc_2085_, 1, v___x_2073_);
v___x_2075_ = v_reuseFailAlloc_2085_;
goto v_reusejp_2074_;
}
v_reusejp_2074_:
{
lean_object* v___x_2077_; 
if (v_isShared_2030_ == 0)
{
lean_ctor_set(v___x_2029_, 1, v___x_2075_);
lean_ctor_set(v___x_2029_, 0, v___x_2065_);
v___x_2077_ = v___x_2029_;
goto v_reusejp_2076_;
}
else
{
lean_object* v_reuseFailAlloc_2084_; 
v_reuseFailAlloc_2084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2084_, 0, v___x_2065_);
lean_ctor_set(v_reuseFailAlloc_2084_, 1, v___x_2075_);
v___x_2077_ = v_reuseFailAlloc_2084_;
goto v_reusejp_2076_;
}
v_reusejp_2076_:
{
lean_object* v___x_2079_; 
if (v_isShared_2025_ == 0)
{
lean_ctor_set(v___x_2024_, 1, v___x_2077_);
lean_ctor_set(v___x_2024_, 0, v___x_2060_);
v___x_2079_ = v___x_2024_;
goto v_reusejp_2078_;
}
else
{
lean_object* v_reuseFailAlloc_2083_; 
v_reuseFailAlloc_2083_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2083_, 0, v___x_2060_);
lean_ctor_set(v_reuseFailAlloc_2083_, 1, v___x_2077_);
v___x_2079_ = v_reuseFailAlloc_2083_;
goto v_reusejp_2078_;
}
v_reusejp_2078_:
{
size_t v___x_2080_; size_t v___x_2081_; 
v___x_2080_ = ((size_t)1ULL);
v___x_2081_ = lean_usize_add(v_i_2013_, v___x_2080_);
v_i_2013_ = v___x_2081_;
v_b_2014_ = v___x_2079_;
goto _start;
}
}
}
}
}
}
else
{
lean_object* v___x_2088_; lean_object* v___x_2090_; 
lean_dec(v_a_2049_);
lean_del_object(v___x_2045_);
lean_dec(v_coe_2043_);
lean_dec_ref(v_expr_2042_);
lean_dec(v_a_2033_);
lean_dec(v_a_2010_);
lean_dec_ref(v_insts_2008_);
v___x_2088_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___closed__1));
if (v_isShared_2057_ == 0)
{
v___x_2090_ = v___x_2056_;
goto v_reusejp_2089_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_fst_2053_);
lean_ctor_set(v_reuseFailAlloc_2103_, 1, v_snd_2054_);
v___x_2090_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2089_;
}
v_reusejp_2089_:
{
lean_object* v___x_2092_; 
if (v_isShared_2038_ == 0)
{
lean_ctor_set(v___x_2037_, 1, v___x_2090_);
v___x_2092_ = v___x_2037_;
goto v_reusejp_2091_;
}
else
{
lean_object* v_reuseFailAlloc_2102_; 
v_reuseFailAlloc_2102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2102_, 0, v_fst_2034_);
lean_ctor_set(v_reuseFailAlloc_2102_, 1, v___x_2090_);
v___x_2092_ = v_reuseFailAlloc_2102_;
goto v_reusejp_2091_;
}
v_reusejp_2091_:
{
lean_object* v___x_2094_; 
if (v_isShared_2030_ == 0)
{
lean_ctor_set(v___x_2029_, 1, v___x_2092_);
v___x_2094_ = v___x_2029_;
goto v_reusejp_2093_;
}
else
{
lean_object* v_reuseFailAlloc_2101_; 
v_reuseFailAlloc_2101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2101_, 0, v_fst_2026_);
lean_ctor_set(v_reuseFailAlloc_2101_, 1, v___x_2092_);
v___x_2094_ = v_reuseFailAlloc_2101_;
goto v_reusejp_2093_;
}
v_reusejp_2093_:
{
lean_object* v___x_2096_; 
if (v_isShared_2025_ == 0)
{
lean_ctor_set(v___x_2024_, 1, v___x_2094_);
lean_ctor_set(v___x_2024_, 0, v___x_2088_);
v___x_2096_ = v___x_2024_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2100_; 
v_reuseFailAlloc_2100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2100_, 0, v___x_2088_);
lean_ctor_set(v_reuseFailAlloc_2100_, 1, v___x_2094_);
v___x_2096_ = v_reuseFailAlloc_2100_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
lean_object* v___x_2098_; 
if (v_isShared_2052_ == 0)
{
lean_ctor_set(v___x_2051_, 0, v___x_2096_);
v___x_2098_ = v___x_2051_;
goto v_reusejp_2097_;
}
else
{
lean_object* v_reuseFailAlloc_2099_; 
v_reuseFailAlloc_2099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2099_, 0, v___x_2096_);
v___x_2098_ = v_reuseFailAlloc_2099_;
goto v_reusejp_2097_;
}
v_reusejp_2097_:
{
return v___x_2098_;
}
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
lean_object* v_a_2106_; lean_object* v___x_2108_; uint8_t v_isShared_2109_; uint8_t v_isSharedCheck_2113_; 
lean_del_object(v___x_2045_);
lean_dec(v_coe_2043_);
lean_dec_ref(v_expr_2042_);
lean_del_object(v___x_2037_);
lean_dec(v_snd_2035_);
lean_dec(v_fst_2034_);
lean_dec(v_a_2033_);
lean_del_object(v___x_2029_);
lean_dec(v_fst_2026_);
lean_del_object(v___x_2024_);
lean_dec(v_a_2010_);
lean_dec_ref(v_insts_2008_);
v_a_2106_ = lean_ctor_get(v___x_2048_, 0);
v_isSharedCheck_2113_ = !lean_is_exclusive(v___x_2048_);
if (v_isSharedCheck_2113_ == 0)
{
v___x_2108_ = v___x_2048_;
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
else
{
lean_inc(v_a_2106_);
lean_dec(v___x_2048_);
v___x_2108_ = lean_box(0);
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
v_resetjp_2107_:
{
lean_object* v___x_2111_; 
if (v_isShared_2109_ == 0)
{
v___x_2111_ = v___x_2108_;
goto v_reusejp_2110_;
}
else
{
lean_object* v_reuseFailAlloc_2112_; 
v_reuseFailAlloc_2112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2112_, 0, v_a_2106_);
v___x_2111_ = v_reuseFailAlloc_2112_;
goto v_reusejp_2110_;
}
v_reusejp_2110_:
{
return v___x_2111_;
}
}
}
}
}
}
else
{
lean_object* v_a_2116_; lean_object* v___x_2118_; uint8_t v_isShared_2119_; uint8_t v_isSharedCheck_2123_; 
lean_del_object(v___x_2029_);
lean_dec(v_snd_2027_);
lean_dec(v_fst_2026_);
lean_del_object(v___x_2024_);
lean_dec(v_a_2010_);
lean_dec_ref(v_insts_2008_);
v_a_2116_ = lean_ctor_get(v___x_2032_, 0);
v_isSharedCheck_2123_ = !lean_is_exclusive(v___x_2032_);
if (v_isSharedCheck_2123_ == 0)
{
v___x_2118_ = v___x_2032_;
v_isShared_2119_ = v_isSharedCheck_2123_;
goto v_resetjp_2117_;
}
else
{
lean_inc(v_a_2116_);
lean_dec(v___x_2032_);
v___x_2118_ = lean_box(0);
v_isShared_2119_ = v_isSharedCheck_2123_;
goto v_resetjp_2117_;
}
v_resetjp_2117_:
{
lean_object* v___x_2121_; 
if (v_isShared_2119_ == 0)
{
v___x_2121_ = v___x_2118_;
goto v_reusejp_2120_;
}
else
{
lean_object* v_reuseFailAlloc_2122_; 
v_reuseFailAlloc_2122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2122_, 0, v_a_2116_);
v___x_2121_ = v_reuseFailAlloc_2122_;
goto v_reusejp_2120_;
}
v_reusejp_2120_:
{
return v___x_2121_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1___boxed(lean_object* v_insts_2127_, lean_object* v_x_2128_, lean_object* v_a_2129_, lean_object* v_as_2130_, lean_object* v_sz_2131_, lean_object* v_i_2132_, lean_object* v_b_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_){
_start:
{
size_t v_sz_boxed_2139_; size_t v_i_boxed_2140_; lean_object* v_res_2141_; 
v_sz_boxed_2139_ = lean_unbox_usize(v_sz_2131_);
lean_dec(v_sz_2131_);
v_i_boxed_2140_ = lean_unbox_usize(v_i_2132_);
lean_dec(v_i_2132_);
v_res_2141_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1(v_insts_2127_, v_x_2128_, v_a_2129_, v_as_2130_, v_sz_boxed_2139_, v_i_boxed_2140_, v_b_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_);
lean_dec(v___y_2137_);
lean_dec_ref(v___y_2136_);
lean_dec(v___y_2135_);
lean_dec_ref(v___y_2134_);
lean_dec_ref(v_as_2130_);
lean_dec_ref(v_x_2128_);
return v_res_2141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition(lean_object* v_fData_2146_, lean_object* v_a_2147_, lean_object* v_a_2148_, lean_object* v_a_2149_, lean_object* v_a_2150_){
_start:
{
lean_object* v_lctx_2152_; lean_object* v_insts_2153_; lean_object* v_fn_2154_; lean_object* v_args_2155_; lean_object* v_mainVar_2156_; lean_object* v_mainArgs_2157_; lean_object* v_xId_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; 
v_lctx_2152_ = lean_ctor_get(v_fData_2146_, 0);
v_insts_2153_ = lean_ctor_get(v_fData_2146_, 1);
v_fn_2154_ = lean_ctor_get(v_fData_2146_, 2);
v_args_2155_ = lean_ctor_get(v_fData_2146_, 3);
v_mainVar_2156_ = lean_ctor_get(v_fData_2146_, 4);
v_mainArgs_2157_ = lean_ctor_get(v_fData_2146_, 5);
v_xId_2158_ = l_Lean_Expr_fvarId_x21(v_mainVar_2156_);
lean_inc(v_xId_2158_);
v___x_2159_ = lean_alloc_closure((void*)(l_Lean_FVarId_getUserName___boxed), 6, 1);
lean_closure_set(v___x_2159_, 0, v_xId_2158_);
lean_inc_ref(v_insts_2153_);
lean_inc_ref(v_lctx_2152_);
v___x_2160_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_2152_, v_insts_2153_, v___x_2159_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2160_) == 0)
{
lean_object* v_a_2161_; lean_object* v___x_2163_; uint8_t v_isShared_2164_; uint8_t v_isSharedCheck_2310_; 
v_a_2161_ = lean_ctor_get(v___x_2160_, 0);
v_isSharedCheck_2310_ = !lean_is_exclusive(v___x_2160_);
if (v_isSharedCheck_2310_ == 0)
{
v___x_2163_ = v___x_2160_;
v_isShared_2164_ = v_isSharedCheck_2310_;
goto v_resetjp_2162_;
}
else
{
lean_inc(v_a_2161_);
lean_dec(v___x_2160_);
v___x_2163_ = lean_box(0);
v_isShared_2164_ = v_isSharedCheck_2310_;
goto v_resetjp_2162_;
}
v_resetjp_2162_:
{
uint8_t v___x_2165_; 
v___x_2165_ = l_Lean_Expr_containsFVar(v_fn_2154_, v_xId_2158_);
lean_dec(v_xId_2158_);
if (v___x_2165_ == 0)
{
lean_object* v___x_2166_; lean_object* v___x_2167_; uint8_t v___x_2168_; 
v___x_2166_ = lean_array_get_size(v_mainArgs_2157_);
v___x_2167_ = lean_unsigned_to_nat(0u);
v___x_2168_ = lean_nat_dec_eq(v___x_2166_, v___x_2167_);
if (v___x_2168_ == 0)
{
lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; size_t v_sz_2174_; size_t v___x_2175_; lean_object* v___x_2176_; 
lean_del_object(v___x_2163_);
v___x_2169_ = lean_box(0);
v___x_2170_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___closed__1));
lean_inc_ref(v_args_2155_);
v___x_2171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2171_, 0, v_args_2155_);
lean_ctor_set(v___x_2171_, 1, v___x_2170_);
lean_inc_ref(v_lctx_2152_);
v___x_2172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2172_, 0, v_lctx_2152_);
lean_ctor_set(v___x_2172_, 1, v___x_2171_);
v___x_2173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2173_, 0, v___x_2169_);
lean_ctor_set(v___x_2173_, 1, v___x_2172_);
v_sz_2174_ = lean_array_size(v_mainArgs_2157_);
v___x_2175_ = ((size_t)0ULL);
lean_inc_ref(v_insts_2153_);
v___x_2176_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__1(v_insts_2153_, v_mainVar_2156_, v_a_2161_, v_mainArgs_2157_, v_sz_2174_, v___x_2175_, v___x_2173_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2176_) == 0)
{
lean_object* v_a_2177_; lean_object* v___x_2179_; uint8_t v_isShared_2180_; uint8_t v_isSharedCheck_2296_; 
v_a_2177_ = lean_ctor_get(v___x_2176_, 0);
v_isSharedCheck_2296_ = !lean_is_exclusive(v___x_2176_);
if (v_isSharedCheck_2296_ == 0)
{
v___x_2179_ = v___x_2176_;
v_isShared_2180_ = v_isSharedCheck_2296_;
goto v_resetjp_2178_;
}
else
{
lean_inc(v_a_2177_);
lean_dec(v___x_2176_);
v___x_2179_ = lean_box(0);
v_isShared_2180_ = v_isSharedCheck_2296_;
goto v_resetjp_2178_;
}
v_resetjp_2178_:
{
lean_object* v_fst_2181_; 
v_fst_2181_ = lean_ctor_get(v_a_2177_, 0);
if (lean_obj_tag(v_fst_2181_) == 0)
{
lean_object* v_snd_2182_; lean_object* v_snd_2183_; lean_object* v_snd_2184_; lean_object* v_fst_2185_; lean_object* v_fst_2186_; lean_object* v_fst_2187_; lean_object* v_snd_2188_; lean_object* v___x_2190_; uint8_t v_isShared_2191_; uint8_t v_isSharedCheck_2291_; 
lean_del_object(v___x_2179_);
v_snd_2182_ = lean_ctor_get(v_a_2177_, 1);
lean_inc(v_snd_2182_);
lean_dec(v_a_2177_);
v_snd_2183_ = lean_ctor_get(v_snd_2182_, 1);
lean_inc(v_snd_2183_);
v_snd_2184_ = lean_ctor_get(v_snd_2183_, 1);
lean_inc(v_snd_2184_);
v_fst_2185_ = lean_ctor_get(v_snd_2182_, 0);
lean_inc(v_fst_2185_);
lean_dec(v_snd_2182_);
v_fst_2186_ = lean_ctor_get(v_snd_2183_, 0);
lean_inc(v_fst_2186_);
lean_dec(v_snd_2183_);
v_fst_2187_ = lean_ctor_get(v_snd_2184_, 0);
v_snd_2188_ = lean_ctor_get(v_snd_2184_, 1);
v_isSharedCheck_2291_ = !lean_is_exclusive(v_snd_2184_);
if (v_isSharedCheck_2291_ == 0)
{
v___x_2190_ = v_snd_2184_;
v_isShared_2191_ = v_isSharedCheck_2291_;
goto v_resetjp_2189_;
}
else
{
lean_inc(v_snd_2188_);
lean_inc(v_fst_2187_);
lean_dec(v_snd_2184_);
v___x_2190_ = lean_box(0);
v_isShared_2191_ = v_isSharedCheck_2291_;
goto v_resetjp_2189_;
}
v_resetjp_2189_:
{
uint8_t v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___f_2195_; lean_object* v___x_2196_; 
v___x_2192_ = 1;
v___x_2193_ = lean_box(v___x_2168_);
v___x_2194_ = lean_box(v___x_2192_);
lean_inc_ref(v_mainVar_2156_);
v___f_2195_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__0___boxed), 9, 4);
lean_closure_set(v___f_2195_, 0, v_fst_2187_);
lean_closure_set(v___f_2195_, 1, v_mainVar_2156_);
lean_closure_set(v___f_2195_, 2, v___x_2193_);
lean_closure_set(v___f_2195_, 3, v___x_2194_);
lean_inc_ref(v_insts_2153_);
lean_inc(v_fst_2185_);
v___x_2196_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_fst_2185_, v_insts_2153_, v___f_2195_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2196_) == 0)
{
lean_object* v_a_2197_; lean_object* v___x_2198_; uint8_t v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___f_2204_; lean_object* v___x_2205_; 
v_a_2197_ = lean_ctor_get(v___x_2196_, 0);
lean_inc(v_a_2197_);
lean_dec_ref_known(v___x_2196_, 1);
lean_inc_ref(v_fn_2154_);
v___x_2198_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_fn_2154_, v_fst_2186_);
lean_dec(v_fst_2186_);
v___x_2199_ = 1;
v___x_2200_ = lean_array_get_size(v_snd_2188_);
v___x_2201_ = lean_box(v___x_2168_);
v___x_2202_ = lean_box(v___x_2192_);
v___x_2203_ = lean_box(v___x_2199_);
v___f_2204_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___lam__1___boxed), 11, 6);
lean_closure_set(v___f_2204_, 0, v_snd_2188_);
lean_closure_set(v___f_2204_, 1, v___x_2198_);
lean_closure_set(v___f_2204_, 2, v___x_2201_);
lean_closure_set(v___f_2204_, 3, v___x_2202_);
lean_closure_set(v___f_2204_, 4, v___x_2203_);
lean_closure_set(v___f_2204_, 5, v___x_2200_);
lean_inc_ref(v_insts_2153_);
v___x_2205_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_fst_2185_, v_insts_2153_, v___f_2204_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2205_) == 0)
{
lean_object* v_a_2206_; lean_object* v___x_2207_; 
v_a_2206_ = lean_ctor_get(v___x_2205_, 0);
lean_inc(v_a_2206_);
lean_dec_ref_known(v___x_2205_, 1);
v___x_2207_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_toExpr(v_fData_2146_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2207_) == 0)
{
lean_object* v_a_2208_; lean_object* v_keyedConfig_2209_; uint8_t v_trackZetaDelta_2210_; lean_object* v_zetaDeltaSet_2211_; lean_object* v_lctx_2212_; lean_object* v_localInstances_2213_; lean_object* v_defEqCtx_x3f_2214_; lean_object* v_synthPendingDepth_2215_; lean_object* v_customCanUnfoldPredicate_x3f_2216_; uint8_t v_univApprox_2217_; uint8_t v_inTypeClassResolution_2218_; uint8_t v_cacheInferType_2219_; uint8_t v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; 
v_a_2208_ = lean_ctor_get(v___x_2207_, 0);
lean_inc_n(v_a_2208_, 2);
lean_dec_ref_known(v___x_2207_, 1);
v_keyedConfig_2209_ = lean_ctor_get(v_a_2147_, 0);
v_trackZetaDelta_2210_ = lean_ctor_get_uint8(v_a_2147_, sizeof(void*)*7);
v_zetaDeltaSet_2211_ = lean_ctor_get(v_a_2147_, 1);
v_lctx_2212_ = lean_ctor_get(v_a_2147_, 2);
v_localInstances_2213_ = lean_ctor_get(v_a_2147_, 3);
v_defEqCtx_x3f_2214_ = lean_ctor_get(v_a_2147_, 4);
v_synthPendingDepth_2215_ = lean_ctor_get(v_a_2147_, 5);
v_customCanUnfoldPredicate_x3f_2216_ = lean_ctor_get(v_a_2147_, 6);
v_univApprox_2217_ = lean_ctor_get_uint8(v_a_2147_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2218_ = lean_ctor_get_uint8(v_a_2147_, sizeof(void*)*7 + 2);
v_cacheInferType_2219_ = lean_ctor_get_uint8(v_a_2147_, sizeof(void*)*7 + 3);
v___x_2220_ = 3;
lean_inc_ref(v_keyedConfig_2209_);
v___x_2221_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2220_, v_keyedConfig_2209_);
lean_inc(v_customCanUnfoldPredicate_x3f_2216_);
lean_inc(v_synthPendingDepth_2215_);
lean_inc(v_defEqCtx_x3f_2214_);
lean_inc_ref(v_localInstances_2213_);
lean_inc_ref(v_lctx_2212_);
lean_inc(v_zetaDeltaSet_2211_);
v___x_2222_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2222_, 0, v___x_2221_);
lean_ctor_set(v___x_2222_, 1, v_zetaDeltaSet_2211_);
lean_ctor_set(v___x_2222_, 2, v_lctx_2212_);
lean_ctor_set(v___x_2222_, 3, v_localInstances_2213_);
lean_ctor_set(v___x_2222_, 4, v_defEqCtx_x3f_2214_);
lean_ctor_set(v___x_2222_, 5, v_synthPendingDepth_2215_);
lean_ctor_set(v___x_2222_, 6, v_customCanUnfoldPredicate_x3f_2216_);
lean_ctor_set_uint8(v___x_2222_, sizeof(void*)*7, v_trackZetaDelta_2210_);
lean_ctor_set_uint8(v___x_2222_, sizeof(void*)*7 + 1, v_univApprox_2217_);
lean_ctor_set_uint8(v___x_2222_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2218_);
lean_ctor_set_uint8(v___x_2222_, sizeof(void*)*7 + 3, v_cacheInferType_2219_);
lean_inc(v_a_2206_);
v___x_2223_ = l_Lean_Meta_isExprDefEq(v_a_2208_, v_a_2206_, v___x_2222_, v_a_2148_, v_a_2149_, v_a_2150_);
if (lean_obj_tag(v___x_2223_) == 0)
{
lean_object* v_a_2224_; lean_object* v___x_2226_; uint8_t v_isShared_2227_; uint8_t v_isSharedCheck_2258_; 
v_a_2224_ = lean_ctor_get(v___x_2223_, 0);
v_isSharedCheck_2258_ = !lean_is_exclusive(v___x_2223_);
if (v_isSharedCheck_2258_ == 0)
{
v___x_2226_ = v___x_2223_;
v_isShared_2227_ = v_isSharedCheck_2258_;
goto v_resetjp_2225_;
}
else
{
lean_inc(v_a_2224_);
lean_dec(v___x_2223_);
v___x_2226_ = lean_box(0);
v_isShared_2227_ = v_isSharedCheck_2258_;
goto v_resetjp_2225_;
}
v_resetjp_2225_:
{
uint8_t v___x_2228_; 
v___x_2228_ = lean_unbox(v_a_2224_);
lean_dec(v_a_2224_);
if (v___x_2228_ == 0)
{
lean_object* v___x_2229_; 
lean_del_object(v___x_2226_);
lean_inc(v_a_2197_);
v___x_2229_ = l_Lean_Meta_isExprDefEq(v_a_2208_, v_a_2197_, v___x_2222_, v_a_2148_, v_a_2149_, v_a_2150_);
lean_dec_ref_known(v___x_2222_, 7);
if (lean_obj_tag(v___x_2229_) == 0)
{
lean_object* v_a_2230_; lean_object* v___x_2232_; uint8_t v_isShared_2233_; uint8_t v_isSharedCheck_2245_; 
v_a_2230_ = lean_ctor_get(v___x_2229_, 0);
v_isSharedCheck_2245_ = !lean_is_exclusive(v___x_2229_);
if (v_isSharedCheck_2245_ == 0)
{
v___x_2232_ = v___x_2229_;
v_isShared_2233_ = v_isSharedCheck_2245_;
goto v_resetjp_2231_;
}
else
{
lean_inc(v_a_2230_);
lean_dec(v___x_2229_);
v___x_2232_ = lean_box(0);
v_isShared_2233_ = v_isSharedCheck_2245_;
goto v_resetjp_2231_;
}
v_resetjp_2231_:
{
uint8_t v___x_2234_; 
v___x_2234_ = lean_unbox(v_a_2230_);
lean_dec(v_a_2230_);
if (v___x_2234_ == 0)
{
lean_object* v___x_2236_; 
if (v_isShared_2191_ == 0)
{
lean_ctor_set(v___x_2190_, 1, v_a_2197_);
lean_ctor_set(v___x_2190_, 0, v_a_2206_);
v___x_2236_ = v___x_2190_;
goto v_reusejp_2235_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v_a_2206_);
lean_ctor_set(v_reuseFailAlloc_2240_, 1, v_a_2197_);
v___x_2236_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2235_;
}
v_reusejp_2235_:
{
lean_object* v___x_2238_; 
if (v_isShared_2233_ == 0)
{
lean_ctor_set(v___x_2232_, 0, v___x_2236_);
v___x_2238_ = v___x_2232_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v___x_2236_);
v___x_2238_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
return v___x_2238_;
}
}
}
else
{
lean_object* v___x_2241_; lean_object* v___x_2243_; 
lean_dec(v_a_2206_);
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
v___x_2241_ = lean_box(2);
if (v_isShared_2233_ == 0)
{
lean_ctor_set(v___x_2232_, 0, v___x_2241_);
v___x_2243_ = v___x_2232_;
goto v_reusejp_2242_;
}
else
{
lean_object* v_reuseFailAlloc_2244_; 
v_reuseFailAlloc_2244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2244_, 0, v___x_2241_);
v___x_2243_ = v_reuseFailAlloc_2244_;
goto v_reusejp_2242_;
}
v_reusejp_2242_:
{
return v___x_2243_;
}
}
}
}
else
{
lean_object* v_a_2246_; lean_object* v___x_2248_; uint8_t v_isShared_2249_; uint8_t v_isSharedCheck_2253_; 
lean_dec(v_a_2206_);
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
v_a_2246_ = lean_ctor_get(v___x_2229_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v___x_2229_);
if (v_isSharedCheck_2253_ == 0)
{
v___x_2248_ = v___x_2229_;
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
else
{
lean_inc(v_a_2246_);
lean_dec(v___x_2229_);
v___x_2248_ = lean_box(0);
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
v_resetjp_2247_:
{
lean_object* v___x_2251_; 
if (v_isShared_2249_ == 0)
{
v___x_2251_ = v___x_2248_;
goto v_reusejp_2250_;
}
else
{
lean_object* v_reuseFailAlloc_2252_; 
v_reuseFailAlloc_2252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2252_, 0, v_a_2246_);
v___x_2251_ = v_reuseFailAlloc_2252_;
goto v_reusejp_2250_;
}
v_reusejp_2250_:
{
return v___x_2251_;
}
}
}
}
else
{
lean_object* v___x_2254_; lean_object* v___x_2256_; 
lean_dec_ref_known(v___x_2222_, 7);
lean_dec(v_a_2208_);
lean_dec(v_a_2206_);
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
v___x_2254_ = lean_box(1);
if (v_isShared_2227_ == 0)
{
lean_ctor_set(v___x_2226_, 0, v___x_2254_);
v___x_2256_ = v___x_2226_;
goto v_reusejp_2255_;
}
else
{
lean_object* v_reuseFailAlloc_2257_; 
v_reuseFailAlloc_2257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2257_, 0, v___x_2254_);
v___x_2256_ = v_reuseFailAlloc_2257_;
goto v_reusejp_2255_;
}
v_reusejp_2255_:
{
return v___x_2256_;
}
}
}
}
else
{
lean_object* v_a_2259_; lean_object* v___x_2261_; uint8_t v_isShared_2262_; uint8_t v_isSharedCheck_2266_; 
lean_dec_ref_known(v___x_2222_, 7);
lean_dec(v_a_2208_);
lean_dec(v_a_2206_);
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
v_a_2259_ = lean_ctor_get(v___x_2223_, 0);
v_isSharedCheck_2266_ = !lean_is_exclusive(v___x_2223_);
if (v_isSharedCheck_2266_ == 0)
{
v___x_2261_ = v___x_2223_;
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
else
{
lean_inc(v_a_2259_);
lean_dec(v___x_2223_);
v___x_2261_ = lean_box(0);
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
v_resetjp_2260_:
{
lean_object* v___x_2264_; 
if (v_isShared_2262_ == 0)
{
v___x_2264_ = v___x_2261_;
goto v_reusejp_2263_;
}
else
{
lean_object* v_reuseFailAlloc_2265_; 
v_reuseFailAlloc_2265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2265_, 0, v_a_2259_);
v___x_2264_ = v_reuseFailAlloc_2265_;
goto v_reusejp_2263_;
}
v_reusejp_2263_:
{
return v___x_2264_;
}
}
}
}
else
{
lean_object* v_a_2267_; lean_object* v___x_2269_; uint8_t v_isShared_2270_; uint8_t v_isSharedCheck_2274_; 
lean_dec(v_a_2206_);
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
v_a_2267_ = lean_ctor_get(v___x_2207_, 0);
v_isSharedCheck_2274_ = !lean_is_exclusive(v___x_2207_);
if (v_isSharedCheck_2274_ == 0)
{
v___x_2269_ = v___x_2207_;
v_isShared_2270_ = v_isSharedCheck_2274_;
goto v_resetjp_2268_;
}
else
{
lean_inc(v_a_2267_);
lean_dec(v___x_2207_);
v___x_2269_ = lean_box(0);
v_isShared_2270_ = v_isSharedCheck_2274_;
goto v_resetjp_2268_;
}
v_resetjp_2268_:
{
lean_object* v___x_2272_; 
if (v_isShared_2270_ == 0)
{
v___x_2272_ = v___x_2269_;
goto v_reusejp_2271_;
}
else
{
lean_object* v_reuseFailAlloc_2273_; 
v_reuseFailAlloc_2273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2273_, 0, v_a_2267_);
v___x_2272_ = v_reuseFailAlloc_2273_;
goto v_reusejp_2271_;
}
v_reusejp_2271_:
{
return v___x_2272_;
}
}
}
}
else
{
lean_object* v_a_2275_; lean_object* v___x_2277_; uint8_t v_isShared_2278_; uint8_t v_isSharedCheck_2282_; 
lean_dec(v_a_2197_);
lean_del_object(v___x_2190_);
lean_dec_ref(v_fData_2146_);
v_a_2275_ = lean_ctor_get(v___x_2205_, 0);
v_isSharedCheck_2282_ = !lean_is_exclusive(v___x_2205_);
if (v_isSharedCheck_2282_ == 0)
{
v___x_2277_ = v___x_2205_;
v_isShared_2278_ = v_isSharedCheck_2282_;
goto v_resetjp_2276_;
}
else
{
lean_inc(v_a_2275_);
lean_dec(v___x_2205_);
v___x_2277_ = lean_box(0);
v_isShared_2278_ = v_isSharedCheck_2282_;
goto v_resetjp_2276_;
}
v_resetjp_2276_:
{
lean_object* v___x_2280_; 
if (v_isShared_2278_ == 0)
{
v___x_2280_ = v___x_2277_;
goto v_reusejp_2279_;
}
else
{
lean_object* v_reuseFailAlloc_2281_; 
v_reuseFailAlloc_2281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2281_, 0, v_a_2275_);
v___x_2280_ = v_reuseFailAlloc_2281_;
goto v_reusejp_2279_;
}
v_reusejp_2279_:
{
return v___x_2280_;
}
}
}
}
else
{
lean_object* v_a_2283_; lean_object* v___x_2285_; uint8_t v_isShared_2286_; uint8_t v_isSharedCheck_2290_; 
lean_del_object(v___x_2190_);
lean_dec(v_snd_2188_);
lean_dec(v_fst_2186_);
lean_dec(v_fst_2185_);
lean_dec_ref(v_fData_2146_);
v_a_2283_ = lean_ctor_get(v___x_2196_, 0);
v_isSharedCheck_2290_ = !lean_is_exclusive(v___x_2196_);
if (v_isSharedCheck_2290_ == 0)
{
v___x_2285_ = v___x_2196_;
v_isShared_2286_ = v_isSharedCheck_2290_;
goto v_resetjp_2284_;
}
else
{
lean_inc(v_a_2283_);
lean_dec(v___x_2196_);
v___x_2285_ = lean_box(0);
v_isShared_2286_ = v_isSharedCheck_2290_;
goto v_resetjp_2284_;
}
v_resetjp_2284_:
{
lean_object* v___x_2288_; 
if (v_isShared_2286_ == 0)
{
v___x_2288_ = v___x_2285_;
goto v_reusejp_2287_;
}
else
{
lean_object* v_reuseFailAlloc_2289_; 
v_reuseFailAlloc_2289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2289_, 0, v_a_2283_);
v___x_2288_ = v_reuseFailAlloc_2289_;
goto v_reusejp_2287_;
}
v_reusejp_2287_:
{
return v___x_2288_;
}
}
}
}
}
else
{
lean_object* v_val_2292_; lean_object* v___x_2294_; 
lean_inc_ref(v_fst_2181_);
lean_dec(v_a_2177_);
lean_dec_ref(v_fData_2146_);
v_val_2292_ = lean_ctor_get(v_fst_2181_, 0);
lean_inc(v_val_2292_);
lean_dec_ref_known(v_fst_2181_, 1);
if (v_isShared_2180_ == 0)
{
lean_ctor_set(v___x_2179_, 0, v_val_2292_);
v___x_2294_ = v___x_2179_;
goto v_reusejp_2293_;
}
else
{
lean_object* v_reuseFailAlloc_2295_; 
v_reuseFailAlloc_2295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2295_, 0, v_val_2292_);
v___x_2294_ = v_reuseFailAlloc_2295_;
goto v_reusejp_2293_;
}
v_reusejp_2293_:
{
return v___x_2294_;
}
}
}
}
else
{
lean_object* v_a_2297_; lean_object* v___x_2299_; uint8_t v_isShared_2300_; uint8_t v_isSharedCheck_2304_; 
lean_dec_ref(v_fData_2146_);
v_a_2297_ = lean_ctor_get(v___x_2176_, 0);
v_isSharedCheck_2304_ = !lean_is_exclusive(v___x_2176_);
if (v_isSharedCheck_2304_ == 0)
{
v___x_2299_ = v___x_2176_;
v_isShared_2300_ = v_isSharedCheck_2304_;
goto v_resetjp_2298_;
}
else
{
lean_inc(v_a_2297_);
lean_dec(v___x_2176_);
v___x_2299_ = lean_box(0);
v_isShared_2300_ = v_isSharedCheck_2304_;
goto v_resetjp_2298_;
}
v_resetjp_2298_:
{
lean_object* v___x_2302_; 
if (v_isShared_2300_ == 0)
{
v___x_2302_ = v___x_2299_;
goto v_reusejp_2301_;
}
else
{
lean_object* v_reuseFailAlloc_2303_; 
v_reuseFailAlloc_2303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2303_, 0, v_a_2297_);
v___x_2302_ = v_reuseFailAlloc_2303_;
goto v_reusejp_2301_;
}
v_reusejp_2301_:
{
return v___x_2302_;
}
}
}
}
else
{
lean_object* v___x_2305_; lean_object* v___x_2307_; 
lean_dec(v_a_2161_);
lean_dec_ref(v_fData_2146_);
v___x_2305_ = lean_box(2);
if (v_isShared_2164_ == 0)
{
lean_ctor_set(v___x_2163_, 0, v___x_2305_);
v___x_2307_ = v___x_2163_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2308_; 
v_reuseFailAlloc_2308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2308_, 0, v___x_2305_);
v___x_2307_ = v_reuseFailAlloc_2308_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
return v___x_2307_;
}
}
}
else
{
lean_object* v___x_2309_; 
lean_del_object(v___x_2163_);
lean_dec(v_a_2161_);
v___x_2309_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_peeloffArgDecomposition(v_fData_2146_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
return v___x_2309_;
}
}
}
else
{
lean_object* v_a_2311_; lean_object* v___x_2313_; uint8_t v_isShared_2314_; uint8_t v_isSharedCheck_2318_; 
lean_dec(v_xId_2158_);
lean_dec_ref(v_fData_2146_);
v_a_2311_ = lean_ctor_get(v___x_2160_, 0);
v_isSharedCheck_2318_ = !lean_is_exclusive(v___x_2160_);
if (v_isSharedCheck_2318_ == 0)
{
v___x_2313_ = v___x_2160_;
v_isShared_2314_ = v_isSharedCheck_2318_;
goto v_resetjp_2312_;
}
else
{
lean_inc(v_a_2311_);
lean_dec(v___x_2160_);
v___x_2313_ = lean_box(0);
v_isShared_2314_ = v_isSharedCheck_2318_;
goto v_resetjp_2312_;
}
v_resetjp_2312_:
{
lean_object* v___x_2316_; 
if (v_isShared_2314_ == 0)
{
v___x_2316_ = v___x_2313_;
goto v_reusejp_2315_;
}
else
{
lean_object* v_reuseFailAlloc_2317_; 
v_reuseFailAlloc_2317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2317_, 0, v_a_2311_);
v___x_2316_ = v_reuseFailAlloc_2317_;
goto v_reusejp_2315_;
}
v_reusejp_2315_:
{
return v___x_2316_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition___boxed(lean_object* v_fData_2319_, lean_object* v_a_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_, lean_object* v_a_2323_, lean_object* v_a_2324_){
_start:
{
lean_object* v_res_2325_; 
v_res_2325_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decomposition(v_fData_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
lean_dec(v_a_2323_);
lean_dec_ref(v_a_2322_);
lean_dec(v_a_2321_);
lean_dec_ref(v_a_2320_);
return v_res_2325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0(lean_object* v___y_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_){
_start:
{
lean_object* v___x_2331_; 
v___x_2331_ = lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___redArg(v___y_2329_);
return v___x_2331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0___boxed(lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_){
_start:
{
lean_object* v_res_2337_; 
v_res_2337_ = lp_mathlib_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Mathlib_Meta_FunProp_FunctionData_decomposition_spec__0_spec__0(v___y_2332_, v___y_2333_, v___y_2334_, v___y_2335_);
lean_dec(v___y_2335_);
lean_dec_ref(v___y_2334_);
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
return v_res_2337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2(lean_object* v_as_2338_, size_t v_i_2339_, size_t v_stop_2340_, lean_object* v_b_2341_){
_start:
{
uint8_t v___x_2342_; 
v___x_2342_ = lean_usize_dec_eq(v_i_2339_, v_stop_2340_);
if (v___x_2342_ == 0)
{
lean_object* v___x_2343_; lean_object* v_fst_2344_; lean_object* v_snd_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v_coe_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2359_; 
v___x_2343_ = lean_array_uget_borrowed(v_as_2338_, v_i_2339_);
v_fst_2344_ = lean_ctor_get(v___x_2343_, 0);
v_snd_2345_ = lean_ctor_get(v___x_2343_, 1);
v___x_2346_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
v___x_2347_ = lean_array_get(v___x_2346_, v_b_2341_, v_fst_2344_);
v_coe_2348_ = lean_ctor_get(v___x_2347_, 1);
v_isSharedCheck_2359_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2359_ == 0)
{
lean_object* v_unused_2360_; 
v_unused_2360_ = lean_ctor_get(v___x_2347_, 0);
lean_dec(v_unused_2360_);
v___x_2350_ = v___x_2347_;
v_isShared_2351_ = v_isSharedCheck_2359_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_coe_2348_);
lean_dec(v___x_2347_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2359_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v___x_2353_; 
lean_inc(v_snd_2345_);
if (v_isShared_2351_ == 0)
{
lean_ctor_set(v___x_2350_, 0, v_snd_2345_);
v___x_2353_ = v___x_2350_;
goto v_reusejp_2352_;
}
else
{
lean_object* v_reuseFailAlloc_2358_; 
v_reuseFailAlloc_2358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2358_, 0, v_snd_2345_);
lean_ctor_set(v_reuseFailAlloc_2358_, 1, v_coe_2348_);
v___x_2353_ = v_reuseFailAlloc_2358_;
goto v_reusejp_2352_;
}
v_reusejp_2352_:
{
lean_object* v___x_2354_; size_t v___x_2355_; size_t v___x_2356_; 
v___x_2354_ = lean_array_set(v_b_2341_, v_fst_2344_, v___x_2353_);
v___x_2355_ = ((size_t)1ULL);
v___x_2356_ = lean_usize_add(v_i_2339_, v___x_2355_);
v_i_2339_ = v___x_2356_;
v_b_2341_ = v___x_2354_;
goto _start;
}
}
}
else
{
return v_b_2341_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2___boxed(lean_object* v_as_2361_, lean_object* v_i_2362_, lean_object* v_stop_2363_, lean_object* v_b_2364_){
_start:
{
size_t v_i_boxed_2365_; size_t v_stop_boxed_2366_; lean_object* v_res_2367_; 
v_i_boxed_2365_ = lean_unbox_usize(v_i_2362_);
lean_dec(v_i_2362_);
v_stop_boxed_2366_ = lean_unbox_usize(v_stop_2363_);
lean_dec(v_stop_2363_);
v_res_2367_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2(v_as_2361_, v_i_boxed_2365_, v_stop_boxed_2366_, v_b_2364_);
lean_dec_ref(v_as_2361_);
return v_res_2367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0(lean_object* v_gxs_2368_, lean_object* v___x_2369_, lean_object* v_fn_2370_, uint8_t v___x_2371_, uint8_t v___x_2372_, uint8_t v___x_2373_, lean_object* v_a_2374_, lean_object* v_args_2375_, lean_object* v_args_2376_, size_t v___x_2377_, lean_object* v_y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_){
_start:
{
lean_object* v___x_2384_; lean_object* v___x_2385_; 
v___x_2384_ = lean_array_get_size(v_gxs_2368_);
lean_inc_ref(v_y_2378_);
v___x_2385_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(v_y_2378_, v___x_2384_, v___y_2379_, v___y_2380_, v___y_2381_, v___y_2382_);
if (lean_obj_tag(v___x_2385_) == 0)
{
lean_object* v_a_2386_; lean_object* v___y_2388_; lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v___x_2412_; uint8_t v___x_2413_; 
v_a_2386_ = lean_ctor_get(v___x_2385_, 0);
lean_inc(v_a_2386_);
lean_dec_ref_known(v___x_2385_, 1);
v___x_2410_ = l_Array_zip___redArg(v_args_2375_, v_a_2386_);
lean_dec(v_a_2386_);
v___x_2411_ = lean_unsigned_to_nat(0u);
v___x_2412_ = lean_array_get_size(v___x_2410_);
v___x_2413_ = lean_nat_dec_lt(v___x_2411_, v___x_2412_);
if (v___x_2413_ == 0)
{
lean_dec_ref(v___x_2410_);
v___y_2388_ = v_args_2376_;
goto v___jp_2387_;
}
else
{
uint8_t v___x_2414_; 
v___x_2414_ = lean_nat_dec_le(v___x_2412_, v___x_2412_);
if (v___x_2414_ == 0)
{
if (v___x_2413_ == 0)
{
lean_dec_ref(v___x_2410_);
v___y_2388_ = v_args_2376_;
goto v___jp_2387_;
}
else
{
size_t v___x_2415_; lean_object* v___x_2416_; 
v___x_2415_ = lean_usize_of_nat(v___x_2412_);
v___x_2416_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2(v___x_2410_, v___x_2377_, v___x_2415_, v_args_2376_);
lean_dec_ref(v___x_2410_);
v___y_2388_ = v___x_2416_;
goto v___jp_2387_;
}
}
else
{
size_t v___x_2417_; lean_object* v___x_2418_; 
v___x_2417_ = lean_usize_of_nat(v___x_2412_);
v___x_2418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__2(v___x_2410_, v___x_2377_, v___x_2417_, v_args_2376_);
lean_dec_ref(v___x_2410_);
v___y_2388_ = v___x_2418_;
goto v___jp_2387_;
}
}
v___jp_2387_:
{
lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; 
v___x_2389_ = lean_array_push(v___x_2369_, v_y_2378_);
v___x_2390_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_fn_2370_, v___y_2388_);
lean_dec_ref(v___y_2388_);
v___x_2391_ = l_Lean_Meta_mkLambdaFVars(v___x_2389_, v___x_2390_, v___x_2371_, v___x_2372_, v___x_2371_, v___x_2372_, v___x_2373_, v___y_2379_, v___y_2380_, v___y_2381_, v___y_2382_);
lean_dec_ref(v___x_2389_);
if (lean_obj_tag(v___x_2391_) == 0)
{
lean_object* v_a_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2401_; 
v_a_2392_ = lean_ctor_get(v___x_2391_, 0);
v_isSharedCheck_2401_ = !lean_is_exclusive(v___x_2391_);
if (v_isSharedCheck_2401_ == 0)
{
v___x_2394_ = v___x_2391_;
v_isShared_2395_ = v_isSharedCheck_2401_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_a_2392_);
lean_dec(v___x_2391_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2401_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___x_2399_; 
v___x_2396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2396_, 0, v_a_2392_);
lean_ctor_set(v___x_2396_, 1, v_a_2374_);
v___x_2397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2397_, 0, v___x_2396_);
if (v_isShared_2395_ == 0)
{
lean_ctor_set(v___x_2394_, 0, v___x_2397_);
v___x_2399_ = v___x_2394_;
goto v_reusejp_2398_;
}
else
{
lean_object* v_reuseFailAlloc_2400_; 
v_reuseFailAlloc_2400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2400_, 0, v___x_2397_);
v___x_2399_ = v_reuseFailAlloc_2400_;
goto v_reusejp_2398_;
}
v_reusejp_2398_:
{
return v___x_2399_;
}
}
}
else
{
lean_object* v_a_2402_; lean_object* v___x_2404_; uint8_t v_isShared_2405_; uint8_t v_isSharedCheck_2409_; 
lean_dec_ref(v_a_2374_);
v_a_2402_ = lean_ctor_get(v___x_2391_, 0);
v_isSharedCheck_2409_ = !lean_is_exclusive(v___x_2391_);
if (v_isSharedCheck_2409_ == 0)
{
v___x_2404_ = v___x_2391_;
v_isShared_2405_ = v_isSharedCheck_2409_;
goto v_resetjp_2403_;
}
else
{
lean_inc(v_a_2402_);
lean_dec(v___x_2391_);
v___x_2404_ = lean_box(0);
v_isShared_2405_ = v_isSharedCheck_2409_;
goto v_resetjp_2403_;
}
v_resetjp_2403_:
{
lean_object* v___x_2407_; 
if (v_isShared_2405_ == 0)
{
v___x_2407_ = v___x_2404_;
goto v_reusejp_2406_;
}
else
{
lean_object* v_reuseFailAlloc_2408_; 
v_reuseFailAlloc_2408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2408_, 0, v_a_2402_);
v___x_2407_ = v_reuseFailAlloc_2408_;
goto v_reusejp_2406_;
}
v_reusejp_2406_:
{
return v___x_2407_;
}
}
}
}
}
else
{
lean_object* v_a_2419_; lean_object* v___x_2421_; uint8_t v_isShared_2422_; uint8_t v_isSharedCheck_2426_; 
lean_dec_ref(v_y_2378_);
lean_dec_ref(v_args_2376_);
lean_dec_ref(v_a_2374_);
lean_dec_ref(v_fn_2370_);
lean_dec_ref(v___x_2369_);
v_a_2419_ = lean_ctor_get(v___x_2385_, 0);
v_isSharedCheck_2426_ = !lean_is_exclusive(v___x_2385_);
if (v_isSharedCheck_2426_ == 0)
{
v___x_2421_ = v___x_2385_;
v_isShared_2422_ = v_isSharedCheck_2426_;
goto v_resetjp_2420_;
}
else
{
lean_inc(v_a_2419_);
lean_dec(v___x_2385_);
v___x_2421_ = lean_box(0);
v_isShared_2422_ = v_isSharedCheck_2426_;
goto v_resetjp_2420_;
}
v_resetjp_2420_:
{
lean_object* v___x_2424_; 
if (v_isShared_2422_ == 0)
{
v___x_2424_ = v___x_2421_;
goto v_reusejp_2423_;
}
else
{
lean_object* v_reuseFailAlloc_2425_; 
v_reuseFailAlloc_2425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2425_, 0, v_a_2419_);
v___x_2424_ = v_reuseFailAlloc_2425_;
goto v_reusejp_2423_;
}
v_reusejp_2423_:
{
return v___x_2424_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0___boxed(lean_object* v_gxs_2427_, lean_object* v___x_2428_, lean_object* v_fn_2429_, lean_object* v___x_2430_, lean_object* v___x_2431_, lean_object* v___x_2432_, lean_object* v_a_2433_, lean_object* v_args_2434_, lean_object* v_args_2435_, lean_object* v___x_2436_, lean_object* v_y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_){
_start:
{
uint8_t v___x_3898__boxed_2443_; uint8_t v___x_3899__boxed_2444_; uint8_t v___x_3900__boxed_2445_; size_t v___x_3902__boxed_2446_; lean_object* v_res_2447_; 
v___x_3898__boxed_2443_ = lean_unbox(v___x_2430_);
v___x_3899__boxed_2444_ = lean_unbox(v___x_2431_);
v___x_3900__boxed_2445_ = lean_unbox(v___x_2432_);
v___x_3902__boxed_2446_ = lean_unbox_usize(v___x_2436_);
lean_dec(v___x_2436_);
v_res_2447_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0(v_gxs_2427_, v___x_2428_, v_fn_2429_, v___x_3898__boxed_2443_, v___x_3899__boxed_2444_, v___x_3900__boxed_2445_, v_a_2433_, v_args_2434_, v_args_2435_, v___x_3902__boxed_2446_, v_y_2437_, v___y_2438_, v___y_2439_, v___y_2440_, v___y_2441_);
lean_dec(v___y_2441_);
lean_dec_ref(v___y_2440_);
lean_dec(v___y_2439_);
lean_dec_ref(v___y_2438_);
lean_dec_ref(v_args_2434_);
lean_dec_ref(v_gxs_2427_);
return v_res_2447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1(lean_object* v_gxs_2451_, lean_object* v_mainVar_2452_, uint8_t v___x_2453_, uint8_t v___x_2454_, lean_object* v_lctx_2455_, lean_object* v_insts_2456_, lean_object* v_fn_2457_, lean_object* v_args_2458_, lean_object* v_args_2459_, size_t v___x_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_){
_start:
{
lean_object* v___y_2467_; uint8_t v___y_2468_; lean_object* v_a_2473_; lean_object* v___x_2476_; 
lean_inc_ref(v_gxs_2451_);
v___x_2476_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(v_gxs_2451_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_);
if (lean_obj_tag(v___x_2476_) == 0)
{
lean_object* v_a_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; uint8_t v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; 
v_a_2477_ = lean_ctor_get(v___x_2476_, 0);
lean_inc_n(v_a_2477_, 2);
lean_dec_ref_known(v___x_2476_, 1);
v___x_2478_ = lean_unsigned_to_nat(1u);
v___x_2479_ = lean_mk_empty_array_with_capacity(v___x_2478_);
lean_inc_ref(v___x_2479_);
v___x_2480_ = lean_array_push(v___x_2479_, v_mainVar_2452_);
v___x_2481_ = 1;
v___x_2482_ = lean_box(v___x_2453_);
v___x_2483_ = lean_box(v___x_2454_);
v___x_2484_ = lean_box(v___x_2453_);
v___x_2485_ = lean_box(v___x_2454_);
v___x_2486_ = lean_box(v___x_2481_);
v___x_2487_ = lean_alloc_closure((void*)(l_Lean_Meta_mkLambdaFVars___boxed), 12, 7);
lean_closure_set(v___x_2487_, 0, v___x_2480_);
lean_closure_set(v___x_2487_, 1, v_a_2477_);
lean_closure_set(v___x_2487_, 2, v___x_2482_);
lean_closure_set(v___x_2487_, 3, v___x_2483_);
lean_closure_set(v___x_2487_, 4, v___x_2484_);
lean_closure_set(v___x_2487_, 5, v___x_2485_);
lean_closure_set(v___x_2487_, 6, v___x_2486_);
v___x_2488_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_2455_, v_insts_2456_, v___x_2487_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_);
if (lean_obj_tag(v___x_2488_) == 0)
{
lean_object* v_a_2489_; lean_object* v___x_2490_; 
v_a_2489_ = lean_ctor_get(v___x_2488_, 0);
lean_inc(v_a_2489_);
lean_dec_ref_known(v___x_2488_, 1);
lean_inc(v___y_2464_);
lean_inc_ref(v___y_2463_);
lean_inc(v___y_2462_);
lean_inc_ref(v___y_2461_);
v___x_2490_ = lean_infer_type(v_a_2477_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_);
if (lean_obj_tag(v___x_2490_) == 0)
{
lean_object* v_a_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___f_2496_; lean_object* v___x_2497_; lean_object* v___x_2498_; 
v_a_2491_ = lean_ctor_get(v___x_2490_, 0);
lean_inc(v_a_2491_);
lean_dec_ref_known(v___x_2490_, 1);
v___x_2492_ = lean_box(v___x_2453_);
v___x_2493_ = lean_box(v___x_2454_);
v___x_2494_ = lean_box(v___x_2481_);
v___x_2495_ = lean_box_usize(v___x_2460_);
v___f_2496_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__0___boxed), 16, 10);
lean_closure_set(v___f_2496_, 0, v_gxs_2451_);
lean_closure_set(v___f_2496_, 1, v___x_2479_);
lean_closure_set(v___f_2496_, 2, v_fn_2457_);
lean_closure_set(v___f_2496_, 3, v___x_2492_);
lean_closure_set(v___f_2496_, 4, v___x_2493_);
lean_closure_set(v___f_2496_, 5, v___x_2494_);
lean_closure_set(v___f_2496_, 6, v_a_2489_);
lean_closure_set(v___f_2496_, 7, v_args_2458_);
lean_closure_set(v___f_2496_, 8, v_args_2459_);
lean_closure_set(v___f_2496_, 9, v___x_2495_);
v___x_2497_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___closed__1));
v___x_2498_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Meta_FunProp_getFunctionData_x3f_spec__2___redArg(v___x_2497_, v_a_2491_, v___f_2496_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_);
lean_dec(v___y_2464_);
lean_dec_ref(v___y_2463_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
if (lean_obj_tag(v___x_2498_) == 0)
{
return v___x_2498_;
}
else
{
lean_object* v_a_2499_; 
v_a_2499_ = lean_ctor_get(v___x_2498_, 0);
lean_inc(v_a_2499_);
lean_dec_ref_known(v___x_2498_, 1);
v_a_2473_ = v_a_2499_;
goto v___jp_2472_;
}
}
else
{
lean_object* v_a_2500_; 
lean_dec(v_a_2489_);
lean_dec_ref(v___x_2479_);
lean_dec(v___y_2464_);
lean_dec_ref(v___y_2463_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
lean_dec_ref(v_args_2459_);
lean_dec_ref(v_args_2458_);
lean_dec_ref(v_fn_2457_);
lean_dec_ref(v_gxs_2451_);
v_a_2500_ = lean_ctor_get(v___x_2490_, 0);
lean_inc(v_a_2500_);
lean_dec_ref_known(v___x_2490_, 1);
v_a_2473_ = v_a_2500_;
goto v___jp_2472_;
}
}
else
{
lean_object* v_a_2501_; 
lean_dec_ref(v___x_2479_);
lean_dec(v_a_2477_);
lean_dec(v___y_2464_);
lean_dec_ref(v___y_2463_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
lean_dec_ref(v_args_2459_);
lean_dec_ref(v_args_2458_);
lean_dec_ref(v_fn_2457_);
lean_dec_ref(v_gxs_2451_);
v_a_2501_ = lean_ctor_get(v___x_2488_, 0);
lean_inc(v_a_2501_);
lean_dec_ref_known(v___x_2488_, 1);
v_a_2473_ = v_a_2501_;
goto v___jp_2472_;
}
}
else
{
lean_object* v_a_2502_; 
lean_dec(v___y_2464_);
lean_dec_ref(v___y_2463_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
lean_dec_ref(v_args_2459_);
lean_dec_ref(v_args_2458_);
lean_dec_ref(v_fn_2457_);
lean_dec_ref(v_insts_2456_);
lean_dec_ref(v_lctx_2455_);
lean_dec_ref(v_mainVar_2452_);
lean_dec_ref(v_gxs_2451_);
v_a_2502_ = lean_ctor_get(v___x_2476_, 0);
lean_inc(v_a_2502_);
lean_dec_ref_known(v___x_2476_, 1);
v_a_2473_ = v_a_2502_;
goto v___jp_2472_;
}
v___jp_2466_:
{
if (v___y_2468_ == 0)
{
lean_object* v___x_2469_; lean_object* v___x_2470_; 
lean_dec_ref(v___y_2467_);
v___x_2469_ = lean_box(0);
v___x_2470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2470_, 0, v___x_2469_);
return v___x_2470_;
}
else
{
lean_object* v___x_2471_; 
v___x_2471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2471_, 0, v___y_2467_);
return v___x_2471_;
}
}
v___jp_2472_:
{
uint8_t v___x_2474_; 
v___x_2474_ = l_Lean_Exception_isInterrupt(v_a_2473_);
if (v___x_2474_ == 0)
{
uint8_t v___x_2475_; 
lean_inc_ref(v_a_2473_);
v___x_2475_ = l_Lean_Exception_isRuntime(v_a_2473_);
v___y_2467_ = v_a_2473_;
v___y_2468_ = v___x_2475_;
goto v___jp_2466_;
}
else
{
v___y_2467_ = v_a_2473_;
v___y_2468_ = v___x_2474_;
goto v___jp_2466_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___boxed(lean_object* v_gxs_2503_, lean_object* v_mainVar_2504_, lean_object* v___x_2505_, lean_object* v___x_2506_, lean_object* v_lctx_2507_, lean_object* v_insts_2508_, lean_object* v_fn_2509_, lean_object* v_args_2510_, lean_object* v_args_2511_, lean_object* v___x_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_){
_start:
{
uint8_t v___x_4022__boxed_2518_; uint8_t v___x_4023__boxed_2519_; size_t v___x_4024__boxed_2520_; lean_object* v_res_2521_; 
v___x_4022__boxed_2518_ = lean_unbox(v___x_2505_);
v___x_4023__boxed_2519_ = lean_unbox(v___x_2506_);
v___x_4024__boxed_2520_ = lean_unbox_usize(v___x_2512_);
lean_dec(v___x_2512_);
v_res_2521_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1(v_gxs_2503_, v_mainVar_2504_, v___x_4022__boxed_2518_, v___x_4023__boxed_2519_, v_lctx_2507_, v_insts_2508_, v_fn_2509_, v_args_2510_, v_args_2511_, v___x_4024__boxed_2520_, v___y_2513_, v___y_2514_, v___y_2515_, v___y_2516_);
return v_res_2521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg(lean_object* v___x_2522_, lean_object* v_a_2523_, lean_object* v_b_2524_, lean_object* v_range_2525_, lean_object* v_b_2526_, lean_object* v_i_2527_){
_start:
{
lean_object* v_stop_2528_; lean_object* v_step_2529_; lean_object* v_a_2531_; uint8_t v___x_2534_; 
v_stop_2528_ = lean_ctor_get(v_range_2525_, 1);
v_step_2529_ = lean_ctor_get(v_range_2525_, 2);
v___x_2534_ = lean_nat_dec_lt(v_i_2527_, v_stop_2528_);
if (v___x_2534_ == 0)
{
lean_dec(v_i_2527_);
return v_b_2526_;
}
else
{
uint8_t v___x_2535_; 
v___x_2535_ = lean_nat_dec_eq(v_b_2526_, v___x_2522_);
if (v___x_2535_ == 0)
{
lean_object* v___x_2536_; lean_object* v___x_2537_; lean_object* v___x_2538_; uint8_t v___x_2539_; 
v___x_2536_ = lean_unsigned_to_nat(0u);
v___x_2537_ = lean_array_get_borrowed(v___x_2536_, v_a_2523_, v_b_2526_);
v___x_2538_ = lean_array_fget_borrowed(v_b_2524_, v_i_2527_);
v___x_2539_ = lean_nat_dec_eq(v___x_2537_, v___x_2538_);
if (v___x_2539_ == 0)
{
v_a_2531_ = v_b_2526_;
goto v___jp_2530_;
}
else
{
lean_object* v___x_2540_; lean_object* v_i_2541_; 
v___x_2540_ = lean_unsigned_to_nat(1u);
v_i_2541_ = lean_nat_add(v_b_2526_, v___x_2540_);
lean_dec(v_b_2526_);
v_a_2531_ = v_i_2541_;
goto v___jp_2530_;
}
}
else
{
lean_dec(v_i_2527_);
return v_b_2526_;
}
}
v___jp_2530_:
{
lean_object* v___x_2532_; 
v___x_2532_ = lean_nat_add(v_i_2527_, v_step_2529_);
lean_dec(v_i_2527_);
v_b_2526_ = v_a_2531_;
v_i_2527_ = v___x_2532_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg___boxed(lean_object* v___x_2542_, lean_object* v_a_2543_, lean_object* v_b_2544_, lean_object* v_range_2545_, lean_object* v_b_2546_, lean_object* v_i_2547_){
_start:
{
lean_object* v_res_2548_; 
v_res_2548_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg(v___x_2542_, v_a_2543_, v_b_2544_, v_range_2545_, v_b_2546_, v_i_2547_);
lean_dec_ref(v_range_2545_);
lean_dec_ref(v_b_2544_);
lean_dec_ref(v_a_2543_);
lean_dec(v___x_2542_);
return v_res_2548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0(lean_object* v_a_2549_, lean_object* v_b_2550_){
_start:
{
lean_object* v___x_2551_; lean_object* v___x_2552_; uint8_t v___x_2553_; 
v___x_2551_ = lean_array_get_size(v_b_2550_);
v___x_2552_ = lean_array_get_size(v_a_2549_);
v___x_2553_ = lean_nat_dec_lt(v___x_2551_, v___x_2552_);
if (v___x_2553_ == 0)
{
lean_object* v_i_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; uint8_t v___x_2558_; 
v_i_2554_ = lean_unsigned_to_nat(0u);
v___x_2555_ = lean_unsigned_to_nat(1u);
v___x_2556_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2556_, 0, v_i_2554_);
lean_ctor_set(v___x_2556_, 1, v___x_2551_);
lean_ctor_set(v___x_2556_, 2, v___x_2555_);
v___x_2557_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg(v___x_2552_, v_a_2549_, v_b_2550_, v___x_2556_, v_i_2554_, v_i_2554_);
lean_dec_ref_known(v___x_2556_, 3);
v___x_2558_ = lean_nat_dec_eq(v___x_2557_, v___x_2552_);
lean_dec(v___x_2557_);
return v___x_2558_;
}
else
{
uint8_t v___x_2559_; 
v___x_2559_ = 0;
return v___x_2559_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0___boxed(lean_object* v_a_2560_, lean_object* v_b_2561_){
_start:
{
uint8_t v_res_2562_; lean_object* v_r_2563_; 
v_res_2562_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0(v_a_2560_, v_b_2561_);
lean_dec_ref(v_b_2561_);
lean_dec_ref(v_a_2560_);
v_r_2563_ = lean_box(v_res_2562_);
return v_r_2563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1(lean_object* v_fData_2564_, size_t v_sz_2565_, size_t v_i_2566_, lean_object* v_bs_2567_){
_start:
{
uint8_t v___x_2568_; 
v___x_2568_ = lean_usize_dec_lt(v_i_2566_, v_sz_2565_);
if (v___x_2568_ == 0)
{
return v_bs_2567_;
}
else
{
lean_object* v_args_2569_; lean_object* v_v_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v_expr_2573_; lean_object* v___x_2574_; lean_object* v_bs_x27_2575_; size_t v___x_2576_; size_t v___x_2577_; lean_object* v___x_2578_; 
v_args_2569_ = lean_ctor_get(v_fData_2564_, 3);
v_v_2570_ = lean_array_uget_borrowed(v_bs_2567_, v_i_2566_);
v___x_2571_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
v___x_2572_ = lean_array_get_borrowed(v___x_2571_, v_args_2569_, v_v_2570_);
v_expr_2573_ = lean_ctor_get(v___x_2572_, 0);
v___x_2574_ = lean_unsigned_to_nat(0u);
v_bs_x27_2575_ = lean_array_uset(v_bs_2567_, v_i_2566_, v___x_2574_);
v___x_2576_ = ((size_t)1ULL);
v___x_2577_ = lean_usize_add(v_i_2566_, v___x_2576_);
lean_inc_ref(v_expr_2573_);
v___x_2578_ = lean_array_uset(v_bs_x27_2575_, v_i_2566_, v_expr_2573_);
v_i_2566_ = v___x_2577_;
v_bs_2567_ = v___x_2578_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1___boxed(lean_object* v_fData_2580_, lean_object* v_sz_2581_, lean_object* v_i_2582_, lean_object* v_bs_2583_){
_start:
{
size_t v_sz_boxed_2584_; size_t v_i_boxed_2585_; lean_object* v_res_2586_; 
v_sz_boxed_2584_ = lean_unbox_usize(v_sz_2581_);
lean_dec(v_sz_2581_);
v_i_boxed_2585_ = lean_unbox_usize(v_i_2582_);
lean_dec(v_i_2582_);
v_res_2586_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1(v_fData_2580_, v_sz_boxed_2584_, v_i_boxed_2585_, v_bs_2583_);
lean_dec_ref(v_fData_2580_);
return v_res_2586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs(lean_object* v_fData_2589_, lean_object* v_args_2590_, lean_object* v_a_2591_, lean_object* v_a_2592_, lean_object* v_a_2593_, lean_object* v_a_2594_){
_start:
{
lean_object* v_lctx_2599_; lean_object* v_insts_2600_; lean_object* v_fn_2601_; lean_object* v_args_2602_; lean_object* v_mainVar_2603_; lean_object* v_mainArgs_2604_; uint8_t v___x_2605_; 
v_lctx_2599_ = lean_ctor_get(v_fData_2589_, 0);
lean_inc_ref(v_lctx_2599_);
v_insts_2600_ = lean_ctor_get(v_fData_2589_, 1);
lean_inc_ref(v_insts_2600_);
v_fn_2601_ = lean_ctor_get(v_fData_2589_, 2);
lean_inc_ref(v_fn_2601_);
v_args_2602_ = lean_ctor_get(v_fData_2589_, 3);
lean_inc_ref(v_args_2602_);
v_mainVar_2603_ = lean_ctor_get(v_fData_2589_, 4);
lean_inc_ref(v_mainVar_2603_);
v_mainArgs_2604_ = lean_ctor_get(v_fData_2589_, 5);
v___x_2605_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0(v_mainArgs_2604_, v_args_2590_);
if (v___x_2605_ == 0)
{
lean_object* v___x_2606_; lean_object* v___x_2607_; 
lean_dec_ref(v_mainVar_2603_);
lean_dec_ref(v_args_2602_);
lean_dec_ref(v_fn_2601_);
lean_dec_ref(v_insts_2600_);
lean_dec_ref(v_lctx_2599_);
lean_dec_ref(v_args_2590_);
lean_dec_ref(v_fData_2589_);
v___x_2606_ = lean_box(0);
v___x_2607_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2607_, 0, v___x_2606_);
return v___x_2607_;
}
else
{
lean_object* v___x_2608_; uint8_t v___x_2609_; 
v___x_2608_ = l_Lean_Expr_fvarId_x21(v_mainVar_2603_);
v___x_2609_ = l_Lean_Expr_containsFVar(v_fn_2601_, v___x_2608_);
lean_dec(v___x_2608_);
if (v___x_2609_ == 0)
{
if (v___x_2605_ == 0)
{
lean_dec_ref(v_mainVar_2603_);
lean_dec_ref(v_args_2602_);
lean_dec_ref(v_fn_2601_);
lean_dec_ref(v_insts_2600_);
lean_dec_ref(v_lctx_2599_);
lean_dec_ref(v_args_2590_);
lean_dec_ref(v_fData_2589_);
goto v___jp_2596_;
}
else
{
size_t v_sz_2610_; size_t v___x_2611_; lean_object* v_gxs_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___f_2616_; lean_object* v___x_2617_; 
v_sz_2610_ = lean_array_size(v_args_2590_);
v___x_2611_ = ((size_t)0ULL);
lean_inc_ref(v_args_2590_);
v_gxs_2612_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__1(v_fData_2589_, v_sz_2610_, v___x_2611_, v_args_2590_);
lean_dec_ref(v_fData_2589_);
v___x_2613_ = lean_box(v___x_2609_);
v___x_2614_ = lean_box(v___x_2605_);
v___x_2615_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed__const__1));
lean_inc_ref(v_insts_2600_);
lean_inc_ref(v_lctx_2599_);
v___f_2616_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___lam__1___boxed), 15, 10);
lean_closure_set(v___f_2616_, 0, v_gxs_2612_);
lean_closure_set(v___f_2616_, 1, v_mainVar_2603_);
lean_closure_set(v___f_2616_, 2, v___x_2613_);
lean_closure_set(v___f_2616_, 3, v___x_2614_);
lean_closure_set(v___f_2616_, 4, v_lctx_2599_);
lean_closure_set(v___f_2616_, 5, v_insts_2600_);
lean_closure_set(v___f_2616_, 6, v_fn_2601_);
lean_closure_set(v___f_2616_, 7, v_args_2590_);
lean_closure_set(v___f_2616_, 8, v_args_2602_);
lean_closure_set(v___f_2616_, 9, v___x_2615_);
v___x_2617_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Meta_FunProp_FunctionData_toExpr_spec__0___redArg(v_lctx_2599_, v_insts_2600_, v___f_2616_, v_a_2591_, v_a_2592_, v_a_2593_, v_a_2594_);
return v___x_2617_;
}
}
else
{
lean_dec_ref(v_mainVar_2603_);
lean_dec_ref(v_args_2602_);
lean_dec_ref(v_fn_2601_);
lean_dec_ref(v_insts_2600_);
lean_dec_ref(v_lctx_2599_);
lean_dec_ref(v_args_2590_);
lean_dec_ref(v_fData_2589_);
goto v___jp_2596_;
}
}
v___jp_2596_:
{
lean_object* v___x_2597_; lean_object* v___x_2598_; 
v___x_2597_ = lean_box(0);
v___x_2598_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2598_, 0, v___x_2597_);
return v___x_2598_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs___boxed(lean_object* v_fData_2618_, lean_object* v_args_2619_, lean_object* v_a_2620_, lean_object* v_a_2621_, lean_object* v_a_2622_, lean_object* v_a_2623_, lean_object* v_a_2624_){
_start:
{
lean_object* v_res_2625_; 
v_res_2625_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs(v_fData_2618_, v_args_2619_, v_a_2620_, v_a_2621_, v_a_2622_, v_a_2623_);
lean_dec(v_a_2623_);
lean_dec_ref(v_a_2622_);
lean_dec(v_a_2621_);
lean_dec_ref(v_a_2620_);
return v_res_2625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0(lean_object* v___x_2626_, lean_object* v_a_2627_, lean_object* v_b_2628_, lean_object* v_range_2629_, lean_object* v_b_2630_, lean_object* v_i_2631_, lean_object* v_hs_2632_, lean_object* v_hl_2633_){
_start:
{
lean_object* v___x_2634_; 
v___x_2634_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___redArg(v___x_2626_, v_a_2627_, v_b_2628_, v_range_2629_, v_b_2630_, v_i_2631_);
return v___x_2634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0___boxed(lean_object* v___x_2635_, lean_object* v_a_2636_, lean_object* v_b_2637_, lean_object* v_range_2638_, lean_object* v_b_2639_, lean_object* v_i_2640_, lean_object* v_hs_2641_, lean_object* v_hl_2642_){
_start:
{
lean_object* v_res_2643_; 
v_res_2643_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Meta_FunProp_isOrderedSubsetOf___at___00Mathlib_Meta_FunProp_FunctionData_decompositionOverArgs_spec__0_spec__0(v___x_2635_, v_a_2636_, v_b_2637_, v_range_2638_, v_b_2639_, v_i_2640_, v_hs_2641_, v_hl_2642_);
lean_dec_ref(v_range_2638_);
lean_dec_ref(v_b_2637_);
lean_dec_ref(v_a_2636_);
lean_dec(v___x_2635_);
return v_res_2643_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication_default();
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedMorApplication();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
}
#ifdef __cplusplus
}
#endif
