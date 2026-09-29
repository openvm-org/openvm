// Lean compiler output
// Module: Mathlib.Tactic.Translate.UnfoldBoundary
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Delta public import Mathlib.Init public import Lean.Meta.Tactic.Simp
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replaceTargetEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_neutralConfig;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_ConstantInfo_name(lean_object*);
uint8_t l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_tryTheorem_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Meta_Simp_Methods_toMethodsRefImpl(lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_ExprStructEq_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_ExprStructEq_beq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_IO_CancelToken_isSet(lean_object*);
extern lean_object* l_Lean_interruptExceptionId;
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_checkApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_throwFunctionExpected___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConst(lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_NameSet_contains___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_delta_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__6_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__7_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__8_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "translate_detail"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__1_value),LEAN_SCALAR_PTR_LITERAL(169, 172, 100, 184, 244, 250, 16, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "unfoldConsts: created the cast "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " to unfold "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__10_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__11_value),LEAN_SCALAR_PTR_LITERAL(183, 66, 254, 161, 210, 133, 94, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "unfoldConsts: added a cast from "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " to "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "refoldConsts: added a cast from "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " does not have type "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "refoldConsts: not a function\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "refoldConsts: created the cast "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "@["};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "] failed to insert a cast to make `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "` have type `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "`\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "` applied to `"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "` well typed\n\n"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transform"};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___closed__0_value;
static const lean_array_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12___boxed(lean_object**);
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_headBetaBody(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_unfold_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_unfold_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_cast_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_cast_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "UnfoldBoundary"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "registerUnfoldBoundaryExt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__3_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__4_value),LEAN_SCALAR_PTR_LITERAL(152, 21, 246, 81, 15, 211, 170, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__5_value),LEAN_SCALAR_PTR_LITERAL(108, 23, 249, 243, 120, 233, 169, 125)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt();
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = l_Lean_NameSet_empty;
v___x_2_ = lean_box(1);
v___x_3_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
lean_ctor_set(v___x_3_, 1, v___x_2_);
lean_ctor_set(v___x_3_, 2, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default;
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre(lean_object* v_b_8_, lean_object* v_e_9_, lean_object* v_a_10_, lean_object* v_a_11_, lean_object* v_a_12_, lean_object* v_a_13_, lean_object* v_a_14_, lean_object* v_a_15_, lean_object* v_a_16_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_18_ = l_Lean_Expr_getAppFn(v_e_9_);
lean_inc(v_a_16_);
lean_inc_ref(v_a_15_);
lean_inc(v_a_14_);
lean_inc_ref(v_a_13_);
v___x_19_ = lean_whnf(v___x_18_, v_a_13_, v_a_14_, v_a_15_, v_a_16_);
if (lean_obj_tag(v___x_19_) == 0)
{
lean_object* v_a_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_65_; 
v_a_20_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_65_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_65_ == 0)
{
v___x_22_ = v___x_19_;
v_isShared_23_ = v_isSharedCheck_65_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_a_20_);
lean_dec(v___x_19_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_65_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
if (lean_obj_tag(v_a_20_) == 4)
{
lean_object* v_declName_24_; lean_object* v_unfolds_25_; lean_object* v___x_26_; 
v_declName_24_ = lean_ctor_get(v_a_20_, 0);
lean_inc(v_declName_24_);
lean_dec_ref_known(v_a_20_, 2);
v_unfolds_25_ = lean_ctor_get(v_b_8_, 0);
v___x_26_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_unfolds_25_, v_declName_24_);
lean_dec(v_declName_24_);
if (lean_obj_tag(v___x_26_) == 1)
{
lean_object* v_val_27_; lean_object* v___x_28_; 
lean_del_object(v___x_22_);
v_val_27_ = lean_ctor_get(v___x_26_, 0);
lean_inc(v_val_27_);
lean_dec_ref_known(v___x_26_, 1);
v___x_28_ = l_Lean_Meta_Simp_tryTheorem_x3f(v_e_9_, v_val_27_, v_a_10_, v_a_11_, v_a_12_, v_a_13_, v_a_14_, v_a_15_, v_a_16_);
if (lean_obj_tag(v___x_28_) == 0)
{
lean_object* v_a_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_48_; 
v_a_29_ = lean_ctor_get(v___x_28_, 0);
v_isSharedCheck_48_ = !lean_is_exclusive(v___x_28_);
if (v_isSharedCheck_48_ == 0)
{
v___x_31_ = v___x_28_;
v_isShared_32_ = v_isSharedCheck_48_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_a_29_);
lean_dec(v___x_28_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_48_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
if (lean_obj_tag(v_a_29_) == 1)
{
lean_object* v_val_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_43_; 
v_val_33_ = lean_ctor_get(v_a_29_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v_a_29_);
if (v_isSharedCheck_43_ == 0)
{
v___x_35_ = v_a_29_;
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_val_33_);
lean_dec(v_a_29_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v___x_38_; 
if (v_isShared_36_ == 0)
{
v___x_38_ = v___x_35_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_val_33_);
v___x_38_ = v_reuseFailAlloc_42_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
lean_object* v___x_40_; 
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 0, v___x_38_);
v___x_40_ = v___x_31_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v___x_38_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
else
{
lean_object* v___x_44_; lean_object* v___x_46_; 
lean_dec(v_a_29_);
v___x_44_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0));
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 0, v___x_44_);
v___x_46_ = v___x_31_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v___x_44_);
v___x_46_ = v_reuseFailAlloc_47_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
return v___x_46_;
}
}
}
}
else
{
lean_object* v_a_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_56_; 
v_a_49_ = lean_ctor_get(v___x_28_, 0);
v_isSharedCheck_56_ = !lean_is_exclusive(v___x_28_);
if (v_isSharedCheck_56_ == 0)
{
v___x_51_ = v___x_28_;
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_a_49_);
lean_dec(v___x_28_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_54_; 
if (v_isShared_52_ == 0)
{
v___x_54_ = v___x_51_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_a_49_);
v___x_54_ = v_reuseFailAlloc_55_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
return v___x_54_;
}
}
}
}
else
{
lean_object* v___x_57_; lean_object* v___x_59_; 
lean_dec(v___x_26_);
lean_dec_ref(v_e_9_);
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0));
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 0, v___x_57_);
v___x_59_ = v___x_22_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___x_57_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
else
{
lean_object* v___x_61_; lean_object* v___x_63_; 
lean_dec(v_a_20_);
lean_dec_ref(v_e_9_);
v___x_61_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___closed__0));
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 0, v___x_61_);
v___x_63_ = v___x_22_;
goto v_reusejp_62_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v___x_61_);
v___x_63_ = v_reuseFailAlloc_64_;
goto v_reusejp_62_;
}
v_reusejp_62_:
{
return v___x_63_;
}
}
}
}
else
{
lean_object* v_a_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_73_; 
lean_dec_ref(v_e_9_);
v_a_66_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_73_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_73_ == 0)
{
v___x_68_ = v___x_19_;
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_a_66_);
lean_dec(v___x_19_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
lean_object* v___x_71_; 
if (v_isShared_69_ == 0)
{
v___x_71_ = v___x_68_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v_a_66_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___boxed(lean_object* v_b_74_, lean_object* v_e_75_, lean_object* v_a_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre(v_b_74_, v_e_75_, v_a_76_, v_a_77_, v_a_78_, v_a_79_, v_a_80_, v_a_81_, v_a_82_);
lean_dec(v_a_82_);
lean_dec_ref(v_a_81_);
lean_dec(v_a_80_);
lean_dec_ref(v_a_79_);
lean_dec(v_a_78_);
lean_dec_ref(v_a_77_);
lean_dec(v_a_76_);
lean_dec_ref(v_b_74_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0(lean_object* v_b_85_, lean_object* v_x_86_, lean_object* v_i_87_, lean_object* v___y_88_, lean_object* v___y_89_){
_start:
{
lean_object* v_unfolds_91_; lean_object* v_casts_92_; lean_object* v___x_93_; uint8_t v___x_94_; 
v_unfolds_91_ = lean_ctor_get(v_b_85_, 0);
v_casts_92_ = lean_ctor_get(v_b_85_, 1);
v___x_93_ = l_Lean_ConstantInfo_name(v_i_87_);
v___x_94_ = l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(v___x_93_, v_unfolds_91_);
if (v___x_94_ == 0)
{
uint8_t v___x_95_; 
v___x_95_ = l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(v___x_93_, v_casts_92_);
lean_dec(v___x_93_);
if (v___x_95_ == 0)
{
uint8_t v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = 1;
v___x_97_ = lean_box(v___x_96_);
v___x_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_box(v___x_94_);
v___x_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
return v___x_100_;
}
}
else
{
uint8_t v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v___x_93_);
v___x_101_ = 0;
v___x_102_ = lean_box(v___x_101_);
v___x_103_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
return v___x_103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0___boxed(lean_object* v_b_104_, lean_object* v_x_105_, lean_object* v_i_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0(v_b_104_, v_x_105_, v_i_106_, v___y_107_, v___y_108_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec_ref(v_i_106_);
lean_dec_ref(v_x_105_);
lean_dec_ref(v_b_104_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1(lean_object* v_x_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___closed__0));
v___x_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___boxed(lean_object* v_x_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1(v_x_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v_x_124_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2(lean_object* v_e_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_143_, 0, v_e_134_);
v___x_144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2___boxed(lean_object* v_e_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__2(v_e_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
lean_dec(v___y_146_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3(lean_object* v_x_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = lean_box(0);
v___x_165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3___boxed(lean_object* v_x_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__3(v_x_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v___y_171_);
lean_dec_ref(v___y_170_);
lean_dec(v___y_169_);
lean_dec_ref(v___y_168_);
lean_dec(v___y_167_);
lean_dec_ref(v_x_166_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4(uint8_t v___x_176_, lean_object* v_e_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_186_ = lean_box(0);
v___x_187_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_187_, 0, v_e_177_);
lean_ctor_set(v___x_187_, 1, v___x_186_);
lean_ctor_set_uint8(v___x_187_, sizeof(void*)*2, v___x_176_);
v___x_188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
v___x_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4___boxed(lean_object* v___x_190_, lean_object* v_e_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_){
_start:
{
uint8_t v___x_2727__boxed_200_; lean_object* v_res_201_; 
v___x_2727__boxed_200_ = lean_unbox(v___x_190_);
v_res_201_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__4(v___x_2727__boxed_200_, v_e_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
lean_dec(v___y_198_);
lean_dec_ref(v___y_197_);
lean_dec(v___y_196_);
lean_dec_ref(v___y_195_);
lean_dec(v___y_194_);
lean_dec_ref(v___y_193_);
lean_dec(v___y_192_);
return v_res_201_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1(void){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_204_ = lean_box(0);
v___x_205_ = lean_unsigned_to_nat(16u);
v___x_206_ = lean_mk_array(v___x_205_, v___x_204_);
return v___x_206_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_207_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__1);
v___x_208_ = lean_unsigned_to_nat(0u);
v___x_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v___x_207_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3(void){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_210_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__3);
v___x_212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; lean_object* v___x_216_; 
v___x_213_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4);
v___x_214_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2);
v___x_215_ = 1;
v___x_216_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_216_, 0, v___x_214_);
lean_ctor_set(v___x_216_, 1, v___x_213_);
lean_ctor_set_uint8(v___x_216_, sizeof(void*)*2, v___x_215_);
return v___x_216_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10(void){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_223_ = lean_unsigned_to_nat(0u);
v___x_224_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4);
v___x_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___x_223_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_226_ = lean_unsigned_to_nat(32u);
v___x_227_ = lean_mk_empty_array_with_capacity(v___x_226_);
v___x_228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_228_, 0, v___x_227_);
return v___x_228_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12(void){
_start:
{
size_t v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_229_ = ((size_t)5ULL);
v___x_230_ = lean_unsigned_to_nat(0u);
v___x_231_ = lean_unsigned_to_nat(32u);
v___x_232_ = lean_mk_empty_array_with_capacity(v___x_231_);
v___x_233_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__11);
v___x_234_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
lean_ctor_set(v___x_234_, 2, v___x_230_);
lean_ctor_set(v___x_234_, 3, v___x_230_);
lean_ctor_set_usize(v___x_234_, 4, v___x_229_);
return v___x_234_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13(void){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_235_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__12);
v___x_236_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__4);
v___x_237_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
lean_ctor_set(v___x_237_, 2, v___x_236_);
lean_ctor_set(v___x_237_, 3, v___x_235_);
return v___x_237_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14(void){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_238_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__13);
v___x_239_ = lean_unsigned_to_nat(0u);
v___x_240_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__10);
v___x_241_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__2);
v___x_242_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5);
v___x_243_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v___x_241_);
lean_ctor_set(v___x_243_, 2, v___x_241_);
lean_ctor_set(v___x_243_, 3, v___x_240_);
lean_ctor_set(v___x_243_, 4, v___x_239_);
lean_ctor_set(v___x_243_, 5, v___x_238_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(lean_object* v_b_244_, lean_object* v_x_245_, lean_object* v_a_246_, lean_object* v_a_247_, lean_object* v_a_248_, lean_object* v_a_249_){
_start:
{
lean_object* v___x_251_; lean_object* v_maxSteps_252_; lean_object* v_maxDischargeDepth_253_; uint8_t v_contextual_254_; uint8_t v_memoize_255_; uint8_t v_singlePass_256_; uint8_t v_zeta_257_; uint8_t v_beta_258_; uint8_t v_eta_259_; uint8_t v_etaStruct_260_; uint8_t v_iota_261_; uint8_t v_proj_262_; uint8_t v_decide_263_; uint8_t v_arith_264_; uint8_t v_autoUnfold_265_; uint8_t v_dsimp_266_; uint8_t v_failIfUnchanged_267_; uint8_t v_ground_268_; uint8_t v_unfoldPartialApp_269_; uint8_t v_zetaDelta_270_; uint8_t v_index_271_; uint8_t v_implicitDefEqProofs_272_; uint8_t v_zetaUnused_273_; uint8_t v_catchRuntime_274_; uint8_t v_zetaHave_275_; uint8_t v_letToHave_276_; uint8_t v_congrConsts_277_; uint8_t v_bitVecOfNat_278_; uint8_t v_warnExponents_279_; uint8_t v_suggestions_280_; lean_object* v_maxSuggestions_281_; uint8_t v_locals_282_; lean_object* v_keyedConfig_283_; uint8_t v_trackZetaDelta_284_; lean_object* v_zetaDeltaSet_285_; lean_object* v_lctx_286_; lean_object* v_localInstances_287_; lean_object* v_defEqCtx_x3f_288_; lean_object* v_synthPendingDepth_289_; uint8_t v_univApprox_290_; uint8_t v_inTypeClassResolution_291_; uint8_t v_cacheInferType_292_; uint8_t v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___f_296_; uint8_t v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_251_ = l_Lean_Meta_Simp_neutralConfig;
v_maxSteps_252_ = lean_ctor_get(v___x_251_, 0);
v_maxDischargeDepth_253_ = lean_ctor_get(v___x_251_, 1);
v_contextual_254_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3);
v_memoize_255_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 1);
v_singlePass_256_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 2);
v_zeta_257_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 3);
v_beta_258_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 4);
v_eta_259_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 5);
v_etaStruct_260_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 6);
v_iota_261_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 7);
v_proj_262_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 8);
v_decide_263_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 9);
v_arith_264_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 10);
v_autoUnfold_265_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 11);
v_dsimp_266_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 12);
v_failIfUnchanged_267_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 13);
v_ground_268_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 14);
v_unfoldPartialApp_269_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 15);
v_zetaDelta_270_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 16);
v_index_271_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 17);
v_implicitDefEqProofs_272_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 18);
v_zetaUnused_273_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 19);
v_catchRuntime_274_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 20);
v_zetaHave_275_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 21);
v_letToHave_276_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 22);
v_congrConsts_277_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 23);
v_bitVecOfNat_278_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 24);
v_warnExponents_279_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 25);
v_suggestions_280_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 26);
v_maxSuggestions_281_ = lean_ctor_get(v___x_251_, 2);
v_locals_282_ = lean_ctor_get_uint8(v___x_251_, sizeof(void*)*3 + 27);
v_keyedConfig_283_ = lean_ctor_get(v_a_246_, 0);
v_trackZetaDelta_284_ = lean_ctor_get_uint8(v_a_246_, sizeof(void*)*7);
v_zetaDeltaSet_285_ = lean_ctor_get(v_a_246_, 1);
v_lctx_286_ = lean_ctor_get(v_a_246_, 2);
v_localInstances_287_ = lean_ctor_get(v_a_246_, 3);
v_defEqCtx_x3f_288_ = lean_ctor_get(v_a_246_, 4);
v_synthPendingDepth_289_ = lean_ctor_get(v_a_246_, 5);
v_univApprox_290_ = lean_ctor_get_uint8(v_a_246_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_291_ = lean_ctor_get_uint8(v_a_246_, sizeof(void*)*7 + 2);
v_cacheInferType_292_ = lean_ctor_get_uint8(v_a_246_, sizeof(void*)*7 + 3);
v___x_293_ = 1;
lean_inc(v_maxSuggestions_281_);
lean_inc(v_maxDischargeDepth_253_);
lean_inc(v_maxSteps_252_);
v___x_294_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_294_, 0, v_maxSteps_252_);
lean_ctor_set(v___x_294_, 1, v_maxDischargeDepth_253_);
lean_ctor_set(v___x_294_, 2, v_maxSuggestions_281_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3, v_contextual_254_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 1, v_memoize_255_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 2, v_singlePass_256_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 3, v_zeta_257_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 4, v_beta_258_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 5, v_eta_259_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 6, v_etaStruct_260_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 7, v_iota_261_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 8, v_proj_262_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 9, v_decide_263_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 10, v_arith_264_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 11, v_autoUnfold_265_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 12, v_dsimp_266_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 13, v_failIfUnchanged_267_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 14, v_ground_268_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 15, v_unfoldPartialApp_269_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 16, v_zetaDelta_270_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 17, v_index_271_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 18, v_implicitDefEqProofs_272_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 19, v_zetaUnused_273_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 20, v_catchRuntime_274_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 21, v_zetaHave_275_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 22, v_letToHave_276_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 23, v_congrConsts_277_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 24, v_bitVecOfNat_278_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 25, v_warnExponents_279_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 26, v_suggestions_280_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 27, v_locals_282_);
lean_ctor_set_uint8(v___x_294_, sizeof(void*)*3 + 28, v___x_293_);
v___x_295_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__0));
lean_inc_ref(v_b_244_);
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_296_, 0, v_b_244_);
v___x_297_ = 0;
v___x_298_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__5);
v___x_299_ = l_Lean_Options_empty;
v___x_300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_300_, 0, v___f_296_);
lean_inc_ref(v_keyedConfig_283_);
v___x_301_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_297_, v_keyedConfig_283_);
lean_inc(v_synthPendingDepth_289_);
lean_inc(v_defEqCtx_x3f_288_);
lean_inc_ref(v_localInstances_287_);
lean_inc_ref(v_lctx_286_);
lean_inc(v_zetaDeltaSet_285_);
v___x_302_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v_zetaDeltaSet_285_);
lean_ctor_set(v___x_302_, 2, v_lctx_286_);
lean_ctor_set(v___x_302_, 3, v_localInstances_287_);
lean_ctor_set(v___x_302_, 4, v_defEqCtx_x3f_288_);
lean_ctor_set(v___x_302_, 5, v_synthPendingDepth_289_);
lean_ctor_set(v___x_302_, 6, v___x_300_);
lean_ctor_set_uint8(v___x_302_, sizeof(void*)*7, v_trackZetaDelta_284_);
lean_ctor_set_uint8(v___x_302_, sizeof(void*)*7 + 1, v_univApprox_290_);
lean_ctor_set_uint8(v___x_302_, sizeof(void*)*7 + 2, v_inTypeClassResolution_291_);
lean_ctor_set_uint8(v___x_302_, sizeof(void*)*7 + 3, v_cacheInferType_292_);
v___x_303_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_294_, v___x_295_, v___x_298_, v___x_299_, v___x_302_, v_a_248_, v_a_249_);
if (lean_obj_tag(v___x_303_) == 0)
{
lean_object* v_a_304_; lean_object* v___f_305_; lean_object* v___f_306_; lean_object* v___f_307_; lean_object* v___f_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; 
v_a_304_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_a_304_);
lean_dec_ref_known(v___x_303_, 1);
v___f_305_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__6));
v___f_306_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__7));
v___f_307_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__8));
v___f_308_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__9));
v___x_309_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run_pre___boxed), 10, 1);
lean_closure_set(v___x_309_, 0, v_b_244_);
v___x_310_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v___f_308_);
lean_ctor_set(v___x_310_, 2, v___f_305_);
lean_ctor_set(v___x_310_, 3, v___f_306_);
lean_ctor_set(v___x_310_, 4, v___f_307_);
lean_ctor_set_uint8(v___x_310_, sizeof(void*)*5, v___x_293_);
v___x_311_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__14);
v___x_312_ = lean_st_mk_ref(v___x_311_);
v___x_313_ = l_Lean_Meta_Simp_Methods_toMethodsRefImpl(v___x_310_);
lean_dec_ref_known(v___x_310_, 5);
lean_inc(v_a_249_);
lean_inc_ref(v_a_248_);
lean_inc(v_a_247_);
lean_inc(v___x_312_);
v___x_314_ = lean_apply_8(v_x_245_, v___x_313_, v_a_304_, v___x_312_, v___x_302_, v_a_247_, v_a_248_, v_a_249_, lean_box(0));
if (lean_obj_tag(v___x_314_) == 0)
{
lean_object* v_a_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_323_; 
v_a_315_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_323_ == 0)
{
v___x_317_ = v___x_314_;
v_isShared_318_ = v_isSharedCheck_323_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_a_315_);
lean_dec(v___x_314_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_323_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_319_; lean_object* v___x_321_; 
v___x_319_ = lean_st_ref_get(v___x_312_);
lean_dec(v___x_312_);
lean_dec(v___x_319_);
if (v_isShared_318_ == 0)
{
v___x_321_ = v___x_317_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_a_315_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
else
{
lean_dec(v___x_312_);
return v___x_314_;
}
}
else
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_331_; 
lean_dec_ref_known(v___x_302_, 7);
lean_dec_ref(v_x_245_);
lean_dec_ref(v_b_244_);
v_a_324_ = lean_ctor_get(v___x_303_, 0);
v_isSharedCheck_331_ = !lean_is_exclusive(v___x_303_);
if (v_isSharedCheck_331_ == 0)
{
v___x_326_ = v___x_303_;
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_303_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_329_; 
if (v_isShared_327_ == 0)
{
v___x_329_ = v___x_326_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v_a_324_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___boxed(lean_object* v_b_332_, lean_object* v_x_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(v_b_332_, v_x_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_);
lean_dec(v_a_337_);
lean_dec_ref(v_a_336_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run(lean_object* v_00_u03b1_340_, lean_object* v_b_341_, lean_object* v_x_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(v_b_341_, v_x_342_, v_a_343_, v_a_344_, v_a_345_, v_a_346_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___boxed(lean_object* v_00_u03b1_349_, lean_object* v_b_350_, lean_object* v_x_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run(v_00_u03b1_349_, v_b_350_, v_x_351_, v_a_352_, v_a_353_, v_a_354_, v_a_355_);
lean_dec(v_a_355_);
lean_dec_ref(v_a_354_);
lean_dec(v_a_353_);
lean_dec_ref(v_a_352_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0(lean_object* v_msgData_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; lean_object* v_env_365_; lean_object* v___x_366_; lean_object* v_mctx_367_; lean_object* v_lctx_368_; lean_object* v_options_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_364_ = lean_st_ref_get(v___y_362_);
v_env_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc_ref(v_env_365_);
lean_dec(v___x_364_);
v___x_366_ = lean_st_ref_get(v___y_360_);
v_mctx_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc_ref(v_mctx_367_);
lean_dec(v___x_366_);
v_lctx_368_ = lean_ctor_get(v___y_359_, 2);
v_options_369_ = lean_ctor_get(v___y_361_, 2);
lean_inc_ref(v_options_369_);
lean_inc_ref(v_lctx_368_);
v___x_370_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_370_, 0, v_env_365_);
lean_ctor_set(v___x_370_, 1, v_mctx_367_);
lean_ctor_set(v___x_370_, 2, v_lctx_368_);
lean_ctor_set(v___x_370_, 3, v_options_369_);
v___x_371_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v_msgData_358_);
v___x_372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0___boxed(lean_object* v_msgData_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0(v_msgData_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_);
lean_dec(v___y_377_);
lean_dec_ref(v___y_376_);
lean_dec(v___y_375_);
lean_dec_ref(v___y_374_);
return v_res_379_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_380_; double v___x_381_; 
v___x_380_ = lean_unsigned_to_nat(0u);
v___x_381_ = lean_float_of_nat(v___x_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(lean_object* v_cls_385_, lean_object* v_msg_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_){
_start:
{
lean_object* v_ref_392_; lean_object* v___x_393_; lean_object* v_a_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_438_; 
v_ref_392_ = lean_ctor_get(v___y_389_, 5);
v___x_393_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0(v_msg_386_, v___y_387_, v___y_388_, v___y_389_, v___y_390_);
v_a_394_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_438_ == 0)
{
v___x_396_ = v___x_393_;
v_isShared_397_ = v_isSharedCheck_438_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_a_394_);
lean_dec(v___x_393_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_438_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
lean_object* v___x_398_; lean_object* v_traceState_399_; lean_object* v_env_400_; lean_object* v_nextMacroScope_401_; lean_object* v_ngen_402_; lean_object* v_auxDeclNGen_403_; lean_object* v_cache_404_; lean_object* v_messages_405_; lean_object* v_infoState_406_; lean_object* v_snapshotTasks_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_437_; 
v___x_398_ = lean_st_ref_take(v___y_390_);
v_traceState_399_ = lean_ctor_get(v___x_398_, 4);
v_env_400_ = lean_ctor_get(v___x_398_, 0);
v_nextMacroScope_401_ = lean_ctor_get(v___x_398_, 1);
v_ngen_402_ = lean_ctor_get(v___x_398_, 2);
v_auxDeclNGen_403_ = lean_ctor_get(v___x_398_, 3);
v_cache_404_ = lean_ctor_get(v___x_398_, 5);
v_messages_405_ = lean_ctor_get(v___x_398_, 6);
v_infoState_406_ = lean_ctor_get(v___x_398_, 7);
v_snapshotTasks_407_ = lean_ctor_get(v___x_398_, 8);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_437_ == 0)
{
v___x_409_ = v___x_398_;
v_isShared_410_ = v_isSharedCheck_437_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_snapshotTasks_407_);
lean_inc(v_infoState_406_);
lean_inc(v_messages_405_);
lean_inc(v_cache_404_);
lean_inc(v_traceState_399_);
lean_inc(v_auxDeclNGen_403_);
lean_inc(v_ngen_402_);
lean_inc(v_nextMacroScope_401_);
lean_inc(v_env_400_);
lean_dec(v___x_398_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_437_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
uint64_t v_tid_411_; lean_object* v_traces_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_436_; 
v_tid_411_ = lean_ctor_get_uint64(v_traceState_399_, sizeof(void*)*1);
v_traces_412_ = lean_ctor_get(v_traceState_399_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v_traceState_399_);
if (v_isSharedCheck_436_ == 0)
{
v___x_414_ = v_traceState_399_;
v_isShared_415_ = v_isSharedCheck_436_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_traces_412_);
lean_dec(v_traceState_399_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_436_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_416_; double v___x_417_; uint8_t v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_426_; 
v___x_416_ = lean_box(0);
v___x_417_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__0);
v___x_418_ = 0;
v___x_419_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__1));
v___x_420_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_420_, 0, v_cls_385_);
lean_ctor_set(v___x_420_, 1, v___x_416_);
lean_ctor_set(v___x_420_, 2, v___x_419_);
lean_ctor_set_float(v___x_420_, sizeof(void*)*3, v___x_417_);
lean_ctor_set_float(v___x_420_, sizeof(void*)*3 + 8, v___x_417_);
lean_ctor_set_uint8(v___x_420_, sizeof(void*)*3 + 16, v___x_418_);
v___x_421_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___closed__2));
v___x_422_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_422_, 0, v___x_420_);
lean_ctor_set(v___x_422_, 1, v_a_394_);
lean_ctor_set(v___x_422_, 2, v___x_421_);
lean_inc(v_ref_392_);
v___x_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_423_, 0, v_ref_392_);
lean_ctor_set(v___x_423_, 1, v___x_422_);
v___x_424_ = l_Lean_PersistentArray_push___redArg(v_traces_412_, v___x_423_);
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 0, v___x_424_);
v___x_426_ = v___x_414_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v___x_424_);
lean_ctor_set_uint64(v_reuseFailAlloc_435_, sizeof(void*)*1, v_tid_411_);
v___x_426_ = v_reuseFailAlloc_435_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
lean_object* v___x_428_; 
if (v_isShared_410_ == 0)
{
lean_ctor_set(v___x_409_, 4, v___x_426_);
v___x_428_ = v___x_409_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_env_400_);
lean_ctor_set(v_reuseFailAlloc_434_, 1, v_nextMacroScope_401_);
lean_ctor_set(v_reuseFailAlloc_434_, 2, v_ngen_402_);
lean_ctor_set(v_reuseFailAlloc_434_, 3, v_auxDeclNGen_403_);
lean_ctor_set(v_reuseFailAlloc_434_, 4, v___x_426_);
lean_ctor_set(v_reuseFailAlloc_434_, 5, v_cache_404_);
lean_ctor_set(v_reuseFailAlloc_434_, 6, v_messages_405_);
lean_ctor_set(v_reuseFailAlloc_434_, 7, v_infoState_406_);
lean_ctor_set(v_reuseFailAlloc_434_, 8, v_snapshotTasks_407_);
v___x_428_ = v_reuseFailAlloc_434_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
v___x_429_ = lean_st_ref_set(v___y_390_, v___x_428_);
v___x_430_ = lean_box(0);
if (v_isShared_397_ == 0)
{
lean_ctor_set(v___x_396_, 0, v___x_430_);
v___x_432_ = v___x_396_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v___x_430_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg___boxed(lean_object* v_cls_439_, lean_object* v_msg_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v_cls_439_, v_msg_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
return v_res_446_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0(void){
_start:
{
lean_object* v___x_447_; lean_object* v_dummy_448_; 
v___x_447_ = lean_box(0);
v_dummy_448_ = l_Lean_Expr_sort___override(v___x_447_);
return v_dummy_448_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_455_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2));
v___x_456_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__4));
v___x_457_ = l_Lean_Name_append(v___x_456_, v___x_455_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7(void){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_459_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__6));
v___x_460_ = l_Lean_stringToMessageData(v___x_459_);
return v___x_460_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__8));
v___x_463_ = l_Lean_stringToMessageData(v___x_462_);
return v___x_463_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_470_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__13));
v___x_471_ = l_Lean_stringToMessageData(v___x_470_);
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16(void){
_start:
{
lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_473_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__15));
v___x_474_ = l_Lean_stringToMessageData(v___x_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts(lean_object* v_b_475_, lean_object* v_e_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_){
_start:
{
lean_object* v_e_486_; lean_object* v___y_487_; lean_object* v___y_488_; lean_object* v___y_489_; lean_object* v___y_490_; lean_object* v___y_491_; lean_object* v___y_492_; lean_object* v___y_493_; lean_object* v___x_557_; 
lean_inc(v_a_483_);
lean_inc_ref(v_a_482_);
lean_inc(v_a_481_);
lean_inc_ref(v_a_480_);
lean_inc_ref(v_e_476_);
v___x_557_ = lean_infer_type(v_e_476_, v_a_480_, v_a_481_, v_a_482_, v_a_483_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_object* v_a_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_618_; 
v_a_558_ = lean_ctor_get(v___x_557_, 0);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_557_);
if (v_isSharedCheck_618_ == 0)
{
v___x_560_ = v___x_557_;
v_isShared_561_ = v_isSharedCheck_618_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_a_558_);
lean_dec(v___x_557_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_618_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_562_; 
lean_inc(v_a_483_);
lean_inc_ref(v_a_482_);
lean_inc(v_a_481_);
lean_inc_ref(v_a_480_);
lean_inc(v_a_479_);
lean_inc_ref(v_a_478_);
lean_inc(v_a_477_);
lean_inc(v_a_558_);
v___x_562_ = lean_simp(v_a_558_, v_a_477_, v_a_478_, v_a_479_, v_a_480_, v_a_481_, v_a_482_, v_a_483_);
if (lean_obj_tag(v___x_562_) == 0)
{
lean_object* v_a_563_; lean_object* v_expr_564_; lean_object* v_proof_x3f_565_; lean_object* v___y_567_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_573_; 
v_a_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_a_563_);
lean_dec_ref_known(v___x_562_, 1);
v_expr_564_ = lean_ctor_get(v_a_563_, 0);
lean_inc_ref(v_expr_564_);
v_proof_x3f_565_ = lean_ctor_get(v_a_563_, 1);
lean_inc(v_proof_x3f_565_);
lean_dec(v_a_563_);
if (lean_obj_tag(v_proof_x3f_565_) == 1)
{
lean_object* v_options_588_; uint8_t v_hasTrace_589_; 
v_options_588_ = lean_ctor_get(v_a_482_, 2);
v_hasTrace_589_ = lean_ctor_get_uint8(v_options_588_, sizeof(void*)*1);
if (v_hasTrace_589_ == 0)
{
v___y_567_ = v_a_477_;
v___y_568_ = v_a_478_;
v___y_569_ = v_a_479_;
v___y_570_ = v_a_480_;
v___y_571_ = v_a_481_;
v___y_572_ = v_a_482_;
v___y_573_ = v_a_483_;
goto v___jp_566_;
}
else
{
lean_object* v_inheritedTraceOptions_590_; lean_object* v___x_591_; lean_object* v___x_592_; uint8_t v___x_593_; 
v_inheritedTraceOptions_590_ = lean_ctor_get(v_a_482_, 13);
v___x_591_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2));
v___x_592_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5);
v___x_593_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_590_, v_options_588_, v___x_592_);
if (v___x_593_ == 0)
{
v___y_567_ = v_a_477_;
v___y_568_ = v_a_478_;
v___y_569_ = v_a_479_;
v___y_570_ = v_a_480_;
v___y_571_ = v_a_481_;
v___y_572_ = v_a_482_;
v___y_573_ = v_a_483_;
goto v___jp_566_;
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; 
v___x_594_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__14);
lean_inc(v_a_558_);
v___x_595_ = l_Lean_MessageData_ofExpr(v_a_558_);
v___x_596_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_594_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
v___x_597_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16);
v___x_598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_596_);
lean_ctor_set(v___x_598_, 1, v___x_597_);
lean_inc_ref(v_expr_564_);
v___x_599_ = l_Lean_MessageData_ofExpr(v_expr_564_);
v___x_600_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_598_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
v___x_601_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v___x_591_, v___x_600_, v_a_480_, v_a_481_, v_a_482_, v_a_483_);
if (lean_obj_tag(v___x_601_) == 0)
{
lean_dec_ref_known(v___x_601_, 1);
v___y_567_ = v_a_477_;
v___y_568_ = v_a_478_;
v___y_569_ = v_a_479_;
v___y_570_ = v_a_480_;
v___y_571_ = v_a_481_;
v___y_572_ = v_a_482_;
v___y_573_ = v_a_483_;
goto v___jp_566_;
}
else
{
lean_object* v_a_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_609_; 
lean_dec_ref_known(v_proof_x3f_565_, 1);
lean_dec_ref(v_expr_564_);
lean_del_object(v___x_560_);
lean_dec(v_a_558_);
lean_dec_ref(v_e_476_);
v_a_602_ = lean_ctor_get(v___x_601_, 0);
v_isSharedCheck_609_ = !lean_is_exclusive(v___x_601_);
if (v_isSharedCheck_609_ == 0)
{
v___x_604_ = v___x_601_;
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_a_602_);
lean_dec(v___x_601_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
lean_object* v___x_607_; 
if (v_isShared_605_ == 0)
{
v___x_607_ = v___x_604_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_a_602_);
v___x_607_ = v_reuseFailAlloc_608_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
return v___x_607_;
}
}
}
}
}
}
else
{
lean_dec(v_proof_x3f_565_);
lean_dec_ref(v_expr_564_);
lean_del_object(v___x_560_);
lean_dec(v_a_558_);
v_e_486_ = v_e_476_;
v___y_487_ = v_a_477_;
v___y_488_ = v_a_478_;
v___y_489_ = v_a_479_;
v___y_490_ = v_a_480_;
v___y_491_ = v_a_481_;
v___y_492_ = v_a_482_;
v___y_493_ = v_a_483_;
goto v___jp_485_;
}
v___jp_566_:
{
lean_object* v___x_574_; lean_object* v___x_576_; 
v___x_574_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__12));
if (v_isShared_561_ == 0)
{
lean_ctor_set_tag(v___x_560_, 1);
v___x_576_ = v___x_560_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v_a_558_);
v___x_576_ = v_reuseFailAlloc_587_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_577_, 0, v_expr_564_);
v___x_578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_578_, 0, v_e_476_);
v___x_579_ = lean_unsigned_to_nat(4u);
v___x_580_ = lean_mk_empty_array_with_capacity(v___x_579_);
v___x_581_ = lean_array_push(v___x_580_, v___x_576_);
v___x_582_ = lean_array_push(v___x_581_, v___x_577_);
v___x_583_ = lean_array_push(v___x_582_, v_proof_x3f_565_);
v___x_584_ = lean_array_push(v___x_583_, v___x_578_);
v___x_585_ = l_Lean_Meta_mkAppOptM(v___x_574_, v___x_584_, v___y_570_, v___y_571_, v___y_572_, v___y_573_);
if (lean_obj_tag(v___x_585_) == 0)
{
lean_object* v_a_586_; 
v_a_586_ = lean_ctor_get(v___x_585_, 0);
lean_inc(v_a_586_);
lean_dec_ref_known(v___x_585_, 1);
v_e_486_ = v_a_586_;
v___y_487_ = v___y_567_;
v___y_488_ = v___y_568_;
v___y_489_ = v___y_569_;
v___y_490_ = v___y_570_;
v___y_491_ = v___y_571_;
v___y_492_ = v___y_572_;
v___y_493_ = v___y_573_;
goto v___jp_485_;
}
else
{
return v___x_585_;
}
}
}
}
else
{
lean_object* v_a_610_; lean_object* v___x_612_; uint8_t v_isShared_613_; uint8_t v_isSharedCheck_617_; 
lean_del_object(v___x_560_);
lean_dec(v_a_558_);
lean_dec_ref(v_e_476_);
v_a_610_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_617_ == 0)
{
v___x_612_ = v___x_562_;
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
else
{
lean_inc(v_a_610_);
lean_dec(v___x_562_);
v___x_612_ = lean_box(0);
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
v_resetjp_611_:
{
lean_object* v___x_615_; 
if (v_isShared_613_ == 0)
{
v___x_615_ = v___x_612_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v_a_610_);
v___x_615_ = v_reuseFailAlloc_616_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
return v___x_615_;
}
}
}
}
}
else
{
lean_dec_ref(v_e_476_);
return v___x_557_;
}
v___jp_485_:
{
lean_object* v___x_494_; 
lean_inc(v___y_493_);
lean_inc_ref(v___y_492_);
lean_inc(v___y_491_);
lean_inc_ref(v___y_490_);
lean_inc_ref(v_e_486_);
v___x_494_ = lean_infer_type(v_e_486_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; lean_object* v___x_496_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc(v_a_495_);
lean_dec_ref_known(v___x_494_, 1);
lean_inc(v___y_493_);
lean_inc_ref(v___y_492_);
lean_inc(v___y_491_);
lean_inc_ref(v___y_490_);
v___x_496_ = lean_whnf(v_a_495_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_a_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_556_; 
v_a_497_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_556_ == 0)
{
v___x_499_ = v___x_496_;
v_isShared_500_ = v_isSharedCheck_556_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_a_497_);
lean_dec(v___x_496_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_556_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___x_501_; 
v___x_501_ = l_Lean_Expr_getAppFn(v_a_497_);
if (lean_obj_tag(v___x_501_) == 4)
{
lean_object* v_declName_502_; lean_object* v_us_503_; lean_object* v_casts_504_; lean_object* v___x_505_; 
v_declName_502_ = lean_ctor_get(v___x_501_, 0);
lean_inc(v_declName_502_);
v_us_503_ = lean_ctor_get(v___x_501_, 1);
lean_inc(v_us_503_);
lean_dec_ref_known(v___x_501_, 2);
v_casts_504_ = lean_ctor_get(v_b_475_, 1);
v___x_505_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_casts_504_, v_declName_502_);
if (lean_obj_tag(v___x_505_) == 1)
{
lean_object* v_val_506_; lean_object* v_options_507_; lean_object* v_fst_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_548_; 
lean_del_object(v___x_499_);
v_val_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc(v_val_506_);
lean_dec_ref_known(v___x_505_, 1);
v_options_507_ = lean_ctor_get(v___y_492_, 2);
v_fst_508_ = lean_ctor_get(v_val_506_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v_val_506_);
if (v_isSharedCheck_548_ == 0)
{
lean_object* v_unused_549_; 
v_unused_549_ = lean_ctor_get(v_val_506_, 1);
lean_dec(v_unused_549_);
v___x_510_ = v_val_506_;
v_isShared_511_ = v_isSharedCheck_548_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_fst_508_);
lean_dec(v_val_506_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_548_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v_inheritedTraceOptions_512_; uint8_t v_hasTrace_513_; lean_object* v_nargs_514_; lean_object* v___x_515_; lean_object* v_dummy_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v_inheritedTraceOptions_512_ = lean_ctor_get(v___y_492_, 13);
v_hasTrace_513_ = lean_ctor_get_uint8(v_options_507_, sizeof(void*)*1);
v_nargs_514_ = l_Lean_Expr_getAppNumArgs(v_a_497_);
v___x_515_ = l_Lean_Expr_const___override(v_fst_508_, v_us_503_);
v_dummy_516_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0);
lean_inc(v_nargs_514_);
v___x_517_ = lean_mk_array(v_nargs_514_, v_dummy_516_);
v___x_518_ = lean_unsigned_to_nat(1u);
v___x_519_ = lean_nat_sub(v_nargs_514_, v___x_518_);
lean_dec(v_nargs_514_);
v___x_520_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_497_, v___x_517_, v___x_519_);
v___x_521_ = l_Lean_mkAppN(v___x_515_, v___x_520_);
lean_dec_ref(v___x_520_);
v___x_522_ = l_Lean_Expr_app___override(v___x_521_, v_e_486_);
if (v_hasTrace_513_ == 0)
{
lean_del_object(v___x_510_);
lean_dec(v_declName_502_);
v_e_476_ = v___x_522_;
v_a_477_ = v___y_487_;
v_a_478_ = v___y_488_;
v_a_479_ = v___y_489_;
v_a_480_ = v___y_490_;
v_a_481_ = v___y_491_;
v_a_482_ = v___y_492_;
v_a_483_ = v___y_493_;
goto _start;
}
else
{
lean_object* v___x_524_; lean_object* v___x_525_; uint8_t v___x_526_; 
v___x_524_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2));
v___x_525_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5);
v___x_526_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_512_, v_options_507_, v___x_525_);
if (v___x_526_ == 0)
{
lean_del_object(v___x_510_);
lean_dec(v_declName_502_);
v_e_476_ = v___x_522_;
v_a_477_ = v___y_487_;
v_a_478_ = v___y_488_;
v_a_479_ = v___y_489_;
v_a_480_ = v___y_490_;
v_a_481_ = v___y_491_;
v_a_482_ = v___y_492_;
v_a_483_ = v___y_493_;
goto _start;
}
else
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_531_; 
v___x_528_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__7);
lean_inc_ref(v___x_522_);
v___x_529_ = l_Lean_MessageData_ofExpr(v___x_522_);
if (v_isShared_511_ == 0)
{
lean_ctor_set_tag(v___x_510_, 7);
lean_ctor_set(v___x_510_, 1, v___x_529_);
lean_ctor_set(v___x_510_, 0, v___x_528_);
v___x_531_ = v___x_510_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_528_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v___x_529_);
v___x_531_ = v_reuseFailAlloc_547_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
lean_object* v___x_532_; lean_object* v___x_533_; uint8_t v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_532_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9);
v___x_533_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_533_, 0, v___x_531_);
lean_ctor_set(v___x_533_, 1, v___x_532_);
v___x_534_ = 0;
v___x_535_ = l_Lean_MessageData_ofConstName(v_declName_502_, v___x_534_);
v___x_536_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_533_);
lean_ctor_set(v___x_536_, 1, v___x_535_);
v___x_537_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v___x_524_, v___x_536_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_537_) == 0)
{
lean_dec_ref_known(v___x_537_, 1);
v_e_476_ = v___x_522_;
v_a_477_ = v___y_487_;
v_a_478_ = v___y_488_;
v_a_479_ = v___y_489_;
v_a_480_ = v___y_490_;
v_a_481_ = v___y_491_;
v_a_482_ = v___y_492_;
v_a_483_ = v___y_493_;
goto _start;
}
else
{
lean_object* v_a_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_546_; 
lean_dec_ref(v___x_522_);
v_a_539_ = lean_ctor_get(v___x_537_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_546_ == 0)
{
v___x_541_ = v___x_537_;
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_a_539_);
lean_dec(v___x_537_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_544_; 
if (v_isShared_542_ == 0)
{
v___x_544_ = v___x_541_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_a_539_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
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
lean_object* v___x_551_; 
lean_dec(v___x_505_);
lean_dec(v_us_503_);
lean_dec(v_declName_502_);
lean_dec(v_a_497_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 0, v_e_486_);
v___x_551_ = v___x_499_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_e_486_);
v___x_551_ = v_reuseFailAlloc_552_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
return v___x_551_;
}
}
}
else
{
lean_object* v___x_554_; 
lean_dec_ref(v___x_501_);
lean_dec(v_a_497_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 0, v_e_486_);
v___x_554_ = v___x_499_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_e_486_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
}
else
{
lean_dec_ref(v_e_486_);
return v___x_496_;
}
}
else
{
lean_dec_ref(v_e_486_);
return v___x_494_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___boxed(lean_object* v_b_619_, lean_object* v_e_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_){
_start:
{
lean_object* v_res_629_; 
v_res_629_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts(v_b_619_, v_e_620_, v_a_621_, v_a_622_, v_a_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
lean_dec(v_a_623_);
lean_dec_ref(v_a_622_);
lean_dec(v_a_621_);
lean_dec_ref(v_b_619_);
return v_res_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0(lean_object* v_cls_630_, lean_object* v_msg_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v_cls_630_, v_msg_631_, v___y_635_, v___y_636_, v___y_637_, v___y_638_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___boxed(lean_object* v_cls_641_, lean_object* v_msg_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0(v_cls_641_, v_msg_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
lean_dec(v___y_643_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0(lean_object* v_k_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v_b_656_, lean_object* v_c_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v___x_663_; 
lean_inc(v___y_661_);
lean_inc_ref(v___y_660_);
lean_inc(v___y_659_);
lean_inc_ref(v___y_658_);
lean_inc(v___y_655_);
lean_inc_ref(v___y_654_);
lean_inc(v___y_653_);
v___x_663_ = lean_apply_10(v_k_652_, v_b_656_, v_c_657_, v___y_653_, v___y_654_, v___y_655_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, lean_box(0));
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0___boxed(lean_object* v_k_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v_b_668_, lean_object* v_c_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v_res_675_; 
v_res_675_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0(v_k_664_, v___y_665_, v___y_666_, v___y_667_, v_b_668_, v_c_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_);
lean_dec(v___y_673_);
lean_dec_ref(v___y_672_);
lean_dec(v___y_671_);
lean_dec_ref(v___y_670_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
lean_dec(v___y_665_);
return v_res_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg(lean_object* v_type_676_, lean_object* v_k_677_, uint8_t v_cleanupAnnotations_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
lean_object* v___f_687_; uint8_t v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
lean_inc(v___y_681_);
lean_inc_ref(v___y_680_);
lean_inc(v___y_679_);
v___f_687_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___lam__0___boxed), 11, 4);
lean_closure_set(v___f_687_, 0, v_k_677_);
lean_closure_set(v___f_687_, 1, v___y_679_);
lean_closure_set(v___f_687_, 2, v___y_680_);
lean_closure_set(v___f_687_, 3, v___y_681_);
v___x_688_ = 0;
v___x_689_ = lean_box(0);
v___x_690_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_688_, v___x_689_, v_type_676_, v___f_687_, v_cleanupAnnotations_678_, v___x_688_, v___y_682_, v___y_683_, v___y_684_, v___y_685_);
if (lean_obj_tag(v___x_690_) == 0)
{
return v___x_690_;
}
else
{
lean_object* v_a_691_; lean_object* v___x_693_; uint8_t v_isShared_694_; uint8_t v_isSharedCheck_698_; 
v_a_691_ = lean_ctor_get(v___x_690_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_690_);
if (v_isSharedCheck_698_ == 0)
{
v___x_693_ = v___x_690_;
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
else
{
lean_inc(v_a_691_);
lean_dec(v___x_690_);
v___x_693_ = lean_box(0);
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
v_resetjp_692_:
{
lean_object* v___x_696_; 
if (v_isShared_694_ == 0)
{
v___x_696_ = v___x_693_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v_a_691_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg___boxed(lean_object* v_type_699_, lean_object* v_k_700_, lean_object* v_cleanupAnnotations_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_710_; lean_object* v_res_711_; 
v_cleanupAnnotations_boxed_710_ = lean_unbox(v_cleanupAnnotations_701_);
v_res_711_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg(v_type_699_, v_k_700_, v_cleanupAnnotations_boxed_710_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_, v___y_708_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec(v___y_706_);
lean_dec_ref(v___y_705_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
lean_dec(v___y_702_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2(lean_object* v_00_u03b1_712_, lean_object* v_type_713_, lean_object* v_k_714_, uint8_t v_cleanupAnnotations_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_){
_start:
{
lean_object* v___x_724_; 
v___x_724_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg(v_type_713_, v_k_714_, v_cleanupAnnotations_715_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___boxed(lean_object* v_00_u03b1_725_, lean_object* v_type_726_, lean_object* v_k_727_, lean_object* v_cleanupAnnotations_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_737_; lean_object* v_res_738_; 
v_cleanupAnnotations_boxed_737_ = lean_unbox(v_cleanupAnnotations_728_);
v_res_738_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2(v_00_u03b1_725_, v_type_726_, v_k_727_, v_cleanupAnnotations_boxed_737_, v___y_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_, v___y_734_, v___y_735_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
return v_res_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(lean_object* v_msg_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v_ref_745_; lean_object* v___x_746_; lean_object* v_a_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_755_; 
v_ref_745_ = lean_ctor_get(v___y_742_, 5);
v___x_746_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0_spec__0(v_msg_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_);
v_a_747_ = lean_ctor_get(v___x_746_, 0);
v_isSharedCheck_755_ = !lean_is_exclusive(v___x_746_);
if (v_isSharedCheck_755_ == 0)
{
v___x_749_ = v___x_746_;
v_isShared_750_ = v_isSharedCheck_755_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_a_747_);
lean_dec(v___x_746_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_755_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_751_; lean_object* v___x_753_; 
lean_inc(v_ref_745_);
v___x_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_751_, 0, v_ref_745_);
lean_ctor_set(v___x_751_, 1, v_a_747_);
if (v_isShared_750_ == 0)
{
lean_ctor_set_tag(v___x_749_, 1);
lean_ctor_set(v___x_749_, 0, v___x_751_);
v___x_753_ = v___x_749_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v___x_751_);
v___x_753_ = v_reuseFailAlloc_754_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
return v___x_753_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg___boxed(lean_object* v_msg_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_){
_start:
{
lean_object* v_res_762_; 
v_res_762_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v_msg_756_, v___y_757_, v___y_758_, v___y_759_, v___y_760_);
lean_dec(v___y_760_);
lean_dec_ref(v___y_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
return v_res_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_x_763_, lean_object* v_x_764_, lean_object* v_x_765_, lean_object* v_x_766_){
_start:
{
lean_object* v_ks_767_; lean_object* v_vs_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_792_; 
v_ks_767_ = lean_ctor_get(v_x_763_, 0);
v_vs_768_ = lean_ctor_get(v_x_763_, 1);
v_isSharedCheck_792_ = !lean_is_exclusive(v_x_763_);
if (v_isSharedCheck_792_ == 0)
{
v___x_770_ = v_x_763_;
v_isShared_771_ = v_isSharedCheck_792_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_vs_768_);
lean_inc(v_ks_767_);
lean_dec(v_x_763_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_792_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_772_; uint8_t v___x_773_; 
v___x_772_ = lean_array_get_size(v_ks_767_);
v___x_773_ = lean_nat_dec_lt(v_x_764_, v___x_772_);
if (v___x_773_ == 0)
{
lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_777_; 
lean_dec(v_x_764_);
v___x_774_ = lean_array_push(v_ks_767_, v_x_765_);
v___x_775_ = lean_array_push(v_vs_768_, v_x_766_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 1, v___x_775_);
lean_ctor_set(v___x_770_, 0, v___x_774_);
v___x_777_ = v___x_770_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_774_);
lean_ctor_set(v_reuseFailAlloc_778_, 1, v___x_775_);
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
lean_object* v_k_x27_779_; uint8_t v___x_780_; 
v_k_x27_779_ = lean_array_fget_borrowed(v_ks_767_, v_x_764_);
v___x_780_ = l_Lean_instBEqMVarId_beq(v_x_765_, v_k_x27_779_);
if (v___x_780_ == 0)
{
lean_object* v___x_782_; 
if (v_isShared_771_ == 0)
{
v___x_782_ = v___x_770_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_ks_767_);
lean_ctor_set(v_reuseFailAlloc_786_, 1, v_vs_768_);
v___x_782_ = v_reuseFailAlloc_786_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
lean_object* v___x_783_; lean_object* v___x_784_; 
v___x_783_ = lean_unsigned_to_nat(1u);
v___x_784_ = lean_nat_add(v_x_764_, v___x_783_);
lean_dec(v_x_764_);
v_x_763_ = v___x_782_;
v_x_764_ = v___x_784_;
goto _start;
}
}
else
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_790_; 
v___x_787_ = lean_array_fset(v_ks_767_, v_x_764_, v_x_765_);
v___x_788_ = lean_array_fset(v_vs_768_, v_x_764_, v_x_766_);
lean_dec(v_x_764_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 1, v___x_788_);
lean_ctor_set(v___x_770_, 0, v___x_787_);
v___x_790_ = v___x_770_;
goto v_reusejp_789_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v___x_787_);
lean_ctor_set(v_reuseFailAlloc_791_, 1, v___x_788_);
v___x_790_ = v_reuseFailAlloc_791_;
goto v_reusejp_789_;
}
v_reusejp_789_:
{
return v___x_790_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4___redArg(lean_object* v_n_793_, lean_object* v_k_794_, lean_object* v_v_795_){
_start:
{
lean_object* v___x_796_; lean_object* v___x_797_; 
v___x_796_ = lean_unsigned_to_nat(0u);
v___x_797_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_n_793_, v___x_796_, v_k_794_, v_v_795_);
return v___x_797_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(lean_object* v_x_799_, size_t v_x_800_, size_t v_x_801_, lean_object* v_x_802_, lean_object* v_x_803_){
_start:
{
if (lean_obj_tag(v_x_799_) == 0)
{
lean_object* v_es_804_; size_t v___x_805_; size_t v___x_806_; lean_object* v_j_807_; lean_object* v___x_808_; uint8_t v___x_809_; 
v_es_804_ = lean_ctor_get(v_x_799_, 0);
v___x_805_ = ((size_t)31ULL);
v___x_806_ = lean_usize_land(v_x_800_, v___x_805_);
v_j_807_ = lean_usize_to_nat(v___x_806_);
v___x_808_ = lean_array_get_size(v_es_804_);
v___x_809_ = lean_nat_dec_lt(v_j_807_, v___x_808_);
if (v___x_809_ == 0)
{
lean_dec(v_j_807_);
lean_dec(v_x_803_);
lean_dec(v_x_802_);
return v_x_799_;
}
else
{
lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_848_; 
lean_inc_ref(v_es_804_);
v_isSharedCheck_848_ = !lean_is_exclusive(v_x_799_);
if (v_isSharedCheck_848_ == 0)
{
lean_object* v_unused_849_; 
v_unused_849_ = lean_ctor_get(v_x_799_, 0);
lean_dec(v_unused_849_);
v___x_811_ = v_x_799_;
v_isShared_812_ = v_isSharedCheck_848_;
goto v_resetjp_810_;
}
else
{
lean_dec(v_x_799_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_848_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v_v_813_; lean_object* v___x_814_; lean_object* v_xs_x27_815_; lean_object* v___y_817_; 
v_v_813_ = lean_array_fget(v_es_804_, v_j_807_);
v___x_814_ = lean_box(0);
v_xs_x27_815_ = lean_array_fset(v_es_804_, v_j_807_, v___x_814_);
switch(lean_obj_tag(v_v_813_))
{
case 0:
{
lean_object* v_key_822_; lean_object* v_val_823_; lean_object* v___x_825_; uint8_t v_isShared_826_; uint8_t v_isSharedCheck_833_; 
v_key_822_ = lean_ctor_get(v_v_813_, 0);
v_val_823_ = lean_ctor_get(v_v_813_, 1);
v_isSharedCheck_833_ = !lean_is_exclusive(v_v_813_);
if (v_isSharedCheck_833_ == 0)
{
v___x_825_ = v_v_813_;
v_isShared_826_ = v_isSharedCheck_833_;
goto v_resetjp_824_;
}
else
{
lean_inc(v_val_823_);
lean_inc(v_key_822_);
lean_dec(v_v_813_);
v___x_825_ = lean_box(0);
v_isShared_826_ = v_isSharedCheck_833_;
goto v_resetjp_824_;
}
v_resetjp_824_:
{
uint8_t v___x_827_; 
v___x_827_ = l_Lean_instBEqMVarId_beq(v_x_802_, v_key_822_);
if (v___x_827_ == 0)
{
lean_object* v___x_828_; lean_object* v___x_829_; 
lean_del_object(v___x_825_);
v___x_828_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_822_, v_val_823_, v_x_802_, v_x_803_);
v___x_829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
v___y_817_ = v___x_829_;
goto v___jp_816_;
}
else
{
lean_object* v___x_831_; 
lean_dec(v_val_823_);
lean_dec(v_key_822_);
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 1, v_x_803_);
lean_ctor_set(v___x_825_, 0, v_x_802_);
v___x_831_ = v___x_825_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v_x_802_);
lean_ctor_set(v_reuseFailAlloc_832_, 1, v_x_803_);
v___x_831_ = v_reuseFailAlloc_832_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
v___y_817_ = v___x_831_;
goto v___jp_816_;
}
}
}
}
case 1:
{
lean_object* v_node_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_846_; 
v_node_834_ = lean_ctor_get(v_v_813_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v_v_813_);
if (v_isSharedCheck_846_ == 0)
{
v___x_836_ = v_v_813_;
v_isShared_837_ = v_isSharedCheck_846_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_node_834_);
lean_dec(v_v_813_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_846_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
size_t v___x_838_; size_t v___x_839_; size_t v___x_840_; size_t v___x_841_; lean_object* v___x_842_; lean_object* v___x_844_; 
v___x_838_ = ((size_t)5ULL);
v___x_839_ = lean_usize_shift_right(v_x_800_, v___x_838_);
v___x_840_ = ((size_t)1ULL);
v___x_841_ = lean_usize_add(v_x_801_, v___x_840_);
v___x_842_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(v_node_834_, v___x_839_, v___x_841_, v_x_802_, v_x_803_);
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 0, v___x_842_);
v___x_844_ = v___x_836_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v___x_842_);
v___x_844_ = v_reuseFailAlloc_845_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
v___y_817_ = v___x_844_;
goto v___jp_816_;
}
}
}
default: 
{
lean_object* v___x_847_; 
v___x_847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_847_, 0, v_x_802_);
lean_ctor_set(v___x_847_, 1, v_x_803_);
v___y_817_ = v___x_847_;
goto v___jp_816_;
}
}
v___jp_816_:
{
lean_object* v___x_818_; lean_object* v___x_820_; 
v___x_818_ = lean_array_fset(v_xs_x27_815_, v_j_807_, v___y_817_);
lean_dec(v_j_807_);
if (v_isShared_812_ == 0)
{
lean_ctor_set(v___x_811_, 0, v___x_818_);
v___x_820_ = v___x_811_;
goto v_reusejp_819_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v___x_818_);
v___x_820_ = v_reuseFailAlloc_821_;
goto v_reusejp_819_;
}
v_reusejp_819_:
{
return v___x_820_;
}
}
}
}
}
else
{
lean_object* v_ks_850_; lean_object* v_vs_851_; lean_object* v___x_853_; uint8_t v_isShared_854_; uint8_t v_isSharedCheck_871_; 
v_ks_850_ = lean_ctor_get(v_x_799_, 0);
v_vs_851_ = lean_ctor_get(v_x_799_, 1);
v_isSharedCheck_871_ = !lean_is_exclusive(v_x_799_);
if (v_isSharedCheck_871_ == 0)
{
v___x_853_ = v_x_799_;
v_isShared_854_ = v_isSharedCheck_871_;
goto v_resetjp_852_;
}
else
{
lean_inc(v_vs_851_);
lean_inc(v_ks_850_);
lean_dec(v_x_799_);
v___x_853_ = lean_box(0);
v_isShared_854_ = v_isSharedCheck_871_;
goto v_resetjp_852_;
}
v_resetjp_852_:
{
lean_object* v___x_856_; 
if (v_isShared_854_ == 0)
{
v___x_856_ = v___x_853_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_ks_850_);
lean_ctor_set(v_reuseFailAlloc_870_, 1, v_vs_851_);
v___x_856_ = v_reuseFailAlloc_870_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
lean_object* v_newNode_857_; uint8_t v___y_859_; size_t v___x_865_; uint8_t v___x_866_; 
v_newNode_857_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4___redArg(v___x_856_, v_x_802_, v_x_803_);
v___x_865_ = ((size_t)7ULL);
v___x_866_ = lean_usize_dec_le(v___x_865_, v_x_801_);
if (v___x_866_ == 0)
{
lean_object* v___x_867_; lean_object* v___x_868_; uint8_t v___x_869_; 
v___x_867_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_857_);
v___x_868_ = lean_unsigned_to_nat(4u);
v___x_869_ = lean_nat_dec_lt(v___x_867_, v___x_868_);
lean_dec(v___x_867_);
v___y_859_ = v___x_869_;
goto v___jp_858_;
}
else
{
v___y_859_ = v___x_866_;
goto v___jp_858_;
}
v___jp_858_:
{
if (v___y_859_ == 0)
{
lean_object* v_ks_860_; lean_object* v_vs_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v_ks_860_ = lean_ctor_get(v_newNode_857_, 0);
lean_inc_ref(v_ks_860_);
v_vs_861_ = lean_ctor_get(v_newNode_857_, 1);
lean_inc_ref(v_vs_861_);
lean_dec_ref(v_newNode_857_);
v___x_862_ = lean_unsigned_to_nat(0u);
v___x_863_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_864_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg(v_x_801_, v_ks_860_, v_vs_861_, v___x_862_, v___x_863_);
lean_dec_ref(v_vs_861_);
lean_dec_ref(v_ks_860_);
return v___x_864_;
}
else
{
return v_newNode_857_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg(size_t v_depth_872_, lean_object* v_keys_873_, lean_object* v_vals_874_, lean_object* v_i_875_, lean_object* v_entries_876_){
_start:
{
lean_object* v___x_877_; uint8_t v___x_878_; 
v___x_877_ = lean_array_get_size(v_keys_873_);
v___x_878_ = lean_nat_dec_lt(v_i_875_, v___x_877_);
if (v___x_878_ == 0)
{
lean_dec(v_i_875_);
return v_entries_876_;
}
else
{
lean_object* v_k_879_; lean_object* v_v_880_; uint64_t v___x_881_; size_t v_h_882_; size_t v___x_883_; lean_object* v___x_884_; size_t v___x_885_; size_t v___x_886_; size_t v___x_887_; size_t v_h_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v_k_879_ = lean_array_fget_borrowed(v_keys_873_, v_i_875_);
v_v_880_ = lean_array_fget_borrowed(v_vals_874_, v_i_875_);
v___x_881_ = l_Lean_instHashableMVarId_hash(v_k_879_);
v_h_882_ = lean_uint64_to_usize(v___x_881_);
v___x_883_ = ((size_t)5ULL);
v___x_884_ = lean_unsigned_to_nat(1u);
v___x_885_ = ((size_t)1ULL);
v___x_886_ = lean_usize_sub(v_depth_872_, v___x_885_);
v___x_887_ = lean_usize_mul(v___x_883_, v___x_886_);
v_h_888_ = lean_usize_shift_right(v_h_882_, v___x_887_);
v___x_889_ = lean_nat_add(v_i_875_, v___x_884_);
lean_dec(v_i_875_);
lean_inc(v_v_880_);
lean_inc(v_k_879_);
v___x_890_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(v_entries_876_, v_h_888_, v_depth_872_, v_k_879_, v_v_880_);
v_i_875_ = v___x_889_;
v_entries_876_ = v___x_890_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_depth_892_, lean_object* v_keys_893_, lean_object* v_vals_894_, lean_object* v_i_895_, lean_object* v_entries_896_){
_start:
{
size_t v_depth_boxed_897_; lean_object* v_res_898_; 
v_depth_boxed_897_ = lean_unbox_usize(v_depth_892_);
lean_dec(v_depth_892_);
v_res_898_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_boxed_897_, v_keys_893_, v_vals_894_, v_i_895_, v_entries_896_);
lean_dec_ref(v_vals_894_);
lean_dec_ref(v_keys_893_);
return v_res_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_x_899_, lean_object* v_x_900_, lean_object* v_x_901_, lean_object* v_x_902_, lean_object* v_x_903_){
_start:
{
size_t v_x_33518__boxed_904_; size_t v_x_33519__boxed_905_; lean_object* v_res_906_; 
v_x_33518__boxed_904_ = lean_unbox_usize(v_x_900_);
lean_dec(v_x_900_);
v_x_33519__boxed_905_ = lean_unbox_usize(v_x_901_);
lean_dec(v_x_901_);
v_res_906_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(v_x_899_, v_x_33518__boxed_904_, v_x_33519__boxed_905_, v_x_902_, v_x_903_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1___redArg(lean_object* v_x_907_, lean_object* v_x_908_, lean_object* v_x_909_){
_start:
{
uint64_t v___x_910_; size_t v___x_911_; size_t v___x_912_; lean_object* v___x_913_; 
v___x_910_ = l_Lean_instHashableMVarId_hash(v_x_908_);
v___x_911_ = lean_uint64_to_usize(v___x_910_);
v___x_912_ = ((size_t)1ULL);
v___x_913_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(v_x_907_, v___x_911_, v___x_912_, v_x_908_, v_x_909_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(lean_object* v_mvarId_914_, lean_object* v_val_915_, lean_object* v___y_916_){
_start:
{
lean_object* v___x_918_; lean_object* v_mctx_919_; lean_object* v_cache_920_; lean_object* v_zetaDeltaFVarIds_921_; lean_object* v_postponed_922_; lean_object* v_diag_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_951_; 
v___x_918_ = lean_st_ref_take(v___y_916_);
v_mctx_919_ = lean_ctor_get(v___x_918_, 0);
v_cache_920_ = lean_ctor_get(v___x_918_, 1);
v_zetaDeltaFVarIds_921_ = lean_ctor_get(v___x_918_, 2);
v_postponed_922_ = lean_ctor_get(v___x_918_, 3);
v_diag_923_ = lean_ctor_get(v___x_918_, 4);
v_isSharedCheck_951_ = !lean_is_exclusive(v___x_918_);
if (v_isSharedCheck_951_ == 0)
{
v___x_925_ = v___x_918_;
v_isShared_926_ = v_isSharedCheck_951_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_diag_923_);
lean_inc(v_postponed_922_);
lean_inc(v_zetaDeltaFVarIds_921_);
lean_inc(v_cache_920_);
lean_inc(v_mctx_919_);
lean_dec(v___x_918_);
v___x_925_ = lean_box(0);
v_isShared_926_ = v_isSharedCheck_951_;
goto v_resetjp_924_;
}
v_resetjp_924_:
{
lean_object* v_depth_927_; lean_object* v_levelAssignDepth_928_; lean_object* v_lmvarCounter_929_; lean_object* v_mvarCounter_930_; lean_object* v_lDecls_931_; lean_object* v_decls_932_; lean_object* v_userNames_933_; lean_object* v_lAssignment_934_; lean_object* v_eAssignment_935_; lean_object* v_dAssignment_936_; lean_object* v___x_938_; uint8_t v_isShared_939_; uint8_t v_isSharedCheck_950_; 
v_depth_927_ = lean_ctor_get(v_mctx_919_, 0);
v_levelAssignDepth_928_ = lean_ctor_get(v_mctx_919_, 1);
v_lmvarCounter_929_ = lean_ctor_get(v_mctx_919_, 2);
v_mvarCounter_930_ = lean_ctor_get(v_mctx_919_, 3);
v_lDecls_931_ = lean_ctor_get(v_mctx_919_, 4);
v_decls_932_ = lean_ctor_get(v_mctx_919_, 5);
v_userNames_933_ = lean_ctor_get(v_mctx_919_, 6);
v_lAssignment_934_ = lean_ctor_get(v_mctx_919_, 7);
v_eAssignment_935_ = lean_ctor_get(v_mctx_919_, 8);
v_dAssignment_936_ = lean_ctor_get(v_mctx_919_, 9);
v_isSharedCheck_950_ = !lean_is_exclusive(v_mctx_919_);
if (v_isSharedCheck_950_ == 0)
{
v___x_938_ = v_mctx_919_;
v_isShared_939_ = v_isSharedCheck_950_;
goto v_resetjp_937_;
}
else
{
lean_inc(v_dAssignment_936_);
lean_inc(v_eAssignment_935_);
lean_inc(v_lAssignment_934_);
lean_inc(v_userNames_933_);
lean_inc(v_decls_932_);
lean_inc(v_lDecls_931_);
lean_inc(v_mvarCounter_930_);
lean_inc(v_lmvarCounter_929_);
lean_inc(v_levelAssignDepth_928_);
lean_inc(v_depth_927_);
lean_dec(v_mctx_919_);
v___x_938_ = lean_box(0);
v_isShared_939_ = v_isSharedCheck_950_;
goto v_resetjp_937_;
}
v_resetjp_937_:
{
lean_object* v___x_940_; lean_object* v___x_942_; 
v___x_940_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1___redArg(v_eAssignment_935_, v_mvarId_914_, v_val_915_);
if (v_isShared_939_ == 0)
{
lean_ctor_set(v___x_938_, 8, v___x_940_);
v___x_942_ = v___x_938_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_949_, 0, v_depth_927_);
lean_ctor_set(v_reuseFailAlloc_949_, 1, v_levelAssignDepth_928_);
lean_ctor_set(v_reuseFailAlloc_949_, 2, v_lmvarCounter_929_);
lean_ctor_set(v_reuseFailAlloc_949_, 3, v_mvarCounter_930_);
lean_ctor_set(v_reuseFailAlloc_949_, 4, v_lDecls_931_);
lean_ctor_set(v_reuseFailAlloc_949_, 5, v_decls_932_);
lean_ctor_set(v_reuseFailAlloc_949_, 6, v_userNames_933_);
lean_ctor_set(v_reuseFailAlloc_949_, 7, v_lAssignment_934_);
lean_ctor_set(v_reuseFailAlloc_949_, 8, v___x_940_);
lean_ctor_set(v_reuseFailAlloc_949_, 9, v_dAssignment_936_);
v___x_942_ = v_reuseFailAlloc_949_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
lean_object* v___x_944_; 
if (v_isShared_926_ == 0)
{
lean_ctor_set(v___x_925_, 0, v___x_942_);
v___x_944_ = v___x_925_;
goto v_reusejp_943_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v___x_942_);
lean_ctor_set(v_reuseFailAlloc_948_, 1, v_cache_920_);
lean_ctor_set(v_reuseFailAlloc_948_, 2, v_zetaDeltaFVarIds_921_);
lean_ctor_set(v_reuseFailAlloc_948_, 3, v_postponed_922_);
lean_ctor_set(v_reuseFailAlloc_948_, 4, v_diag_923_);
v___x_944_ = v_reuseFailAlloc_948_;
goto v_reusejp_943_;
}
v_reusejp_943_:
{
lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_945_ = lean_st_ref_set(v___y_916_, v___x_944_);
v___x_946_ = lean_box(0);
v___x_947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_947_, 0, v___x_946_);
return v___x_947_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg___boxed(lean_object* v_mvarId_952_, lean_object* v_val_953_, lean_object* v___y_954_, lean_object* v___y_955_){
_start:
{
lean_object* v_res_956_; 
v_res_956_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(v_mvarId_952_, v_val_953_, v___y_954_);
lean_dec(v___y_954_);
return v_res_956_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1(void){
_start:
{
lean_object* v___x_958_; lean_object* v___x_959_; 
v___x_958_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__0));
v___x_959_ = l_Lean_stringToMessageData(v___x_958_);
return v___x_959_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1(void){
_start:
{
lean_object* v___x_961_; lean_object* v___x_962_; 
v___x_961_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__0));
v___x_962_ = l_Lean_stringToMessageData(v___x_961_);
return v___x_962_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3(void){
_start:
{
lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_964_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__2));
v___x_965_ = l_Lean_stringToMessageData(v___x_964_);
return v___x_965_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5(void){
_start:
{
lean_object* v___x_967_; lean_object* v___x_968_; 
v___x_967_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__4));
v___x_968_ = l_Lean_stringToMessageData(v___x_967_);
return v___x_968_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7(void){
_start:
{
lean_object* v___x_970_; lean_object* v___x_971_; 
v___x_970_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__6));
v___x_971_ = l_Lean_stringToMessageData(v___x_970_);
return v___x_971_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9(void){
_start:
{
lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_973_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__8));
v___x_974_ = l_Lean_stringToMessageData(v___x_973_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0(lean_object* v_goal_975_, lean_object* v_e_976_, lean_object* v_b_977_, lean_object* v_xs_978_, lean_object* v_tgt_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_){
_start:
{
lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_991_; lean_object* v___y_992_; lean_object* v___y_993_; lean_object* v___y_994_; lean_object* v___y_995_; lean_object* v___x_1061_; 
lean_inc(v___y_986_);
lean_inc_ref(v___y_985_);
lean_inc(v___y_984_);
lean_inc_ref(v___y_983_);
v___x_1061_ = lean_whnf(v_tgt_979_, v___y_983_, v___y_984_, v___y_985_, v___y_986_);
if (lean_obj_tag(v___x_1061_) == 0)
{
lean_object* v_a_1062_; lean_object* v___x_1063_; 
v_a_1062_ = lean_ctor_get(v___x_1061_, 0);
lean_inc(v_a_1062_);
lean_dec_ref_known(v___x_1061_, 1);
v___x_1063_ = l_Lean_Expr_getAppFn(v_a_1062_);
if (lean_obj_tag(v___x_1063_) == 4)
{
lean_object* v_declName_1064_; lean_object* v_us_1065_; lean_object* v_casts_1066_; lean_object* v___x_1067_; 
v_declName_1064_ = lean_ctor_get(v___x_1063_, 0);
lean_inc(v_declName_1064_);
v_us_1065_ = lean_ctor_get(v___x_1063_, 1);
lean_inc(v_us_1065_);
lean_dec_ref_known(v___x_1063_, 2);
v_casts_1066_ = lean_ctor_get(v_b_977_, 1);
v___x_1067_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_casts_1066_, v_declName_1064_);
if (lean_obj_tag(v___x_1067_) == 1)
{
lean_object* v_val_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1168_; 
v_val_1068_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1168_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1168_ == 0)
{
v___x_1070_ = v___x_1067_;
v_isShared_1071_ = v_isSharedCheck_1168_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_val_1068_);
lean_dec(v___x_1067_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1168_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v_options_1072_; lean_object* v_snd_1073_; lean_object* v___x_1075_; uint8_t v_isShared_1076_; uint8_t v_isSharedCheck_1166_; 
v_options_1072_ = lean_ctor_get(v___y_985_, 2);
v_snd_1073_ = lean_ctor_get(v_val_1068_, 1);
v_isSharedCheck_1166_ = !lean_is_exclusive(v_val_1068_);
if (v_isSharedCheck_1166_ == 0)
{
lean_object* v_unused_1167_; 
v_unused_1167_ = lean_ctor_get(v_val_1068_, 0);
lean_dec(v_unused_1167_);
v___x_1075_ = v_val_1068_;
v_isShared_1076_ = v_isSharedCheck_1166_;
goto v_resetjp_1074_;
}
else
{
lean_inc(v_snd_1073_);
lean_dec(v_val_1068_);
v___x_1075_ = lean_box(0);
v_isShared_1076_ = v_isSharedCheck_1166_;
goto v_resetjp_1074_;
}
v_resetjp_1074_:
{
lean_object* v_inheritedTraceOptions_1077_; uint8_t v_hasTrace_1078_; lean_object* v_nargs_1079_; lean_object* v___x_1080_; lean_object* v_dummy_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___y_1088_; lean_object* v___y_1089_; lean_object* v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; lean_object* v___y_1093_; lean_object* v___y_1094_; 
v_inheritedTraceOptions_1077_ = lean_ctor_get(v___y_985_, 13);
v_hasTrace_1078_ = lean_ctor_get_uint8(v_options_1072_, sizeof(void*)*1);
v_nargs_1079_ = l_Lean_Expr_getAppNumArgs(v_a_1062_);
v___x_1080_ = l_Lean_Expr_const___override(v_snd_1073_, v_us_1065_);
v_dummy_1081_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0);
lean_inc(v_nargs_1079_);
v___x_1082_ = lean_mk_array(v_nargs_1079_, v_dummy_1081_);
v___x_1083_ = lean_unsigned_to_nat(1u);
v___x_1084_ = lean_nat_sub(v_nargs_1079_, v___x_1083_);
lean_dec(v_nargs_1079_);
v___x_1085_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_1062_, v___x_1082_, v___x_1084_);
v___x_1086_ = l_Lean_mkAppN(v___x_1080_, v___x_1085_);
lean_dec_ref(v___x_1085_);
if (v_hasTrace_1078_ == 0)
{
lean_dec(v_declName_1064_);
v___y_1088_ = v___y_980_;
v___y_1089_ = v___y_981_;
v___y_1090_ = v___y_982_;
v___y_1091_ = v___y_983_;
v___y_1092_ = v___y_984_;
v___y_1093_ = v___y_985_;
v___y_1094_ = v___y_986_;
goto v___jp_1087_;
}
else
{
lean_object* v___x_1154_; lean_object* v___x_1155_; uint8_t v___x_1156_; 
v___x_1154_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2));
v___x_1155_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5);
v___x_1156_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1077_, v_options_1072_, v___x_1155_);
if (v___x_1156_ == 0)
{
lean_dec(v_declName_1064_);
v___y_1088_ = v___y_980_;
v___y_1089_ = v___y_981_;
v___y_1090_ = v___y_982_;
v___y_1091_ = v___y_983_;
v___y_1092_ = v___y_984_;
v___y_1093_ = v___y_985_;
v___y_1094_ = v___y_986_;
goto v___jp_1087_;
}
else
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1157_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__9);
lean_inc_ref(v___x_1086_);
v___x_1158_ = l_Lean_MessageData_ofExpr(v___x_1086_);
v___x_1159_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1157_);
lean_ctor_set(v___x_1159_, 1, v___x_1158_);
v___x_1160_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__9);
v___x_1161_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1159_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
v___x_1162_ = 0;
v___x_1163_ = l_Lean_MessageData_ofConstName(v_declName_1064_, v___x_1162_);
v___x_1164_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1164_, 0, v___x_1161_);
lean_ctor_set(v___x_1164_, 1, v___x_1163_);
v___x_1165_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v___x_1154_, v___x_1164_, v___y_983_, v___y_984_, v___y_985_, v___y_986_);
if (lean_obj_tag(v___x_1165_) == 0)
{
lean_dec_ref_known(v___x_1165_, 1);
v___y_1088_ = v___y_980_;
v___y_1089_ = v___y_981_;
v___y_1090_ = v___y_982_;
v___y_1091_ = v___y_983_;
v___y_1092_ = v___y_984_;
v___y_1093_ = v___y_985_;
v___y_1094_ = v___y_986_;
goto v___jp_1087_;
}
else
{
lean_dec_ref(v___x_1086_);
lean_del_object(v___x_1075_);
lean_del_object(v___x_1070_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
return v___x_1165_;
}
}
}
v___jp_1087_:
{
lean_object* v___x_1095_; 
lean_inc(v___y_1094_);
lean_inc_ref(v___y_1093_);
lean_inc(v___y_1092_);
lean_inc_ref(v___y_1091_);
lean_inc_ref(v___x_1086_);
v___x_1095_ = lean_infer_type(v___x_1086_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
lean_inc(v_a_1096_);
lean_dec_ref_known(v___x_1095_, 1);
if (lean_obj_tag(v_a_1096_) == 7)
{
lean_object* v_binderType_1097_; lean_object* v___x_1099_; 
lean_del_object(v___x_1075_);
v_binderType_1097_ = lean_ctor_get(v_a_1096_, 1);
lean_inc_ref(v_binderType_1097_);
lean_dec_ref_known(v_a_1096_, 3);
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 0, v_binderType_1097_);
v___x_1099_ = v___x_1070_;
goto v_reusejp_1098_;
}
else
{
lean_object* v_reuseFailAlloc_1139_; 
v_reuseFailAlloc_1139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1139_, 0, v_binderType_1097_);
v___x_1099_ = v_reuseFailAlloc_1139_;
goto v_reusejp_1098_;
}
v_reusejp_1098_:
{
uint8_t v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; 
v___x_1100_ = 0;
v___x_1101_ = lean_box(0);
v___x_1102_ = l_Lean_Meta_mkFreshExprMVar(v___x_1099_, v___x_1100_, v___x_1101_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
if (lean_obj_tag(v___x_1102_) == 0)
{
lean_object* v_a_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; 
v_a_1103_ = lean_ctor_get(v___x_1102_, 0);
lean_inc(v_a_1103_);
lean_dec_ref_known(v___x_1102_, 1);
lean_inc_ref(v_xs_978_);
v___x_1104_ = l_Lean_Expr_beta(v_e_976_, v_xs_978_);
v___x_1105_ = l_Lean_Expr_mvarId_x21(v_a_1103_);
v___x_1106_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go(v_b_977_, v___x_1104_, v___x_1105_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
if (lean_obj_tag(v___x_1106_) == 0)
{
lean_object* v___x_1107_; uint8_t v___x_1108_; uint8_t v___x_1109_; uint8_t v___x_1110_; lean_object* v___x_1111_; 
lean_dec_ref_known(v___x_1106_, 1);
v___x_1107_ = l_Lean_Expr_app___override(v___x_1086_, v_a_1103_);
v___x_1108_ = 0;
v___x_1109_ = 1;
v___x_1110_ = 1;
v___x_1111_ = l_Lean_Meta_mkLambdaFVars(v_xs_978_, v___x_1107_, v___x_1108_, v___x_1109_, v___x_1108_, v___x_1109_, v___x_1110_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
lean_dec_ref(v_xs_978_);
if (lean_obj_tag(v___x_1111_) == 0)
{
lean_object* v_a_1112_; lean_object* v___x_1113_; 
v_a_1112_ = lean_ctor_get(v___x_1111_, 0);
lean_inc(v_a_1112_);
lean_dec_ref_known(v___x_1111_, 1);
v___x_1113_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(v_goal_975_, v_a_1112_, v___y_1092_);
if (lean_obj_tag(v___x_1113_) == 0)
{
lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1121_; 
v_isSharedCheck_1121_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1121_ == 0)
{
lean_object* v_unused_1122_; 
v_unused_1122_ = lean_ctor_get(v___x_1113_, 0);
lean_dec(v_unused_1122_);
v___x_1115_ = v___x_1113_;
v_isShared_1116_ = v_isSharedCheck_1121_;
goto v_resetjp_1114_;
}
else
{
lean_dec(v___x_1113_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1121_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1117_; lean_object* v___x_1119_; 
v___x_1117_ = lean_box(0);
if (v_isShared_1116_ == 0)
{
lean_ctor_set(v___x_1115_, 0, v___x_1117_);
v___x_1119_ = v___x_1115_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v___x_1117_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
return v___x_1119_;
}
}
}
else
{
return v___x_1113_;
}
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
lean_dec(v_goal_975_);
v_a_1123_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1111_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1111_);
v___x_1125_ = lean_box(0);
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
v_resetjp_1124_:
{
lean_object* v___x_1128_; 
if (v_isShared_1126_ == 0)
{
v___x_1128_ = v___x_1125_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_a_1123_);
v___x_1128_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1127_;
}
v_reusejp_1127_:
{
return v___x_1128_;
}
}
}
}
else
{
lean_dec(v_a_1103_);
lean_dec_ref(v___x_1086_);
lean_dec_ref(v_xs_978_);
lean_dec(v_goal_975_);
return v___x_1106_;
}
}
else
{
lean_object* v_a_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1138_; 
lean_dec_ref(v___x_1086_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1131_ = lean_ctor_get(v___x_1102_, 0);
v_isSharedCheck_1138_ = !lean_is_exclusive(v___x_1102_);
if (v_isSharedCheck_1138_ == 0)
{
v___x_1133_ = v___x_1102_;
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_a_1131_);
lean_dec(v___x_1102_);
v___x_1133_ = lean_box(0);
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
v_resetjp_1132_:
{
lean_object* v___x_1136_; 
if (v_isShared_1134_ == 0)
{
v___x_1136_ = v___x_1133_;
goto v_reusejp_1135_;
}
else
{
lean_object* v_reuseFailAlloc_1137_; 
v_reuseFailAlloc_1137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1137_, 0, v_a_1131_);
v___x_1136_ = v_reuseFailAlloc_1137_;
goto v_reusejp_1135_;
}
v_reusejp_1135_:
{
return v___x_1136_;
}
}
}
}
}
else
{
lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1143_; 
lean_dec(v_a_1096_);
lean_del_object(v___x_1070_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
v___x_1140_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__7);
v___x_1141_ = l_Lean_MessageData_ofExpr(v___x_1086_);
if (v_isShared_1076_ == 0)
{
lean_ctor_set_tag(v___x_1075_, 7);
lean_ctor_set(v___x_1075_, 1, v___x_1141_);
lean_ctor_set(v___x_1075_, 0, v___x_1140_);
v___x_1143_ = v___x_1075_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v___x_1140_);
lean_ctor_set(v_reuseFailAlloc_1145_, 1, v___x_1141_);
v___x_1143_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
lean_object* v___x_1144_; 
v___x_1144_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v___x_1143_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
if (lean_obj_tag(v___x_1144_) == 0)
{
lean_dec_ref_known(v___x_1144_, 1);
v___y_989_ = v___y_1088_;
v___y_990_ = v___y_1089_;
v___y_991_ = v___y_1090_;
v___y_992_ = v___y_1091_;
v___y_993_ = v___y_1092_;
v___y_994_ = v___y_1093_;
v___y_995_ = v___y_1094_;
goto v___jp_988_;
}
else
{
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
return v___x_1144_;
}
}
}
}
else
{
lean_object* v_a_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1153_; 
lean_dec_ref(v___x_1086_);
lean_del_object(v___x_1075_);
lean_del_object(v___x_1070_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1146_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1153_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1153_ == 0)
{
v___x_1148_ = v___x_1095_;
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_a_1146_);
lean_dec(v___x_1095_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v___x_1151_; 
if (v_isShared_1149_ == 0)
{
v___x_1151_ = v___x_1148_;
goto v_reusejp_1150_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v_a_1146_);
v___x_1151_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1150_;
}
v_reusejp_1150_:
{
return v___x_1151_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_1067_);
lean_dec(v_us_1065_);
lean_dec(v_declName_1064_);
lean_dec(v_a_1062_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
v___y_989_ = v___y_980_;
v___y_990_ = v___y_981_;
v___y_991_ = v___y_982_;
v___y_992_ = v___y_983_;
v___y_993_ = v___y_984_;
v___y_994_ = v___y_985_;
v___y_995_ = v___y_986_;
goto v___jp_988_;
}
}
else
{
lean_dec_ref(v___x_1063_);
lean_dec(v_a_1062_);
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
v___y_989_ = v___y_980_;
v___y_990_ = v___y_981_;
v___y_991_ = v___y_982_;
v___y_992_ = v___y_983_;
v___y_993_ = v___y_984_;
v___y_994_ = v___y_985_;
v___y_995_ = v___y_986_;
goto v___jp_988_;
}
}
else
{
lean_object* v_a_1169_; lean_object* v___x_1171_; uint8_t v_isShared_1172_; uint8_t v_isSharedCheck_1176_; 
lean_dec_ref(v_xs_978_);
lean_dec_ref(v_b_977_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1169_ = lean_ctor_get(v___x_1061_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1061_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1171_ = v___x_1061_;
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
else
{
lean_inc(v_a_1169_);
lean_dec(v___x_1061_);
v___x_1171_ = lean_box(0);
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
v_resetjp_1170_:
{
lean_object* v___x_1174_; 
if (v_isShared_1172_ == 0)
{
v___x_1174_ = v___x_1171_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v_a_1169_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
v___jp_988_:
{
lean_object* v___x_996_; 
lean_inc(v_goal_975_);
v___x_996_ = l_Lean_MVarId_getType(v_goal_975_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_996_) == 0)
{
lean_object* v_a_997_; lean_object* v___x_998_; 
v_a_997_ = lean_ctor_get(v___x_996_, 0);
lean_inc(v_a_997_);
lean_dec_ref_known(v___x_996_, 1);
lean_inc(v___y_995_);
lean_inc_ref(v___y_994_);
lean_inc(v___y_993_);
lean_inc_ref(v___y_992_);
lean_inc_ref(v_e_976_);
v___x_998_ = lean_infer_type(v_e_976_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_998_) == 0)
{
lean_object* v_a_999_; lean_object* v___x_1000_; 
v_a_999_ = lean_ctor_get(v___x_998_, 0);
lean_inc(v_a_999_);
lean_dec_ref_known(v___x_998_, 1);
v___x_1000_ = l_Lean_Meta_isExprDefEq(v_a_997_, v_a_999_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_1000_) == 0)
{
lean_object* v_a_1001_; uint8_t v___x_1002_; 
v_a_1001_ = lean_ctor_get(v___x_1000_, 0);
lean_inc(v_a_1001_);
lean_dec_ref_known(v___x_1000_, 1);
v___x_1002_ = lean_unbox(v_a_1001_);
lean_dec(v_a_1001_);
if (v___x_1002_ == 0)
{
lean_object* v___x_1003_; 
lean_inc(v___y_995_);
lean_inc_ref(v___y_994_);
lean_inc(v___y_993_);
lean_inc_ref(v___y_992_);
lean_inc_ref(v_e_976_);
v___x_1003_ = lean_infer_type(v_e_976_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_1003_) == 0)
{
lean_object* v_a_1004_; lean_object* v___x_1005_; 
v_a_1004_ = lean_ctor_get(v___x_1003_, 0);
lean_inc(v_a_1004_);
lean_dec_ref_known(v___x_1003_, 1);
lean_inc(v_goal_975_);
v___x_1005_ = l_Lean_MVarId_getType(v_goal_975_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_1005_) == 0)
{
lean_object* v_a_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v_a_1006_ = lean_ctor_get(v___x_1005_, 0);
lean_inc(v_a_1006_);
lean_dec_ref_known(v___x_1005_, 1);
lean_inc_ref(v_e_976_);
v___x_1007_ = l_Lean_MessageData_ofExpr(v_e_976_);
v___x_1008_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__1);
v___x_1009_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1009_, 0, v___x_1007_);
lean_ctor_set(v___x_1009_, 1, v___x_1008_);
v___x_1010_ = l_Lean_MessageData_ofExpr(v_a_1004_);
v___x_1011_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1011_, 0, v___x_1009_);
lean_ctor_set(v___x_1011_, 1, v___x_1010_);
v___x_1012_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__3);
v___x_1013_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1013_, 0, v___x_1011_);
lean_ctor_set(v___x_1013_, 1, v___x_1012_);
v___x_1014_ = l_Lean_MessageData_ofExpr(v_a_1006_);
v___x_1015_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1013_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
v___x_1016_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___closed__5);
v___x_1017_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1017_, 0, v___x_1015_);
lean_ctor_set(v___x_1017_, 1, v___x_1016_);
v___x_1018_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v___x_1017_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
if (lean_obj_tag(v___x_1018_) == 0)
{
lean_object* v___x_1019_; 
lean_dec_ref_known(v___x_1018_, 1);
v___x_1019_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(v_goal_975_, v_e_976_, v___y_993_);
return v___x_1019_;
}
else
{
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
return v___x_1018_;
}
}
else
{
lean_object* v_a_1020_; lean_object* v___x_1022_; uint8_t v_isShared_1023_; uint8_t v_isSharedCheck_1027_; 
lean_dec(v_a_1004_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1020_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1027_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1027_ == 0)
{
v___x_1022_ = v___x_1005_;
v_isShared_1023_ = v_isSharedCheck_1027_;
goto v_resetjp_1021_;
}
else
{
lean_inc(v_a_1020_);
lean_dec(v___x_1005_);
v___x_1022_ = lean_box(0);
v_isShared_1023_ = v_isSharedCheck_1027_;
goto v_resetjp_1021_;
}
v_resetjp_1021_:
{
lean_object* v___x_1025_; 
if (v_isShared_1023_ == 0)
{
v___x_1025_ = v___x_1022_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v_a_1020_);
v___x_1025_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
return v___x_1025_;
}
}
}
}
else
{
lean_object* v_a_1028_; lean_object* v___x_1030_; uint8_t v_isShared_1031_; uint8_t v_isSharedCheck_1035_; 
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1028_ = lean_ctor_get(v___x_1003_, 0);
v_isSharedCheck_1035_ = !lean_is_exclusive(v___x_1003_);
if (v_isSharedCheck_1035_ == 0)
{
v___x_1030_ = v___x_1003_;
v_isShared_1031_ = v_isSharedCheck_1035_;
goto v_resetjp_1029_;
}
else
{
lean_inc(v_a_1028_);
lean_dec(v___x_1003_);
v___x_1030_ = lean_box(0);
v_isShared_1031_ = v_isSharedCheck_1035_;
goto v_resetjp_1029_;
}
v_resetjp_1029_:
{
lean_object* v___x_1033_; 
if (v_isShared_1031_ == 0)
{
v___x_1033_ = v___x_1030_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v_a_1028_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
return v___x_1033_;
}
}
}
}
else
{
lean_object* v___x_1036_; 
v___x_1036_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(v_goal_975_, v_e_976_, v___y_993_);
return v___x_1036_;
}
}
else
{
lean_object* v_a_1037_; lean_object* v___x_1039_; uint8_t v_isShared_1040_; uint8_t v_isSharedCheck_1044_; 
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1037_ = lean_ctor_get(v___x_1000_, 0);
v_isSharedCheck_1044_ = !lean_is_exclusive(v___x_1000_);
if (v_isSharedCheck_1044_ == 0)
{
v___x_1039_ = v___x_1000_;
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
else
{
lean_inc(v_a_1037_);
lean_dec(v___x_1000_);
v___x_1039_ = lean_box(0);
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
v_resetjp_1038_:
{
lean_object* v___x_1042_; 
if (v_isShared_1040_ == 0)
{
v___x_1042_ = v___x_1039_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v_a_1037_);
v___x_1042_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
return v___x_1042_;
}
}
}
}
else
{
lean_object* v_a_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1052_; 
lean_dec(v_a_997_);
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1045_ = lean_ctor_get(v___x_998_, 0);
v_isSharedCheck_1052_ = !lean_is_exclusive(v___x_998_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1047_ = v___x_998_;
v_isShared_1048_ = v_isSharedCheck_1052_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_a_1045_);
lean_dec(v___x_998_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1052_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v___x_1050_; 
if (v_isShared_1048_ == 0)
{
v___x_1050_ = v___x_1047_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v_a_1045_);
v___x_1050_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
return v___x_1050_;
}
}
}
}
else
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1060_; 
lean_dec_ref(v_e_976_);
lean_dec(v_goal_975_);
v_a_1053_ = lean_ctor_get(v___x_996_, 0);
v_isSharedCheck_1060_ = !lean_is_exclusive(v___x_996_);
if (v_isSharedCheck_1060_ == 0)
{
v___x_1055_ = v___x_996_;
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_996_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v___x_1058_; 
if (v_isShared_1056_ == 0)
{
v___x_1058_ = v___x_1055_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v_a_1053_);
v___x_1058_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
return v___x_1058_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___boxed(lean_object* v_goal_1177_, lean_object* v_e_1178_, lean_object* v_b_1179_, lean_object* v_xs_1180_, lean_object* v_tgt_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0(v_goal_1177_, v_e_1178_, v_b_1179_, v_xs_1180_, v_tgt_1181_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
lean_dec(v___y_1186_);
lean_dec_ref(v___y_1185_);
lean_dec(v___y_1184_);
lean_dec_ref(v___y_1183_);
lean_dec(v___y_1182_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go(lean_object* v_b_1191_, lean_object* v_e_1192_, lean_object* v_goal_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_, lean_object* v_a_1199_, lean_object* v_a_1200_){
_start:
{
lean_object* v_goal_1203_; lean_object* v___y_1204_; lean_object* v___y_1205_; lean_object* v___y_1206_; lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___x_1224_; 
lean_inc(v_goal_1193_);
v___x_1224_ = l_Lean_MVarId_getType(v_goal_1193_, v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_);
if (lean_obj_tag(v___x_1224_) == 0)
{
lean_object* v_a_1225_; lean_object* v___x_1226_; 
v_a_1225_ = lean_ctor_get(v___x_1224_, 0);
lean_inc(v_a_1225_);
lean_dec_ref_known(v___x_1224_, 1);
lean_inc(v_a_1200_);
lean_inc_ref(v_a_1199_);
lean_inc(v_a_1198_);
lean_inc_ref(v_a_1197_);
lean_inc(v_a_1196_);
lean_inc_ref(v_a_1195_);
lean_inc(v_a_1194_);
v___x_1226_ = lean_simp(v_a_1225_, v_a_1194_, v_a_1195_, v_a_1196_, v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v_a_1227_; lean_object* v_proof_x3f_1228_; 
v_a_1227_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_a_1227_);
lean_dec_ref_known(v___x_1226_, 1);
v_proof_x3f_1228_ = lean_ctor_get(v_a_1227_, 1);
lean_inc(v_proof_x3f_1228_);
if (lean_obj_tag(v_proof_x3f_1228_) == 1)
{
lean_object* v_expr_1229_; lean_object* v_val_1230_; lean_object* v___y_1232_; lean_object* v___y_1233_; lean_object* v___y_1234_; lean_object* v___y_1235_; lean_object* v___y_1236_; lean_object* v___y_1237_; lean_object* v___y_1238_; lean_object* v_options_1249_; uint8_t v_hasTrace_1250_; 
v_expr_1229_ = lean_ctor_get(v_a_1227_, 0);
lean_inc_ref(v_expr_1229_);
lean_dec(v_a_1227_);
v_val_1230_ = lean_ctor_get(v_proof_x3f_1228_, 0);
lean_inc(v_val_1230_);
lean_dec_ref_known(v_proof_x3f_1228_, 1);
v_options_1249_ = lean_ctor_get(v_a_1199_, 2);
v_hasTrace_1250_ = lean_ctor_get_uint8(v_options_1249_, sizeof(void*)*1);
if (v_hasTrace_1250_ == 0)
{
v___y_1232_ = v_a_1194_;
v___y_1233_ = v_a_1195_;
v___y_1234_ = v_a_1196_;
v___y_1235_ = v_a_1197_;
v___y_1236_ = v_a_1198_;
v___y_1237_ = v_a_1199_;
v___y_1238_ = v_a_1200_;
goto v___jp_1231_;
}
else
{
lean_object* v_inheritedTraceOptions_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; uint8_t v___x_1254_; 
v_inheritedTraceOptions_1251_ = lean_ctor_get(v_a_1199_, 13);
v___x_1252_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__2));
v___x_1253_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__5);
v___x_1254_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1251_, v_options_1249_, v___x_1253_);
if (v___x_1254_ == 0)
{
v___y_1232_ = v_a_1194_;
v___y_1233_ = v_a_1195_;
v___y_1234_ = v_a_1196_;
v___y_1235_ = v_a_1197_;
v___y_1236_ = v_a_1198_;
v___y_1237_ = v_a_1199_;
v___y_1238_ = v_a_1200_;
goto v___jp_1231_;
}
else
{
lean_object* v___x_1255_; 
lean_inc(v_goal_1193_);
v___x_1255_ = l_Lean_MVarId_getType(v_goal_1193_, v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_);
if (lean_obj_tag(v___x_1255_) == 0)
{
lean_object* v_a_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; 
v_a_1256_ = lean_ctor_get(v___x_1255_, 0);
lean_inc(v_a_1256_);
lean_dec_ref_known(v___x_1255_, 1);
v___x_1257_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___closed__1);
v___x_1258_ = l_Lean_MessageData_ofExpr(v_a_1256_);
v___x_1259_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1259_, 0, v___x_1257_);
lean_ctor_set(v___x_1259_, 1, v___x_1258_);
v___x_1260_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__16);
v___x_1261_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1261_, 0, v___x_1259_);
lean_ctor_set(v___x_1261_, 1, v___x_1260_);
lean_inc_ref(v_expr_1229_);
v___x_1262_ = l_Lean_MessageData_ofExpr(v_expr_1229_);
v___x_1263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1261_);
lean_ctor_set(v___x_1263_, 1, v___x_1262_);
v___x_1264_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts_spec__0___redArg(v___x_1252_, v___x_1263_, v_a_1197_, v_a_1198_, v_a_1199_, v_a_1200_);
if (lean_obj_tag(v___x_1264_) == 0)
{
lean_dec_ref_known(v___x_1264_, 1);
v___y_1232_ = v_a_1194_;
v___y_1233_ = v_a_1195_;
v___y_1234_ = v_a_1196_;
v___y_1235_ = v_a_1197_;
v___y_1236_ = v_a_1198_;
v___y_1237_ = v_a_1199_;
v___y_1238_ = v_a_1200_;
goto v___jp_1231_;
}
else
{
lean_dec(v_val_1230_);
lean_dec_ref(v_expr_1229_);
lean_dec(v_goal_1193_);
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
return v___x_1264_;
}
}
else
{
lean_object* v_a_1265_; lean_object* v___x_1267_; uint8_t v_isShared_1268_; uint8_t v_isSharedCheck_1272_; 
lean_dec(v_val_1230_);
lean_dec_ref(v_expr_1229_);
lean_dec(v_goal_1193_);
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
v_a_1265_ = lean_ctor_get(v___x_1255_, 0);
v_isSharedCheck_1272_ = !lean_is_exclusive(v___x_1255_);
if (v_isSharedCheck_1272_ == 0)
{
v___x_1267_ = v___x_1255_;
v_isShared_1268_ = v_isSharedCheck_1272_;
goto v_resetjp_1266_;
}
else
{
lean_inc(v_a_1265_);
lean_dec(v___x_1255_);
v___x_1267_ = lean_box(0);
v_isShared_1268_ = v_isSharedCheck_1272_;
goto v_resetjp_1266_;
}
v_resetjp_1266_:
{
lean_object* v___x_1270_; 
if (v_isShared_1268_ == 0)
{
v___x_1270_ = v___x_1267_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1271_; 
v_reuseFailAlloc_1271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1271_, 0, v_a_1265_);
v___x_1270_ = v_reuseFailAlloc_1271_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
return v___x_1270_;
}
}
}
}
}
v___jp_1231_:
{
lean_object* v___x_1239_; 
v___x_1239_ = l_Lean_MVarId_replaceTargetEq(v_goal_1193_, v_expr_1229_, v_val_1230_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
if (lean_obj_tag(v___x_1239_) == 0)
{
lean_object* v_a_1240_; 
v_a_1240_ = lean_ctor_get(v___x_1239_, 0);
lean_inc(v_a_1240_);
lean_dec_ref_known(v___x_1239_, 1);
v_goal_1203_ = v_a_1240_;
v___y_1204_ = v___y_1232_;
v___y_1205_ = v___y_1233_;
v___y_1206_ = v___y_1234_;
v___y_1207_ = v___y_1235_;
v___y_1208_ = v___y_1236_;
v___y_1209_ = v___y_1237_;
v___y_1210_ = v___y_1238_;
goto v___jp_1202_;
}
else
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
v_a_1241_ = lean_ctor_get(v___x_1239_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1239_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1239_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v___x_1239_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_a_1241_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
}
}
else
{
lean_dec(v_proof_x3f_1228_);
lean_dec(v_a_1227_);
v_goal_1203_ = v_goal_1193_;
v___y_1204_ = v_a_1194_;
v___y_1205_ = v_a_1195_;
v___y_1206_ = v_a_1196_;
v___y_1207_ = v_a_1197_;
v___y_1208_ = v_a_1198_;
v___y_1209_ = v_a_1199_;
v___y_1210_ = v_a_1200_;
goto v___jp_1202_;
}
}
else
{
lean_object* v_a_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1280_; 
lean_dec(v_goal_1193_);
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
v_a_1273_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1280_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1280_ == 0)
{
v___x_1275_ = v___x_1226_;
v_isShared_1276_ = v_isSharedCheck_1280_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_a_1273_);
lean_dec(v___x_1226_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1280_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
lean_object* v___x_1278_; 
if (v_isShared_1276_ == 0)
{
v___x_1278_ = v___x_1275_;
goto v_reusejp_1277_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v_a_1273_);
v___x_1278_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1277_;
}
v_reusejp_1277_:
{
return v___x_1278_;
}
}
}
}
else
{
lean_object* v_a_1281_; lean_object* v___x_1283_; uint8_t v_isShared_1284_; uint8_t v_isSharedCheck_1288_; 
lean_dec(v_goal_1193_);
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
v_a_1281_ = lean_ctor_get(v___x_1224_, 0);
v_isSharedCheck_1288_ = !lean_is_exclusive(v___x_1224_);
if (v_isSharedCheck_1288_ == 0)
{
v___x_1283_ = v___x_1224_;
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
else
{
lean_inc(v_a_1281_);
lean_dec(v___x_1224_);
v___x_1283_ = lean_box(0);
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
v_resetjp_1282_:
{
lean_object* v___x_1286_; 
if (v_isShared_1284_ == 0)
{
v___x_1286_ = v___x_1283_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v_a_1281_);
v___x_1286_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
return v___x_1286_;
}
}
}
v___jp_1202_:
{
lean_object* v___x_1211_; 
lean_inc(v_goal_1203_);
v___x_1211_ = l_Lean_MVarId_getType(v_goal_1203_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_);
if (lean_obj_tag(v___x_1211_) == 0)
{
lean_object* v_a_1212_; lean_object* v___f_1213_; uint8_t v___x_1214_; lean_object* v___x_1215_; 
v_a_1212_ = lean_ctor_get(v___x_1211_, 0);
lean_inc(v_a_1212_);
lean_dec_ref_known(v___x_1211_, 1);
v___f_1213_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___lam__0___boxed), 13, 3);
lean_closure_set(v___f_1213_, 0, v_goal_1203_);
lean_closure_set(v___f_1213_, 1, v_e_1192_);
lean_closure_set(v___f_1213_, 2, v_b_1191_);
v___x_1214_ = 0;
v___x_1215_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__2___redArg(v_a_1212_, v___f_1213_, v___x_1214_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_);
return v___x_1215_;
}
else
{
lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
lean_dec(v_goal_1203_);
lean_dec_ref(v_e_1192_);
lean_dec_ref(v_b_1191_);
v_a_1216_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1218_ = v___x_1211_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1211_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1221_; 
if (v_isShared_1219_ == 0)
{
v___x_1221_ = v___x_1218_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v_a_1216_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go___boxed(lean_object* v_b_1289_, lean_object* v_e_1290_, lean_object* v_goal_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_){
_start:
{
lean_object* v_res_1300_; 
v_res_1300_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go(v_b_1289_, v_e_1290_, v_goal_1291_, v_a_1292_, v_a_1293_, v_a_1294_, v_a_1295_, v_a_1296_, v_a_1297_, v_a_1298_);
lean_dec(v_a_1298_);
lean_dec_ref(v_a_1297_);
lean_dec(v_a_1296_);
lean_dec_ref(v_a_1295_);
lean_dec(v_a_1294_);
lean_dec_ref(v_a_1293_);
lean_dec(v_a_1292_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0(lean_object* v_00_u03b1_1301_, lean_object* v_msg_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_){
_start:
{
lean_object* v___x_1311_; 
v___x_1311_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v_msg_1302_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_);
return v___x_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___boxed(lean_object* v_00_u03b1_1312_, lean_object* v_msg_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
lean_object* v_res_1322_; 
v_res_1322_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0(v_00_u03b1_1312_, v_msg_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
lean_dec(v___y_1314_);
return v_res_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1(lean_object* v_mvarId_1323_, lean_object* v_val_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_){
_start:
{
lean_object* v___x_1333_; 
v___x_1333_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___redArg(v_mvarId_1323_, v_val_1324_, v___y_1329_);
return v___x_1333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1___boxed(lean_object* v_mvarId_1334_, lean_object* v_val_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_){
_start:
{
lean_object* v_res_1344_; 
v_res_1344_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1(v_mvarId_1334_, v_val_1335_, v___y_1336_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_);
lean_dec(v___y_1342_);
lean_dec_ref(v___y_1341_);
lean_dec(v___y_1340_);
lean_dec_ref(v___y_1339_);
lean_dec(v___y_1338_);
lean_dec_ref(v___y_1337_);
lean_dec(v___y_1336_);
return v_res_1344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1(lean_object* v_00_u03b2_1345_, lean_object* v_x_1346_, lean_object* v_x_1347_, lean_object* v_x_1348_){
_start:
{
lean_object* v___x_1349_; 
v___x_1349_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1___redArg(v_x_1346_, v_x_1347_, v_x_1348_);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_1350_, lean_object* v_x_1351_, size_t v_x_1352_, size_t v_x_1353_, lean_object* v_x_1354_, lean_object* v_x_1355_){
_start:
{
lean_object* v___x_1356_; 
v___x_1356_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___redArg(v_x_1351_, v_x_1352_, v_x_1353_, v_x_1354_, v_x_1355_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_1357_, lean_object* v_x_1358_, lean_object* v_x_1359_, lean_object* v_x_1360_, lean_object* v_x_1361_, lean_object* v_x_1362_){
_start:
{
size_t v_x_34445__boxed_1363_; size_t v_x_34446__boxed_1364_; lean_object* v_res_1365_; 
v_x_34445__boxed_1363_ = lean_unbox_usize(v_x_1359_);
lean_dec(v_x_1359_);
v_x_34446__boxed_1364_ = lean_unbox_usize(v_x_1360_);
lean_dec(v_x_1360_);
v_res_1365_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3(v_00_u03b2_1357_, v_x_1358_, v_x_34445__boxed_1363_, v_x_34446__boxed_1364_, v_x_1361_, v_x_1362_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4(lean_object* v_00_u03b2_1366_, lean_object* v_n_1367_, lean_object* v_k_1368_, lean_object* v_v_1369_){
_start:
{
lean_object* v___x_1370_; 
v___x_1370_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4___redArg(v_n_1367_, v_k_1368_, v_v_1369_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_1371_, size_t v_depth_1372_, lean_object* v_keys_1373_, lean_object* v_vals_1374_, lean_object* v_heq_1375_, lean_object* v_i_1376_, lean_object* v_entries_1377_){
_start:
{
lean_object* v___x_1378_; 
v___x_1378_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_1372_, v_keys_1373_, v_vals_1374_, v_i_1376_, v_entries_1377_);
return v___x_1378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1379_, lean_object* v_depth_1380_, lean_object* v_keys_1381_, lean_object* v_vals_1382_, lean_object* v_heq_1383_, lean_object* v_i_1384_, lean_object* v_entries_1385_){
_start:
{
size_t v_depth_boxed_1386_; lean_object* v_res_1387_; 
v_depth_boxed_1386_ = lean_unbox_usize(v_depth_1380_);
lean_dec(v_depth_1380_);
v_res_1387_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__5(v_00_u03b2_1379_, v_depth_boxed_1386_, v_keys_1381_, v_vals_1382_, v_heq_1383_, v_i_1384_, v_entries_1385_);
lean_dec_ref(v_vals_1382_);
lean_dec_ref(v_keys_1381_);
return v_res_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_1388_, lean_object* v_x_1389_, lean_object* v_x_1390_, lean_object* v_x_1391_, lean_object* v_x_1392_){
_start:
{
lean_object* v___x_1393_; 
v___x_1393_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_x_1389_, v_x_1390_, v_x_1391_, v_x_1392_);
return v___x_1393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg(lean_object* v_e_1394_, lean_object* v___y_1395_){
_start:
{
uint8_t v___x_1397_; 
v___x_1397_ = l_Lean_Expr_hasMVar(v_e_1394_);
if (v___x_1397_ == 0)
{
lean_object* v___x_1398_; 
v___x_1398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1398_, 0, v_e_1394_);
return v___x_1398_;
}
else
{
lean_object* v___x_1399_; lean_object* v_mctx_1400_; lean_object* v___x_1401_; lean_object* v_fst_1402_; lean_object* v_snd_1403_; lean_object* v___x_1404_; lean_object* v_cache_1405_; lean_object* v_zetaDeltaFVarIds_1406_; lean_object* v_postponed_1407_; lean_object* v_diag_1408_; lean_object* v___x_1410_; uint8_t v_isShared_1411_; uint8_t v_isSharedCheck_1417_; 
v___x_1399_ = lean_st_ref_get(v___y_1395_);
v_mctx_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc_ref(v_mctx_1400_);
lean_dec(v___x_1399_);
v___x_1401_ = l_Lean_instantiateMVarsCore(v_mctx_1400_, v_e_1394_);
v_fst_1402_ = lean_ctor_get(v___x_1401_, 0);
lean_inc(v_fst_1402_);
v_snd_1403_ = lean_ctor_get(v___x_1401_, 1);
lean_inc(v_snd_1403_);
lean_dec_ref(v___x_1401_);
v___x_1404_ = lean_st_ref_take(v___y_1395_);
v_cache_1405_ = lean_ctor_get(v___x_1404_, 1);
v_zetaDeltaFVarIds_1406_ = lean_ctor_get(v___x_1404_, 2);
v_postponed_1407_ = lean_ctor_get(v___x_1404_, 3);
v_diag_1408_ = lean_ctor_get(v___x_1404_, 4);
v_isSharedCheck_1417_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1417_ == 0)
{
lean_object* v_unused_1418_; 
v_unused_1418_ = lean_ctor_get(v___x_1404_, 0);
lean_dec(v_unused_1418_);
v___x_1410_ = v___x_1404_;
v_isShared_1411_ = v_isSharedCheck_1417_;
goto v_resetjp_1409_;
}
else
{
lean_inc(v_diag_1408_);
lean_inc(v_postponed_1407_);
lean_inc(v_zetaDeltaFVarIds_1406_);
lean_inc(v_cache_1405_);
lean_dec(v___x_1404_);
v___x_1410_ = lean_box(0);
v_isShared_1411_ = v_isSharedCheck_1417_;
goto v_resetjp_1409_;
}
v_resetjp_1409_:
{
lean_object* v___x_1413_; 
if (v_isShared_1411_ == 0)
{
lean_ctor_set(v___x_1410_, 0, v_snd_1403_);
v___x_1413_ = v___x_1410_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1416_; 
v_reuseFailAlloc_1416_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1416_, 0, v_snd_1403_);
lean_ctor_set(v_reuseFailAlloc_1416_, 1, v_cache_1405_);
lean_ctor_set(v_reuseFailAlloc_1416_, 2, v_zetaDeltaFVarIds_1406_);
lean_ctor_set(v_reuseFailAlloc_1416_, 3, v_postponed_1407_);
lean_ctor_set(v_reuseFailAlloc_1416_, 4, v_diag_1408_);
v___x_1413_ = v_reuseFailAlloc_1416_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
lean_object* v___x_1414_; lean_object* v___x_1415_; 
v___x_1414_ = lean_st_ref_set(v___y_1395_, v___x_1413_);
v___x_1415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1415_, 0, v_fst_1402_);
return v___x_1415_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg___boxed(lean_object* v_e_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_){
_start:
{
lean_object* v_res_1422_; 
v_res_1422_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg(v_e_1419_, v___y_1420_);
lean_dec(v___y_1420_);
return v_res_1422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0(lean_object* v_e_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_){
_start:
{
lean_object* v___x_1432_; 
v___x_1432_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg(v_e_1423_, v___y_1428_);
return v___x_1432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___boxed(lean_object* v_e_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_){
_start:
{
lean_object* v_res_1442_; 
v_res_1442_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0(v_e_1433_, v___y_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
lean_dec(v___y_1440_);
lean_dec_ref(v___y_1439_);
lean_dec(v___y_1438_);
lean_dec_ref(v___y_1437_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec(v___y_1434_);
return v_res_1442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts(lean_object* v_b_1443_, lean_object* v_e_1444_, lean_object* v_expectedType_1445_, lean_object* v_a_1446_, lean_object* v_a_1447_, lean_object* v_a_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_){
_start:
{
lean_object* v___x_1454_; uint8_t v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; 
v___x_1454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1454_, 0, v_expectedType_1445_);
v___x_1455_ = 0;
v___x_1456_ = lean_box(0);
v___x_1457_ = l_Lean_Meta_mkFreshExprMVar(v___x_1454_, v___x_1455_, v___x_1456_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_);
if (lean_obj_tag(v___x_1457_) == 0)
{
lean_object* v_a_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; 
v_a_1458_ = lean_ctor_get(v___x_1457_, 0);
lean_inc(v_a_1458_);
lean_dec_ref_known(v___x_1457_, 1);
v___x_1459_ = l_Lean_Expr_mvarId_x21(v_a_1458_);
v___x_1460_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go(v_b_1443_, v_e_1444_, v___x_1459_, v_a_1446_, v_a_1447_, v_a_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_);
if (lean_obj_tag(v___x_1460_) == 0)
{
lean_object* v___x_1461_; 
lean_dec_ref_known(v___x_1460_, 1);
v___x_1461_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_spec__0___redArg(v_a_1458_, v_a_1450_);
return v___x_1461_;
}
else
{
lean_object* v_a_1462_; lean_object* v___x_1464_; uint8_t v_isShared_1465_; uint8_t v_isSharedCheck_1469_; 
lean_dec(v_a_1458_);
v_a_1462_ = lean_ctor_get(v___x_1460_, 0);
v_isSharedCheck_1469_ = !lean_is_exclusive(v___x_1460_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1464_ = v___x_1460_;
v_isShared_1465_ = v_isSharedCheck_1469_;
goto v_resetjp_1463_;
}
else
{
lean_inc(v_a_1462_);
lean_dec(v___x_1460_);
v___x_1464_ = lean_box(0);
v_isShared_1465_ = v_isSharedCheck_1469_;
goto v_resetjp_1463_;
}
v_resetjp_1463_:
{
lean_object* v___x_1467_; 
if (v_isShared_1465_ == 0)
{
v___x_1467_ = v___x_1464_;
goto v_reusejp_1466_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v_a_1462_);
v___x_1467_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1466_;
}
v_reusejp_1466_:
{
return v___x_1467_;
}
}
}
}
else
{
lean_dec_ref(v_e_1444_);
lean_dec_ref(v_b_1443_);
return v___x_1457_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts___boxed(lean_object* v_b_1470_, lean_object* v_e_1471_, lean_object* v_expectedType_1472_, lean_object* v_a_1473_, lean_object* v_a_1474_, lean_object* v_a_1475_, lean_object* v_a_1476_, lean_object* v_a_1477_, lean_object* v_a_1478_, lean_object* v_a_1479_, lean_object* v_a_1480_){
_start:
{
lean_object* v_res_1481_; 
v_res_1481_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts(v_b_1470_, v_e_1471_, v_expectedType_1472_, v_a_1473_, v_a_1474_, v_a_1475_, v_a_1476_, v_a_1477_, v_a_1478_, v_a_1479_);
lean_dec(v_a_1479_);
lean_dec_ref(v_a_1478_);
lean_dec(v_a_1477_);
lean_dec_ref(v_a_1476_);
lean_dec(v_a_1475_);
lean_dec_ref(v_a_1474_);
lean_dec(v_a_1473_);
return v_res_1481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast(lean_object* v_b_1482_, lean_object* v_e_1483_, lean_object* v_expectedType_1484_, lean_object* v_a_1485_, lean_object* v_a_1486_, lean_object* v_a_1487_, lean_object* v_a_1488_, lean_object* v_a_1489_, lean_object* v_a_1490_, lean_object* v_a_1491_){
_start:
{
lean_object* v___x_1493_; 
lean_inc(v_a_1491_);
lean_inc_ref(v_a_1490_);
lean_inc(v_a_1489_);
lean_inc_ref(v_a_1488_);
lean_inc_ref(v_e_1483_);
v___x_1493_ = lean_infer_type(v_e_1483_, v_a_1488_, v_a_1489_, v_a_1490_, v_a_1491_);
if (lean_obj_tag(v___x_1493_) == 0)
{
lean_object* v_a_1494_; lean_object* v___x_1495_; 
v_a_1494_ = lean_ctor_get(v___x_1493_, 0);
lean_inc(v_a_1494_);
lean_dec_ref_known(v___x_1493_, 1);
lean_inc_ref(v_expectedType_1484_);
v___x_1495_ = l_Lean_Meta_isExprDefEq(v_a_1494_, v_expectedType_1484_, v_a_1488_, v_a_1489_, v_a_1490_, v_a_1491_);
if (lean_obj_tag(v___x_1495_) == 0)
{
lean_object* v_a_1496_; lean_object* v___x_1498_; uint8_t v_isShared_1499_; uint8_t v_isSharedCheck_1507_; 
v_a_1496_ = lean_ctor_get(v___x_1495_, 0);
v_isSharedCheck_1507_ = !lean_is_exclusive(v___x_1495_);
if (v_isSharedCheck_1507_ == 0)
{
v___x_1498_ = v___x_1495_;
v_isShared_1499_ = v_isSharedCheck_1507_;
goto v_resetjp_1497_;
}
else
{
lean_inc(v_a_1496_);
lean_dec(v___x_1495_);
v___x_1498_ = lean_box(0);
v_isShared_1499_ = v_isSharedCheck_1507_;
goto v_resetjp_1497_;
}
v_resetjp_1497_:
{
uint8_t v___x_1500_; 
v___x_1500_ = lean_unbox(v_a_1496_);
lean_dec(v_a_1496_);
if (v___x_1500_ == 0)
{
lean_object* v___x_1501_; 
lean_del_object(v___x_1498_);
v___x_1501_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts(v_b_1482_, v_e_1483_, v_a_1485_, v_a_1486_, v_a_1487_, v_a_1488_, v_a_1489_, v_a_1490_, v_a_1491_);
if (lean_obj_tag(v___x_1501_) == 0)
{
lean_object* v_a_1502_; lean_object* v___x_1503_; 
v_a_1502_ = lean_ctor_get(v___x_1501_, 0);
lean_inc(v_a_1502_);
lean_dec_ref_known(v___x_1501_, 1);
v___x_1503_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts(v_b_1482_, v_a_1502_, v_expectedType_1484_, v_a_1485_, v_a_1486_, v_a_1487_, v_a_1488_, v_a_1489_, v_a_1490_, v_a_1491_);
return v___x_1503_;
}
else
{
lean_dec_ref(v_expectedType_1484_);
lean_dec_ref(v_b_1482_);
return v___x_1501_;
}
}
else
{
lean_object* v___x_1505_; 
lean_dec_ref(v_expectedType_1484_);
lean_dec_ref(v_b_1482_);
if (v_isShared_1499_ == 0)
{
lean_ctor_set(v___x_1498_, 0, v_e_1483_);
v___x_1505_ = v___x_1498_;
goto v_reusejp_1504_;
}
else
{
lean_object* v_reuseFailAlloc_1506_; 
v_reuseFailAlloc_1506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1506_, 0, v_e_1483_);
v___x_1505_ = v_reuseFailAlloc_1506_;
goto v_reusejp_1504_;
}
v_reusejp_1504_:
{
return v___x_1505_;
}
}
}
}
else
{
lean_object* v_a_1508_; lean_object* v___x_1510_; uint8_t v_isShared_1511_; uint8_t v_isSharedCheck_1515_; 
lean_dec_ref(v_expectedType_1484_);
lean_dec_ref(v_e_1483_);
lean_dec_ref(v_b_1482_);
v_a_1508_ = lean_ctor_get(v___x_1495_, 0);
v_isSharedCheck_1515_ = !lean_is_exclusive(v___x_1495_);
if (v_isSharedCheck_1515_ == 0)
{
v___x_1510_ = v___x_1495_;
v_isShared_1511_ = v_isSharedCheck_1515_;
goto v_resetjp_1509_;
}
else
{
lean_inc(v_a_1508_);
lean_dec(v___x_1495_);
v___x_1510_ = lean_box(0);
v_isShared_1511_ = v_isSharedCheck_1515_;
goto v_resetjp_1509_;
}
v_resetjp_1509_:
{
lean_object* v___x_1513_; 
if (v_isShared_1511_ == 0)
{
v___x_1513_ = v___x_1510_;
goto v_reusejp_1512_;
}
else
{
lean_object* v_reuseFailAlloc_1514_; 
v_reuseFailAlloc_1514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1514_, 0, v_a_1508_);
v___x_1513_ = v_reuseFailAlloc_1514_;
goto v_reusejp_1512_;
}
v_reusejp_1512_:
{
return v___x_1513_;
}
}
}
}
else
{
lean_dec_ref(v_expectedType_1484_);
lean_dec_ref(v_e_1483_);
lean_dec_ref(v_b_1482_);
return v___x_1493_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast___boxed(lean_object* v_b_1516_, lean_object* v_e_1517_, lean_object* v_expectedType_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_, lean_object* v_a_1524_, lean_object* v_a_1525_, lean_object* v_a_1526_){
_start:
{
lean_object* v_res_1527_; 
v_res_1527_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast(v_b_1516_, v_e_1517_, v_expectedType_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_, v_a_1525_);
lean_dec(v_a_1525_);
lean_dec_ref(v_a_1524_);
lean_dec(v_a_1523_);
lean_dec_ref(v_a_1522_);
lean_dec(v_a_1521_);
lean_dec_ref(v_a_1520_);
lean_dec(v_a_1519_);
return v_res_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast(lean_object* v_b_1528_, lean_object* v_f_1529_, lean_object* v_a_1530_, lean_object* v_a_1531_, lean_object* v_a_1532_, lean_object* v_a_1533_, lean_object* v_a_1534_, lean_object* v_a_1535_, lean_object* v_a_1536_, lean_object* v_a_1537_){
_start:
{
lean_object* v___x_1539_; 
lean_inc_ref(v_a_1530_);
lean_inc_ref(v_f_1529_);
v___x_1539_ = l_Lean_Meta_checkApp(v_f_1529_, v_a_1530_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
if (lean_obj_tag(v___x_1539_) == 0)
{
lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1547_; 
lean_dec_ref(v_b_1528_);
v_isSharedCheck_1547_ = !lean_is_exclusive(v___x_1539_);
if (v_isSharedCheck_1547_ == 0)
{
lean_object* v_unused_1548_; 
v_unused_1548_ = lean_ctor_get(v___x_1539_, 0);
lean_dec(v_unused_1548_);
v___x_1541_ = v___x_1539_;
v_isShared_1542_ = v_isSharedCheck_1547_;
goto v_resetjp_1540_;
}
else
{
lean_dec(v___x_1539_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1547_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1543_; lean_object* v___x_1545_; 
v___x_1543_ = l_Lean_Expr_app___override(v_f_1529_, v_a_1530_);
if (v_isShared_1542_ == 0)
{
lean_ctor_set(v___x_1541_, 0, v___x_1543_);
v___x_1545_ = v___x_1541_;
goto v_reusejp_1544_;
}
else
{
lean_object* v_reuseFailAlloc_1546_; 
v_reuseFailAlloc_1546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1546_, 0, v___x_1543_);
v___x_1545_ = v_reuseFailAlloc_1546_;
goto v_reusejp_1544_;
}
v_reusejp_1544_:
{
return v___x_1545_;
}
}
}
else
{
lean_object* v_a_1549_; lean_object* v___x_1551_; uint8_t v_isShared_1552_; uint8_t v_isSharedCheck_1578_; 
v_a_1549_ = lean_ctor_get(v___x_1539_, 0);
v_isSharedCheck_1578_ = !lean_is_exclusive(v___x_1539_);
if (v_isSharedCheck_1578_ == 0)
{
v___x_1551_ = v___x_1539_;
v_isShared_1552_ = v_isSharedCheck_1578_;
goto v_resetjp_1550_;
}
else
{
lean_inc(v_a_1549_);
lean_dec(v___x_1539_);
v___x_1551_ = lean_box(0);
v_isShared_1552_ = v_isSharedCheck_1578_;
goto v_resetjp_1550_;
}
v_resetjp_1550_:
{
uint8_t v___y_1554_; uint8_t v___x_1576_; 
v___x_1576_ = l_Lean_Exception_isInterrupt(v_a_1549_);
if (v___x_1576_ == 0)
{
uint8_t v___x_1577_; 
lean_inc(v_a_1549_);
v___x_1577_ = l_Lean_Exception_isRuntime(v_a_1549_);
v___y_1554_ = v___x_1577_;
goto v___jp_1553_;
}
else
{
v___y_1554_ = v___x_1576_;
goto v___jp_1553_;
}
v___jp_1553_:
{
if (v___y_1554_ == 0)
{
lean_object* v___x_1555_; 
lean_del_object(v___x_1551_);
lean_dec(v_a_1549_);
v___x_1555_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts(v_b_1528_, v_f_1529_, v_a_1531_, v_a_1532_, v_a_1533_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
if (lean_obj_tag(v___x_1555_) == 0)
{
lean_object* v_a_1556_; lean_object* v___x_1557_; 
v_a_1556_ = lean_ctor_get(v___x_1555_, 0);
lean_inc_n(v_a_1556_, 2);
lean_dec_ref_known(v___x_1555_, 1);
lean_inc(v_a_1537_);
lean_inc_ref(v_a_1536_);
lean_inc(v_a_1535_);
lean_inc_ref(v_a_1534_);
v___x_1557_ = lean_infer_type(v_a_1556_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
if (lean_obj_tag(v___x_1557_) == 0)
{
lean_object* v_a_1558_; lean_object* v___x_1559_; 
v_a_1558_ = lean_ctor_get(v___x_1557_, 0);
lean_inc(v_a_1558_);
lean_dec_ref_known(v___x_1557_, 1);
lean_inc(v_a_1537_);
lean_inc_ref(v_a_1536_);
lean_inc(v_a_1535_);
lean_inc_ref(v_a_1534_);
v___x_1559_ = lean_whnf(v_a_1558_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
if (lean_obj_tag(v___x_1559_) == 0)
{
lean_object* v_a_1560_; 
v_a_1560_ = lean_ctor_get(v___x_1559_, 0);
lean_inc(v_a_1560_);
lean_dec_ref_known(v___x_1559_, 1);
if (lean_obj_tag(v_a_1560_) == 7)
{
lean_object* v_binderType_1561_; lean_object* v___x_1562_; 
v_binderType_1561_ = lean_ctor_get(v_a_1560_, 1);
lean_inc_ref(v_binderType_1561_);
lean_dec_ref_known(v_a_1560_, 3);
v___x_1562_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast(v_b_1528_, v_a_1530_, v_binderType_1561_, v_a_1531_, v_a_1532_, v_a_1533_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
if (lean_obj_tag(v___x_1562_) == 0)
{
lean_object* v_a_1563_; lean_object* v___x_1565_; uint8_t v_isShared_1566_; uint8_t v_isSharedCheck_1571_; 
v_a_1563_ = lean_ctor_get(v___x_1562_, 0);
v_isSharedCheck_1571_ = !lean_is_exclusive(v___x_1562_);
if (v_isSharedCheck_1571_ == 0)
{
v___x_1565_ = v___x_1562_;
v_isShared_1566_ = v_isSharedCheck_1571_;
goto v_resetjp_1564_;
}
else
{
lean_inc(v_a_1563_);
lean_dec(v___x_1562_);
v___x_1565_ = lean_box(0);
v_isShared_1566_ = v_isSharedCheck_1571_;
goto v_resetjp_1564_;
}
v_resetjp_1564_:
{
lean_object* v___x_1567_; lean_object* v___x_1569_; 
v___x_1567_ = l_Lean_Expr_app___override(v_a_1556_, v_a_1563_);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 0, v___x_1567_);
v___x_1569_ = v___x_1565_;
goto v_reusejp_1568_;
}
else
{
lean_object* v_reuseFailAlloc_1570_; 
v_reuseFailAlloc_1570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1570_, 0, v___x_1567_);
v___x_1569_ = v_reuseFailAlloc_1570_;
goto v_reusejp_1568_;
}
v_reusejp_1568_:
{
return v___x_1569_;
}
}
}
else
{
lean_dec(v_a_1556_);
return v___x_1562_;
}
}
else
{
lean_object* v___x_1572_; 
lean_dec(v_a_1560_);
lean_dec_ref(v_a_1530_);
lean_dec_ref(v_b_1528_);
v___x_1572_ = l_Lean_Meta_throwFunctionExpected___redArg(v_a_1556_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
return v___x_1572_;
}
}
else
{
lean_dec(v_a_1556_);
lean_dec_ref(v_a_1530_);
lean_dec_ref(v_b_1528_);
return v___x_1559_;
}
}
else
{
lean_dec(v_a_1556_);
lean_dec_ref(v_a_1530_);
lean_dec_ref(v_b_1528_);
return v___x_1557_;
}
}
else
{
lean_dec_ref(v_a_1530_);
lean_dec_ref(v_b_1528_);
return v___x_1555_;
}
}
else
{
lean_object* v___x_1574_; 
lean_dec_ref(v_a_1530_);
lean_dec_ref(v_f_1529_);
lean_dec_ref(v_b_1528_);
if (v_isShared_1552_ == 0)
{
v___x_1574_ = v___x_1551_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v_a_1549_);
v___x_1574_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
return v___x_1574_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast___boxed(lean_object* v_b_1579_, lean_object* v_f_1580_, lean_object* v_a_1581_, lean_object* v_a_1582_, lean_object* v_a_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_, lean_object* v_a_1586_, lean_object* v_a_1587_, lean_object* v_a_1588_, lean_object* v_a_1589_){
_start:
{
lean_object* v_res_1590_; 
v_res_1590_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast(v_b_1579_, v_f_1580_, v_a_1581_, v_a_1582_, v_a_1583_, v_a_1584_, v_a_1585_, v_a_1586_, v_a_1587_, v_a_1588_);
lean_dec(v_a_1588_);
lean_dec_ref(v_a_1587_);
lean_dec(v_a_1586_);
lean_dec_ref(v_a_1585_);
lean_dec(v_a_1584_);
lean_dec_ref(v_a_1583_);
lean_dec(v_a_1582_);
return v_res_1590_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1592_; lean_object* v___x_1593_; 
v___x_1592_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__0));
v___x_1593_ = l_Lean_stringToMessageData(v___x_1592_);
return v___x_1593_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1595_; lean_object* v___x_1596_; 
v___x_1595_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__2));
v___x_1596_ = l_Lean_stringToMessageData(v___x_1595_);
return v___x_1596_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1598_; lean_object* v___x_1599_; 
v___x_1598_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__4));
v___x_1599_ = l_Lean_stringToMessageData(v___x_1598_);
return v___x_1599_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7(void){
_start:
{
lean_object* v___x_1601_; lean_object* v___x_1602_; 
v___x_1601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__6));
v___x_1602_ = l_Lean_stringToMessageData(v___x_1601_);
return v___x_1602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0(lean_object* v_b_1603_, lean_object* v_e_1604_, lean_object* v_expectedType_1605_, lean_object* v_attr_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_){
_start:
{
lean_object* v___x_1615_; 
lean_inc_ref(v_expectedType_1605_);
lean_inc_ref(v_e_1604_);
v___x_1615_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkCast(v_b_1603_, v_e_1604_, v_expectedType_1605_, v___y_1607_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_);
if (lean_obj_tag(v___x_1615_) == 0)
{
lean_dec(v_attr_1606_);
lean_dec_ref(v_expectedType_1605_);
lean_dec_ref(v_e_1604_);
return v___x_1615_;
}
else
{
lean_object* v_a_1616_; uint8_t v___y_1618_; uint8_t v___x_1635_; 
v_a_1616_ = lean_ctor_get(v___x_1615_, 0);
lean_inc(v_a_1616_);
v___x_1635_ = l_Lean_Exception_isInterrupt(v_a_1616_);
if (v___x_1635_ == 0)
{
uint8_t v___x_1636_; 
lean_inc(v_a_1616_);
v___x_1636_ = l_Lean_Exception_isRuntime(v_a_1616_);
v___y_1618_ = v___x_1636_;
goto v___jp_1617_;
}
else
{
v___y_1618_ = v___x_1635_;
goto v___jp_1617_;
}
v___jp_1617_:
{
if (v___y_1618_ == 0)
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; 
lean_dec_ref_known(v___x_1615_, 1);
v___x_1619_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1);
v___x_1620_ = l_Lean_MessageData_ofName(v_attr_1606_);
v___x_1621_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1621_, 0, v___x_1619_);
lean_ctor_set(v___x_1621_, 1, v___x_1620_);
v___x_1622_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3);
v___x_1623_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1623_, 0, v___x_1621_);
lean_ctor_set(v___x_1623_, 1, v___x_1622_);
v___x_1624_ = l_Lean_MessageData_ofExpr(v_e_1604_);
v___x_1625_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1625_, 0, v___x_1623_);
lean_ctor_set(v___x_1625_, 1, v___x_1624_);
v___x_1626_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__5);
v___x_1627_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1627_, 0, v___x_1625_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = l_Lean_MessageData_ofExpr(v_expectedType_1605_);
v___x_1629_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1629_, 0, v___x_1627_);
lean_ctor_set(v___x_1629_, 1, v___x_1628_);
v___x_1630_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__7);
v___x_1631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1629_);
lean_ctor_set(v___x_1631_, 1, v___x_1630_);
v___x_1632_ = l_Lean_Exception_toMessageData(v_a_1616_);
v___x_1633_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1633_, 0, v___x_1631_);
lean_ctor_set(v___x_1633_, 1, v___x_1632_);
v___x_1634_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v___x_1633_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_);
return v___x_1634_;
}
else
{
lean_dec(v_a_1616_);
lean_dec(v_attr_1606_);
lean_dec_ref(v_expectedType_1605_);
lean_dec_ref(v_e_1604_);
return v___x_1615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___boxed(lean_object* v_b_1637_, lean_object* v_e_1638_, lean_object* v_expectedType_1639_, lean_object* v_attr_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_){
_start:
{
lean_object* v_res_1649_; 
v_res_1649_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0(v_b_1637_, v_e_1638_, v_expectedType_1639_, v_attr_1640_, v___y_1641_, v___y_1642_, v___y_1643_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
lean_dec(v___y_1647_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v___y_1643_);
lean_dec_ref(v___y_1642_);
lean_dec(v___y_1641_);
return v_res_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast(lean_object* v_b_1650_, lean_object* v_e_1651_, lean_object* v_expectedType_1652_, lean_object* v_attr_1653_, lean_object* v_a_1654_, lean_object* v_a_1655_, lean_object* v_a_1656_, lean_object* v_a_1657_){
_start:
{
lean_object* v___f_1659_; lean_object* v___x_1660_; 
lean_inc_ref(v_b_1650_);
v___f_1659_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___boxed), 12, 4);
lean_closure_set(v___f_1659_, 0, v_b_1650_);
lean_closure_set(v___f_1659_, 1, v_e_1651_);
lean_closure_set(v___f_1659_, 2, v_expectedType_1652_);
lean_closure_set(v___f_1659_, 3, v_attr_1653_);
v___x_1660_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(v_b_1650_, v___f_1659_, v_a_1654_, v_a_1655_, v_a_1656_, v_a_1657_);
return v___x_1660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___boxed(lean_object* v_b_1661_, lean_object* v_e_1662_, lean_object* v_expectedType_1663_, lean_object* v_attr_1664_, lean_object* v_a_1665_, lean_object* v_a_1666_, lean_object* v_a_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_){
_start:
{
lean_object* v_res_1670_; 
v_res_1670_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast(v_b_1661_, v_e_1662_, v_expectedType_1663_, v_attr_1664_, v_a_1665_, v_a_1666_, v_a_1667_, v_a_1668_);
lean_dec(v_a_1668_);
lean_dec_ref(v_a_1667_);
lean_dec(v_a_1666_);
lean_dec_ref(v_a_1665_);
return v_res_1670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(lean_object* v_b_1671_, lean_object* v_as_1672_, size_t v_i_1673_, size_t v_stop_1674_, lean_object* v_b_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_){
_start:
{
uint8_t v___x_1684_; 
v___x_1684_ = lean_usize_dec_eq(v_i_1673_, v_stop_1674_);
if (v___x_1684_ == 0)
{
lean_object* v___x_1685_; lean_object* v___x_1686_; 
v___x_1685_ = lean_array_uget_borrowed(v_as_1672_, v_i_1673_);
lean_inc(v___x_1685_);
lean_inc_ref(v_b_1671_);
v___x_1686_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_mkAppWithCast(v_b_1671_, v_b_1675_, v___x_1685_, v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_, v___y_1682_);
if (lean_obj_tag(v___x_1686_) == 0)
{
lean_object* v_a_1687_; size_t v___x_1688_; size_t v___x_1689_; 
v_a_1687_ = lean_ctor_get(v___x_1686_, 0);
lean_inc(v_a_1687_);
lean_dec_ref_known(v___x_1686_, 1);
v___x_1688_ = ((size_t)1ULL);
v___x_1689_ = lean_usize_add(v_i_1673_, v___x_1688_);
v_i_1673_ = v___x_1689_;
v_b_1675_ = v_a_1687_;
goto _start;
}
else
{
lean_dec_ref(v_b_1671_);
return v___x_1686_;
}
}
else
{
lean_object* v___x_1691_; 
lean_dec_ref(v_b_1671_);
v___x_1691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1691_, 0, v_b_1675_);
return v___x_1691_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1___boxed(lean_object* v_b_1692_, lean_object* v_as_1693_, lean_object* v_i_1694_, lean_object* v_stop_1695_, lean_object* v_b_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_){
_start:
{
size_t v_i_boxed_1705_; size_t v_stop_boxed_1706_; lean_object* v_res_1707_; 
v_i_boxed_1705_ = lean_unbox_usize(v_i_1694_);
lean_dec(v_i_1694_);
v_stop_boxed_1706_ = lean_unbox_usize(v_stop_1695_);
lean_dec(v_stop_1695_);
v_res_1707_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(v_b_1692_, v_as_1693_, v_i_boxed_1705_, v_stop_boxed_1706_, v_b_1696_, v___y_1697_, v___y_1698_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
lean_dec(v___y_1699_);
lean_dec_ref(v___y_1698_);
lean_dec(v___y_1697_);
lean_dec_ref(v_as_1693_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__0(lean_object* v_a_1708_, lean_object* v_a_1709_){
_start:
{
if (lean_obj_tag(v_a_1708_) == 0)
{
lean_object* v___x_1710_; 
v___x_1710_ = l_List_reverse___redArg(v_a_1709_);
return v___x_1710_;
}
else
{
lean_object* v_head_1711_; lean_object* v_tail_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1721_; 
v_head_1711_ = lean_ctor_get(v_a_1708_, 0);
v_tail_1712_ = lean_ctor_get(v_a_1708_, 1);
v_isSharedCheck_1721_ = !lean_is_exclusive(v_a_1708_);
if (v_isSharedCheck_1721_ == 0)
{
v___x_1714_ = v_a_1708_;
v_isShared_1715_ = v_isSharedCheck_1721_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_tail_1712_);
lean_inc(v_head_1711_);
lean_dec(v_a_1708_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1721_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v___x_1716_; lean_object* v___x_1718_; 
v___x_1716_ = l_Lean_MessageData_ofExpr(v_head_1711_);
if (v_isShared_1715_ == 0)
{
lean_ctor_set(v___x_1714_, 1, v_a_1709_);
lean_ctor_set(v___x_1714_, 0, v___x_1716_);
v___x_1718_ = v___x_1714_;
goto v_reusejp_1717_;
}
else
{
lean_object* v_reuseFailAlloc_1720_; 
v_reuseFailAlloc_1720_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1720_, 0, v___x_1716_);
lean_ctor_set(v_reuseFailAlloc_1720_, 1, v_a_1709_);
v___x_1718_ = v_reuseFailAlloc_1720_;
goto v_reusejp_1717_;
}
v_reusejp_1717_:
{
v_a_1708_ = v_tail_1712_;
v_a_1709_ = v___x_1718_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; 
v___x_1723_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__0));
v___x_1724_ = l_Lean_stringToMessageData(v___x_1723_);
return v___x_1724_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1726_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__2));
v___x_1727_ = l_Lean_stringToMessageData(v___x_1726_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2(lean_object* v_attr_1728_, lean_object* v_b_1729_, lean_object* v_x_1730_, lean_object* v_x_1731_, lean_object* v_x_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_){
_start:
{
lean_object* v_a_1742_; lean_object* v___y_1746_; uint8_t v___y_1747_; lean_object* v___y_1769_; 
if (lean_obj_tag(v_x_1730_) == 5)
{
lean_object* v_fn_1774_; lean_object* v_arg_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; 
v_fn_1774_ = lean_ctor_get(v_x_1730_, 0);
lean_inc_ref(v_fn_1774_);
v_arg_1775_ = lean_ctor_get(v_x_1730_, 1);
lean_inc_ref(v_arg_1775_);
lean_dec_ref_known(v_x_1730_, 2);
v___x_1776_ = lean_array_set(v_x_1731_, v_x_1732_, v_arg_1775_);
v___x_1777_ = lean_unsigned_to_nat(1u);
v___x_1778_ = lean_nat_sub(v_x_1732_, v___x_1777_);
lean_dec(v_x_1732_);
v_x_1730_ = v_fn_1774_;
v_x_1731_ = v___x_1776_;
v_x_1732_ = v___x_1778_;
goto _start;
}
else
{
lean_object* v___x_1780_; lean_object* v___x_1781_; uint8_t v___x_1782_; 
lean_dec(v_x_1732_);
v___x_1780_ = lean_unsigned_to_nat(0u);
v___x_1781_ = lean_array_get_size(v_x_1731_);
v___x_1782_ = lean_nat_dec_lt(v___x_1780_, v___x_1781_);
if (v___x_1782_ == 0)
{
lean_dec_ref(v_x_1731_);
lean_dec_ref(v_b_1729_);
lean_dec(v_attr_1728_);
v_a_1742_ = v_x_1730_;
goto v___jp_1741_;
}
else
{
uint8_t v___x_1783_; 
v___x_1783_ = lean_nat_dec_le(v___x_1781_, v___x_1781_);
if (v___x_1783_ == 0)
{
if (v___x_1782_ == 0)
{
lean_dec_ref(v_x_1731_);
lean_dec_ref(v_b_1729_);
lean_dec(v_attr_1728_);
v_a_1742_ = v_x_1730_;
goto v___jp_1741_;
}
else
{
size_t v___x_1784_; size_t v___x_1785_; lean_object* v___x_1786_; 
v___x_1784_ = ((size_t)0ULL);
v___x_1785_ = lean_usize_of_nat(v___x_1781_);
lean_inc_ref(v_x_1730_);
v___x_1786_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(v_b_1729_, v_x_1731_, v___x_1784_, v___x_1785_, v_x_1730_, v___y_1733_, v___y_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_);
v___y_1769_ = v___x_1786_;
goto v___jp_1768_;
}
}
else
{
size_t v___x_1787_; size_t v___x_1788_; lean_object* v___x_1789_; 
v___x_1787_ = ((size_t)0ULL);
v___x_1788_ = lean_usize_of_nat(v___x_1781_);
lean_inc_ref(v_x_1730_);
v___x_1789_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(v_b_1729_, v_x_1731_, v___x_1787_, v___x_1788_, v_x_1730_, v___y_1733_, v___y_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_);
v___y_1769_ = v___x_1789_;
goto v___jp_1768_;
}
}
}
v___jp_1741_:
{
lean_object* v___x_1743_; lean_object* v___x_1744_; 
v___x_1743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1743_, 0, v_a_1742_);
v___x_1744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1744_, 0, v___x_1743_);
return v___x_1744_;
}
v___jp_1745_:
{
if (v___y_1747_ == 0)
{
lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; 
v___x_1748_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1);
v___x_1749_ = l_Lean_MessageData_ofName(v_attr_1728_);
v___x_1750_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1750_, 0, v___x_1748_);
lean_ctor_set(v___x_1750_, 1, v___x_1749_);
v___x_1751_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3);
v___x_1752_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1752_, 0, v___x_1750_);
lean_ctor_set(v___x_1752_, 1, v___x_1751_);
v___x_1753_ = l_Lean_MessageData_ofExpr(v_x_1730_);
v___x_1754_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1754_, 0, v___x_1752_);
lean_ctor_set(v___x_1754_, 1, v___x_1753_);
v___x_1755_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1);
v___x_1756_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1756_, 0, v___x_1754_);
lean_ctor_set(v___x_1756_, 1, v___x_1755_);
v___x_1757_ = lean_array_to_list(v_x_1731_);
v___x_1758_ = lean_box(0);
v___x_1759_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__0(v___x_1757_, v___x_1758_);
v___x_1760_ = l_Lean_MessageData_ofList(v___x_1759_);
v___x_1761_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1761_, 0, v___x_1756_);
lean_ctor_set(v___x_1761_, 1, v___x_1760_);
v___x_1762_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3);
v___x_1763_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1763_, 0, v___x_1761_);
lean_ctor_set(v___x_1763_, 1, v___x_1762_);
v___x_1764_ = l_Lean_Exception_toMessageData(v___y_1746_);
v___x_1765_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1765_, 0, v___x_1763_);
lean_ctor_set(v___x_1765_, 1, v___x_1764_);
v___x_1766_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v___x_1765_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_);
return v___x_1766_;
}
else
{
lean_object* v___x_1767_; 
lean_dec_ref(v_x_1731_);
lean_dec_ref(v_x_1730_);
lean_dec(v_attr_1728_);
v___x_1767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1767_, 0, v___y_1746_);
return v___x_1767_;
}
}
v___jp_1768_:
{
if (lean_obj_tag(v___y_1769_) == 0)
{
lean_object* v_a_1770_; 
lean_dec_ref(v_x_1731_);
lean_dec_ref(v_x_1730_);
lean_dec(v_attr_1728_);
v_a_1770_ = lean_ctor_get(v___y_1769_, 0);
lean_inc(v_a_1770_);
lean_dec_ref_known(v___y_1769_, 1);
v_a_1742_ = v_a_1770_;
goto v___jp_1741_;
}
else
{
lean_object* v_a_1771_; uint8_t v___x_1772_; 
v_a_1771_ = lean_ctor_get(v___y_1769_, 0);
lean_inc(v_a_1771_);
lean_dec_ref_known(v___y_1769_, 1);
v___x_1772_ = l_Lean_Exception_isInterrupt(v_a_1771_);
if (v___x_1772_ == 0)
{
uint8_t v___x_1773_; 
lean_inc(v_a_1771_);
v___x_1773_ = l_Lean_Exception_isRuntime(v_a_1771_);
v___y_1746_ = v_a_1771_;
v___y_1747_ = v___x_1773_;
goto v___jp_1745_;
}
else
{
v___y_1746_ = v_a_1771_;
v___y_1747_ = v___x_1772_;
goto v___jp_1745_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___boxed(lean_object* v_attr_1790_, lean_object* v_b_1791_, lean_object* v_x_1792_, lean_object* v_x_1793_, lean_object* v_x_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_){
_start:
{
lean_object* v_res_1803_; 
v_res_1803_ = lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2(v_attr_1790_, v_b_1791_, v_x_1792_, v_x_1793_, v_x_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_);
lean_dec(v___y_1801_);
lean_dec_ref(v___y_1800_);
lean_dec(v___y_1799_);
lean_dec_ref(v___y_1798_);
lean_dec(v___y_1797_);
lean_dec_ref(v___y_1796_);
lean_dec(v___y_1795_);
return v_res_1803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2(lean_object* v_b_1804_, lean_object* v_attr_1805_, lean_object* v_x_1806_, lean_object* v_x_1807_, lean_object* v_x_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_){
_start:
{
lean_object* v_a_1818_; lean_object* v___y_1822_; uint8_t v___y_1823_; lean_object* v___y_1845_; 
if (lean_obj_tag(v_x_1806_) == 5)
{
lean_object* v_fn_1850_; lean_object* v_arg_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; 
v_fn_1850_ = lean_ctor_get(v_x_1806_, 0);
lean_inc_ref(v_fn_1850_);
v_arg_1851_ = lean_ctor_get(v_x_1806_, 1);
lean_inc_ref(v_arg_1851_);
lean_dec_ref_known(v_x_1806_, 2);
v___x_1852_ = lean_array_set(v_x_1807_, v_x_1808_, v_arg_1851_);
v___x_1853_ = lean_unsigned_to_nat(1u);
v___x_1854_ = lean_nat_sub(v_x_1808_, v___x_1853_);
v___x_1855_ = lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2(v_attr_1805_, v_b_1804_, v_fn_1850_, v___x_1852_, v___x_1854_, v___y_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
return v___x_1855_;
}
else
{
lean_object* v___x_1856_; lean_object* v___x_1857_; uint8_t v___x_1858_; 
v___x_1856_ = lean_unsigned_to_nat(0u);
v___x_1857_ = lean_array_get_size(v_x_1807_);
v___x_1858_ = lean_nat_dec_lt(v___x_1856_, v___x_1857_);
if (v___x_1858_ == 0)
{
lean_dec_ref(v_x_1807_);
lean_dec(v_attr_1805_);
lean_dec_ref(v_b_1804_);
v_a_1818_ = v_x_1806_;
goto v___jp_1817_;
}
else
{
uint8_t v___x_1859_; 
v___x_1859_ = lean_nat_dec_le(v___x_1857_, v___x_1857_);
if (v___x_1859_ == 0)
{
if (v___x_1858_ == 0)
{
lean_dec_ref(v_x_1807_);
lean_dec(v_attr_1805_);
lean_dec_ref(v_b_1804_);
v_a_1818_ = v_x_1806_;
goto v___jp_1817_;
}
else
{
size_t v___x_1860_; size_t v___x_1861_; lean_object* v___x_1862_; 
v___x_1860_ = ((size_t)0ULL);
v___x_1861_ = lean_usize_of_nat(v___x_1857_);
lean_inc_ref(v_x_1806_);
v___x_1862_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(v_b_1804_, v_x_1807_, v___x_1860_, v___x_1861_, v_x_1806_, v___y_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
v___y_1845_ = v___x_1862_;
goto v___jp_1844_;
}
}
else
{
size_t v___x_1863_; size_t v___x_1864_; lean_object* v___x_1865_; 
v___x_1863_ = ((size_t)0ULL);
v___x_1864_ = lean_usize_of_nat(v___x_1857_);
lean_inc_ref(v_x_1806_);
v___x_1865_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__1(v_b_1804_, v_x_1807_, v___x_1863_, v___x_1864_, v_x_1806_, v___y_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
v___y_1845_ = v___x_1865_;
goto v___jp_1844_;
}
}
}
v___jp_1817_:
{
lean_object* v___x_1819_; lean_object* v___x_1820_; 
v___x_1819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1819_, 0, v_a_1818_);
v___x_1820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1820_, 0, v___x_1819_);
return v___x_1820_;
}
v___jp_1821_:
{
if (v___y_1823_ == 0)
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; 
v___x_1824_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__1);
v___x_1825_ = l_Lean_MessageData_ofName(v_attr_1805_);
v___x_1826_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1826_, 0, v___x_1824_);
lean_ctor_set(v___x_1826_, 1, v___x_1825_);
v___x_1827_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_cast___lam__0___closed__3);
v___x_1828_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1828_, 0, v___x_1826_);
lean_ctor_set(v___x_1828_, 1, v___x_1827_);
v___x_1829_ = l_Lean_MessageData_ofExpr(v_x_1806_);
v___x_1830_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1830_, 0, v___x_1828_);
lean_ctor_set(v___x_1830_, 1, v___x_1829_);
v___x_1831_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__1);
v___x_1832_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1832_, 0, v___x_1830_);
lean_ctor_set(v___x_1832_, 1, v___x_1831_);
v___x_1833_ = lean_array_to_list(v_x_1807_);
v___x_1834_ = lean_box(0);
v___x_1835_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__0(v___x_1833_, v___x_1834_);
v___x_1836_ = l_Lean_MessageData_ofList(v___x_1835_);
v___x_1837_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1837_, 0, v___x_1832_);
lean_ctor_set(v___x_1837_, 1, v___x_1836_);
v___x_1838_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2_spec__2___closed__3);
v___x_1839_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1839_, 0, v___x_1837_);
lean_ctor_set(v___x_1839_, 1, v___x_1838_);
v___x_1840_ = l_Lean_Exception_toMessageData(v___y_1822_);
v___x_1841_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1841_, 0, v___x_1839_);
lean_ctor_set(v___x_1841_, 1, v___x_1840_);
v___x_1842_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_refoldConsts_go_spec__0___redArg(v___x_1841_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
return v___x_1842_;
}
else
{
lean_object* v___x_1843_; 
lean_dec_ref(v_x_1807_);
lean_dec_ref(v_x_1806_);
lean_dec(v_attr_1805_);
v___x_1843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1843_, 0, v___y_1822_);
return v___x_1843_;
}
}
v___jp_1844_:
{
if (lean_obj_tag(v___y_1845_) == 0)
{
lean_object* v_a_1846_; 
lean_dec_ref(v_x_1807_);
lean_dec_ref(v_x_1806_);
lean_dec(v_attr_1805_);
v_a_1846_ = lean_ctor_get(v___y_1845_, 0);
lean_inc(v_a_1846_);
lean_dec_ref_known(v___y_1845_, 1);
v_a_1818_ = v_a_1846_;
goto v___jp_1817_;
}
else
{
lean_object* v_a_1847_; uint8_t v___x_1848_; 
v_a_1847_ = lean_ctor_get(v___y_1845_, 0);
lean_inc(v_a_1847_);
lean_dec_ref_known(v___y_1845_, 1);
v___x_1848_ = l_Lean_Exception_isInterrupt(v_a_1847_);
if (v___x_1848_ == 0)
{
uint8_t v___x_1849_; 
lean_inc(v_a_1847_);
v___x_1849_ = l_Lean_Exception_isRuntime(v_a_1847_);
v___y_1822_ = v_a_1847_;
v___y_1823_ = v___x_1849_;
goto v___jp_1821_;
}
else
{
v___y_1822_ = v_a_1847_;
v___y_1823_ = v___x_1848_;
goto v___jp_1821_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2___boxed(lean_object* v_b_1866_, lean_object* v_attr_1867_, lean_object* v_x_1868_, lean_object* v_x_1869_, lean_object* v_x_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_){
_start:
{
lean_object* v_res_1879_; 
v_res_1879_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2(v_b_1866_, v_attr_1867_, v_x_1868_, v_x_1869_, v_x_1870_, v___y_1871_, v___y_1872_, v___y_1873_, v___y_1874_, v___y_1875_, v___y_1876_, v___y_1877_);
lean_dec(v___y_1877_);
lean_dec_ref(v___y_1876_);
lean_dec(v___y_1875_);
lean_dec_ref(v___y_1874_);
lean_dec(v___y_1873_);
lean_dec_ref(v___y_1872_);
lean_dec(v___y_1871_);
lean_dec(v_x_1870_);
return v_res_1879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1(lean_object* v_b_1880_, lean_object* v_attr_1881_, lean_object* v_e_1882_, lean_object* v___y_1883_, lean_object* v___y_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_){
_start:
{
lean_object* v_dummy_1891_; lean_object* v_nargs_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; 
v_dummy_1891_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0);
v_nargs_1892_ = l_Lean_Expr_getAppNumArgs(v_e_1882_);
lean_inc(v_nargs_1892_);
v___x_1893_ = lean_mk_array(v_nargs_1892_, v_dummy_1891_);
v___x_1894_ = lean_unsigned_to_nat(1u);
v___x_1895_ = lean_nat_sub(v_nargs_1892_, v___x_1894_);
lean_dec(v_nargs_1892_);
v___x_1896_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__2(v_b_1880_, v_attr_1881_, v_e_1882_, v___x_1893_, v___x_1895_, v___y_1883_, v___y_1884_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_, v___y_1889_);
lean_dec(v___x_1895_);
return v___x_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1___boxed(lean_object* v_b_1897_, lean_object* v_attr_1898_, lean_object* v_e_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_){
_start:
{
lean_object* v_res_1908_; 
v_res_1908_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1(v_b_1897_, v_attr_1898_, v_e_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_);
lean_dec(v___y_1906_);
lean_dec_ref(v___y_1905_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
lean_dec(v___y_1902_);
lean_dec_ref(v___y_1901_);
lean_dec(v___y_1900_);
return v_res_1908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0(lean_object* v_00_u03b1_1909_, lean_object* v_x_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v___x_1919_; lean_object* v___x_1920_; 
v___x_1919_ = lean_apply_1(v_x_1910_, lean_box(0));
v___x_1920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1920_, 0, v___x_1919_);
return v___x_1920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0___boxed(lean_object* v_00_u03b1_1921_, lean_object* v_x_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_){
_start:
{
lean_object* v_res_1931_; 
v_res_1931_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0(v_00_u03b1_1921_, v_x_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_);
lean_dec(v___y_1929_);
lean_dec_ref(v___y_1928_);
lean_dec(v___y_1927_);
lean_dec_ref(v___y_1926_);
lean_dec(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec(v___y_1923_);
return v_res_1931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg(lean_object* v_a_1932_, lean_object* v_x_1933_){
_start:
{
if (lean_obj_tag(v_x_1933_) == 0)
{
lean_object* v___x_1934_; 
v___x_1934_ = lean_box(0);
return v___x_1934_;
}
else
{
lean_object* v_key_1935_; lean_object* v_value_1936_; lean_object* v_tail_1937_; uint8_t v___x_1938_; 
v_key_1935_ = lean_ctor_get(v_x_1933_, 0);
v_value_1936_ = lean_ctor_get(v_x_1933_, 1);
v_tail_1937_ = lean_ctor_get(v_x_1933_, 2);
v___x_1938_ = l_Lean_ExprStructEq_beq(v_key_1935_, v_a_1932_);
if (v___x_1938_ == 0)
{
v_x_1933_ = v_tail_1937_;
goto _start;
}
else
{
lean_object* v___x_1940_; 
lean_inc(v_value_1936_);
v___x_1940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1940_, 0, v_value_1936_);
return v___x_1940_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg___boxed(lean_object* v_a_1941_, lean_object* v_x_1942_){
_start:
{
lean_object* v_res_1943_; 
v_res_1943_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg(v_a_1941_, v_x_1942_);
lean_dec(v_x_1942_);
lean_dec_ref(v_a_1941_);
return v_res_1943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(lean_object* v_m_1944_, lean_object* v_a_1945_){
_start:
{
lean_object* v_buckets_1946_; lean_object* v___x_1947_; uint64_t v___x_1948_; uint64_t v___x_1949_; uint64_t v___x_1950_; uint64_t v_fold_1951_; uint64_t v___x_1952_; uint64_t v___x_1953_; uint64_t v___x_1954_; size_t v___x_1955_; size_t v___x_1956_; size_t v___x_1957_; size_t v___x_1958_; size_t v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; 
v_buckets_1946_ = lean_ctor_get(v_m_1944_, 1);
v___x_1947_ = lean_array_get_size(v_buckets_1946_);
v___x_1948_ = l_Lean_ExprStructEq_hash(v_a_1945_);
v___x_1949_ = 32ULL;
v___x_1950_ = lean_uint64_shift_right(v___x_1948_, v___x_1949_);
v_fold_1951_ = lean_uint64_xor(v___x_1948_, v___x_1950_);
v___x_1952_ = 16ULL;
v___x_1953_ = lean_uint64_shift_right(v_fold_1951_, v___x_1952_);
v___x_1954_ = lean_uint64_xor(v_fold_1951_, v___x_1953_);
v___x_1955_ = lean_uint64_to_usize(v___x_1954_);
v___x_1956_ = lean_usize_of_nat(v___x_1947_);
v___x_1957_ = ((size_t)1ULL);
v___x_1958_ = lean_usize_sub(v___x_1956_, v___x_1957_);
v___x_1959_ = lean_usize_land(v___x_1955_, v___x_1958_);
v___x_1960_ = lean_array_uget_borrowed(v_buckets_1946_, v___x_1959_);
v___x_1961_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg(v_a_1945_, v___x_1960_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg___boxed(lean_object* v_m_1962_, lean_object* v_a_1963_){
_start:
{
lean_object* v_res_1964_; 
v_res_1964_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(v_m_1962_, v_a_1963_);
lean_dec_ref(v_a_1963_);
lean_dec_ref(v_m_1962_);
return v_res_1964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21___redArg(lean_object* v_a_1965_, lean_object* v_b_1966_, lean_object* v_x_1967_){
_start:
{
if (lean_obj_tag(v_x_1967_) == 0)
{
lean_dec(v_b_1966_);
lean_dec_ref(v_a_1965_);
return v_x_1967_;
}
else
{
lean_object* v_key_1968_; lean_object* v_value_1969_; lean_object* v_tail_1970_; lean_object* v___x_1972_; uint8_t v_isShared_1973_; uint8_t v_isSharedCheck_1982_; 
v_key_1968_ = lean_ctor_get(v_x_1967_, 0);
v_value_1969_ = lean_ctor_get(v_x_1967_, 1);
v_tail_1970_ = lean_ctor_get(v_x_1967_, 2);
v_isSharedCheck_1982_ = !lean_is_exclusive(v_x_1967_);
if (v_isSharedCheck_1982_ == 0)
{
v___x_1972_ = v_x_1967_;
v_isShared_1973_ = v_isSharedCheck_1982_;
goto v_resetjp_1971_;
}
else
{
lean_inc(v_tail_1970_);
lean_inc(v_value_1969_);
lean_inc(v_key_1968_);
lean_dec(v_x_1967_);
v___x_1972_ = lean_box(0);
v_isShared_1973_ = v_isSharedCheck_1982_;
goto v_resetjp_1971_;
}
v_resetjp_1971_:
{
uint8_t v___x_1974_; 
v___x_1974_ = l_Lean_ExprStructEq_beq(v_key_1968_, v_a_1965_);
if (v___x_1974_ == 0)
{
lean_object* v___x_1975_; lean_object* v___x_1977_; 
v___x_1975_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21___redArg(v_a_1965_, v_b_1966_, v_tail_1970_);
if (v_isShared_1973_ == 0)
{
lean_ctor_set(v___x_1972_, 2, v___x_1975_);
v___x_1977_ = v___x_1972_;
goto v_reusejp_1976_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v_key_1968_);
lean_ctor_set(v_reuseFailAlloc_1978_, 1, v_value_1969_);
lean_ctor_set(v_reuseFailAlloc_1978_, 2, v___x_1975_);
v___x_1977_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1976_;
}
v_reusejp_1976_:
{
return v___x_1977_;
}
}
else
{
lean_object* v___x_1980_; 
lean_dec(v_value_1969_);
lean_dec(v_key_1968_);
if (v_isShared_1973_ == 0)
{
lean_ctor_set(v___x_1972_, 1, v_b_1966_);
lean_ctor_set(v___x_1972_, 0, v_a_1965_);
v___x_1980_ = v___x_1972_;
goto v_reusejp_1979_;
}
else
{
lean_object* v_reuseFailAlloc_1981_; 
v_reuseFailAlloc_1981_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1981_, 0, v_a_1965_);
lean_ctor_set(v_reuseFailAlloc_1981_, 1, v_b_1966_);
lean_ctor_set(v_reuseFailAlloc_1981_, 2, v_tail_1970_);
v___x_1980_ = v_reuseFailAlloc_1981_;
goto v_reusejp_1979_;
}
v_reusejp_1979_:
{
return v___x_1980_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg(lean_object* v_a_1983_, lean_object* v_x_1984_){
_start:
{
if (lean_obj_tag(v_x_1984_) == 0)
{
uint8_t v___x_1985_; 
v___x_1985_ = 0;
return v___x_1985_;
}
else
{
lean_object* v_key_1986_; lean_object* v_tail_1987_; uint8_t v___x_1988_; 
v_key_1986_ = lean_ctor_get(v_x_1984_, 0);
v_tail_1987_ = lean_ctor_get(v_x_1984_, 2);
v___x_1988_ = l_Lean_ExprStructEq_beq(v_key_1986_, v_a_1983_);
if (v___x_1988_ == 0)
{
v_x_1984_ = v_tail_1987_;
goto _start;
}
else
{
return v___x_1988_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg___boxed(lean_object* v_a_1990_, lean_object* v_x_1991_){
_start:
{
uint8_t v_res_1992_; lean_object* v_r_1993_; 
v_res_1992_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg(v_a_1990_, v_x_1991_);
lean_dec(v_x_1991_);
lean_dec_ref(v_a_1990_);
v_r_1993_ = lean_box(v_res_1992_);
return v_r_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22___redArg(lean_object* v_x_1994_, lean_object* v_x_1995_){
_start:
{
if (lean_obj_tag(v_x_1995_) == 0)
{
return v_x_1994_;
}
else
{
lean_object* v_key_1996_; lean_object* v_value_1997_; lean_object* v_tail_1998_; lean_object* v___x_2000_; uint8_t v_isShared_2001_; uint8_t v_isSharedCheck_2021_; 
v_key_1996_ = lean_ctor_get(v_x_1995_, 0);
v_value_1997_ = lean_ctor_get(v_x_1995_, 1);
v_tail_1998_ = lean_ctor_get(v_x_1995_, 2);
v_isSharedCheck_2021_ = !lean_is_exclusive(v_x_1995_);
if (v_isSharedCheck_2021_ == 0)
{
v___x_2000_ = v_x_1995_;
v_isShared_2001_ = v_isSharedCheck_2021_;
goto v_resetjp_1999_;
}
else
{
lean_inc(v_tail_1998_);
lean_inc(v_value_1997_);
lean_inc(v_key_1996_);
lean_dec(v_x_1995_);
v___x_2000_ = lean_box(0);
v_isShared_2001_ = v_isSharedCheck_2021_;
goto v_resetjp_1999_;
}
v_resetjp_1999_:
{
lean_object* v___x_2002_; uint64_t v___x_2003_; uint64_t v___x_2004_; uint64_t v___x_2005_; uint64_t v_fold_2006_; uint64_t v___x_2007_; uint64_t v___x_2008_; uint64_t v___x_2009_; size_t v___x_2010_; size_t v___x_2011_; size_t v___x_2012_; size_t v___x_2013_; size_t v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2017_; 
v___x_2002_ = lean_array_get_size(v_x_1994_);
v___x_2003_ = l_Lean_ExprStructEq_hash(v_key_1996_);
v___x_2004_ = 32ULL;
v___x_2005_ = lean_uint64_shift_right(v___x_2003_, v___x_2004_);
v_fold_2006_ = lean_uint64_xor(v___x_2003_, v___x_2005_);
v___x_2007_ = 16ULL;
v___x_2008_ = lean_uint64_shift_right(v_fold_2006_, v___x_2007_);
v___x_2009_ = lean_uint64_xor(v_fold_2006_, v___x_2008_);
v___x_2010_ = lean_uint64_to_usize(v___x_2009_);
v___x_2011_ = lean_usize_of_nat(v___x_2002_);
v___x_2012_ = ((size_t)1ULL);
v___x_2013_ = lean_usize_sub(v___x_2011_, v___x_2012_);
v___x_2014_ = lean_usize_land(v___x_2010_, v___x_2013_);
v___x_2015_ = lean_array_uget_borrowed(v_x_1994_, v___x_2014_);
lean_inc(v___x_2015_);
if (v_isShared_2001_ == 0)
{
lean_ctor_set(v___x_2000_, 2, v___x_2015_);
v___x_2017_ = v___x_2000_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v_key_1996_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v_value_1997_);
lean_ctor_set(v_reuseFailAlloc_2020_, 2, v___x_2015_);
v___x_2017_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
lean_object* v___x_2018_; 
v___x_2018_ = lean_array_uset(v_x_1994_, v___x_2014_, v___x_2017_);
v_x_1994_ = v___x_2018_;
v_x_1995_ = v_tail_1998_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21___redArg(lean_object* v_i_2022_, lean_object* v_source_2023_, lean_object* v_target_2024_){
_start:
{
lean_object* v___x_2025_; uint8_t v___x_2026_; 
v___x_2025_ = lean_array_get_size(v_source_2023_);
v___x_2026_ = lean_nat_dec_lt(v_i_2022_, v___x_2025_);
if (v___x_2026_ == 0)
{
lean_dec_ref(v_source_2023_);
lean_dec(v_i_2022_);
return v_target_2024_;
}
else
{
lean_object* v_es_2027_; lean_object* v___x_2028_; lean_object* v_source_2029_; lean_object* v_target_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; 
v_es_2027_ = lean_array_fget(v_source_2023_, v_i_2022_);
v___x_2028_ = lean_box(0);
v_source_2029_ = lean_array_fset(v_source_2023_, v_i_2022_, v___x_2028_);
v_target_2030_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22___redArg(v_target_2024_, v_es_2027_);
v___x_2031_ = lean_unsigned_to_nat(1u);
v___x_2032_ = lean_nat_add(v_i_2022_, v___x_2031_);
lean_dec(v_i_2022_);
v_i_2022_ = v___x_2032_;
v_source_2023_ = v_source_2029_;
v_target_2024_ = v_target_2030_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20___redArg(lean_object* v_data_2034_){
_start:
{
lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v_nbuckets_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; 
v___x_2035_ = lean_array_get_size(v_data_2034_);
v___x_2036_ = lean_unsigned_to_nat(2u);
v_nbuckets_2037_ = lean_nat_mul(v___x_2035_, v___x_2036_);
v___x_2038_ = lean_unsigned_to_nat(0u);
v___x_2039_ = lean_box(0);
v___x_2040_ = lean_mk_array(v_nbuckets_2037_, v___x_2039_);
v___x_2041_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21___redArg(v___x_2038_, v_data_2034_, v___x_2040_);
return v___x_2041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14___redArg(lean_object* v_m_2042_, lean_object* v_a_2043_, lean_object* v_b_2044_){
_start:
{
lean_object* v_size_2045_; lean_object* v_buckets_2046_; lean_object* v___x_2048_; uint8_t v_isShared_2049_; uint8_t v_isSharedCheck_2089_; 
v_size_2045_ = lean_ctor_get(v_m_2042_, 0);
v_buckets_2046_ = lean_ctor_get(v_m_2042_, 1);
v_isSharedCheck_2089_ = !lean_is_exclusive(v_m_2042_);
if (v_isSharedCheck_2089_ == 0)
{
v___x_2048_ = v_m_2042_;
v_isShared_2049_ = v_isSharedCheck_2089_;
goto v_resetjp_2047_;
}
else
{
lean_inc(v_buckets_2046_);
lean_inc(v_size_2045_);
lean_dec(v_m_2042_);
v___x_2048_ = lean_box(0);
v_isShared_2049_ = v_isSharedCheck_2089_;
goto v_resetjp_2047_;
}
v_resetjp_2047_:
{
lean_object* v___x_2050_; uint64_t v___x_2051_; uint64_t v___x_2052_; uint64_t v___x_2053_; uint64_t v_fold_2054_; uint64_t v___x_2055_; uint64_t v___x_2056_; uint64_t v___x_2057_; size_t v___x_2058_; size_t v___x_2059_; size_t v___x_2060_; size_t v___x_2061_; size_t v___x_2062_; lean_object* v_bkt_2063_; uint8_t v___x_2064_; 
v___x_2050_ = lean_array_get_size(v_buckets_2046_);
v___x_2051_ = l_Lean_ExprStructEq_hash(v_a_2043_);
v___x_2052_ = 32ULL;
v___x_2053_ = lean_uint64_shift_right(v___x_2051_, v___x_2052_);
v_fold_2054_ = lean_uint64_xor(v___x_2051_, v___x_2053_);
v___x_2055_ = 16ULL;
v___x_2056_ = lean_uint64_shift_right(v_fold_2054_, v___x_2055_);
v___x_2057_ = lean_uint64_xor(v_fold_2054_, v___x_2056_);
v___x_2058_ = lean_uint64_to_usize(v___x_2057_);
v___x_2059_ = lean_usize_of_nat(v___x_2050_);
v___x_2060_ = ((size_t)1ULL);
v___x_2061_ = lean_usize_sub(v___x_2059_, v___x_2060_);
v___x_2062_ = lean_usize_land(v___x_2058_, v___x_2061_);
v_bkt_2063_ = lean_array_uget_borrowed(v_buckets_2046_, v___x_2062_);
v___x_2064_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg(v_a_2043_, v_bkt_2063_);
if (v___x_2064_ == 0)
{
lean_object* v___x_2065_; lean_object* v_size_x27_2066_; lean_object* v___x_2067_; lean_object* v_buckets_x27_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; uint8_t v___x_2074_; 
v___x_2065_ = lean_unsigned_to_nat(1u);
v_size_x27_2066_ = lean_nat_add(v_size_2045_, v___x_2065_);
lean_dec(v_size_2045_);
lean_inc(v_bkt_2063_);
v___x_2067_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2067_, 0, v_a_2043_);
lean_ctor_set(v___x_2067_, 1, v_b_2044_);
lean_ctor_set(v___x_2067_, 2, v_bkt_2063_);
v_buckets_x27_2068_ = lean_array_uset(v_buckets_2046_, v___x_2062_, v___x_2067_);
v___x_2069_ = lean_unsigned_to_nat(4u);
v___x_2070_ = lean_nat_mul(v_size_x27_2066_, v___x_2069_);
v___x_2071_ = lean_unsigned_to_nat(3u);
v___x_2072_ = lean_nat_div(v___x_2070_, v___x_2071_);
lean_dec(v___x_2070_);
v___x_2073_ = lean_array_get_size(v_buckets_x27_2068_);
v___x_2074_ = lean_nat_dec_le(v___x_2072_, v___x_2073_);
lean_dec(v___x_2072_);
if (v___x_2074_ == 0)
{
lean_object* v_val_2075_; lean_object* v___x_2077_; 
v_val_2075_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20___redArg(v_buckets_x27_2068_);
if (v_isShared_2049_ == 0)
{
lean_ctor_set(v___x_2048_, 1, v_val_2075_);
lean_ctor_set(v___x_2048_, 0, v_size_x27_2066_);
v___x_2077_ = v___x_2048_;
goto v_reusejp_2076_;
}
else
{
lean_object* v_reuseFailAlloc_2078_; 
v_reuseFailAlloc_2078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2078_, 0, v_size_x27_2066_);
lean_ctor_set(v_reuseFailAlloc_2078_, 1, v_val_2075_);
v___x_2077_ = v_reuseFailAlloc_2078_;
goto v_reusejp_2076_;
}
v_reusejp_2076_:
{
return v___x_2077_;
}
}
else
{
lean_object* v___x_2080_; 
if (v_isShared_2049_ == 0)
{
lean_ctor_set(v___x_2048_, 1, v_buckets_x27_2068_);
lean_ctor_set(v___x_2048_, 0, v_size_x27_2066_);
v___x_2080_ = v___x_2048_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2081_; 
v_reuseFailAlloc_2081_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2081_, 0, v_size_x27_2066_);
lean_ctor_set(v_reuseFailAlloc_2081_, 1, v_buckets_x27_2068_);
v___x_2080_ = v_reuseFailAlloc_2081_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
return v___x_2080_;
}
}
}
else
{
lean_object* v___x_2082_; lean_object* v_buckets_x27_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2087_; 
lean_inc(v_bkt_2063_);
v___x_2082_ = lean_box(0);
v_buckets_x27_2083_ = lean_array_uset(v_buckets_2046_, v___x_2062_, v___x_2082_);
v___x_2084_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21___redArg(v_a_2043_, v_b_2044_, v_bkt_2063_);
v___x_2085_ = lean_array_uset(v_buckets_x27_2083_, v___x_2062_, v___x_2084_);
if (v_isShared_2049_ == 0)
{
lean_ctor_set(v___x_2048_, 1, v___x_2085_);
v___x_2087_ = v___x_2048_;
goto v_reusejp_2086_;
}
else
{
lean_object* v_reuseFailAlloc_2088_; 
v_reuseFailAlloc_2088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2088_, 0, v_size_2045_);
lean_ctor_set(v_reuseFailAlloc_2088_, 1, v___x_2085_);
v___x_2087_ = v_reuseFailAlloc_2088_;
goto v_reusejp_2086_;
}
v_reusejp_2086_:
{
return v___x_2087_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2(lean_object* v_a_2090_, lean_object* v_e_2091_, lean_object* v_a_2092_){
_start:
{
lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; 
v___x_2094_ = lean_st_ref_take(v_a_2090_);
v___x_2095_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14___redArg(v___x_2094_, v_e_2091_, v_a_2092_);
v___x_2096_ = lean_st_ref_set(v_a_2090_, v___x_2095_);
v___x_2097_ = lean_box(0);
return v___x_2097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2___boxed(lean_object* v_a_2098_, lean_object* v_e_2099_, lean_object* v_a_2100_, lean_object* v___y_2101_){
_start:
{
lean_object* v_res_2102_; 
v_res_2102_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2(v_a_2098_, v_e_2099_, v_a_2100_);
lean_dec(v_a_2098_);
return v_res_2102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0(lean_object* v_00_u03b1_2103_, lean_object* v_x_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_){
_start:
{
lean_object* v___x_2113_; lean_object* v___x_2114_; 
v___x_2113_ = lean_apply_1(v_x_2104_, lean_box(0));
v___x_2114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2114_, 0, v___x_2113_);
return v___x_2114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0___boxed(lean_object* v_00_u03b1_2115_, lean_object* v_x_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_){
_start:
{
lean_object* v_res_2125_; 
v_res_2125_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0(v_00_u03b1_2115_, v_x_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
lean_dec(v___y_2121_);
lean_dec_ref(v___y_2120_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
lean_dec(v___y_2117_);
return v_res_2125_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3(void){
_start:
{
lean_object* v___x_2131_; lean_object* v___x_2132_; 
v___x_2131_ = l_Lean_maxRecDepthErrorMessage;
v___x_2132_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2132_, 0, v___x_2131_);
return v___x_2132_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4(void){
_start:
{
lean_object* v___x_2133_; lean_object* v___x_2134_; 
v___x_2133_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__3);
v___x_2134_ = l_Lean_MessageData_ofFormat(v___x_2133_);
return v___x_2134_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5(void){
_start:
{
lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; 
v___x_2135_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__4);
v___x_2136_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__2));
v___x_2137_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2136_);
lean_ctor_set(v___x_2137_, 1, v___x_2135_);
return v___x_2137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg(lean_object* v_ref_2138_){
_start:
{
lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; 
v___x_2140_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5);
v___x_2141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2141_, 0, v_ref_2138_);
lean_ctor_set(v___x_2141_, 1, v___x_2140_);
v___x_2142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2142_, 0, v___x_2141_);
return v___x_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___boxed(lean_object* v_ref_2143_, lean_object* v___y_2144_){
_start:
{
lean_object* v_res_2145_; 
v_res_2145_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg(v_ref_2143_);
return v_res_2145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg(lean_object* v_x_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_){
_start:
{
lean_object* v___y_2157_; lean_object* v_fileName_2166_; lean_object* v_fileMap_2167_; lean_object* v_options_2168_; lean_object* v_currRecDepth_2169_; lean_object* v_maxRecDepth_2170_; lean_object* v_ref_2171_; lean_object* v_currNamespace_2172_; lean_object* v_openDecls_2173_; lean_object* v_initHeartbeats_2174_; lean_object* v_maxHeartbeats_2175_; lean_object* v_quotContext_2176_; lean_object* v_currMacroScope_2177_; uint8_t v_diag_2178_; lean_object* v_cancelTk_x3f_2179_; uint8_t v_suppressElabErrors_2180_; lean_object* v_inheritedTraceOptions_2181_; lean_object* v___x_2187_; uint8_t v___x_2188_; 
v_fileName_2166_ = lean_ctor_get(v___y_2153_, 0);
v_fileMap_2167_ = lean_ctor_get(v___y_2153_, 1);
v_options_2168_ = lean_ctor_get(v___y_2153_, 2);
v_currRecDepth_2169_ = lean_ctor_get(v___y_2153_, 3);
v_maxRecDepth_2170_ = lean_ctor_get(v___y_2153_, 4);
v_ref_2171_ = lean_ctor_get(v___y_2153_, 5);
v_currNamespace_2172_ = lean_ctor_get(v___y_2153_, 6);
v_openDecls_2173_ = lean_ctor_get(v___y_2153_, 7);
v_initHeartbeats_2174_ = lean_ctor_get(v___y_2153_, 8);
v_maxHeartbeats_2175_ = lean_ctor_get(v___y_2153_, 9);
v_quotContext_2176_ = lean_ctor_get(v___y_2153_, 10);
v_currMacroScope_2177_ = lean_ctor_get(v___y_2153_, 11);
v_diag_2178_ = lean_ctor_get_uint8(v___y_2153_, sizeof(void*)*14);
v_cancelTk_x3f_2179_ = lean_ctor_get(v___y_2153_, 12);
v_suppressElabErrors_2180_ = lean_ctor_get_uint8(v___y_2153_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2181_ = lean_ctor_get(v___y_2153_, 13);
v___x_2187_ = lean_unsigned_to_nat(0u);
v___x_2188_ = lean_nat_dec_eq(v_maxRecDepth_2170_, v___x_2187_);
if (v___x_2188_ == 0)
{
uint8_t v___x_2189_; 
v___x_2189_ = lean_nat_dec_eq(v_currRecDepth_2169_, v_maxRecDepth_2170_);
if (v___x_2189_ == 0)
{
goto v___jp_2182_;
}
else
{
lean_object* v___x_2190_; 
lean_dec_ref(v_x_2146_);
lean_inc(v_ref_2171_);
v___x_2190_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg(v_ref_2171_);
v___y_2157_ = v___x_2190_;
goto v___jp_2156_;
}
}
else
{
goto v___jp_2182_;
}
v___jp_2156_:
{
if (lean_obj_tag(v___y_2157_) == 0)
{
return v___y_2157_;
}
else
{
lean_object* v_a_2158_; lean_object* v___x_2160_; uint8_t v_isShared_2161_; uint8_t v_isSharedCheck_2165_; 
v_a_2158_ = lean_ctor_get(v___y_2157_, 0);
v_isSharedCheck_2165_ = !lean_is_exclusive(v___y_2157_);
if (v_isSharedCheck_2165_ == 0)
{
v___x_2160_ = v___y_2157_;
v_isShared_2161_ = v_isSharedCheck_2165_;
goto v_resetjp_2159_;
}
else
{
lean_inc(v_a_2158_);
lean_dec(v___y_2157_);
v___x_2160_ = lean_box(0);
v_isShared_2161_ = v_isSharedCheck_2165_;
goto v_resetjp_2159_;
}
v_resetjp_2159_:
{
lean_object* v___x_2163_; 
if (v_isShared_2161_ == 0)
{
v___x_2163_ = v___x_2160_;
goto v_reusejp_2162_;
}
else
{
lean_object* v_reuseFailAlloc_2164_; 
v_reuseFailAlloc_2164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2164_, 0, v_a_2158_);
v___x_2163_ = v_reuseFailAlloc_2164_;
goto v_reusejp_2162_;
}
v_reusejp_2162_:
{
return v___x_2163_;
}
}
}
}
v___jp_2182_:
{
lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; 
v___x_2183_ = lean_unsigned_to_nat(1u);
v___x_2184_ = lean_nat_add(v_currRecDepth_2169_, v___x_2183_);
lean_inc_ref(v_inheritedTraceOptions_2181_);
lean_inc(v_cancelTk_x3f_2179_);
lean_inc(v_currMacroScope_2177_);
lean_inc(v_quotContext_2176_);
lean_inc(v_maxHeartbeats_2175_);
lean_inc(v_initHeartbeats_2174_);
lean_inc(v_openDecls_2173_);
lean_inc(v_currNamespace_2172_);
lean_inc(v_ref_2171_);
lean_inc(v_maxRecDepth_2170_);
lean_inc_ref(v_options_2168_);
lean_inc_ref(v_fileMap_2167_);
lean_inc_ref(v_fileName_2166_);
v___x_2185_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2185_, 0, v_fileName_2166_);
lean_ctor_set(v___x_2185_, 1, v_fileMap_2167_);
lean_ctor_set(v___x_2185_, 2, v_options_2168_);
lean_ctor_set(v___x_2185_, 3, v___x_2184_);
lean_ctor_set(v___x_2185_, 4, v_maxRecDepth_2170_);
lean_ctor_set(v___x_2185_, 5, v_ref_2171_);
lean_ctor_set(v___x_2185_, 6, v_currNamespace_2172_);
lean_ctor_set(v___x_2185_, 7, v_openDecls_2173_);
lean_ctor_set(v___x_2185_, 8, v_initHeartbeats_2174_);
lean_ctor_set(v___x_2185_, 9, v_maxHeartbeats_2175_);
lean_ctor_set(v___x_2185_, 10, v_quotContext_2176_);
lean_ctor_set(v___x_2185_, 11, v_currMacroScope_2177_);
lean_ctor_set(v___x_2185_, 12, v_cancelTk_x3f_2179_);
lean_ctor_set(v___x_2185_, 13, v_inheritedTraceOptions_2181_);
lean_ctor_set_uint8(v___x_2185_, sizeof(void*)*14, v_diag_2178_);
lean_ctor_set_uint8(v___x_2185_, sizeof(void*)*14 + 1, v_suppressElabErrors_2180_);
lean_inc(v___y_2154_);
lean_inc(v___y_2152_);
lean_inc_ref(v___y_2151_);
lean_inc(v___y_2150_);
lean_inc_ref(v___y_2149_);
lean_inc(v___y_2148_);
lean_inc(v___y_2147_);
v___x_2186_ = lean_apply_9(v_x_2146_, v___y_2147_, v___y_2148_, v___y_2149_, v___y_2150_, v___y_2151_, v___y_2152_, v___x_2185_, v___y_2154_, lean_box(0));
v___y_2157_ = v___x_2186_;
goto v___jp_2156_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg___boxed(lean_object* v_x_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_){
_start:
{
lean_object* v_res_2201_; 
v_res_2201_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg(v_x_2191_, v___y_2192_, v___y_2193_, v___y_2194_, v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_);
lean_dec(v___y_2199_);
lean_dec_ref(v___y_2198_);
lean_dec(v___y_2197_);
lean_dec_ref(v___y_2196_);
lean_dec(v___y_2195_);
lean_dec_ref(v___y_2194_);
lean_dec(v___y_2193_);
lean_dec(v___y_2192_);
return v_res_2201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2(lean_object* v___x_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_, lean_object* v___y_2208_, lean_object* v___y_2209_){
_start:
{
lean_object* v___x_2211_; 
v___x_2211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2211_, 0, v___x_2202_);
return v___x_2211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2___boxed(lean_object* v___x_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_, lean_object* v___y_2216_, lean_object* v___y_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_){
_start:
{
lean_object* v_res_2221_; 
v_res_2221_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2(v___x_2212_, v___y_2213_, v___y_2214_, v___y_2215_, v___y_2216_, v___y_2217_, v___y_2218_, v___y_2219_);
lean_dec(v___y_2219_);
lean_dec_ref(v___y_2218_);
lean_dec(v___y_2217_);
lean_dec_ref(v___y_2216_);
lean_dec(v___y_2215_);
lean_dec_ref(v___y_2214_);
lean_dec(v___y_2213_);
return v_res_2221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0(lean_object* v_k_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v_b_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_){
_start:
{
lean_object* v___x_2233_; 
lean_inc(v___y_2231_);
lean_inc_ref(v___y_2230_);
lean_inc(v___y_2229_);
lean_inc_ref(v___y_2228_);
lean_inc(v___y_2226_);
lean_inc_ref(v___y_2225_);
lean_inc(v___y_2224_);
lean_inc(v___y_2223_);
v___x_2233_ = lean_apply_10(v_k_2222_, v_b_2227_, v___y_2223_, v___y_2224_, v___y_2225_, v___y_2226_, v___y_2228_, v___y_2229_, v___y_2230_, v___y_2231_, lean_box(0));
return v___x_2233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0___boxed(lean_object* v_k_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v_b_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_){
_start:
{
lean_object* v_res_2245_; 
v_res_2245_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0(v_k_2234_, v___y_2235_, v___y_2236_, v___y_2237_, v___y_2238_, v_b_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_);
lean_dec(v___y_2243_);
lean_dec_ref(v___y_2242_);
lean_dec(v___y_2241_);
lean_dec_ref(v___y_2240_);
lean_dec(v___y_2238_);
lean_dec_ref(v___y_2237_);
lean_dec(v___y_2236_);
lean_dec(v___y_2235_);
return v_res_2245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(lean_object* v_name_2246_, uint8_t v_bi_2247_, lean_object* v_type_2248_, lean_object* v_k_2249_, uint8_t v_kind_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_){
_start:
{
lean_object* v___f_2260_; lean_object* v___x_2261_; 
lean_inc(v___y_2254_);
lean_inc_ref(v___y_2253_);
lean_inc(v___y_2252_);
lean_inc(v___y_2251_);
v___f_2260_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0___boxed), 11, 5);
lean_closure_set(v___f_2260_, 0, v_k_2249_);
lean_closure_set(v___f_2260_, 1, v___y_2251_);
lean_closure_set(v___f_2260_, 2, v___y_2252_);
lean_closure_set(v___f_2260_, 3, v___y_2253_);
lean_closure_set(v___f_2260_, 4, v___y_2254_);
v___x_2261_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_2246_, v_bi_2247_, v_type_2248_, v___f_2260_, v_kind_2250_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_);
if (lean_obj_tag(v___x_2261_) == 0)
{
return v___x_2261_;
}
else
{
lean_object* v_a_2262_; lean_object* v___x_2264_; uint8_t v_isShared_2265_; uint8_t v_isSharedCheck_2269_; 
v_a_2262_ = lean_ctor_get(v___x_2261_, 0);
v_isSharedCheck_2269_ = !lean_is_exclusive(v___x_2261_);
if (v_isSharedCheck_2269_ == 0)
{
v___x_2264_ = v___x_2261_;
v_isShared_2265_ = v_isSharedCheck_2269_;
goto v_resetjp_2263_;
}
else
{
lean_inc(v_a_2262_);
lean_dec(v___x_2261_);
v___x_2264_ = lean_box(0);
v_isShared_2265_ = v_isSharedCheck_2269_;
goto v_resetjp_2263_;
}
v_resetjp_2263_:
{
lean_object* v___x_2267_; 
if (v_isShared_2265_ == 0)
{
v___x_2267_ = v___x_2264_;
goto v_reusejp_2266_;
}
else
{
lean_object* v_reuseFailAlloc_2268_; 
v_reuseFailAlloc_2268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2268_, 0, v_a_2262_);
v___x_2267_ = v_reuseFailAlloc_2268_;
goto v_reusejp_2266_;
}
v_reusejp_2266_:
{
return v___x_2267_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___boxed(lean_object* v_name_2270_, lean_object* v_bi_2271_, lean_object* v_type_2272_, lean_object* v_k_2273_, lean_object* v_kind_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_){
_start:
{
uint8_t v_bi_boxed_2284_; uint8_t v_kind_boxed_2285_; lean_object* v_res_2286_; 
v_bi_boxed_2284_ = lean_unbox(v_bi_2271_);
v_kind_boxed_2285_ = lean_unbox(v_kind_2274_);
v_res_2286_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(v_name_2270_, v_bi_boxed_2284_, v_type_2272_, v_k_2273_, v_kind_boxed_2285_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_, v___y_2280_, v___y_2281_, v___y_2282_);
lean_dec(v___y_2282_);
lean_dec_ref(v___y_2281_);
lean_dec(v___y_2280_);
lean_dec_ref(v___y_2279_);
lean_dec(v___y_2278_);
lean_dec_ref(v___y_2277_);
lean_dec(v___y_2276_);
lean_dec(v___y_2275_);
return v_res_2286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg(lean_object* v_name_2287_, lean_object* v_type_2288_, lean_object* v_val_2289_, lean_object* v_k_2290_, uint8_t v_nondep_2291_, uint8_t v_kind_2292_, lean_object* v___y_2293_, lean_object* v___y_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_){
_start:
{
lean_object* v___f_2302_; lean_object* v___x_2303_; 
lean_inc(v___y_2296_);
lean_inc_ref(v___y_2295_);
lean_inc(v___y_2294_);
lean_inc(v___y_2293_);
v___f_2302_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg___lam__0___boxed), 11, 5);
lean_closure_set(v___f_2302_, 0, v_k_2290_);
lean_closure_set(v___f_2302_, 1, v___y_2293_);
lean_closure_set(v___f_2302_, 2, v___y_2294_);
lean_closure_set(v___f_2302_, 3, v___y_2295_);
lean_closure_set(v___f_2302_, 4, v___y_2296_);
v___x_2303_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_2287_, v_type_2288_, v_val_2289_, v___f_2302_, v_nondep_2291_, v_kind_2292_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_);
if (lean_obj_tag(v___x_2303_) == 0)
{
return v___x_2303_;
}
else
{
lean_object* v_a_2304_; lean_object* v___x_2306_; uint8_t v_isShared_2307_; uint8_t v_isSharedCheck_2311_; 
v_a_2304_ = lean_ctor_get(v___x_2303_, 0);
v_isSharedCheck_2311_ = !lean_is_exclusive(v___x_2303_);
if (v_isSharedCheck_2311_ == 0)
{
v___x_2306_ = v___x_2303_;
v_isShared_2307_ = v_isSharedCheck_2311_;
goto v_resetjp_2305_;
}
else
{
lean_inc(v_a_2304_);
lean_dec(v___x_2303_);
v___x_2306_ = lean_box(0);
v_isShared_2307_ = v_isSharedCheck_2311_;
goto v_resetjp_2305_;
}
v_resetjp_2305_:
{
lean_object* v___x_2309_; 
if (v_isShared_2307_ == 0)
{
v___x_2309_ = v___x_2306_;
goto v_reusejp_2308_;
}
else
{
lean_object* v_reuseFailAlloc_2310_; 
v_reuseFailAlloc_2310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2310_, 0, v_a_2304_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg___boxed(lean_object* v_name_2312_, lean_object* v_type_2313_, lean_object* v_val_2314_, lean_object* v_k_2315_, lean_object* v_nondep_2316_, lean_object* v_kind_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_, lean_object* v___y_2325_, lean_object* v___y_2326_){
_start:
{
uint8_t v_nondep_boxed_2327_; uint8_t v_kind_boxed_2328_; lean_object* v_res_2329_; 
v_nondep_boxed_2327_ = lean_unbox(v_nondep_2316_);
v_kind_boxed_2328_ = lean_unbox(v_kind_2317_);
v_res_2329_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg(v_name_2312_, v_type_2313_, v_val_2314_, v_k_2315_, v_nondep_boxed_2327_, v_kind_boxed_2328_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_, v___y_2323_, v___y_2324_, v___y_2325_);
lean_dec(v___y_2325_);
lean_dec_ref(v___y_2324_);
lean_dec(v___y_2323_);
lean_dec_ref(v___y_2322_);
lean_dec(v___y_2321_);
lean_dec_ref(v___y_2320_);
lean_dec(v___y_2319_);
lean_dec(v___y_2318_);
return v_res_2329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0(lean_object* v_fvars_2333_, lean_object* v_pre_2334_, lean_object* v_post_2335_, uint8_t v_usedLetOnly_2336_, uint8_t v_skipConstInApp_2337_, uint8_t v_skipInstances_2338_, lean_object* v_body_2339_, lean_object* v_x_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_){
_start:
{
lean_object* v___x_2350_; lean_object* v___x_2351_; 
v___x_2350_ = lean_array_push(v_fvars_2333_, v_x_2340_);
v___x_2351_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10(v_pre_2334_, v_post_2335_, v_usedLetOnly_2336_, v_skipConstInApp_2337_, v_skipInstances_2338_, v___x_2350_, v_body_2339_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_, v___y_2347_, v___y_2348_);
return v___x_2351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0___boxed(lean_object** _args){
lean_object* v_fvars_2352_ = _args[0];
lean_object* v_pre_2353_ = _args[1];
lean_object* v_post_2354_ = _args[2];
lean_object* v_usedLetOnly_2355_ = _args[3];
lean_object* v_skipConstInApp_2356_ = _args[4];
lean_object* v_skipInstances_2357_ = _args[5];
lean_object* v_body_2358_ = _args[6];
lean_object* v_x_2359_ = _args[7];
lean_object* v___y_2360_ = _args[8];
lean_object* v___y_2361_ = _args[9];
lean_object* v___y_2362_ = _args[10];
lean_object* v___y_2363_ = _args[11];
lean_object* v___y_2364_ = _args[12];
lean_object* v___y_2365_ = _args[13];
lean_object* v___y_2366_ = _args[14];
lean_object* v___y_2367_ = _args[15];
lean_object* v___y_2368_ = _args[16];
_start:
{
uint8_t v_usedLetOnly_boxed_2369_; uint8_t v_skipConstInApp_boxed_2370_; uint8_t v_skipInstances_boxed_2371_; lean_object* v_res_2372_; 
v_usedLetOnly_boxed_2369_ = lean_unbox(v_usedLetOnly_2355_);
v_skipConstInApp_boxed_2370_ = lean_unbox(v_skipConstInApp_2356_);
v_skipInstances_boxed_2371_ = lean_unbox(v_skipInstances_2357_);
v_res_2372_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0(v_fvars_2352_, v_pre_2353_, v_post_2354_, v_usedLetOnly_boxed_2369_, v_skipConstInApp_boxed_2370_, v_skipInstances_boxed_2371_, v_body_2358_, v_x_2359_, v___y_2360_, v___y_2361_, v___y_2362_, v___y_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2362_);
lean_dec(v___y_2361_);
lean_dec(v___y_2360_);
return v_res_2372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(lean_object* v_pre_2373_, lean_object* v_post_2374_, uint8_t v_usedLetOnly_2375_, uint8_t v_skipConstInApp_2376_, uint8_t v_skipInstances_2377_, lean_object* v_e_2378_, lean_object* v_a_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_){
_start:
{
lean_object* v___x_2388_; 
lean_inc_ref(v_post_2374_);
lean_inc(v___y_2386_);
lean_inc_ref(v___y_2385_);
lean_inc(v___y_2384_);
lean_inc_ref(v___y_2383_);
lean_inc(v___y_2382_);
lean_inc_ref(v___y_2381_);
lean_inc(v___y_2380_);
lean_inc_ref(v_e_2378_);
v___x_2388_ = lean_apply_9(v_post_2374_, v_e_2378_, v___y_2380_, v___y_2381_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_, v___y_2386_, lean_box(0));
if (lean_obj_tag(v___x_2388_) == 0)
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2407_; 
v_a_2389_ = lean_ctor_get(v___x_2388_, 0);
v_isSharedCheck_2407_ = !lean_is_exclusive(v___x_2388_);
if (v_isSharedCheck_2407_ == 0)
{
v___x_2391_ = v___x_2388_;
v_isShared_2392_ = v_isSharedCheck_2407_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___x_2388_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2407_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
switch(lean_obj_tag(v_a_2389_))
{
case 0:
{
lean_object* v_e_2393_; lean_object* v___x_2395_; 
lean_dec_ref(v_e_2378_);
lean_dec_ref(v_post_2374_);
lean_dec_ref(v_pre_2373_);
v_e_2393_ = lean_ctor_get(v_a_2389_, 0);
lean_inc_ref(v_e_2393_);
lean_dec_ref_known(v_a_2389_, 1);
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v_e_2393_);
v___x_2395_ = v___x_2391_;
goto v_reusejp_2394_;
}
else
{
lean_object* v_reuseFailAlloc_2396_; 
v_reuseFailAlloc_2396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2396_, 0, v_e_2393_);
v___x_2395_ = v_reuseFailAlloc_2396_;
goto v_reusejp_2394_;
}
v_reusejp_2394_:
{
return v___x_2395_;
}
}
case 1:
{
lean_object* v_e_2397_; lean_object* v___x_2398_; 
lean_del_object(v___x_2391_);
lean_dec_ref(v_e_2378_);
v_e_2397_ = lean_ctor_get(v_a_2389_, 0);
lean_inc_ref(v_e_2397_);
lean_dec_ref_known(v_a_2389_, 1);
v___x_2398_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2373_, v_post_2374_, v_usedLetOnly_2375_, v_skipConstInApp_2376_, v_skipInstances_2377_, v_e_2397_, v_a_2379_, v___y_2380_, v___y_2381_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_, v___y_2386_);
return v___x_2398_;
}
default: 
{
lean_object* v_e_x3f_2399_; 
lean_dec_ref(v_post_2374_);
lean_dec_ref(v_pre_2373_);
v_e_x3f_2399_ = lean_ctor_get(v_a_2389_, 0);
lean_inc(v_e_x3f_2399_);
lean_dec_ref_known(v_a_2389_, 1);
if (lean_obj_tag(v_e_x3f_2399_) == 0)
{
lean_object* v___x_2401_; 
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v_e_2378_);
v___x_2401_ = v___x_2391_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v_e_2378_);
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
lean_object* v_val_2403_; lean_object* v___x_2405_; 
lean_dec_ref(v_e_2378_);
v_val_2403_ = lean_ctor_get(v_e_x3f_2399_, 0);
lean_inc(v_val_2403_);
lean_dec_ref_known(v_e_x3f_2399_, 1);
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v_val_2403_);
v___x_2405_ = v___x_2391_;
goto v_reusejp_2404_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v_val_2403_);
v___x_2405_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2404_;
}
v_reusejp_2404_:
{
return v___x_2405_;
}
}
}
}
}
}
else
{
lean_object* v_a_2408_; lean_object* v___x_2410_; uint8_t v_isShared_2411_; uint8_t v_isSharedCheck_2415_; 
lean_dec_ref(v_e_2378_);
lean_dec_ref(v_post_2374_);
lean_dec_ref(v_pre_2373_);
v_a_2408_ = lean_ctor_get(v___x_2388_, 0);
v_isSharedCheck_2415_ = !lean_is_exclusive(v___x_2388_);
if (v_isSharedCheck_2415_ == 0)
{
v___x_2410_ = v___x_2388_;
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
else
{
lean_inc(v_a_2408_);
lean_dec(v___x_2388_);
v___x_2410_ = lean_box(0);
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
v_resetjp_2409_:
{
lean_object* v___x_2413_; 
if (v_isShared_2411_ == 0)
{
v___x_2413_ = v___x_2410_;
goto v_reusejp_2412_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v_a_2408_);
v___x_2413_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2412_;
}
v_reusejp_2412_:
{
return v___x_2413_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10(lean_object* v_pre_2416_, lean_object* v_post_2417_, uint8_t v_usedLetOnly_2418_, uint8_t v_skipConstInApp_2419_, uint8_t v_skipInstances_2420_, lean_object* v_fvars_2421_, lean_object* v_e_2422_, lean_object* v_a_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_, lean_object* v___y_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_, lean_object* v___y_2430_){
_start:
{
if (lean_obj_tag(v_e_2422_) == 6)
{
lean_object* v_binderName_2432_; lean_object* v_binderType_2433_; lean_object* v_body_2434_; uint8_t v_binderInfo_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; 
v_binderName_2432_ = lean_ctor_get(v_e_2422_, 0);
lean_inc(v_binderName_2432_);
v_binderType_2433_ = lean_ctor_get(v_e_2422_, 1);
lean_inc_ref(v_binderType_2433_);
v_body_2434_ = lean_ctor_get(v_e_2422_, 2);
lean_inc_ref(v_body_2434_);
v_binderInfo_2435_ = lean_ctor_get_uint8(v_e_2422_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2422_, 3);
v___x_2436_ = lean_expr_instantiate_rev(v_binderType_2433_, v_fvars_2421_);
lean_dec_ref(v_binderType_2433_);
lean_inc_ref(v_post_2417_);
lean_inc_ref(v_pre_2416_);
v___x_2437_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2416_, v_post_2417_, v_usedLetOnly_2418_, v_skipConstInApp_2419_, v_skipInstances_2420_, v___x_2436_, v_a_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_);
if (lean_obj_tag(v___x_2437_) == 0)
{
lean_object* v_a_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___f_2442_; uint8_t v___x_2443_; lean_object* v___x_2444_; 
v_a_2438_ = lean_ctor_get(v___x_2437_, 0);
lean_inc(v_a_2438_);
lean_dec_ref_known(v___x_2437_, 1);
v___x_2439_ = lean_box(v_usedLetOnly_2418_);
v___x_2440_ = lean_box(v_skipConstInApp_2419_);
v___x_2441_ = lean_box(v_skipInstances_2420_);
v___f_2442_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___lam__0___boxed), 17, 7);
lean_closure_set(v___f_2442_, 0, v_fvars_2421_);
lean_closure_set(v___f_2442_, 1, v_pre_2416_);
lean_closure_set(v___f_2442_, 2, v_post_2417_);
lean_closure_set(v___f_2442_, 3, v___x_2439_);
lean_closure_set(v___f_2442_, 4, v___x_2440_);
lean_closure_set(v___f_2442_, 5, v___x_2441_);
lean_closure_set(v___f_2442_, 6, v_body_2434_);
v___x_2443_ = 0;
v___x_2444_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(v_binderName_2432_, v_binderInfo_2435_, v_a_2438_, v___f_2442_, v___x_2443_, v_a_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_);
return v___x_2444_;
}
else
{
lean_dec_ref(v_body_2434_);
lean_dec(v_binderName_2432_);
lean_dec_ref(v_fvars_2421_);
lean_dec_ref(v_post_2417_);
lean_dec_ref(v_pre_2416_);
return v___x_2437_;
}
}
else
{
lean_object* v___x_2445_; lean_object* v___x_2446_; 
v___x_2445_ = lean_expr_instantiate_rev(v_e_2422_, v_fvars_2421_);
lean_dec_ref(v_e_2422_);
lean_inc_ref(v_post_2417_);
lean_inc_ref(v_pre_2416_);
v___x_2446_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2416_, v_post_2417_, v_usedLetOnly_2418_, v_skipConstInApp_2419_, v_skipInstances_2420_, v___x_2445_, v_a_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_);
if (lean_obj_tag(v___x_2446_) == 0)
{
lean_object* v_a_2447_; uint8_t v___x_2448_; uint8_t v___x_2449_; uint8_t v___x_2450_; lean_object* v___x_2451_; 
v_a_2447_ = lean_ctor_get(v___x_2446_, 0);
lean_inc(v_a_2447_);
lean_dec_ref_known(v___x_2446_, 1);
v___x_2448_ = 0;
v___x_2449_ = 1;
v___x_2450_ = 1;
v___x_2451_ = l_Lean_Meta_mkLambdaFVars(v_fvars_2421_, v_a_2447_, v___x_2448_, v_usedLetOnly_2418_, v___x_2448_, v___x_2449_, v___x_2450_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_);
lean_dec_ref(v_fvars_2421_);
if (lean_obj_tag(v___x_2451_) == 0)
{
lean_object* v_a_2452_; lean_object* v___x_2453_; 
v_a_2452_ = lean_ctor_get(v___x_2451_, 0);
lean_inc(v_a_2452_);
lean_dec_ref_known(v___x_2451_, 1);
v___x_2453_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2416_, v_post_2417_, v_usedLetOnly_2418_, v_skipConstInApp_2419_, v_skipInstances_2420_, v_a_2452_, v_a_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_);
return v___x_2453_;
}
else
{
lean_dec_ref(v_post_2417_);
lean_dec_ref(v_pre_2416_);
return v___x_2451_;
}
}
else
{
lean_dec_ref(v_fvars_2421_);
lean_dec_ref(v_post_2417_);
lean_dec_ref(v_pre_2416_);
return v___x_2446_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0(lean_object* v_fvars_2454_, lean_object* v_pre_2455_, lean_object* v_post_2456_, uint8_t v_usedLetOnly_2457_, uint8_t v_skipConstInApp_2458_, uint8_t v_skipInstances_2459_, lean_object* v_body_2460_, lean_object* v_x_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_){
_start:
{
lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2471_ = lean_array_push(v_fvars_2454_, v_x_2461_);
v___x_2472_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11(v_pre_2455_, v_post_2456_, v_usedLetOnly_2457_, v_skipConstInApp_2458_, v_skipInstances_2459_, v___x_2471_, v_body_2460_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_, v___y_2466_, v___y_2467_, v___y_2468_, v___y_2469_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0___boxed(lean_object** _args){
lean_object* v_fvars_2473_ = _args[0];
lean_object* v_pre_2474_ = _args[1];
lean_object* v_post_2475_ = _args[2];
lean_object* v_usedLetOnly_2476_ = _args[3];
lean_object* v_skipConstInApp_2477_ = _args[4];
lean_object* v_skipInstances_2478_ = _args[5];
lean_object* v_body_2479_ = _args[6];
lean_object* v_x_2480_ = _args[7];
lean_object* v___y_2481_ = _args[8];
lean_object* v___y_2482_ = _args[9];
lean_object* v___y_2483_ = _args[10];
lean_object* v___y_2484_ = _args[11];
lean_object* v___y_2485_ = _args[12];
lean_object* v___y_2486_ = _args[13];
lean_object* v___y_2487_ = _args[14];
lean_object* v___y_2488_ = _args[15];
lean_object* v___y_2489_ = _args[16];
_start:
{
uint8_t v_usedLetOnly_boxed_2490_; uint8_t v_skipConstInApp_boxed_2491_; uint8_t v_skipInstances_boxed_2492_; lean_object* v_res_2493_; 
v_usedLetOnly_boxed_2490_ = lean_unbox(v_usedLetOnly_2476_);
v_skipConstInApp_boxed_2491_ = lean_unbox(v_skipConstInApp_2477_);
v_skipInstances_boxed_2492_ = lean_unbox(v_skipInstances_2478_);
v_res_2493_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0(v_fvars_2473_, v_pre_2474_, v_post_2475_, v_usedLetOnly_boxed_2490_, v_skipConstInApp_boxed_2491_, v_skipInstances_boxed_2492_, v_body_2479_, v_x_2480_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_);
lean_dec(v___y_2488_);
lean_dec_ref(v___y_2487_);
lean_dec(v___y_2486_);
lean_dec_ref(v___y_2485_);
lean_dec(v___y_2484_);
lean_dec_ref(v___y_2483_);
lean_dec(v___y_2482_);
lean_dec(v___y_2481_);
return v_res_2493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11(lean_object* v_pre_2494_, lean_object* v_post_2495_, uint8_t v_usedLetOnly_2496_, uint8_t v_skipConstInApp_2497_, uint8_t v_skipInstances_2498_, lean_object* v_fvars_2499_, lean_object* v_e_2500_, lean_object* v_a_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_){
_start:
{
if (lean_obj_tag(v_e_2500_) == 8)
{
lean_object* v_declName_2510_; lean_object* v_type_2511_; lean_object* v_value_2512_; lean_object* v_body_2513_; uint8_t v_nondep_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; 
v_declName_2510_ = lean_ctor_get(v_e_2500_, 0);
lean_inc(v_declName_2510_);
v_type_2511_ = lean_ctor_get(v_e_2500_, 1);
lean_inc_ref(v_type_2511_);
v_value_2512_ = lean_ctor_get(v_e_2500_, 2);
lean_inc_ref(v_value_2512_);
v_body_2513_ = lean_ctor_get(v_e_2500_, 3);
lean_inc_ref(v_body_2513_);
v_nondep_2514_ = lean_ctor_get_uint8(v_e_2500_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_2500_, 4);
v___x_2515_ = lean_expr_instantiate_rev(v_type_2511_, v_fvars_2499_);
lean_dec_ref(v_type_2511_);
lean_inc_ref(v_post_2495_);
lean_inc_ref(v_pre_2494_);
v___x_2516_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2494_, v_post_2495_, v_usedLetOnly_2496_, v_skipConstInApp_2497_, v_skipInstances_2498_, v___x_2515_, v_a_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
if (lean_obj_tag(v___x_2516_) == 0)
{
lean_object* v_a_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; 
v_a_2517_ = lean_ctor_get(v___x_2516_, 0);
lean_inc(v_a_2517_);
lean_dec_ref_known(v___x_2516_, 1);
v___x_2518_ = lean_expr_instantiate_rev(v_value_2512_, v_fvars_2499_);
lean_dec_ref(v_value_2512_);
lean_inc_ref(v_post_2495_);
lean_inc_ref(v_pre_2494_);
v___x_2519_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2494_, v_post_2495_, v_usedLetOnly_2496_, v_skipConstInApp_2497_, v_skipInstances_2498_, v___x_2518_, v_a_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
if (lean_obj_tag(v___x_2519_) == 0)
{
lean_object* v_a_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___f_2524_; uint8_t v___x_2525_; lean_object* v___x_2526_; 
v_a_2520_ = lean_ctor_get(v___x_2519_, 0);
lean_inc(v_a_2520_);
lean_dec_ref_known(v___x_2519_, 1);
v___x_2521_ = lean_box(v_usedLetOnly_2496_);
v___x_2522_ = lean_box(v_skipConstInApp_2497_);
v___x_2523_ = lean_box(v_skipInstances_2498_);
v___f_2524_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___lam__0___boxed), 17, 7);
lean_closure_set(v___f_2524_, 0, v_fvars_2499_);
lean_closure_set(v___f_2524_, 1, v_pre_2494_);
lean_closure_set(v___f_2524_, 2, v_post_2495_);
lean_closure_set(v___f_2524_, 3, v___x_2521_);
lean_closure_set(v___f_2524_, 4, v___x_2522_);
lean_closure_set(v___f_2524_, 5, v___x_2523_);
lean_closure_set(v___f_2524_, 6, v_body_2513_);
v___x_2525_ = 0;
v___x_2526_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg(v_declName_2510_, v_a_2517_, v_a_2520_, v___f_2524_, v_nondep_2514_, v___x_2525_, v_a_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
return v___x_2526_;
}
else
{
lean_dec(v_a_2517_);
lean_dec_ref(v_body_2513_);
lean_dec(v_declName_2510_);
lean_dec_ref(v_fvars_2499_);
lean_dec_ref(v_post_2495_);
lean_dec_ref(v_pre_2494_);
return v___x_2519_;
}
}
else
{
lean_dec_ref(v_body_2513_);
lean_dec_ref(v_value_2512_);
lean_dec(v_declName_2510_);
lean_dec_ref(v_fvars_2499_);
lean_dec_ref(v_post_2495_);
lean_dec_ref(v_pre_2494_);
return v___x_2516_;
}
}
else
{
lean_object* v___x_2527_; lean_object* v___x_2528_; 
v___x_2527_ = lean_expr_instantiate_rev(v_e_2500_, v_fvars_2499_);
lean_dec_ref(v_e_2500_);
lean_inc_ref(v_post_2495_);
lean_inc_ref(v_pre_2494_);
v___x_2528_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2494_, v_post_2495_, v_usedLetOnly_2496_, v_skipConstInApp_2497_, v_skipInstances_2498_, v___x_2527_, v_a_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
if (lean_obj_tag(v___x_2528_) == 0)
{
lean_object* v_a_2529_; uint8_t v___x_2530_; uint8_t v___x_2531_; lean_object* v___x_2532_; 
v_a_2529_ = lean_ctor_get(v___x_2528_, 0);
lean_inc(v_a_2529_);
lean_dec_ref_known(v___x_2528_, 1);
v___x_2530_ = 0;
v___x_2531_ = 1;
v___x_2532_ = l_Lean_Meta_mkLetFVars(v_fvars_2499_, v_a_2529_, v_usedLetOnly_2496_, v___x_2530_, v___x_2531_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
lean_dec_ref(v_fvars_2499_);
if (lean_obj_tag(v___x_2532_) == 0)
{
lean_object* v_a_2533_; lean_object* v___x_2534_; 
v_a_2533_ = lean_ctor_get(v___x_2532_, 0);
lean_inc(v_a_2533_);
lean_dec_ref_known(v___x_2532_, 1);
v___x_2534_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2494_, v_post_2495_, v_usedLetOnly_2496_, v_skipConstInApp_2497_, v_skipInstances_2498_, v_a_2533_, v_a_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
return v___x_2534_;
}
else
{
lean_dec_ref(v_post_2495_);
lean_dec_ref(v_pre_2494_);
return v___x_2532_;
}
}
else
{
lean_dec_ref(v_fvars_2499_);
lean_dec_ref(v_post_2495_);
lean_dec_ref(v_pre_2494_);
return v___x_2528_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5(lean_object* v_pre_2535_, lean_object* v_post_2536_, uint8_t v_usedLetOnly_2537_, uint8_t v_skipConstInApp_2538_, uint8_t v_skipInstances_2539_, size_t v_sz_2540_, size_t v_i_2541_, lean_object* v_bs_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_){
_start:
{
uint8_t v___x_2552_; 
v___x_2552_ = lean_usize_dec_lt(v_i_2541_, v_sz_2540_);
if (v___x_2552_ == 0)
{
lean_object* v___x_2553_; 
lean_dec_ref(v_post_2536_);
lean_dec_ref(v_pre_2535_);
v___x_2553_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2553_, 0, v_bs_2542_);
return v___x_2553_;
}
else
{
lean_object* v_v_2554_; lean_object* v___x_2555_; 
v_v_2554_ = lean_array_uget_borrowed(v_bs_2542_, v_i_2541_);
lean_inc(v_v_2554_);
lean_inc_ref(v_post_2536_);
lean_inc_ref(v_pre_2535_);
v___x_2555_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2535_, v_post_2536_, v_usedLetOnly_2537_, v_skipConstInApp_2538_, v_skipInstances_2539_, v_v_2554_, v___y_2543_, v___y_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_, v___y_2549_, v___y_2550_);
if (lean_obj_tag(v___x_2555_) == 0)
{
lean_object* v_a_2556_; lean_object* v___x_2557_; lean_object* v_bs_x27_2558_; size_t v___x_2559_; size_t v___x_2560_; lean_object* v___x_2561_; 
v_a_2556_ = lean_ctor_get(v___x_2555_, 0);
lean_inc(v_a_2556_);
lean_dec_ref_known(v___x_2555_, 1);
v___x_2557_ = lean_unsigned_to_nat(0u);
v_bs_x27_2558_ = lean_array_uset(v_bs_2542_, v_i_2541_, v___x_2557_);
v___x_2559_ = ((size_t)1ULL);
v___x_2560_ = lean_usize_add(v_i_2541_, v___x_2559_);
v___x_2561_ = lean_array_uset(v_bs_x27_2558_, v_i_2541_, v_a_2556_);
v_i_2541_ = v___x_2560_;
v_bs_2542_ = v___x_2561_;
goto _start;
}
else
{
lean_object* v_a_2563_; lean_object* v___x_2565_; uint8_t v_isShared_2566_; uint8_t v_isSharedCheck_2570_; 
lean_dec_ref(v_bs_2542_);
lean_dec_ref(v_post_2536_);
lean_dec_ref(v_pre_2535_);
v_a_2563_ = lean_ctor_get(v___x_2555_, 0);
v_isSharedCheck_2570_ = !lean_is_exclusive(v___x_2555_);
if (v_isSharedCheck_2570_ == 0)
{
v___x_2565_ = v___x_2555_;
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
else
{
lean_inc(v_a_2563_);
lean_dec(v___x_2555_);
v___x_2565_ = lean_box(0);
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
v_resetjp_2564_:
{
lean_object* v___x_2568_; 
if (v_isShared_2566_ == 0)
{
v___x_2568_ = v___x_2565_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2569_; 
v_reuseFailAlloc_2569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2569_, 0, v_a_2563_);
v___x_2568_ = v_reuseFailAlloc_2569_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
return v___x_2568_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0(lean_object* v_pre_2571_, lean_object* v_post_2572_, uint8_t v_usedLetOnly_2573_, uint8_t v_skipConstInApp_2574_, uint8_t v_skipInstances_2575_, lean_object* v___x_2576_, lean_object* v___y_2577_, lean_object* v_b_2578_, lean_object* v_a_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v___x_2588_; 
v___x_2588_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2571_, v_post_2572_, v_usedLetOnly_2573_, v_skipConstInApp_2574_, v_skipInstances_2575_, v___x_2576_, v___y_2577_, v___y_2580_, v___y_2581_, v___y_2582_, v___y_2583_, v___y_2584_, v___y_2585_, v___y_2586_);
if (lean_obj_tag(v___x_2588_) == 0)
{
lean_object* v_a_2589_; lean_object* v___x_2591_; uint8_t v_isShared_2592_; uint8_t v_isSharedCheck_2598_; 
v_a_2589_ = lean_ctor_get(v___x_2588_, 0);
v_isSharedCheck_2598_ = !lean_is_exclusive(v___x_2588_);
if (v_isSharedCheck_2598_ == 0)
{
v___x_2591_ = v___x_2588_;
v_isShared_2592_ = v_isSharedCheck_2598_;
goto v_resetjp_2590_;
}
else
{
lean_inc(v_a_2589_);
lean_dec(v___x_2588_);
v___x_2591_ = lean_box(0);
v_isShared_2592_ = v_isSharedCheck_2598_;
goto v_resetjp_2590_;
}
v_resetjp_2590_:
{
lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2596_; 
v___x_2593_ = lean_array_fset(v_b_2578_, v_a_2579_, v_a_2589_);
v___x_2594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2594_, 0, v___x_2593_);
if (v_isShared_2592_ == 0)
{
lean_ctor_set(v___x_2591_, 0, v___x_2594_);
v___x_2596_ = v___x_2591_;
goto v_reusejp_2595_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v___x_2594_);
v___x_2596_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2595_;
}
v_reusejp_2595_:
{
return v___x_2596_;
}
}
}
else
{
lean_object* v_a_2599_; lean_object* v___x_2601_; uint8_t v_isShared_2602_; uint8_t v_isSharedCheck_2606_; 
lean_dec_ref(v_b_2578_);
v_a_2599_ = lean_ctor_get(v___x_2588_, 0);
v_isSharedCheck_2606_ = !lean_is_exclusive(v___x_2588_);
if (v_isSharedCheck_2606_ == 0)
{
v___x_2601_ = v___x_2588_;
v_isShared_2602_ = v_isSharedCheck_2606_;
goto v_resetjp_2600_;
}
else
{
lean_inc(v_a_2599_);
lean_dec(v___x_2588_);
v___x_2601_ = lean_box(0);
v_isShared_2602_ = v_isSharedCheck_2606_;
goto v_resetjp_2600_;
}
v_resetjp_2600_:
{
lean_object* v___x_2604_; 
if (v_isShared_2602_ == 0)
{
v___x_2604_ = v___x_2601_;
goto v_reusejp_2603_;
}
else
{
lean_object* v_reuseFailAlloc_2605_; 
v_reuseFailAlloc_2605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2605_, 0, v_a_2599_);
v___x_2604_ = v_reuseFailAlloc_2605_;
goto v_reusejp_2603_;
}
v_reusejp_2603_:
{
return v___x_2604_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0___boxed(lean_object** _args){
lean_object* v_pre_2607_ = _args[0];
lean_object* v_post_2608_ = _args[1];
lean_object* v_usedLetOnly_2609_ = _args[2];
lean_object* v_skipConstInApp_2610_ = _args[3];
lean_object* v_skipInstances_2611_ = _args[4];
lean_object* v___x_2612_ = _args[5];
lean_object* v___y_2613_ = _args[6];
lean_object* v_b_2614_ = _args[7];
lean_object* v_a_2615_ = _args[8];
lean_object* v___y_2616_ = _args[9];
lean_object* v___y_2617_ = _args[10];
lean_object* v___y_2618_ = _args[11];
lean_object* v___y_2619_ = _args[12];
lean_object* v___y_2620_ = _args[13];
lean_object* v___y_2621_ = _args[14];
lean_object* v___y_2622_ = _args[15];
lean_object* v___y_2623_ = _args[16];
_start:
{
uint8_t v_usedLetOnly_boxed_2624_; uint8_t v_skipConstInApp_boxed_2625_; uint8_t v_skipInstances_boxed_2626_; lean_object* v_res_2627_; 
v_usedLetOnly_boxed_2624_ = lean_unbox(v_usedLetOnly_2609_);
v_skipConstInApp_boxed_2625_ = lean_unbox(v_skipConstInApp_2610_);
v_skipInstances_boxed_2626_ = lean_unbox(v_skipInstances_2611_);
v_res_2627_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0(v_pre_2607_, v_post_2608_, v_usedLetOnly_boxed_2624_, v_skipConstInApp_boxed_2625_, v_skipInstances_boxed_2626_, v___x_2612_, v___y_2613_, v_b_2614_, v_a_2615_, v___y_2616_, v___y_2617_, v___y_2618_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_);
lean_dec(v___y_2622_);
lean_dec_ref(v___y_2621_);
lean_dec(v___y_2620_);
lean_dec_ref(v___y_2619_);
lean_dec(v___y_2618_);
lean_dec_ref(v___y_2617_);
lean_dec(v___y_2616_);
lean_dec(v_a_2615_);
lean_dec(v___y_2613_);
return v_res_2627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg(lean_object* v_upperBound_2628_, lean_object* v___x_2629_, lean_object* v_pre_2630_, lean_object* v_post_2631_, uint8_t v_usedLetOnly_2632_, uint8_t v_skipConstInApp_2633_, uint8_t v_skipInstances_2634_, lean_object* v_a_2635_, lean_object* v_b_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_){
_start:
{
lean_object* v___y_2647_; uint8_t v___x_2670_; 
v___x_2670_ = lean_nat_dec_lt(v_a_2635_, v_upperBound_2628_);
if (v___x_2670_ == 0)
{
lean_object* v___x_2671_; 
lean_dec(v_a_2635_);
lean_dec_ref(v_post_2631_);
lean_dec_ref(v_pre_2630_);
v___x_2671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2671_, 0, v_b_2636_);
return v___x_2671_;
}
else
{
lean_object* v___x_2672_; lean_object* v___x_2673_; uint8_t v___x_2674_; 
v___x_2672_ = lean_array_fget_borrowed(v_b_2636_, v_a_2635_);
v___x_2673_ = lean_array_get_size(v___x_2629_);
v___x_2674_ = lean_nat_dec_lt(v_a_2635_, v___x_2673_);
if (v___x_2674_ == 0)
{
lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___f_2678_; 
lean_inc(v___x_2672_);
v___x_2675_ = lean_box(v_usedLetOnly_2632_);
v___x_2676_ = lean_box(v_skipConstInApp_2633_);
v___x_2677_ = lean_box(v_skipInstances_2634_);
lean_inc(v_a_2635_);
lean_inc(v___y_2637_);
lean_inc_ref(v_post_2631_);
lean_inc_ref(v_pre_2630_);
v___f_2678_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0___boxed), 17, 9);
lean_closure_set(v___f_2678_, 0, v_pre_2630_);
lean_closure_set(v___f_2678_, 1, v_post_2631_);
lean_closure_set(v___f_2678_, 2, v___x_2675_);
lean_closure_set(v___f_2678_, 3, v___x_2676_);
lean_closure_set(v___f_2678_, 4, v___x_2677_);
lean_closure_set(v___f_2678_, 5, v___x_2672_);
lean_closure_set(v___f_2678_, 6, v___y_2637_);
lean_closure_set(v___f_2678_, 7, v_b_2636_);
lean_closure_set(v___f_2678_, 8, v_a_2635_);
v___y_2647_ = v___f_2678_;
goto v___jp_2646_;
}
else
{
lean_object* v___x_2679_; uint8_t v_isInstance_2680_; 
v___x_2679_ = lean_array_fget_borrowed(v___x_2629_, v_a_2635_);
v_isInstance_2680_ = lean_ctor_get_uint8(v___x_2679_, sizeof(void*)*1 + 4);
if (v_isInstance_2680_ == 0)
{
lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___f_2684_; 
lean_inc(v___x_2672_);
v___x_2681_ = lean_box(v_usedLetOnly_2632_);
v___x_2682_ = lean_box(v_skipConstInApp_2633_);
v___x_2683_ = lean_box(v_skipInstances_2634_);
lean_inc(v_a_2635_);
lean_inc(v___y_2637_);
lean_inc_ref(v_post_2631_);
lean_inc_ref(v_pre_2630_);
v___f_2684_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__0___boxed), 17, 9);
lean_closure_set(v___f_2684_, 0, v_pre_2630_);
lean_closure_set(v___f_2684_, 1, v_post_2631_);
lean_closure_set(v___f_2684_, 2, v___x_2681_);
lean_closure_set(v___f_2684_, 3, v___x_2682_);
lean_closure_set(v___f_2684_, 4, v___x_2683_);
lean_closure_set(v___f_2684_, 5, v___x_2672_);
lean_closure_set(v___f_2684_, 6, v___y_2637_);
lean_closure_set(v___f_2684_, 7, v_b_2636_);
lean_closure_set(v___f_2684_, 8, v_a_2635_);
v___y_2647_ = v___f_2684_;
goto v___jp_2646_;
}
else
{
lean_object* v___x_2685_; lean_object* v___f_2686_; 
v___x_2685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2685_, 0, v_b_2636_);
v___f_2686_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___lam__2___boxed), 9, 1);
lean_closure_set(v___f_2686_, 0, v___x_2685_);
v___y_2647_ = v___f_2686_;
goto v___jp_2646_;
}
}
}
v___jp_2646_:
{
lean_object* v___x_2648_; 
lean_inc(v___y_2644_);
lean_inc_ref(v___y_2643_);
lean_inc(v___y_2642_);
lean_inc_ref(v___y_2641_);
lean_inc(v___y_2640_);
lean_inc_ref(v___y_2639_);
lean_inc(v___y_2638_);
v___x_2648_ = lean_apply_8(v___y_2647_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, lean_box(0));
if (lean_obj_tag(v___x_2648_) == 0)
{
lean_object* v_a_2649_; lean_object* v___x_2651_; uint8_t v_isShared_2652_; uint8_t v_isSharedCheck_2661_; 
v_a_2649_ = lean_ctor_get(v___x_2648_, 0);
v_isSharedCheck_2661_ = !lean_is_exclusive(v___x_2648_);
if (v_isSharedCheck_2661_ == 0)
{
v___x_2651_ = v___x_2648_;
v_isShared_2652_ = v_isSharedCheck_2661_;
goto v_resetjp_2650_;
}
else
{
lean_inc(v_a_2649_);
lean_dec(v___x_2648_);
v___x_2651_ = lean_box(0);
v_isShared_2652_ = v_isSharedCheck_2661_;
goto v_resetjp_2650_;
}
v_resetjp_2650_:
{
if (lean_obj_tag(v_a_2649_) == 0)
{
lean_object* v_a_2653_; lean_object* v___x_2655_; 
lean_dec(v_a_2635_);
lean_dec_ref(v_post_2631_);
lean_dec_ref(v_pre_2630_);
v_a_2653_ = lean_ctor_get(v_a_2649_, 0);
lean_inc(v_a_2653_);
lean_dec_ref_known(v_a_2649_, 1);
if (v_isShared_2652_ == 0)
{
lean_ctor_set(v___x_2651_, 0, v_a_2653_);
v___x_2655_ = v___x_2651_;
goto v_reusejp_2654_;
}
else
{
lean_object* v_reuseFailAlloc_2656_; 
v_reuseFailAlloc_2656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2656_, 0, v_a_2653_);
v___x_2655_ = v_reuseFailAlloc_2656_;
goto v_reusejp_2654_;
}
v_reusejp_2654_:
{
return v___x_2655_;
}
}
else
{
lean_object* v_a_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; 
lean_del_object(v___x_2651_);
v_a_2657_ = lean_ctor_get(v_a_2649_, 0);
lean_inc(v_a_2657_);
lean_dec_ref_known(v_a_2649_, 1);
v___x_2658_ = lean_unsigned_to_nat(1u);
v___x_2659_ = lean_nat_add(v_a_2635_, v___x_2658_);
lean_dec(v_a_2635_);
v_a_2635_ = v___x_2659_;
v_b_2636_ = v_a_2657_;
goto _start;
}
}
}
else
{
lean_object* v_a_2662_; lean_object* v___x_2664_; uint8_t v_isShared_2665_; uint8_t v_isSharedCheck_2669_; 
lean_dec(v_a_2635_);
lean_dec_ref(v_post_2631_);
lean_dec_ref(v_pre_2630_);
v_a_2662_ = lean_ctor_get(v___x_2648_, 0);
v_isSharedCheck_2669_ = !lean_is_exclusive(v___x_2648_);
if (v_isSharedCheck_2669_ == 0)
{
v___x_2664_ = v___x_2648_;
v_isShared_2665_ = v_isSharedCheck_2669_;
goto v_resetjp_2663_;
}
else
{
lean_inc(v_a_2662_);
lean_dec(v___x_2648_);
v___x_2664_ = lean_box(0);
v_isShared_2665_ = v_isSharedCheck_2669_;
goto v_resetjp_2663_;
}
v_resetjp_2663_:
{
lean_object* v___x_2667_; 
if (v_isShared_2665_ == 0)
{
v___x_2667_ = v___x_2664_;
goto v_reusejp_2666_;
}
else
{
lean_object* v_reuseFailAlloc_2668_; 
v_reuseFailAlloc_2668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2668_, 0, v_a_2662_);
v___x_2667_ = v_reuseFailAlloc_2668_;
goto v_reusejp_2666_;
}
v_reusejp_2666_:
{
return v___x_2667_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12(uint8_t v_skipInstances_2687_, lean_object* v_pre_2688_, lean_object* v_post_2689_, uint8_t v_usedLetOnly_2690_, uint8_t v_skipConstInApp_2691_, lean_object* v_x_2692_, lean_object* v_x_2693_, lean_object* v_x_2694_, lean_object* v___y_2695_, lean_object* v___y_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_, lean_object* v___y_2700_, lean_object* v___y_2701_, lean_object* v___y_2702_){
_start:
{
lean_object* v_f_2705_; lean_object* v___y_2706_; lean_object* v___y_2707_; lean_object* v___y_2708_; lean_object* v___y_2709_; lean_object* v___y_2710_; lean_object* v___y_2711_; lean_object* v___y_2712_; lean_object* v___y_2713_; 
if (lean_obj_tag(v_x_2692_) == 5)
{
lean_object* v_fn_2756_; lean_object* v_arg_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; 
v_fn_2756_ = lean_ctor_get(v_x_2692_, 0);
lean_inc_ref(v_fn_2756_);
v_arg_2757_ = lean_ctor_get(v_x_2692_, 1);
lean_inc_ref(v_arg_2757_);
lean_dec_ref_known(v_x_2692_, 2);
v___x_2758_ = lean_array_set(v_x_2693_, v_x_2694_, v_arg_2757_);
v___x_2759_ = lean_unsigned_to_nat(1u);
v___x_2760_ = lean_nat_sub(v_x_2694_, v___x_2759_);
lean_dec(v_x_2694_);
v_x_2692_ = v_fn_2756_;
v_x_2693_ = v___x_2758_;
v_x_2694_ = v___x_2760_;
goto _start;
}
else
{
lean_dec(v_x_2694_);
if (v_skipConstInApp_2691_ == 0)
{
goto v___jp_2753_;
}
else
{
uint8_t v___x_2762_; 
v___x_2762_ = l_Lean_Expr_isConst(v_x_2692_);
if (v___x_2762_ == 0)
{
goto v___jp_2753_;
}
else
{
v_f_2705_ = v_x_2692_;
v___y_2706_ = v___y_2695_;
v___y_2707_ = v___y_2696_;
v___y_2708_ = v___y_2697_;
v___y_2709_ = v___y_2698_;
v___y_2710_ = v___y_2699_;
v___y_2711_ = v___y_2700_;
v___y_2712_ = v___y_2701_;
v___y_2713_ = v___y_2702_;
goto v___jp_2704_;
}
}
}
v___jp_2704_:
{
if (v_skipInstances_2687_ == 0)
{
size_t v_sz_2714_; size_t v___x_2715_; lean_object* v___x_2716_; 
v_sz_2714_ = lean_array_size(v_x_2693_);
v___x_2715_ = ((size_t)0ULL);
lean_inc_ref(v_post_2689_);
lean_inc_ref(v_pre_2688_);
v___x_2716_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5(v_pre_2688_, v_post_2689_, v_usedLetOnly_2690_, v_skipConstInApp_2691_, v_skipInstances_2687_, v_sz_2714_, v___x_2715_, v_x_2693_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
if (lean_obj_tag(v___x_2716_) == 0)
{
lean_object* v_a_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; 
v_a_2717_ = lean_ctor_get(v___x_2716_, 0);
lean_inc(v_a_2717_);
lean_dec_ref_known(v___x_2716_, 1);
v___x_2718_ = l_Lean_mkAppN(v_f_2705_, v_a_2717_);
lean_dec(v_a_2717_);
v___x_2719_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2688_, v_post_2689_, v_usedLetOnly_2690_, v_skipConstInApp_2691_, v_skipInstances_2687_, v___x_2718_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
return v___x_2719_;
}
else
{
lean_object* v_a_2720_; lean_object* v___x_2722_; uint8_t v_isShared_2723_; uint8_t v_isSharedCheck_2727_; 
lean_dec_ref(v_f_2705_);
lean_dec_ref(v_post_2689_);
lean_dec_ref(v_pre_2688_);
v_a_2720_ = lean_ctor_get(v___x_2716_, 0);
v_isSharedCheck_2727_ = !lean_is_exclusive(v___x_2716_);
if (v_isSharedCheck_2727_ == 0)
{
v___x_2722_ = v___x_2716_;
v_isShared_2723_ = v_isSharedCheck_2727_;
goto v_resetjp_2721_;
}
else
{
lean_inc(v_a_2720_);
lean_dec(v___x_2716_);
v___x_2722_ = lean_box(0);
v_isShared_2723_ = v_isSharedCheck_2727_;
goto v_resetjp_2721_;
}
v_resetjp_2721_:
{
lean_object* v___x_2725_; 
if (v_isShared_2723_ == 0)
{
v___x_2725_ = v___x_2722_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2726_; 
v_reuseFailAlloc_2726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2726_, 0, v_a_2720_);
v___x_2725_ = v_reuseFailAlloc_2726_;
goto v_reusejp_2724_;
}
v_reusejp_2724_:
{
return v___x_2725_;
}
}
}
}
else
{
lean_object* v___x_2728_; lean_object* v___x_2729_; 
v___x_2728_ = lean_array_get_size(v_x_2693_);
lean_inc_ref(v_f_2705_);
v___x_2729_ = l_Lean_Meta_getFunInfoNArgs(v_f_2705_, v___x_2728_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
if (lean_obj_tag(v___x_2729_) == 0)
{
lean_object* v_a_2730_; lean_object* v_paramInfo_2731_; lean_object* v___x_2732_; lean_object* v___x_2733_; 
v_a_2730_ = lean_ctor_get(v___x_2729_, 0);
lean_inc(v_a_2730_);
lean_dec_ref_known(v___x_2729_, 1);
v_paramInfo_2731_ = lean_ctor_get(v_a_2730_, 0);
lean_inc_ref(v_paramInfo_2731_);
lean_dec(v_a_2730_);
v___x_2732_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_post_2689_);
lean_inc_ref(v_pre_2688_);
v___x_2733_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg(v___x_2728_, v_paramInfo_2731_, v_pre_2688_, v_post_2689_, v_usedLetOnly_2690_, v_skipConstInApp_2691_, v_skipInstances_2687_, v___x_2732_, v_x_2693_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
lean_dec_ref(v_paramInfo_2731_);
if (lean_obj_tag(v___x_2733_) == 0)
{
lean_object* v_a_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; 
v_a_2734_ = lean_ctor_get(v___x_2733_, 0);
lean_inc(v_a_2734_);
lean_dec_ref_known(v___x_2733_, 1);
v___x_2735_ = l_Lean_mkAppN(v_f_2705_, v_a_2734_);
lean_dec(v_a_2734_);
v___x_2736_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2688_, v_post_2689_, v_usedLetOnly_2690_, v_skipConstInApp_2691_, v_skipInstances_2687_, v___x_2735_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
return v___x_2736_;
}
else
{
lean_object* v_a_2737_; lean_object* v___x_2739_; uint8_t v_isShared_2740_; uint8_t v_isSharedCheck_2744_; 
lean_dec_ref(v_f_2705_);
lean_dec_ref(v_post_2689_);
lean_dec_ref(v_pre_2688_);
v_a_2737_ = lean_ctor_get(v___x_2733_, 0);
v_isSharedCheck_2744_ = !lean_is_exclusive(v___x_2733_);
if (v_isSharedCheck_2744_ == 0)
{
v___x_2739_ = v___x_2733_;
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
else
{
lean_inc(v_a_2737_);
lean_dec(v___x_2733_);
v___x_2739_ = lean_box(0);
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
v_resetjp_2738_:
{
lean_object* v___x_2742_; 
if (v_isShared_2740_ == 0)
{
v___x_2742_ = v___x_2739_;
goto v_reusejp_2741_;
}
else
{
lean_object* v_reuseFailAlloc_2743_; 
v_reuseFailAlloc_2743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2743_, 0, v_a_2737_);
v___x_2742_ = v_reuseFailAlloc_2743_;
goto v_reusejp_2741_;
}
v_reusejp_2741_:
{
return v___x_2742_;
}
}
}
}
else
{
lean_object* v_a_2745_; lean_object* v___x_2747_; uint8_t v_isShared_2748_; uint8_t v_isSharedCheck_2752_; 
lean_dec_ref(v_f_2705_);
lean_dec_ref(v_x_2693_);
lean_dec_ref(v_post_2689_);
lean_dec_ref(v_pre_2688_);
v_a_2745_ = lean_ctor_get(v___x_2729_, 0);
v_isSharedCheck_2752_ = !lean_is_exclusive(v___x_2729_);
if (v_isSharedCheck_2752_ == 0)
{
v___x_2747_ = v___x_2729_;
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
else
{
lean_inc(v_a_2745_);
lean_dec(v___x_2729_);
v___x_2747_ = lean_box(0);
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
v_resetjp_2746_:
{
lean_object* v___x_2750_; 
if (v_isShared_2748_ == 0)
{
v___x_2750_ = v___x_2747_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2751_; 
v_reuseFailAlloc_2751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2751_, 0, v_a_2745_);
v___x_2750_ = v_reuseFailAlloc_2751_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
return v___x_2750_;
}
}
}
}
}
v___jp_2753_:
{
lean_object* v___x_2754_; 
lean_inc_ref(v_post_2689_);
lean_inc_ref(v_pre_2688_);
v___x_2754_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2688_, v_post_2689_, v_usedLetOnly_2690_, v_skipConstInApp_2691_, v_skipInstances_2687_, v_x_2692_, v___y_2695_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_, v___y_2700_, v___y_2701_, v___y_2702_);
if (lean_obj_tag(v___x_2754_) == 0)
{
lean_object* v_a_2755_; 
v_a_2755_ = lean_ctor_get(v___x_2754_, 0);
lean_inc(v_a_2755_);
lean_dec_ref_known(v___x_2754_, 1);
v_f_2705_ = v_a_2755_;
v___y_2706_ = v___y_2695_;
v___y_2707_ = v___y_2696_;
v___y_2708_ = v___y_2697_;
v___y_2709_ = v___y_2698_;
v___y_2710_ = v___y_2699_;
v___y_2711_ = v___y_2700_;
v___y_2712_ = v___y_2701_;
v___y_2713_ = v___y_2702_;
goto v___jp_2704_;
}
else
{
lean_dec_ref(v_x_2693_);
lean_dec_ref(v_post_2689_);
lean_dec_ref(v_pre_2688_);
return v___x_2754_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1(lean_object* v___x_2763_, lean_object* v_pre_2764_, lean_object* v_e_2765_, lean_object* v_post_2766_, uint8_t v_usedLetOnly_2767_, uint8_t v_skipConstInApp_2768_, uint8_t v_skipInstances_2769_, lean_object* v___y_2770_, lean_object* v___y_2771_, lean_object* v___y_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_){
_start:
{
lean_object* v___x_2779_; 
v___x_2779_ = l_Lean_Core_checkSystem(v___x_2763_, v___y_2776_, v___y_2777_);
if (lean_obj_tag(v___x_2779_) == 0)
{
lean_object* v___x_2780_; 
lean_dec_ref_known(v___x_2779_, 1);
lean_inc_ref(v_pre_2764_);
lean_inc(v___y_2777_);
lean_inc_ref(v___y_2776_);
lean_inc(v___y_2775_);
lean_inc_ref(v___y_2774_);
lean_inc(v___y_2773_);
lean_inc_ref(v___y_2772_);
lean_inc(v___y_2771_);
lean_inc_ref(v_e_2765_);
v___x_2780_ = lean_apply_9(v_pre_2764_, v_e_2765_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_, lean_box(0));
if (lean_obj_tag(v___x_2780_) == 0)
{
lean_object* v_a_2781_; lean_object* v___x_2783_; uint8_t v_isShared_2784_; uint8_t v_isSharedCheck_2829_; 
v_a_2781_ = lean_ctor_get(v___x_2780_, 0);
v_isSharedCheck_2829_ = !lean_is_exclusive(v___x_2780_);
if (v_isSharedCheck_2829_ == 0)
{
v___x_2783_ = v___x_2780_;
v_isShared_2784_ = v_isSharedCheck_2829_;
goto v_resetjp_2782_;
}
else
{
lean_inc(v_a_2781_);
lean_dec(v___x_2780_);
v___x_2783_ = lean_box(0);
v_isShared_2784_ = v_isSharedCheck_2829_;
goto v_resetjp_2782_;
}
v_resetjp_2782_:
{
lean_object* v___y_2786_; 
switch(lean_obj_tag(v_a_2781_))
{
case 0:
{
lean_object* v_e_2821_; lean_object* v___x_2823_; 
lean_dec_ref(v_post_2766_);
lean_dec_ref(v_e_2765_);
lean_dec_ref(v_pre_2764_);
v_e_2821_ = lean_ctor_get(v_a_2781_, 0);
lean_inc_ref(v_e_2821_);
lean_dec_ref_known(v_a_2781_, 1);
if (v_isShared_2784_ == 0)
{
lean_ctor_set(v___x_2783_, 0, v_e_2821_);
v___x_2823_ = v___x_2783_;
goto v_reusejp_2822_;
}
else
{
lean_object* v_reuseFailAlloc_2824_; 
v_reuseFailAlloc_2824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2824_, 0, v_e_2821_);
v___x_2823_ = v_reuseFailAlloc_2824_;
goto v_reusejp_2822_;
}
v_reusejp_2822_:
{
return v___x_2823_;
}
}
case 1:
{
lean_object* v_e_2825_; lean_object* v___x_2826_; 
lean_del_object(v___x_2783_);
lean_dec_ref(v_e_2765_);
v_e_2825_ = lean_ctor_get(v_a_2781_, 0);
lean_inc_ref(v_e_2825_);
lean_dec_ref_known(v_a_2781_, 1);
v___x_2826_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v_e_2825_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2826_;
}
default: 
{
lean_object* v_e_x3f_2827_; 
lean_del_object(v___x_2783_);
v_e_x3f_2827_ = lean_ctor_get(v_a_2781_, 0);
lean_inc(v_e_x3f_2827_);
lean_dec_ref_known(v_a_2781_, 1);
if (lean_obj_tag(v_e_x3f_2827_) == 0)
{
v___y_2786_ = v_e_2765_;
goto v___jp_2785_;
}
else
{
lean_object* v_val_2828_; 
lean_dec_ref(v_e_2765_);
v_val_2828_ = lean_ctor_get(v_e_x3f_2827_, 0);
lean_inc(v_val_2828_);
lean_dec_ref_known(v_e_x3f_2827_, 1);
v___y_2786_ = v_val_2828_;
goto v___jp_2785_;
}
}
}
v___jp_2785_:
{
switch(lean_obj_tag(v___y_2786_))
{
case 7:
{
lean_object* v___x_2787_; lean_object* v___x_2788_; 
v___x_2787_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0));
v___x_2788_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___x_2787_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2788_;
}
case 6:
{
lean_object* v___x_2789_; lean_object* v___x_2790_; 
v___x_2789_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0));
v___x_2790_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___x_2789_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2790_;
}
case 8:
{
lean_object* v___x_2791_; lean_object* v___x_2792_; 
v___x_2791_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___closed__0));
v___x_2792_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___x_2791_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2792_;
}
case 5:
{
lean_object* v_dummy_2793_; lean_object* v_nargs_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; 
v_dummy_2793_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0);
v_nargs_2794_ = l_Lean_Expr_getAppNumArgs(v___y_2786_);
lean_inc(v_nargs_2794_);
v___x_2795_ = lean_mk_array(v_nargs_2794_, v_dummy_2793_);
v___x_2796_ = lean_unsigned_to_nat(1u);
v___x_2797_ = lean_nat_sub(v_nargs_2794_, v___x_2796_);
lean_dec(v_nargs_2794_);
v___x_2798_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12(v_skipInstances_2769_, v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v___y_2786_, v___x_2795_, v___x_2797_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2798_;
}
case 10:
{
lean_object* v_data_2799_; lean_object* v_expr_2800_; lean_object* v___x_2801_; 
v_data_2799_ = lean_ctor_get(v___y_2786_, 0);
v_expr_2800_ = lean_ctor_get(v___y_2786_, 1);
lean_inc_ref(v_expr_2800_);
lean_inc_ref(v_post_2766_);
lean_inc_ref(v_pre_2764_);
v___x_2801_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v_expr_2800_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
if (lean_obj_tag(v___x_2801_) == 0)
{
lean_object* v_a_2802_; size_t v___x_2803_; size_t v___x_2804_; uint8_t v___x_2805_; 
v_a_2802_ = lean_ctor_get(v___x_2801_, 0);
lean_inc(v_a_2802_);
lean_dec_ref_known(v___x_2801_, 1);
v___x_2803_ = lean_ptr_addr(v_expr_2800_);
v___x_2804_ = lean_ptr_addr(v_a_2802_);
v___x_2805_ = lean_usize_dec_eq(v___x_2803_, v___x_2804_);
if (v___x_2805_ == 0)
{
lean_object* v___x_2806_; lean_object* v___x_2807_; 
lean_inc(v_data_2799_);
lean_dec_ref_known(v___y_2786_, 2);
v___x_2806_ = l_Lean_Expr_mdata___override(v_data_2799_, v_a_2802_);
v___x_2807_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___x_2806_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2807_;
}
else
{
lean_object* v___x_2808_; 
lean_dec(v_a_2802_);
v___x_2808_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2808_;
}
}
else
{
lean_dec_ref_known(v___y_2786_, 2);
lean_dec_ref(v_post_2766_);
lean_dec_ref(v_pre_2764_);
return v___x_2801_;
}
}
case 11:
{
lean_object* v_typeName_2809_; lean_object* v_idx_2810_; lean_object* v_struct_2811_; lean_object* v___x_2812_; 
v_typeName_2809_ = lean_ctor_get(v___y_2786_, 0);
v_idx_2810_ = lean_ctor_get(v___y_2786_, 1);
v_struct_2811_ = lean_ctor_get(v___y_2786_, 2);
lean_inc_ref(v_struct_2811_);
lean_inc_ref(v_post_2766_);
lean_inc_ref(v_pre_2764_);
v___x_2812_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v_struct_2811_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
if (lean_obj_tag(v___x_2812_) == 0)
{
lean_object* v_a_2813_; size_t v___x_2814_; size_t v___x_2815_; uint8_t v___x_2816_; 
v_a_2813_ = lean_ctor_get(v___x_2812_, 0);
lean_inc(v_a_2813_);
lean_dec_ref_known(v___x_2812_, 1);
v___x_2814_ = lean_ptr_addr(v_struct_2811_);
v___x_2815_ = lean_ptr_addr(v_a_2813_);
v___x_2816_ = lean_usize_dec_eq(v___x_2814_, v___x_2815_);
if (v___x_2816_ == 0)
{
lean_object* v___x_2817_; lean_object* v___x_2818_; 
lean_inc(v_idx_2810_);
lean_inc(v_typeName_2809_);
lean_dec_ref_known(v___y_2786_, 3);
v___x_2817_ = l_Lean_Expr_proj___override(v_typeName_2809_, v_idx_2810_, v_a_2813_);
v___x_2818_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___x_2817_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2818_;
}
else
{
lean_object* v___x_2819_; 
lean_dec(v_a_2813_);
v___x_2819_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2819_;
}
}
else
{
lean_dec_ref_known(v___y_2786_, 3);
lean_dec_ref(v_post_2766_);
lean_dec_ref(v_pre_2764_);
return v___x_2812_;
}
}
default: 
{
lean_object* v___x_2820_; 
v___x_2820_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2764_, v_post_2766_, v_usedLetOnly_2767_, v_skipConstInApp_2768_, v_skipInstances_2769_, v___y_2786_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_);
return v___x_2820_;
}
}
}
}
}
else
{
lean_object* v_a_2830_; lean_object* v___x_2832_; uint8_t v_isShared_2833_; uint8_t v_isSharedCheck_2837_; 
lean_dec_ref(v_post_2766_);
lean_dec_ref(v_e_2765_);
lean_dec_ref(v_pre_2764_);
v_a_2830_ = lean_ctor_get(v___x_2780_, 0);
v_isSharedCheck_2837_ = !lean_is_exclusive(v___x_2780_);
if (v_isSharedCheck_2837_ == 0)
{
v___x_2832_ = v___x_2780_;
v_isShared_2833_ = v_isSharedCheck_2837_;
goto v_resetjp_2831_;
}
else
{
lean_inc(v_a_2830_);
lean_dec(v___x_2780_);
v___x_2832_ = lean_box(0);
v_isShared_2833_ = v_isSharedCheck_2837_;
goto v_resetjp_2831_;
}
v_resetjp_2831_:
{
lean_object* v___x_2835_; 
if (v_isShared_2833_ == 0)
{
v___x_2835_ = v___x_2832_;
goto v_reusejp_2834_;
}
else
{
lean_object* v_reuseFailAlloc_2836_; 
v_reuseFailAlloc_2836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2836_, 0, v_a_2830_);
v___x_2835_ = v_reuseFailAlloc_2836_;
goto v_reusejp_2834_;
}
v_reusejp_2834_:
{
return v___x_2835_;
}
}
}
}
else
{
lean_object* v_a_2838_; lean_object* v___x_2840_; uint8_t v_isShared_2841_; uint8_t v_isSharedCheck_2845_; 
lean_dec_ref(v_post_2766_);
lean_dec_ref(v_e_2765_);
lean_dec_ref(v_pre_2764_);
v_a_2838_ = lean_ctor_get(v___x_2779_, 0);
v_isSharedCheck_2845_ = !lean_is_exclusive(v___x_2779_);
if (v_isSharedCheck_2845_ == 0)
{
v___x_2840_ = v___x_2779_;
v_isShared_2841_ = v_isSharedCheck_2845_;
goto v_resetjp_2839_;
}
else
{
lean_inc(v_a_2838_);
lean_dec(v___x_2779_);
v___x_2840_ = lean_box(0);
v_isShared_2841_ = v_isSharedCheck_2845_;
goto v_resetjp_2839_;
}
v_resetjp_2839_:
{
lean_object* v___x_2843_; 
if (v_isShared_2841_ == 0)
{
v___x_2843_ = v___x_2840_;
goto v_reusejp_2842_;
}
else
{
lean_object* v_reuseFailAlloc_2844_; 
v_reuseFailAlloc_2844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2844_, 0, v_a_2838_);
v___x_2843_ = v_reuseFailAlloc_2844_;
goto v_reusejp_2842_;
}
v_reusejp_2842_:
{
return v___x_2843_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___boxed(lean_object* v___x_2846_, lean_object* v_pre_2847_, lean_object* v_e_2848_, lean_object* v_post_2849_, lean_object* v_usedLetOnly_2850_, lean_object* v_skipConstInApp_2851_, lean_object* v_skipInstances_2852_, lean_object* v___y_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_, lean_object* v___y_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_, lean_object* v___y_2859_, lean_object* v___y_2860_, lean_object* v___y_2861_){
_start:
{
uint8_t v_usedLetOnly_boxed_2862_; uint8_t v_skipConstInApp_boxed_2863_; uint8_t v_skipInstances_boxed_2864_; lean_object* v_res_2865_; 
v_usedLetOnly_boxed_2862_ = lean_unbox(v_usedLetOnly_2850_);
v_skipConstInApp_boxed_2863_ = lean_unbox(v_skipConstInApp_2851_);
v_skipInstances_boxed_2864_ = lean_unbox(v_skipInstances_2852_);
v_res_2865_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1(v___x_2846_, v_pre_2847_, v_e_2848_, v_post_2849_, v_usedLetOnly_boxed_2862_, v_skipConstInApp_boxed_2863_, v_skipInstances_boxed_2864_, v___y_2853_, v___y_2854_, v___y_2855_, v___y_2856_, v___y_2857_, v___y_2858_, v___y_2859_, v___y_2860_);
lean_dec(v___y_2860_);
lean_dec_ref(v___y_2859_);
lean_dec(v___y_2858_);
lean_dec_ref(v___y_2857_);
lean_dec(v___y_2856_);
lean_dec_ref(v___y_2855_);
lean_dec(v___y_2854_);
lean_dec(v___y_2853_);
return v_res_2865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(lean_object* v_pre_2866_, lean_object* v_post_2867_, uint8_t v_usedLetOnly_2868_, uint8_t v_skipConstInApp_2869_, uint8_t v_skipInstances_2870_, lean_object* v_e_2871_, lean_object* v_a_2872_, lean_object* v___y_2873_, lean_object* v___y_2874_, lean_object* v___y_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_){
_start:
{
lean_object* v___x_2881_; lean_object* v___x_2882_; 
lean_inc(v_a_2872_);
v___x_2881_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2881_, 0, lean_box(0));
lean_closure_set(v___x_2881_, 1, lean_box(0));
lean_closure_set(v___x_2881_, 2, v_a_2872_);
v___x_2882_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0(lean_box(0), v___x_2881_, v___y_2873_, v___y_2874_, v___y_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
if (lean_obj_tag(v___x_2882_) == 0)
{
lean_object* v_a_2883_; lean_object* v___x_2885_; uint8_t v_isShared_2886_; uint8_t v_isSharedCheck_2917_; 
v_a_2883_ = lean_ctor_get(v___x_2882_, 0);
v_isSharedCheck_2917_ = !lean_is_exclusive(v___x_2882_);
if (v_isSharedCheck_2917_ == 0)
{
v___x_2885_ = v___x_2882_;
v_isShared_2886_ = v_isSharedCheck_2917_;
goto v_resetjp_2884_;
}
else
{
lean_inc(v_a_2883_);
lean_dec(v___x_2882_);
v___x_2885_ = lean_box(0);
v_isShared_2886_ = v_isSharedCheck_2917_;
goto v_resetjp_2884_;
}
v_resetjp_2884_:
{
lean_object* v___x_2887_; 
v___x_2887_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(v_a_2883_, v_e_2871_);
lean_dec(v_a_2883_);
if (lean_obj_tag(v___x_2887_) == 0)
{
lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___f_2892_; lean_object* v___x_2893_; 
lean_del_object(v___x_2885_);
v___x_2888_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___closed__0));
v___x_2889_ = lean_box(v_usedLetOnly_2868_);
v___x_2890_ = lean_box(v_skipConstInApp_2869_);
v___x_2891_ = lean_box(v_skipInstances_2870_);
lean_inc_ref(v_e_2871_);
v___f_2892_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__1___boxed), 16, 7);
lean_closure_set(v___f_2892_, 0, v___x_2888_);
lean_closure_set(v___f_2892_, 1, v_pre_2866_);
lean_closure_set(v___f_2892_, 2, v_e_2871_);
lean_closure_set(v___f_2892_, 3, v_post_2867_);
lean_closure_set(v___f_2892_, 4, v___x_2889_);
lean_closure_set(v___f_2892_, 5, v___x_2890_);
lean_closure_set(v___f_2892_, 6, v___x_2891_);
v___x_2893_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg(v___f_2892_, v_a_2872_, v___y_2873_, v___y_2874_, v___y_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
if (lean_obj_tag(v___x_2893_) == 0)
{
lean_object* v_a_2894_; lean_object* v___f_2895_; lean_object* v___x_2896_; 
v_a_2894_ = lean_ctor_get(v___x_2893_, 0);
lean_inc_n(v_a_2894_, 2);
lean_dec_ref_known(v___x_2893_, 1);
lean_inc(v_a_2872_);
v___f_2895_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2___boxed), 4, 3);
lean_closure_set(v___f_2895_, 0, v_a_2872_);
lean_closure_set(v___f_2895_, 1, v_e_2871_);
lean_closure_set(v___f_2895_, 2, v_a_2894_);
v___x_2896_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__0(lean_box(0), v___f_2895_, v___y_2873_, v___y_2874_, v___y_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
if (lean_obj_tag(v___x_2896_) == 0)
{
lean_object* v___x_2898_; uint8_t v_isShared_2899_; uint8_t v_isSharedCheck_2903_; 
v_isSharedCheck_2903_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2903_ == 0)
{
lean_object* v_unused_2904_; 
v_unused_2904_ = lean_ctor_get(v___x_2896_, 0);
lean_dec(v_unused_2904_);
v___x_2898_ = v___x_2896_;
v_isShared_2899_ = v_isSharedCheck_2903_;
goto v_resetjp_2897_;
}
else
{
lean_dec(v___x_2896_);
v___x_2898_ = lean_box(0);
v_isShared_2899_ = v_isSharedCheck_2903_;
goto v_resetjp_2897_;
}
v_resetjp_2897_:
{
lean_object* v___x_2901_; 
if (v_isShared_2899_ == 0)
{
lean_ctor_set(v___x_2898_, 0, v_a_2894_);
v___x_2901_ = v___x_2898_;
goto v_reusejp_2900_;
}
else
{
lean_object* v_reuseFailAlloc_2902_; 
v_reuseFailAlloc_2902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2902_, 0, v_a_2894_);
v___x_2901_ = v_reuseFailAlloc_2902_;
goto v_reusejp_2900_;
}
v_reusejp_2900_:
{
return v___x_2901_;
}
}
}
else
{
lean_object* v_a_2905_; lean_object* v___x_2907_; uint8_t v_isShared_2908_; uint8_t v_isSharedCheck_2912_; 
lean_dec(v_a_2894_);
v_a_2905_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_2912_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2912_ == 0)
{
v___x_2907_ = v___x_2896_;
v_isShared_2908_ = v_isSharedCheck_2912_;
goto v_resetjp_2906_;
}
else
{
lean_inc(v_a_2905_);
lean_dec(v___x_2896_);
v___x_2907_ = lean_box(0);
v_isShared_2908_ = v_isSharedCheck_2912_;
goto v_resetjp_2906_;
}
v_resetjp_2906_:
{
lean_object* v___x_2910_; 
if (v_isShared_2908_ == 0)
{
v___x_2910_ = v___x_2907_;
goto v_reusejp_2909_;
}
else
{
lean_object* v_reuseFailAlloc_2911_; 
v_reuseFailAlloc_2911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2911_, 0, v_a_2905_);
v___x_2910_ = v_reuseFailAlloc_2911_;
goto v_reusejp_2909_;
}
v_reusejp_2909_:
{
return v___x_2910_;
}
}
}
}
else
{
lean_dec_ref(v_e_2871_);
return v___x_2893_;
}
}
else
{
lean_object* v_val_2913_; lean_object* v___x_2915_; 
lean_dec_ref(v_e_2871_);
lean_dec_ref(v_post_2867_);
lean_dec_ref(v_pre_2866_);
v_val_2913_ = lean_ctor_get(v___x_2887_, 0);
lean_inc(v_val_2913_);
lean_dec_ref_known(v___x_2887_, 1);
if (v_isShared_2886_ == 0)
{
lean_ctor_set(v___x_2885_, 0, v_val_2913_);
v___x_2915_ = v___x_2885_;
goto v_reusejp_2914_;
}
else
{
lean_object* v_reuseFailAlloc_2916_; 
v_reuseFailAlloc_2916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2916_, 0, v_val_2913_);
v___x_2915_ = v_reuseFailAlloc_2916_;
goto v_reusejp_2914_;
}
v_reusejp_2914_:
{
return v___x_2915_;
}
}
}
}
else
{
lean_object* v_a_2918_; lean_object* v___x_2920_; uint8_t v_isShared_2921_; uint8_t v_isSharedCheck_2925_; 
lean_dec_ref(v_e_2871_);
lean_dec_ref(v_post_2867_);
lean_dec_ref(v_pre_2866_);
v_a_2918_ = lean_ctor_get(v___x_2882_, 0);
v_isSharedCheck_2925_ = !lean_is_exclusive(v___x_2882_);
if (v_isSharedCheck_2925_ == 0)
{
v___x_2920_ = v___x_2882_;
v_isShared_2921_ = v_isSharedCheck_2925_;
goto v_resetjp_2919_;
}
else
{
lean_inc(v_a_2918_);
lean_dec(v___x_2882_);
v___x_2920_ = lean_box(0);
v_isShared_2921_ = v_isSharedCheck_2925_;
goto v_resetjp_2919_;
}
v_resetjp_2919_:
{
lean_object* v___x_2923_; 
if (v_isShared_2921_ == 0)
{
v___x_2923_ = v___x_2920_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_2924_; 
v_reuseFailAlloc_2924_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2924_, 0, v_a_2918_);
v___x_2923_ = v_reuseFailAlloc_2924_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
return v___x_2923_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0___boxed(lean_object** _args){
lean_object* v_fvars_2926_ = _args[0];
lean_object* v_pre_2927_ = _args[1];
lean_object* v_post_2928_ = _args[2];
lean_object* v_usedLetOnly_2929_ = _args[3];
lean_object* v_skipConstInApp_2930_ = _args[4];
lean_object* v_skipInstances_2931_ = _args[5];
lean_object* v_body_2932_ = _args[6];
lean_object* v_x_2933_ = _args[7];
lean_object* v___y_2934_ = _args[8];
lean_object* v___y_2935_ = _args[9];
lean_object* v___y_2936_ = _args[10];
lean_object* v___y_2937_ = _args[11];
lean_object* v___y_2938_ = _args[12];
lean_object* v___y_2939_ = _args[13];
lean_object* v___y_2940_ = _args[14];
lean_object* v___y_2941_ = _args[15];
lean_object* v___y_2942_ = _args[16];
_start:
{
uint8_t v_usedLetOnly_boxed_2943_; uint8_t v_skipConstInApp_boxed_2944_; uint8_t v_skipInstances_boxed_2945_; lean_object* v_res_2946_; 
v_usedLetOnly_boxed_2943_ = lean_unbox(v_usedLetOnly_2929_);
v_skipConstInApp_boxed_2944_ = lean_unbox(v_skipConstInApp_2930_);
v_skipInstances_boxed_2945_ = lean_unbox(v_skipInstances_2931_);
v_res_2946_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0(v_fvars_2926_, v_pre_2927_, v_post_2928_, v_usedLetOnly_boxed_2943_, v_skipConstInApp_boxed_2944_, v_skipInstances_boxed_2945_, v_body_2932_, v_x_2933_, v___y_2934_, v___y_2935_, v___y_2936_, v___y_2937_, v___y_2938_, v___y_2939_, v___y_2940_, v___y_2941_);
lean_dec(v___y_2941_);
lean_dec_ref(v___y_2940_);
lean_dec(v___y_2939_);
lean_dec_ref(v___y_2938_);
lean_dec(v___y_2937_);
lean_dec_ref(v___y_2936_);
lean_dec(v___y_2935_);
lean_dec(v___y_2934_);
return v_res_2946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9(lean_object* v_pre_2947_, lean_object* v_post_2948_, uint8_t v_usedLetOnly_2949_, uint8_t v_skipConstInApp_2950_, uint8_t v_skipInstances_2951_, lean_object* v_fvars_2952_, lean_object* v_e_2953_, lean_object* v_a_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_, lean_object* v___y_2958_, lean_object* v___y_2959_, lean_object* v___y_2960_, lean_object* v___y_2961_){
_start:
{
if (lean_obj_tag(v_e_2953_) == 7)
{
lean_object* v_binderName_2963_; lean_object* v_binderType_2964_; lean_object* v_body_2965_; uint8_t v_binderInfo_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; 
v_binderName_2963_ = lean_ctor_get(v_e_2953_, 0);
lean_inc(v_binderName_2963_);
v_binderType_2964_ = lean_ctor_get(v_e_2953_, 1);
lean_inc_ref(v_binderType_2964_);
v_body_2965_ = lean_ctor_get(v_e_2953_, 2);
lean_inc_ref(v_body_2965_);
v_binderInfo_2966_ = lean_ctor_get_uint8(v_e_2953_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2953_, 3);
v___x_2967_ = lean_expr_instantiate_rev(v_binderType_2964_, v_fvars_2952_);
lean_dec_ref(v_binderType_2964_);
lean_inc_ref(v_post_2948_);
lean_inc_ref(v_pre_2947_);
v___x_2968_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2947_, v_post_2948_, v_usedLetOnly_2949_, v_skipConstInApp_2950_, v_skipInstances_2951_, v___x_2967_, v_a_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_, v___y_2961_);
if (lean_obj_tag(v___x_2968_) == 0)
{
lean_object* v_a_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___f_2973_; uint8_t v___x_2974_; lean_object* v___x_2975_; 
v_a_2969_ = lean_ctor_get(v___x_2968_, 0);
lean_inc(v_a_2969_);
lean_dec_ref_known(v___x_2968_, 1);
v___x_2970_ = lean_box(v_usedLetOnly_2949_);
v___x_2971_ = lean_box(v_skipConstInApp_2950_);
v___x_2972_ = lean_box(v_skipInstances_2951_);
v___f_2973_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0___boxed), 17, 7);
lean_closure_set(v___f_2973_, 0, v_fvars_2952_);
lean_closure_set(v___f_2973_, 1, v_pre_2947_);
lean_closure_set(v___f_2973_, 2, v_post_2948_);
lean_closure_set(v___f_2973_, 3, v___x_2970_);
lean_closure_set(v___f_2973_, 4, v___x_2971_);
lean_closure_set(v___f_2973_, 5, v___x_2972_);
lean_closure_set(v___f_2973_, 6, v_body_2965_);
v___x_2974_ = 0;
v___x_2975_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(v_binderName_2963_, v_binderInfo_2966_, v_a_2969_, v___f_2973_, v___x_2974_, v_a_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_, v___y_2961_);
return v___x_2975_;
}
else
{
lean_dec_ref(v_body_2965_);
lean_dec(v_binderName_2963_);
lean_dec_ref(v_fvars_2952_);
lean_dec_ref(v_post_2948_);
lean_dec_ref(v_pre_2947_);
return v___x_2968_;
}
}
else
{
lean_object* v___x_2976_; lean_object* v___x_2977_; 
v___x_2976_ = lean_expr_instantiate_rev(v_e_2953_, v_fvars_2952_);
lean_dec_ref(v_e_2953_);
lean_inc_ref(v_post_2948_);
lean_inc_ref(v_pre_2947_);
v___x_2977_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_2947_, v_post_2948_, v_usedLetOnly_2949_, v_skipConstInApp_2950_, v_skipInstances_2951_, v___x_2976_, v_a_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_, v___y_2961_);
if (lean_obj_tag(v___x_2977_) == 0)
{
lean_object* v_a_2978_; uint8_t v___x_2979_; uint8_t v___x_2980_; uint8_t v___x_2981_; lean_object* v___x_2982_; 
v_a_2978_ = lean_ctor_get(v___x_2977_, 0);
lean_inc(v_a_2978_);
lean_dec_ref_known(v___x_2977_, 1);
v___x_2979_ = 0;
v___x_2980_ = 1;
v___x_2981_ = 1;
v___x_2982_ = l_Lean_Meta_mkForallFVars(v_fvars_2952_, v_a_2978_, v___x_2979_, v_usedLetOnly_2949_, v___x_2980_, v___x_2981_, v___y_2958_, v___y_2959_, v___y_2960_, v___y_2961_);
lean_dec_ref(v_fvars_2952_);
if (lean_obj_tag(v___x_2982_) == 0)
{
lean_object* v_a_2983_; lean_object* v___x_2984_; 
v_a_2983_ = lean_ctor_get(v___x_2982_, 0);
lean_inc(v_a_2983_);
lean_dec_ref_known(v___x_2982_, 1);
v___x_2984_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_2947_, v_post_2948_, v_usedLetOnly_2949_, v_skipConstInApp_2950_, v_skipInstances_2951_, v_a_2983_, v_a_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_, v___y_2961_);
return v___x_2984_;
}
else
{
lean_dec_ref(v_post_2948_);
lean_dec_ref(v_pre_2947_);
return v___x_2982_;
}
}
else
{
lean_dec_ref(v_fvars_2952_);
lean_dec_ref(v_post_2948_);
lean_dec_ref(v_pre_2947_);
return v___x_2977_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___lam__0(lean_object* v_fvars_2985_, lean_object* v_pre_2986_, lean_object* v_post_2987_, uint8_t v_usedLetOnly_2988_, uint8_t v_skipConstInApp_2989_, uint8_t v_skipInstances_2990_, lean_object* v_body_2991_, lean_object* v_x_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_){
_start:
{
lean_object* v___x_3002_; lean_object* v___x_3003_; 
v___x_3002_ = lean_array_push(v_fvars_2985_, v_x_2992_);
v___x_3003_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9(v_pre_2986_, v_post_2987_, v_usedLetOnly_2988_, v_skipConstInApp_2989_, v_skipInstances_2990_, v___x_3002_, v_body_2991_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_, v___y_2998_, v___y_2999_, v___y_3000_);
return v___x_3003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6___boxed(lean_object* v_pre_3004_, lean_object* v_post_3005_, lean_object* v_usedLetOnly_3006_, lean_object* v_skipConstInApp_3007_, lean_object* v_skipInstances_3008_, lean_object* v_e_3009_, lean_object* v_a_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_){
_start:
{
uint8_t v_usedLetOnly_boxed_3019_; uint8_t v_skipConstInApp_boxed_3020_; uint8_t v_skipInstances_boxed_3021_; lean_object* v_res_3022_; 
v_usedLetOnly_boxed_3019_ = lean_unbox(v_usedLetOnly_3006_);
v_skipConstInApp_boxed_3020_ = lean_unbox(v_skipConstInApp_3007_);
v_skipInstances_boxed_3021_ = lean_unbox(v_skipInstances_3008_);
v_res_3022_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__6(v_pre_3004_, v_post_3005_, v_usedLetOnly_boxed_3019_, v_skipConstInApp_boxed_3020_, v_skipInstances_boxed_3021_, v_e_3009_, v_a_3010_, v___y_3011_, v___y_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_, v___y_3017_);
lean_dec(v___y_3017_);
lean_dec_ref(v___y_3016_);
lean_dec(v___y_3015_);
lean_dec_ref(v___y_3014_);
lean_dec(v___y_3013_);
lean_dec_ref(v___y_3012_);
lean_dec(v___y_3011_);
lean_dec(v_a_3010_);
return v_res_3022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5___boxed(lean_object** _args){
lean_object* v_pre_3023_ = _args[0];
lean_object* v_post_3024_ = _args[1];
lean_object* v_usedLetOnly_3025_ = _args[2];
lean_object* v_skipConstInApp_3026_ = _args[3];
lean_object* v_skipInstances_3027_ = _args[4];
lean_object* v_sz_3028_ = _args[5];
lean_object* v_i_3029_ = _args[6];
lean_object* v_bs_3030_ = _args[7];
lean_object* v___y_3031_ = _args[8];
lean_object* v___y_3032_ = _args[9];
lean_object* v___y_3033_ = _args[10];
lean_object* v___y_3034_ = _args[11];
lean_object* v___y_3035_ = _args[12];
lean_object* v___y_3036_ = _args[13];
lean_object* v___y_3037_ = _args[14];
lean_object* v___y_3038_ = _args[15];
lean_object* v___y_3039_ = _args[16];
_start:
{
uint8_t v_usedLetOnly_boxed_3040_; uint8_t v_skipConstInApp_boxed_3041_; uint8_t v_skipInstances_boxed_3042_; size_t v_sz_boxed_3043_; size_t v_i_boxed_3044_; lean_object* v_res_3045_; 
v_usedLetOnly_boxed_3040_ = lean_unbox(v_usedLetOnly_3025_);
v_skipConstInApp_boxed_3041_ = lean_unbox(v_skipConstInApp_3026_);
v_skipInstances_boxed_3042_ = lean_unbox(v_skipInstances_3027_);
v_sz_boxed_3043_ = lean_unbox_usize(v_sz_3028_);
lean_dec(v_sz_3028_);
v_i_boxed_3044_ = lean_unbox_usize(v_i_3029_);
lean_dec(v_i_3029_);
v_res_3045_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__5(v_pre_3023_, v_post_3024_, v_usedLetOnly_boxed_3040_, v_skipConstInApp_boxed_3041_, v_skipInstances_boxed_3042_, v_sz_boxed_3043_, v_i_boxed_3044_, v_bs_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_, v___y_3036_, v___y_3037_, v___y_3038_);
lean_dec(v___y_3038_);
lean_dec_ref(v___y_3037_);
lean_dec(v___y_3036_);
lean_dec_ref(v___y_3035_);
lean_dec(v___y_3034_);
lean_dec_ref(v___y_3033_);
lean_dec(v___y_3032_);
lean_dec(v___y_3031_);
return v_res_3045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___boxed(lean_object* v_pre_3046_, lean_object* v_post_3047_, lean_object* v_usedLetOnly_3048_, lean_object* v_skipConstInApp_3049_, lean_object* v_skipInstances_3050_, lean_object* v_e_3051_, lean_object* v_a_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_, lean_object* v___y_3058_, lean_object* v___y_3059_, lean_object* v___y_3060_){
_start:
{
uint8_t v_usedLetOnly_boxed_3061_; uint8_t v_skipConstInApp_boxed_3062_; uint8_t v_skipInstances_boxed_3063_; lean_object* v_res_3064_; 
v_usedLetOnly_boxed_3061_ = lean_unbox(v_usedLetOnly_3048_);
v_skipConstInApp_boxed_3062_ = lean_unbox(v_skipConstInApp_3049_);
v_skipInstances_boxed_3063_ = lean_unbox(v_skipInstances_3050_);
v_res_3064_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_3046_, v_post_3047_, v_usedLetOnly_boxed_3061_, v_skipConstInApp_boxed_3062_, v_skipInstances_boxed_3063_, v_e_3051_, v_a_3052_, v___y_3053_, v___y_3054_, v___y_3055_, v___y_3056_, v___y_3057_, v___y_3058_, v___y_3059_);
lean_dec(v___y_3059_);
lean_dec_ref(v___y_3058_);
lean_dec(v___y_3057_);
lean_dec_ref(v___y_3056_);
lean_dec(v___y_3055_);
lean_dec_ref(v___y_3054_);
lean_dec(v___y_3053_);
lean_dec(v_a_3052_);
return v_res_3064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9___boxed(lean_object* v_pre_3065_, lean_object* v_post_3066_, lean_object* v_usedLetOnly_3067_, lean_object* v_skipConstInApp_3068_, lean_object* v_skipInstances_3069_, lean_object* v_fvars_3070_, lean_object* v_e_3071_, lean_object* v_a_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_, lean_object* v___y_3080_){
_start:
{
uint8_t v_usedLetOnly_boxed_3081_; uint8_t v_skipConstInApp_boxed_3082_; uint8_t v_skipInstances_boxed_3083_; lean_object* v_res_3084_; 
v_usedLetOnly_boxed_3081_ = lean_unbox(v_usedLetOnly_3067_);
v_skipConstInApp_boxed_3082_ = lean_unbox(v_skipConstInApp_3068_);
v_skipInstances_boxed_3083_ = lean_unbox(v_skipInstances_3069_);
v_res_3084_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9(v_pre_3065_, v_post_3066_, v_usedLetOnly_boxed_3081_, v_skipConstInApp_boxed_3082_, v_skipInstances_boxed_3083_, v_fvars_3070_, v_e_3071_, v_a_3072_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_, v___y_3079_);
lean_dec(v___y_3079_);
lean_dec_ref(v___y_3078_);
lean_dec(v___y_3077_);
lean_dec_ref(v___y_3076_);
lean_dec(v___y_3075_);
lean_dec_ref(v___y_3074_);
lean_dec(v___y_3073_);
lean_dec(v_a_3072_);
return v_res_3084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10___boxed(lean_object* v_pre_3085_, lean_object* v_post_3086_, lean_object* v_usedLetOnly_3087_, lean_object* v_skipConstInApp_3088_, lean_object* v_skipInstances_3089_, lean_object* v_fvars_3090_, lean_object* v_e_3091_, lean_object* v_a_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_){
_start:
{
uint8_t v_usedLetOnly_boxed_3101_; uint8_t v_skipConstInApp_boxed_3102_; uint8_t v_skipInstances_boxed_3103_; lean_object* v_res_3104_; 
v_usedLetOnly_boxed_3101_ = lean_unbox(v_usedLetOnly_3087_);
v_skipConstInApp_boxed_3102_ = lean_unbox(v_skipConstInApp_3088_);
v_skipInstances_boxed_3103_ = lean_unbox(v_skipInstances_3089_);
v_res_3104_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__10(v_pre_3085_, v_post_3086_, v_usedLetOnly_boxed_3101_, v_skipConstInApp_boxed_3102_, v_skipInstances_boxed_3103_, v_fvars_3090_, v_e_3091_, v_a_3092_, v___y_3093_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, v___y_3099_);
lean_dec(v___y_3099_);
lean_dec_ref(v___y_3098_);
lean_dec(v___y_3097_);
lean_dec_ref(v___y_3096_);
lean_dec(v___y_3095_);
lean_dec_ref(v___y_3094_);
lean_dec(v___y_3093_);
lean_dec(v_a_3092_);
return v_res_3104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11___boxed(lean_object* v_pre_3105_, lean_object* v_post_3106_, lean_object* v_usedLetOnly_3107_, lean_object* v_skipConstInApp_3108_, lean_object* v_skipInstances_3109_, lean_object* v_fvars_3110_, lean_object* v_e_3111_, lean_object* v_a_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_){
_start:
{
uint8_t v_usedLetOnly_boxed_3121_; uint8_t v_skipConstInApp_boxed_3122_; uint8_t v_skipInstances_boxed_3123_; lean_object* v_res_3124_; 
v_usedLetOnly_boxed_3121_ = lean_unbox(v_usedLetOnly_3107_);
v_skipConstInApp_boxed_3122_ = lean_unbox(v_skipConstInApp_3108_);
v_skipInstances_boxed_3123_ = lean_unbox(v_skipInstances_3109_);
v_res_3124_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11(v_pre_3105_, v_post_3106_, v_usedLetOnly_boxed_3121_, v_skipConstInApp_boxed_3122_, v_skipInstances_boxed_3123_, v_fvars_3110_, v_e_3111_, v_a_3112_, v___y_3113_, v___y_3114_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_, v___y_3119_);
lean_dec(v___y_3119_);
lean_dec_ref(v___y_3118_);
lean_dec(v___y_3117_);
lean_dec_ref(v___y_3116_);
lean_dec(v___y_3115_);
lean_dec_ref(v___y_3114_);
lean_dec(v___y_3113_);
lean_dec(v_a_3112_);
return v_res_3124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg___boxed(lean_object** _args){
lean_object* v_upperBound_3125_ = _args[0];
lean_object* v___x_3126_ = _args[1];
lean_object* v_pre_3127_ = _args[2];
lean_object* v_post_3128_ = _args[3];
lean_object* v_usedLetOnly_3129_ = _args[4];
lean_object* v_skipConstInApp_3130_ = _args[5];
lean_object* v_skipInstances_3131_ = _args[6];
lean_object* v_a_3132_ = _args[7];
lean_object* v_b_3133_ = _args[8];
lean_object* v___y_3134_ = _args[9];
lean_object* v___y_3135_ = _args[10];
lean_object* v___y_3136_ = _args[11];
lean_object* v___y_3137_ = _args[12];
lean_object* v___y_3138_ = _args[13];
lean_object* v___y_3139_ = _args[14];
lean_object* v___y_3140_ = _args[15];
lean_object* v___y_3141_ = _args[16];
lean_object* v___y_3142_ = _args[17];
_start:
{
uint8_t v_usedLetOnly_boxed_3143_; uint8_t v_skipConstInApp_boxed_3144_; uint8_t v_skipInstances_boxed_3145_; lean_object* v_res_3146_; 
v_usedLetOnly_boxed_3143_ = lean_unbox(v_usedLetOnly_3129_);
v_skipConstInApp_boxed_3144_ = lean_unbox(v_skipConstInApp_3130_);
v_skipInstances_boxed_3145_ = lean_unbox(v_skipInstances_3131_);
v_res_3146_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg(v_upperBound_3125_, v___x_3126_, v_pre_3127_, v_post_3128_, v_usedLetOnly_boxed_3143_, v_skipConstInApp_boxed_3144_, v_skipInstances_boxed_3145_, v_a_3132_, v_b_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
lean_dec(v___y_3141_);
lean_dec_ref(v___y_3140_);
lean_dec(v___y_3139_);
lean_dec_ref(v___y_3138_);
lean_dec(v___y_3137_);
lean_dec_ref(v___y_3136_);
lean_dec(v___y_3135_);
lean_dec(v___y_3134_);
lean_dec_ref(v___x_3126_);
lean_dec(v_upperBound_3125_);
return v_res_3146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12___boxed(lean_object** _args){
lean_object* v_skipInstances_3147_ = _args[0];
lean_object* v_pre_3148_ = _args[1];
lean_object* v_post_3149_ = _args[2];
lean_object* v_usedLetOnly_3150_ = _args[3];
lean_object* v_skipConstInApp_3151_ = _args[4];
lean_object* v_x_3152_ = _args[5];
lean_object* v_x_3153_ = _args[6];
lean_object* v_x_3154_ = _args[7];
lean_object* v___y_3155_ = _args[8];
lean_object* v___y_3156_ = _args[9];
lean_object* v___y_3157_ = _args[10];
lean_object* v___y_3158_ = _args[11];
lean_object* v___y_3159_ = _args[12];
lean_object* v___y_3160_ = _args[13];
lean_object* v___y_3161_ = _args[14];
lean_object* v___y_3162_ = _args[15];
lean_object* v___y_3163_ = _args[16];
_start:
{
uint8_t v_skipInstances_boxed_3164_; uint8_t v_usedLetOnly_boxed_3165_; uint8_t v_skipConstInApp_boxed_3166_; lean_object* v_res_3167_; 
v_skipInstances_boxed_3164_ = lean_unbox(v_skipInstances_3147_);
v_usedLetOnly_boxed_3165_ = lean_unbox(v_usedLetOnly_3150_);
v_skipConstInApp_boxed_3166_ = lean_unbox(v_skipConstInApp_3151_);
v_res_3167_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__12(v_skipInstances_boxed_3164_, v_pre_3148_, v_post_3149_, v_usedLetOnly_boxed_3165_, v_skipConstInApp_boxed_3166_, v_x_3152_, v_x_3153_, v_x_3154_, v___y_3155_, v___y_3156_, v___y_3157_, v___y_3158_, v___y_3159_, v___y_3160_, v___y_3161_, v___y_3162_);
lean_dec(v___y_3162_);
lean_dec_ref(v___y_3161_);
lean_dec(v___y_3160_);
lean_dec_ref(v___y_3159_);
lean_dec(v___y_3158_);
lean_dec_ref(v___y_3157_);
lean_dec(v___y_3156_);
lean_dec(v___y_3155_);
return v_res_3167_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0(void){
_start:
{
lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; 
v___x_3168_ = lean_box(0);
v___x_3169_ = lean_unsigned_to_nat(16u);
v___x_3170_ = lean_mk_array(v___x_3169_, v___x_3168_);
return v___x_3170_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1(void){
_start:
{
lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; 
v___x_3171_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__0);
v___x_3172_ = lean_unsigned_to_nat(0u);
v___x_3173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3173_, 0, v___x_3172_);
lean_ctor_set(v___x_3173_, 1, v___x_3171_);
return v___x_3173_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2(void){
_start:
{
lean_object* v___x_3174_; lean_object* v___x_3175_; 
v___x_3174_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__1);
v___x_3175_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_3175_, 0, lean_box(0));
lean_closure_set(v___x_3175_, 1, lean_box(0));
lean_closure_set(v___x_3175_, 2, v___x_3174_);
return v___x_3175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3(lean_object* v_input_3176_, lean_object* v_pre_3177_, lean_object* v_post_3178_, uint8_t v_usedLetOnly_3179_, uint8_t v_skipConstInApp_3180_, lean_object* v___y_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_){
_start:
{
lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v_a_3191_; uint8_t v___x_3192_; lean_object* v___x_3193_; 
v___x_3189_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2);
v___x_3190_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0(lean_box(0), v___x_3189_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_);
v_a_3191_ = lean_ctor_get(v___x_3190_, 0);
lean_inc(v_a_3191_);
lean_dec_ref(v___x_3190_);
v___x_3192_ = 0;
v___x_3193_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4(v_pre_3177_, v_post_3178_, v_usedLetOnly_3179_, v_skipConstInApp_3180_, v___x_3192_, v_input_3176_, v_a_3191_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_);
if (lean_obj_tag(v___x_3193_) == 0)
{
lean_object* v_a_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3198_; uint8_t v_isShared_3199_; uint8_t v_isSharedCheck_3203_; 
v_a_3194_ = lean_ctor_get(v___x_3193_, 0);
lean_inc(v_a_3194_);
lean_dec_ref_known(v___x_3193_, 1);
v___x_3195_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_3195_, 0, lean_box(0));
lean_closure_set(v___x_3195_, 1, lean_box(0));
lean_closure_set(v___x_3195_, 2, v_a_3191_);
v___x_3196_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___lam__0(lean_box(0), v___x_3195_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_);
v_isSharedCheck_3203_ = !lean_is_exclusive(v___x_3196_);
if (v_isSharedCheck_3203_ == 0)
{
lean_object* v_unused_3204_; 
v_unused_3204_ = lean_ctor_get(v___x_3196_, 0);
lean_dec(v_unused_3204_);
v___x_3198_ = v___x_3196_;
v_isShared_3199_ = v_isSharedCheck_3203_;
goto v_resetjp_3197_;
}
else
{
lean_dec(v___x_3196_);
v___x_3198_ = lean_box(0);
v_isShared_3199_ = v_isSharedCheck_3203_;
goto v_resetjp_3197_;
}
v_resetjp_3197_:
{
lean_object* v___x_3201_; 
if (v_isShared_3199_ == 0)
{
lean_ctor_set(v___x_3198_, 0, v_a_3194_);
v___x_3201_ = v___x_3198_;
goto v_reusejp_3200_;
}
else
{
lean_object* v_reuseFailAlloc_3202_; 
v_reuseFailAlloc_3202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3202_, 0, v_a_3194_);
v___x_3201_ = v_reuseFailAlloc_3202_;
goto v_reusejp_3200_;
}
v_reusejp_3200_:
{
return v___x_3201_;
}
}
}
else
{
lean_dec(v_a_3191_);
return v___x_3193_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___boxed(lean_object* v_input_3205_, lean_object* v_pre_3206_, lean_object* v_post_3207_, lean_object* v_usedLetOnly_3208_, lean_object* v_skipConstInApp_3209_, lean_object* v___y_3210_, lean_object* v___y_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_){
_start:
{
uint8_t v_usedLetOnly_boxed_3218_; uint8_t v_skipConstInApp_boxed_3219_; lean_object* v_res_3220_; 
v_usedLetOnly_boxed_3218_ = lean_unbox(v_usedLetOnly_3208_);
v_skipConstInApp_boxed_3219_ = lean_unbox(v_skipConstInApp_3209_);
v_res_3220_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3(v_input_3205_, v_pre_3206_, v_post_3207_, v_usedLetOnly_boxed_3218_, v_skipConstInApp_boxed_3219_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
lean_dec(v___y_3216_);
lean_dec_ref(v___y_3215_);
lean_dec(v___y_3214_);
lean_dec_ref(v___y_3213_);
lean_dec(v___y_3212_);
lean_dec_ref(v___y_3211_);
lean_dec(v___y_3210_);
return v_res_3220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries(lean_object* v_b_3221_, lean_object* v_e_3222_, lean_object* v_attr_3223_, lean_object* v_a_3224_, lean_object* v_a_3225_, lean_object* v_a_3226_, lean_object* v_a_3227_){
_start:
{
lean_object* v___f_3229_; lean_object* v___f_3230_; uint8_t v___x_3231_; lean_object* v___x_3232_; lean_object* v___x_3233_; lean_object* v___x_3234_; lean_object* v___x_3235_; 
v___f_3229_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___closed__6));
lean_inc_ref(v_b_3221_);
v___f_3230_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___lam__1___boxed), 11, 2);
lean_closure_set(v___f_3230_, 0, v_b_3221_);
lean_closure_set(v___f_3230_, 1, v_attr_3223_);
v___x_3231_ = 0;
v___x_3232_ = lean_box(v___x_3231_);
v___x_3233_ = lean_box(v___x_3231_);
v___x_3234_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___boxed), 13, 5);
lean_closure_set(v___x_3234_, 0, v_e_3222_);
lean_closure_set(v___x_3234_, 1, v___f_3229_);
lean_closure_set(v___x_3234_, 2, v___f_3230_);
lean_closure_set(v___x_3234_, 3, v___x_3232_);
lean_closure_set(v___x_3234_, 4, v___x_3233_);
v___x_3235_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg(v_b_3221_, v___x_3234_, v_a_3224_, v_a_3225_, v_a_3226_, v_a_3227_);
return v___x_3235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries___boxed(lean_object* v_b_3236_, lean_object* v_e_3237_, lean_object* v_attr_3238_, lean_object* v_a_3239_, lean_object* v_a_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_){
_start:
{
lean_object* v_res_3244_; 
v_res_3244_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries(v_b_3236_, v_e_3237_, v_attr_3238_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_);
lean_dec(v_a_3242_);
lean_dec_ref(v_a_3241_);
lean_dec(v_a_3240_);
lean_dec_ref(v_a_3239_);
return v_res_3244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7(lean_object* v_upperBound_3245_, lean_object* v___x_3246_, lean_object* v_pre_3247_, lean_object* v_post_3248_, uint8_t v_usedLetOnly_3249_, uint8_t v_skipConstInApp_3250_, uint8_t v_skipInstances_3251_, lean_object* v___x_3252_, lean_object* v_inst_3253_, lean_object* v_R_3254_, lean_object* v_a_3255_, lean_object* v_b_3256_, lean_object* v_c_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_, lean_object* v___y_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_){
_start:
{
lean_object* v___x_3267_; 
v___x_3267_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___redArg(v_upperBound_3245_, v___x_3246_, v_pre_3247_, v_post_3248_, v_usedLetOnly_3249_, v_skipConstInApp_3250_, v_skipInstances_3251_, v_a_3255_, v_b_3256_, v___y_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_, v___y_3263_, v___y_3264_, v___y_3265_);
return v___x_3267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7___boxed(lean_object** _args){
lean_object* v_upperBound_3268_ = _args[0];
lean_object* v___x_3269_ = _args[1];
lean_object* v_pre_3270_ = _args[2];
lean_object* v_post_3271_ = _args[3];
lean_object* v_usedLetOnly_3272_ = _args[4];
lean_object* v_skipConstInApp_3273_ = _args[5];
lean_object* v_skipInstances_3274_ = _args[6];
lean_object* v___x_3275_ = _args[7];
lean_object* v_inst_3276_ = _args[8];
lean_object* v_R_3277_ = _args[9];
lean_object* v_a_3278_ = _args[10];
lean_object* v_b_3279_ = _args[11];
lean_object* v_c_3280_ = _args[12];
lean_object* v___y_3281_ = _args[13];
lean_object* v___y_3282_ = _args[14];
lean_object* v___y_3283_ = _args[15];
lean_object* v___y_3284_ = _args[16];
lean_object* v___y_3285_ = _args[17];
lean_object* v___y_3286_ = _args[18];
lean_object* v___y_3287_ = _args[19];
lean_object* v___y_3288_ = _args[20];
lean_object* v___y_3289_ = _args[21];
_start:
{
uint8_t v_usedLetOnly_boxed_3290_; uint8_t v_skipConstInApp_boxed_3291_; uint8_t v_skipInstances_boxed_3292_; lean_object* v_res_3293_; 
v_usedLetOnly_boxed_3290_ = lean_unbox(v_usedLetOnly_3272_);
v_skipConstInApp_boxed_3291_ = lean_unbox(v_skipConstInApp_3273_);
v_skipInstances_boxed_3292_ = lean_unbox(v_skipInstances_3274_);
v_res_3293_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__7(v_upperBound_3268_, v___x_3269_, v_pre_3270_, v_post_3271_, v_usedLetOnly_boxed_3290_, v_skipConstInApp_boxed_3291_, v_skipInstances_boxed_3292_, v___x_3275_, v_inst_3276_, v_R_3277_, v_a_3278_, v_b_3279_, v_c_3280_, v___y_3281_, v___y_3282_, v___y_3283_, v___y_3284_, v___y_3285_, v___y_3286_, v___y_3287_, v___y_3288_);
lean_dec(v___y_3288_);
lean_dec_ref(v___y_3287_);
lean_dec(v___y_3286_);
lean_dec_ref(v___y_3285_);
lean_dec(v___y_3284_);
lean_dec_ref(v___y_3283_);
lean_dec(v___y_3282_);
lean_dec(v___y_3281_);
lean_dec(v___x_3275_);
lean_dec_ref(v___x_3269_);
lean_dec(v_upperBound_3268_);
return v_res_3293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8(lean_object* v_00_u03b2_3294_, lean_object* v_m_3295_, lean_object* v_a_3296_){
_start:
{
lean_object* v___x_3297_; 
v___x_3297_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(v_m_3295_, v_a_3296_);
return v___x_3297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___boxed(lean_object* v_00_u03b2_3298_, lean_object* v_m_3299_, lean_object* v_a_3300_){
_start:
{
lean_object* v_res_3301_; 
v_res_3301_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8(v_00_u03b2_3298_, v_m_3299_, v_a_3300_);
lean_dec_ref(v_a_3300_);
lean_dec_ref(v_m_3299_);
return v_res_3301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11(lean_object* v_00_u03b1_3302_, lean_object* v_name_3303_, uint8_t v_bi_3304_, lean_object* v_type_3305_, lean_object* v_k_3306_, uint8_t v_kind_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_, lean_object* v___y_3313_, lean_object* v___y_3314_, lean_object* v___y_3315_){
_start:
{
lean_object* v___x_3317_; 
v___x_3317_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___redArg(v_name_3303_, v_bi_3304_, v_type_3305_, v_k_3306_, v_kind_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_, v___y_3315_);
return v___x_3317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11___boxed(lean_object* v_00_u03b1_3318_, lean_object* v_name_3319_, lean_object* v_bi_3320_, lean_object* v_type_3321_, lean_object* v_k_3322_, lean_object* v_kind_3323_, lean_object* v___y_3324_, lean_object* v___y_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_, lean_object* v___y_3332_){
_start:
{
uint8_t v_bi_boxed_3333_; uint8_t v_kind_boxed_3334_; lean_object* v_res_3335_; 
v_bi_boxed_3333_ = lean_unbox(v_bi_3320_);
v_kind_boxed_3334_ = lean_unbox(v_kind_3323_);
v_res_3335_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__9_spec__11(v_00_u03b1_3318_, v_name_3319_, v_bi_boxed_3333_, v_type_3321_, v_k_3322_, v_kind_boxed_3334_, v___y_3324_, v___y_3325_, v___y_3326_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_, v___y_3331_);
lean_dec(v___y_3331_);
lean_dec_ref(v___y_3330_);
lean_dec(v___y_3329_);
lean_dec_ref(v___y_3328_);
lean_dec(v___y_3327_);
lean_dec_ref(v___y_3326_);
lean_dec(v___y_3325_);
lean_dec(v___y_3324_);
return v_res_3335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14(lean_object* v_00_u03b1_3336_, lean_object* v_name_3337_, lean_object* v_type_3338_, lean_object* v_val_3339_, lean_object* v_k_3340_, uint8_t v_nondep_3341_, uint8_t v_kind_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_, lean_object* v___y_3345_, lean_object* v___y_3346_, lean_object* v___y_3347_, lean_object* v___y_3348_, lean_object* v___y_3349_, lean_object* v___y_3350_){
_start:
{
lean_object* v___x_3352_; 
v___x_3352_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___redArg(v_name_3337_, v_type_3338_, v_val_3339_, v_k_3340_, v_nondep_3341_, v_kind_3342_, v___y_3343_, v___y_3344_, v___y_3345_, v___y_3346_, v___y_3347_, v___y_3348_, v___y_3349_, v___y_3350_);
return v___x_3352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14___boxed(lean_object* v_00_u03b1_3353_, lean_object* v_name_3354_, lean_object* v_type_3355_, lean_object* v_val_3356_, lean_object* v_k_3357_, lean_object* v_nondep_3358_, lean_object* v_kind_3359_, lean_object* v___y_3360_, lean_object* v___y_3361_, lean_object* v___y_3362_, lean_object* v___y_3363_, lean_object* v___y_3364_, lean_object* v___y_3365_, lean_object* v___y_3366_, lean_object* v___y_3367_, lean_object* v___y_3368_){
_start:
{
uint8_t v_nondep_boxed_3369_; uint8_t v_kind_boxed_3370_; lean_object* v_res_3371_; 
v_nondep_boxed_3369_ = lean_unbox(v_nondep_3358_);
v_kind_boxed_3370_ = lean_unbox(v_kind_3359_);
v_res_3371_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__11_spec__14(v_00_u03b1_3353_, v_name_3354_, v_type_3355_, v_val_3356_, v_k_3357_, v_nondep_boxed_3369_, v_kind_boxed_3370_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_, v___y_3364_, v___y_3365_, v___y_3366_, v___y_3367_);
lean_dec(v___y_3367_);
lean_dec_ref(v___y_3366_);
lean_dec(v___y_3365_);
lean_dec_ref(v___y_3364_);
lean_dec(v___y_3363_);
lean_dec_ref(v___y_3362_);
lean_dec(v___y_3361_);
lean_dec(v___y_3360_);
return v_res_3371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17(lean_object* v_00_u03b1_3372_, lean_object* v_ref_3373_, lean_object* v___y_3374_, lean_object* v___y_3375_, lean_object* v___y_3376_, lean_object* v___y_3377_){
_start:
{
lean_object* v___x_3379_; 
v___x_3379_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg(v_ref_3373_);
return v___x_3379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___boxed(lean_object* v_00_u03b1_3380_, lean_object* v_ref_3381_, lean_object* v___y_3382_, lean_object* v___y_3383_, lean_object* v___y_3384_, lean_object* v___y_3385_, lean_object* v___y_3386_){
_start:
{
lean_object* v_res_3387_; 
v_res_3387_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17(v_00_u03b1_3380_, v_ref_3381_, v___y_3382_, v___y_3383_, v___y_3384_, v___y_3385_);
lean_dec(v___y_3385_);
lean_dec_ref(v___y_3384_);
lean_dec(v___y_3383_);
lean_dec_ref(v___y_3382_);
return v_res_3387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13(lean_object* v_00_u03b1_3388_, lean_object* v_x_3389_, lean_object* v___y_3390_, lean_object* v___y_3391_, lean_object* v___y_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_){
_start:
{
lean_object* v___x_3399_; 
v___x_3399_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___redArg(v_x_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_, v___y_3397_);
return v___x_3399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13___boxed(lean_object* v_00_u03b1_3400_, lean_object* v_x_3401_, lean_object* v___y_3402_, lean_object* v___y_3403_, lean_object* v___y_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_, lean_object* v___y_3408_, lean_object* v___y_3409_, lean_object* v___y_3410_){
_start:
{
lean_object* v_res_3411_; 
v_res_3411_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13(v_00_u03b1_3400_, v_x_3401_, v___y_3402_, v___y_3403_, v___y_3404_, v___y_3405_, v___y_3406_, v___y_3407_, v___y_3408_, v___y_3409_);
lean_dec(v___y_3409_);
lean_dec_ref(v___y_3408_);
lean_dec(v___y_3407_);
lean_dec_ref(v___y_3406_);
lean_dec(v___y_3405_);
lean_dec_ref(v___y_3404_);
lean_dec(v___y_3403_);
lean_dec(v___y_3402_);
return v_res_3411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14(lean_object* v_00_u03b2_3412_, lean_object* v_m_3413_, lean_object* v_a_3414_, lean_object* v_b_3415_){
_start:
{
lean_object* v___x_3416_; 
v___x_3416_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14___redArg(v_m_3413_, v_a_3414_, v_b_3415_);
return v___x_3416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9(lean_object* v_00_u03b2_3417_, lean_object* v_a_3418_, lean_object* v_x_3419_){
_start:
{
lean_object* v___x_3420_; 
v___x_3420_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___redArg(v_a_3418_, v_x_3419_);
return v___x_3420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9___boxed(lean_object* v_00_u03b2_3421_, lean_object* v_a_3422_, lean_object* v_x_3423_){
_start:
{
lean_object* v_res_3424_; 
v_res_3424_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8_spec__9(v_00_u03b2_3421_, v_a_3422_, v_x_3423_);
lean_dec(v_x_3423_);
lean_dec_ref(v_a_3422_);
return v_res_3424_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19(lean_object* v_00_u03b2_3425_, lean_object* v_a_3426_, lean_object* v_x_3427_){
_start:
{
uint8_t v___x_3428_; 
v___x_3428_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___redArg(v_a_3426_, v_x_3427_);
return v___x_3428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19___boxed(lean_object* v_00_u03b2_3429_, lean_object* v_a_3430_, lean_object* v_x_3431_){
_start:
{
uint8_t v_res_3432_; lean_object* v_r_3433_; 
v_res_3432_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__19(v_00_u03b2_3429_, v_a_3430_, v_x_3431_);
lean_dec(v_x_3431_);
lean_dec_ref(v_a_3430_);
v_r_3433_ = lean_box(v_res_3432_);
return v_r_3433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20(lean_object* v_00_u03b2_3434_, lean_object* v_data_3435_){
_start:
{
lean_object* v___x_3436_; 
v___x_3436_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20___redArg(v_data_3435_);
return v___x_3436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21(lean_object* v_00_u03b2_3437_, lean_object* v_a_3438_, lean_object* v_b_3439_, lean_object* v_x_3440_){
_start:
{
lean_object* v___x_3441_; 
v___x_3441_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__21___redArg(v_a_3438_, v_b_3439_, v_x_3440_);
return v___x_3441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21(lean_object* v_00_u03b2_3442_, lean_object* v_i_3443_, lean_object* v_source_3444_, lean_object* v_target_3445_){
_start:
{
lean_object* v___x_3446_; 
v___x_3446_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21___redArg(v_i_3443_, v_source_3444_, v_target_3445_);
return v___x_3446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22(lean_object* v_00_u03b2_3447_, lean_object* v_x_3448_, lean_object* v_x_3449_){
_start:
{
lean_object* v___x_3450_; 
v___x_3450_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__14_spec__20_spec__21_spec__22___redArg(v_x_3448_, v_x_3449_);
return v___x_3450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_headBetaBody(lean_object* v_e_3451_){
_start:
{
if (lean_obj_tag(v_e_3451_) == 6)
{
lean_object* v_binderName_3452_; lean_object* v_binderType_3453_; lean_object* v_body_3454_; uint8_t v_binderInfo_3455_; lean_object* v___x_3456_; uint8_t v___y_3458_; size_t v___x_3462_; uint8_t v___x_3463_; 
v_binderName_3452_ = lean_ctor_get(v_e_3451_, 0);
v_binderType_3453_ = lean_ctor_get(v_e_3451_, 1);
v_body_3454_ = lean_ctor_get(v_e_3451_, 2);
v_binderInfo_3455_ = lean_ctor_get_uint8(v_e_3451_, sizeof(void*)*3 + 8);
lean_inc_ref(v_body_3454_);
v___x_3456_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_headBetaBody(v_body_3454_);
v___x_3462_ = lean_ptr_addr(v_binderType_3453_);
v___x_3463_ = lean_usize_dec_eq(v___x_3462_, v___x_3462_);
if (v___x_3463_ == 0)
{
v___y_3458_ = v___x_3463_;
goto v___jp_3457_;
}
else
{
size_t v___x_3464_; size_t v___x_3465_; uint8_t v___x_3466_; 
v___x_3464_ = lean_ptr_addr(v_body_3454_);
v___x_3465_ = lean_ptr_addr(v___x_3456_);
v___x_3466_ = lean_usize_dec_eq(v___x_3464_, v___x_3465_);
v___y_3458_ = v___x_3466_;
goto v___jp_3457_;
}
v___jp_3457_:
{
if (v___y_3458_ == 0)
{
lean_object* v___x_3459_; 
lean_inc_ref(v_binderType_3453_);
lean_inc(v_binderName_3452_);
lean_dec_ref_known(v_e_3451_, 3);
v___x_3459_ = l_Lean_Expr_lam___override(v_binderName_3452_, v_binderType_3453_, v___x_3456_, v_binderInfo_3455_);
return v___x_3459_;
}
else
{
uint8_t v___x_3460_; 
v___x_3460_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_3455_, v_binderInfo_3455_);
if (v___x_3460_ == 0)
{
lean_object* v___x_3461_; 
lean_inc_ref(v_binderType_3453_);
lean_inc(v_binderName_3452_);
lean_dec_ref_known(v_e_3451_, 3);
v___x_3461_ = l_Lean_Expr_lam___override(v_binderName_3452_, v_binderType_3453_, v___x_3456_, v_binderInfo_3455_);
return v___x_3461_;
}
else
{
lean_dec_ref(v___x_3456_);
return v_e_3451_;
}
}
}
}
else
{
lean_object* v___x_3467_; 
v___x_3467_ = l_Lean_Expr_headBeta(v_e_3451_);
return v___x_3467_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0(lean_object* v_b_3468_, lean_object* v_e_3469_, lean_object* v___y_3470_, lean_object* v___y_3471_){
_start:
{
lean_object* v_insertionFuns_3473_; lean_object* v___x_3474_; uint8_t v___x_3475_; lean_object* v___x_3476_; 
v_insertionFuns_3473_ = lean_ctor_get(v_b_3468_, 2);
lean_inc(v_insertionFuns_3473_);
lean_dec_ref(v_b_3468_);
v___x_3474_ = lean_alloc_closure((void*)(l_Lean_NameSet_contains___boxed), 2, 1);
lean_closure_set(v___x_3474_, 0, v_insertionFuns_3473_);
v___x_3475_ = 0;
v___x_3476_ = l_Lean_Meta_delta_x3f(v_e_3469_, v___x_3474_, v___x_3475_, v___y_3470_, v___y_3471_);
if (lean_obj_tag(v___x_3476_) == 0)
{
lean_object* v_a_3477_; lean_object* v___x_3479_; uint8_t v_isShared_3480_; uint8_t v_isSharedCheck_3497_; 
v_a_3477_ = lean_ctor_get(v___x_3476_, 0);
v_isSharedCheck_3497_ = !lean_is_exclusive(v___x_3476_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3479_ = v___x_3476_;
v_isShared_3480_ = v_isSharedCheck_3497_;
goto v_resetjp_3478_;
}
else
{
lean_inc(v_a_3477_);
lean_dec(v___x_3476_);
v___x_3479_ = lean_box(0);
v_isShared_3480_ = v_isSharedCheck_3497_;
goto v_resetjp_3478_;
}
v_resetjp_3478_:
{
if (lean_obj_tag(v_a_3477_) == 1)
{
lean_object* v_val_3481_; lean_object* v___x_3483_; uint8_t v_isShared_3484_; uint8_t v_isSharedCheck_3492_; 
v_val_3481_ = lean_ctor_get(v_a_3477_, 0);
v_isSharedCheck_3492_ = !lean_is_exclusive(v_a_3477_);
if (v_isSharedCheck_3492_ == 0)
{
v___x_3483_ = v_a_3477_;
v_isShared_3484_ = v_isSharedCheck_3492_;
goto v_resetjp_3482_;
}
else
{
lean_inc(v_val_3481_);
lean_dec(v_a_3477_);
v___x_3483_ = lean_box(0);
v_isShared_3484_ = v_isSharedCheck_3492_;
goto v_resetjp_3482_;
}
v_resetjp_3482_:
{
lean_object* v___x_3485_; lean_object* v___x_3487_; 
v___x_3485_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_headBetaBody(v_val_3481_);
if (v_isShared_3484_ == 0)
{
lean_ctor_set(v___x_3483_, 0, v___x_3485_);
v___x_3487_ = v___x_3483_;
goto v_reusejp_3486_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v___x_3485_);
v___x_3487_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3486_;
}
v_reusejp_3486_:
{
lean_object* v___x_3489_; 
if (v_isShared_3480_ == 0)
{
lean_ctor_set(v___x_3479_, 0, v___x_3487_);
v___x_3489_ = v___x_3479_;
goto v_reusejp_3488_;
}
else
{
lean_object* v_reuseFailAlloc_3490_; 
v_reuseFailAlloc_3490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3490_, 0, v___x_3487_);
v___x_3489_ = v_reuseFailAlloc_3490_;
goto v_reusejp_3488_;
}
v_reusejp_3488_:
{
return v___x_3489_;
}
}
}
}
else
{
lean_object* v___x_3493_; lean_object* v___x_3495_; 
lean_dec(v_a_3477_);
v___x_3493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_run___redArg___lam__1___closed__0));
if (v_isShared_3480_ == 0)
{
lean_ctor_set(v___x_3479_, 0, v___x_3493_);
v___x_3495_ = v___x_3479_;
goto v_reusejp_3494_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v___x_3493_);
v___x_3495_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3494_;
}
v_reusejp_3494_:
{
return v___x_3495_;
}
}
}
}
else
{
lean_object* v_a_3498_; lean_object* v___x_3500_; uint8_t v_isShared_3501_; uint8_t v_isSharedCheck_3505_; 
v_a_3498_ = lean_ctor_get(v___x_3476_, 0);
v_isSharedCheck_3505_ = !lean_is_exclusive(v___x_3476_);
if (v_isSharedCheck_3505_ == 0)
{
v___x_3500_ = v___x_3476_;
v_isShared_3501_ = v_isSharedCheck_3505_;
goto v_resetjp_3499_;
}
else
{
lean_inc(v_a_3498_);
lean_dec(v___x_3476_);
v___x_3500_ = lean_box(0);
v_isShared_3501_ = v_isSharedCheck_3505_;
goto v_resetjp_3499_;
}
v_resetjp_3499_:
{
lean_object* v___x_3503_; 
if (v_isShared_3501_ == 0)
{
v___x_3503_ = v___x_3500_;
goto v_reusejp_3502_;
}
else
{
lean_object* v_reuseFailAlloc_3504_; 
v_reuseFailAlloc_3504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3504_, 0, v_a_3498_);
v___x_3503_ = v_reuseFailAlloc_3504_;
goto v_reusejp_3502_;
}
v_reusejp_3502_:
{
return v___x_3503_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0___boxed(lean_object* v_b_3506_, lean_object* v_e_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_){
_start:
{
lean_object* v_res_3511_; 
v_res_3511_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0(v_b_3506_, v_e_3507_, v___y_3508_, v___y_3509_);
lean_dec(v___y_3509_);
lean_dec_ref(v___y_3508_);
return v_res_3511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1(lean_object* v_e_3512_, lean_object* v___y_3513_, lean_object* v___y_3514_){
_start:
{
lean_object* v___x_3516_; lean_object* v___x_3517_; 
v___x_3516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3516_, 0, v_e_3512_);
v___x_3517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3517_, 0, v___x_3516_);
return v___x_3517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1___boxed(lean_object* v_e_3518_, lean_object* v___y_3519_, lean_object* v___y_3520_, lean_object* v___y_3521_){
_start:
{
lean_object* v_res_3522_; 
v_res_3522_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__1(v_e_3518_, v___y_3519_, v___y_3520_);
lean_dec(v___y_3520_);
lean_dec_ref(v___y_3519_);
return v_res_3522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0(lean_object* v_00_u03b1_3523_, lean_object* v_x_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_){
_start:
{
lean_object* v___x_3528_; lean_object* v___x_3529_; 
v___x_3528_ = lean_apply_1(v_x_3524_, lean_box(0));
v___x_3529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3529_, 0, v___x_3528_);
return v___x_3529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0___boxed(lean_object* v_00_u03b1_3530_, lean_object* v_x_3531_, lean_object* v___y_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_){
_start:
{
lean_object* v_res_3535_; 
v_res_3535_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0(v_00_u03b1_3530_, v_x_3531_, v___y_3532_, v___y_3533_);
lean_dec(v___y_3533_);
lean_dec_ref(v___y_3532_);
return v_res_3535_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_3536_; lean_object* v___x_3537_; lean_object* v___x_3538_; 
v___x_3536_ = lean_box(0);
v___x_3537_ = l_Lean_interruptExceptionId;
v___x_3538_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3538_, 0, v___x_3537_);
lean_ctor_set(v___x_3538_, 1, v___x_3536_);
return v___x_3538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg(){
_start:
{
lean_object* v___x_3540_; lean_object* v___x_3541_; 
v___x_3540_ = lean_obj_once(&lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0, &lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___closed__0);
v___x_3541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3541_, 0, v___x_3540_);
return v___x_3541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg___boxed(lean_object* v___y_3542_){
_start:
{
lean_object* v_res_3543_; 
v_res_3543_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg();
return v_res_3543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg(lean_object* v_ref_3544_){
_start:
{
lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; 
v___x_3546_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__13_spec__17___redArg___closed__5);
v___x_3547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3547_, 0, v_ref_3544_);
lean_ctor_set(v___x_3547_, 1, v___x_3546_);
v___x_3548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3548_, 0, v___x_3547_);
return v___x_3548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object* v_ref_3549_, lean_object* v___y_3550_){
_start:
{
lean_object* v_res_3551_; 
v_res_3551_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg(v_ref_3549_);
return v_res_3551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg(lean_object* v_x_3552_, lean_object* v___y_3553_, lean_object* v___y_3554_, lean_object* v___y_3555_){
_start:
{
lean_object* v___y_3558_; lean_object* v___y_3568_; lean_object* v___y_3569_; lean_object* v___y_3570_; uint8_t v___y_3571_; uint8_t v___y_3572_; lean_object* v___y_3573_; lean_object* v___y_3574_; lean_object* v___y_3575_; lean_object* v___y_3576_; lean_object* v___y_3577_; lean_object* v___y_3578_; lean_object* v___y_3579_; lean_object* v___y_3580_; lean_object* v___y_3581_; lean_object* v___y_3582_; lean_object* v___y_3583_; lean_object* v_fileName_3588_; lean_object* v_fileMap_3589_; lean_object* v_options_3590_; lean_object* v_currRecDepth_3591_; lean_object* v_maxRecDepth_3592_; lean_object* v_ref_3593_; lean_object* v_currNamespace_3594_; lean_object* v_openDecls_3595_; lean_object* v_initHeartbeats_3596_; lean_object* v_maxHeartbeats_3597_; lean_object* v_quotContext_3598_; lean_object* v_currMacroScope_3599_; uint8_t v_diag_3600_; lean_object* v_cancelTk_x3f_3601_; uint8_t v_suppressElabErrors_3602_; lean_object* v_inheritedTraceOptions_3603_; 
v_fileName_3588_ = lean_ctor_get(v___y_3554_, 0);
v_fileMap_3589_ = lean_ctor_get(v___y_3554_, 1);
v_options_3590_ = lean_ctor_get(v___y_3554_, 2);
v_currRecDepth_3591_ = lean_ctor_get(v___y_3554_, 3);
v_maxRecDepth_3592_ = lean_ctor_get(v___y_3554_, 4);
v_ref_3593_ = lean_ctor_get(v___y_3554_, 5);
v_currNamespace_3594_ = lean_ctor_get(v___y_3554_, 6);
v_openDecls_3595_ = lean_ctor_get(v___y_3554_, 7);
v_initHeartbeats_3596_ = lean_ctor_get(v___y_3554_, 8);
v_maxHeartbeats_3597_ = lean_ctor_get(v___y_3554_, 9);
v_quotContext_3598_ = lean_ctor_get(v___y_3554_, 10);
v_currMacroScope_3599_ = lean_ctor_get(v___y_3554_, 11);
v_diag_3600_ = lean_ctor_get_uint8(v___y_3554_, sizeof(void*)*14);
v_cancelTk_x3f_3601_ = lean_ctor_get(v___y_3554_, 12);
v_suppressElabErrors_3602_ = lean_ctor_get_uint8(v___y_3554_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3603_ = lean_ctor_get(v___y_3554_, 13);
if (lean_obj_tag(v_cancelTk_x3f_3601_) == 1)
{
lean_object* v_val_3609_; uint8_t v___x_3610_; 
v_val_3609_ = lean_ctor_get(v_cancelTk_x3f_3601_, 0);
v___x_3610_ = l_IO_CancelToken_isSet(v_val_3609_);
if (v___x_3610_ == 0)
{
goto v___jp_3604_;
}
else
{
lean_object* v___x_3611_; lean_object* v_a_3612_; lean_object* v___x_3614_; uint8_t v_isShared_3615_; uint8_t v_isSharedCheck_3619_; 
lean_dec_ref(v_x_3552_);
v___x_3611_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg();
v_a_3612_ = lean_ctor_get(v___x_3611_, 0);
v_isSharedCheck_3619_ = !lean_is_exclusive(v___x_3611_);
if (v_isSharedCheck_3619_ == 0)
{
v___x_3614_ = v___x_3611_;
v_isShared_3615_ = v_isSharedCheck_3619_;
goto v_resetjp_3613_;
}
else
{
lean_inc(v_a_3612_);
lean_dec(v___x_3611_);
v___x_3614_ = lean_box(0);
v_isShared_3615_ = v_isSharedCheck_3619_;
goto v_resetjp_3613_;
}
v_resetjp_3613_:
{
lean_object* v___x_3617_; 
if (v_isShared_3615_ == 0)
{
v___x_3617_ = v___x_3614_;
goto v_reusejp_3616_;
}
else
{
lean_object* v_reuseFailAlloc_3618_; 
v_reuseFailAlloc_3618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3618_, 0, v_a_3612_);
v___x_3617_ = v_reuseFailAlloc_3618_;
goto v_reusejp_3616_;
}
v_reusejp_3616_:
{
return v___x_3617_;
}
}
}
}
else
{
goto v___jp_3604_;
}
v___jp_3557_:
{
if (lean_obj_tag(v___y_3558_) == 0)
{
return v___y_3558_;
}
else
{
lean_object* v_a_3559_; lean_object* v___x_3561_; uint8_t v_isShared_3562_; uint8_t v_isSharedCheck_3566_; 
v_a_3559_ = lean_ctor_get(v___y_3558_, 0);
v_isSharedCheck_3566_ = !lean_is_exclusive(v___y_3558_);
if (v_isSharedCheck_3566_ == 0)
{
v___x_3561_ = v___y_3558_;
v_isShared_3562_ = v_isSharedCheck_3566_;
goto v_resetjp_3560_;
}
else
{
lean_inc(v_a_3559_);
lean_dec(v___y_3558_);
v___x_3561_ = lean_box(0);
v_isShared_3562_ = v_isSharedCheck_3566_;
goto v_resetjp_3560_;
}
v_resetjp_3560_:
{
lean_object* v___x_3564_; 
if (v_isShared_3562_ == 0)
{
v___x_3564_ = v___x_3561_;
goto v_reusejp_3563_;
}
else
{
lean_object* v_reuseFailAlloc_3565_; 
v_reuseFailAlloc_3565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3565_, 0, v_a_3559_);
v___x_3564_ = v_reuseFailAlloc_3565_;
goto v_reusejp_3563_;
}
v_reusejp_3563_:
{
return v___x_3564_;
}
}
}
}
v___jp_3567_:
{
lean_object* v___x_3584_; lean_object* v___x_3585_; lean_object* v___x_3586_; lean_object* v___x_3587_; 
v___x_3584_ = lean_unsigned_to_nat(1u);
v___x_3585_ = lean_nat_add(v___y_3577_, v___x_3584_);
lean_inc_ref(v___y_3578_);
lean_inc(v___y_3574_);
lean_inc(v___y_3568_);
lean_inc(v___y_3581_);
lean_inc(v___y_3575_);
lean_inc(v___y_3583_);
lean_inc(v___y_3573_);
lean_inc(v___y_3576_);
lean_inc(v___y_3580_);
lean_inc_ref(v___y_3570_);
lean_inc_ref(v___y_3579_);
lean_inc_ref(v___y_3569_);
v___x_3586_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3586_, 0, v___y_3569_);
lean_ctor_set(v___x_3586_, 1, v___y_3579_);
lean_ctor_set(v___x_3586_, 2, v___y_3570_);
lean_ctor_set(v___x_3586_, 3, v___x_3585_);
lean_ctor_set(v___x_3586_, 4, v___y_3580_);
lean_ctor_set(v___x_3586_, 5, v___y_3582_);
lean_ctor_set(v___x_3586_, 6, v___y_3576_);
lean_ctor_set(v___x_3586_, 7, v___y_3573_);
lean_ctor_set(v___x_3586_, 8, v___y_3583_);
lean_ctor_set(v___x_3586_, 9, v___y_3575_);
lean_ctor_set(v___x_3586_, 10, v___y_3581_);
lean_ctor_set(v___x_3586_, 11, v___y_3568_);
lean_ctor_set(v___x_3586_, 12, v___y_3574_);
lean_ctor_set(v___x_3586_, 13, v___y_3578_);
lean_ctor_set_uint8(v___x_3586_, sizeof(void*)*14, v___y_3571_);
lean_ctor_set_uint8(v___x_3586_, sizeof(void*)*14 + 1, v___y_3572_);
lean_inc(v___y_3555_);
lean_inc(v___y_3553_);
v___x_3587_ = lean_apply_4(v_x_3552_, v___y_3553_, v___x_3586_, v___y_3555_, lean_box(0));
v___y_3558_ = v___x_3587_;
goto v___jp_3557_;
}
v___jp_3604_:
{
lean_object* v___x_3605_; uint8_t v___x_3606_; 
v___x_3605_ = lean_unsigned_to_nat(0u);
v___x_3606_ = lean_nat_dec_eq(v_maxRecDepth_3592_, v___x_3605_);
if (v___x_3606_ == 0)
{
uint8_t v___x_3607_; 
v___x_3607_ = lean_nat_dec_eq(v_currRecDepth_3591_, v_maxRecDepth_3592_);
if (v___x_3607_ == 0)
{
lean_inc(v_ref_3593_);
v___y_3568_ = v_currMacroScope_3599_;
v___y_3569_ = v_fileName_3588_;
v___y_3570_ = v_options_3590_;
v___y_3571_ = v_diag_3600_;
v___y_3572_ = v_suppressElabErrors_3602_;
v___y_3573_ = v_openDecls_3595_;
v___y_3574_ = v_cancelTk_x3f_3601_;
v___y_3575_ = v_maxHeartbeats_3597_;
v___y_3576_ = v_currNamespace_3594_;
v___y_3577_ = v_currRecDepth_3591_;
v___y_3578_ = v_inheritedTraceOptions_3603_;
v___y_3579_ = v_fileMap_3589_;
v___y_3580_ = v_maxRecDepth_3592_;
v___y_3581_ = v_quotContext_3598_;
v___y_3582_ = v_ref_3593_;
v___y_3583_ = v_initHeartbeats_3596_;
goto v___jp_3567_;
}
else
{
lean_object* v___x_3608_; 
lean_dec_ref(v_x_3552_);
lean_inc(v_ref_3593_);
v___x_3608_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg(v_ref_3593_);
v___y_3558_ = v___x_3608_;
goto v___jp_3557_;
}
}
else
{
lean_inc(v_ref_3593_);
v___y_3568_ = v_currMacroScope_3599_;
v___y_3569_ = v_fileName_3588_;
v___y_3570_ = v_options_3590_;
v___y_3571_ = v_diag_3600_;
v___y_3572_ = v_suppressElabErrors_3602_;
v___y_3573_ = v_openDecls_3595_;
v___y_3574_ = v_cancelTk_x3f_3601_;
v___y_3575_ = v_maxHeartbeats_3597_;
v___y_3576_ = v_currNamespace_3594_;
v___y_3577_ = v_currRecDepth_3591_;
v___y_3578_ = v_inheritedTraceOptions_3603_;
v___y_3579_ = v_fileMap_3589_;
v___y_3580_ = v_maxRecDepth_3592_;
v___y_3581_ = v_quotContext_3598_;
v___y_3582_ = v_ref_3593_;
v___y_3583_ = v_initHeartbeats_3596_;
goto v___jp_3567_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_x_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_, lean_object* v___y_3623_, lean_object* v___y_3624_){
_start:
{
lean_object* v_res_3625_; 
v_res_3625_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg(v_x_3620_, v___y_3621_, v___y_3622_, v___y_3623_);
lean_dec(v___y_3623_);
lean_dec_ref(v___y_3622_);
lean_dec(v___y_3621_);
return v_res_3625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1(lean_object* v_pre_3626_, lean_object* v_post_3627_, size_t v_sz_3628_, size_t v_i_3629_, lean_object* v_bs_3630_, lean_object* v___y_3631_, lean_object* v___y_3632_, lean_object* v___y_3633_){
_start:
{
uint8_t v___x_3635_; 
v___x_3635_ = lean_usize_dec_lt(v_i_3629_, v_sz_3628_);
if (v___x_3635_ == 0)
{
lean_object* v___x_3636_; 
lean_dec_ref(v_post_3627_);
lean_dec_ref(v_pre_3626_);
v___x_3636_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3636_, 0, v_bs_3630_);
return v___x_3636_;
}
else
{
lean_object* v_v_3637_; lean_object* v___x_3638_; 
v_v_3637_ = lean_array_uget_borrowed(v_bs_3630_, v_i_3629_);
lean_inc(v_v_3637_);
lean_inc_ref(v_post_3627_);
lean_inc_ref(v_pre_3626_);
v___x_3638_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3626_, v_post_3627_, v_v_3637_, v___y_3631_, v___y_3632_, v___y_3633_);
if (lean_obj_tag(v___x_3638_) == 0)
{
lean_object* v_a_3639_; lean_object* v___x_3640_; lean_object* v_bs_x27_3641_; size_t v___x_3642_; size_t v___x_3643_; lean_object* v___x_3644_; 
v_a_3639_ = lean_ctor_get(v___x_3638_, 0);
lean_inc(v_a_3639_);
lean_dec_ref_known(v___x_3638_, 1);
v___x_3640_ = lean_unsigned_to_nat(0u);
v_bs_x27_3641_ = lean_array_uset(v_bs_3630_, v_i_3629_, v___x_3640_);
v___x_3642_ = ((size_t)1ULL);
v___x_3643_ = lean_usize_add(v_i_3629_, v___x_3642_);
v___x_3644_ = lean_array_uset(v_bs_x27_3641_, v_i_3629_, v_a_3639_);
v_i_3629_ = v___x_3643_;
v_bs_3630_ = v___x_3644_;
goto _start;
}
else
{
lean_object* v_a_3646_; lean_object* v___x_3648_; uint8_t v_isShared_3649_; uint8_t v_isSharedCheck_3653_; 
lean_dec_ref(v_bs_3630_);
lean_dec_ref(v_post_3627_);
lean_dec_ref(v_pre_3626_);
v_a_3646_ = lean_ctor_get(v___x_3638_, 0);
v_isSharedCheck_3653_ = !lean_is_exclusive(v___x_3638_);
if (v_isSharedCheck_3653_ == 0)
{
v___x_3648_ = v___x_3638_;
v_isShared_3649_ = v_isSharedCheck_3653_;
goto v_resetjp_3647_;
}
else
{
lean_inc(v_a_3646_);
lean_dec(v___x_3638_);
v___x_3648_ = lean_box(0);
v_isShared_3649_ = v_isSharedCheck_3653_;
goto v_resetjp_3647_;
}
v_resetjp_3647_:
{
lean_object* v___x_3651_; 
if (v_isShared_3649_ == 0)
{
v___x_3651_ = v___x_3648_;
goto v_reusejp_3650_;
}
else
{
lean_object* v_reuseFailAlloc_3652_; 
v_reuseFailAlloc_3652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3652_, 0, v_a_3646_);
v___x_3651_ = v_reuseFailAlloc_3652_;
goto v_reusejp_3650_;
}
v_reusejp_3650_:
{
return v___x_3651_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3(lean_object* v_pre_3654_, lean_object* v_post_3655_, lean_object* v_x_3656_, lean_object* v_x_3657_, lean_object* v_x_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_){
_start:
{
if (lean_obj_tag(v_x_3656_) == 5)
{
lean_object* v_fn_3663_; lean_object* v_arg_3664_; lean_object* v___x_3665_; lean_object* v___x_3666_; lean_object* v___x_3667_; 
v_fn_3663_ = lean_ctor_get(v_x_3656_, 0);
lean_inc_ref(v_fn_3663_);
v_arg_3664_ = lean_ctor_get(v_x_3656_, 1);
lean_inc_ref(v_arg_3664_);
lean_dec_ref_known(v_x_3656_, 2);
v___x_3665_ = lean_array_set(v_x_3657_, v_x_3658_, v_arg_3664_);
v___x_3666_ = lean_unsigned_to_nat(1u);
v___x_3667_ = lean_nat_sub(v_x_3658_, v___x_3666_);
lean_dec(v_x_3658_);
v_x_3656_ = v_fn_3663_;
v_x_3657_ = v___x_3665_;
v_x_3658_ = v___x_3667_;
goto _start;
}
else
{
lean_object* v___x_3669_; 
lean_dec(v_x_3658_);
lean_inc_ref(v_post_3655_);
lean_inc_ref(v_pre_3654_);
v___x_3669_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3654_, v_post_3655_, v_x_3656_, v___y_3659_, v___y_3660_, v___y_3661_);
if (lean_obj_tag(v___x_3669_) == 0)
{
lean_object* v_a_3670_; size_t v_sz_3671_; size_t v___x_3672_; lean_object* v___x_3673_; 
v_a_3670_ = lean_ctor_get(v___x_3669_, 0);
lean_inc(v_a_3670_);
lean_dec_ref_known(v___x_3669_, 1);
v_sz_3671_ = lean_array_size(v_x_3657_);
v___x_3672_ = ((size_t)0ULL);
lean_inc_ref(v_post_3655_);
lean_inc_ref(v_pre_3654_);
v___x_3673_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1(v_pre_3654_, v_post_3655_, v_sz_3671_, v___x_3672_, v_x_3657_, v___y_3659_, v___y_3660_, v___y_3661_);
if (lean_obj_tag(v___x_3673_) == 0)
{
lean_object* v_a_3674_; lean_object* v___x_3675_; lean_object* v___x_3676_; 
v_a_3674_ = lean_ctor_get(v___x_3673_, 0);
lean_inc(v_a_3674_);
lean_dec_ref_known(v___x_3673_, 1);
v___x_3675_ = l_Lean_mkAppN(v_a_3670_, v_a_3674_);
lean_dec(v_a_3674_);
v___x_3676_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3654_, v_post_3655_, v___x_3675_, v___y_3659_, v___y_3660_, v___y_3661_);
return v___x_3676_;
}
else
{
lean_object* v_a_3677_; lean_object* v___x_3679_; uint8_t v_isShared_3680_; uint8_t v_isSharedCheck_3684_; 
lean_dec(v_a_3670_);
lean_dec_ref(v_post_3655_);
lean_dec_ref(v_pre_3654_);
v_a_3677_ = lean_ctor_get(v___x_3673_, 0);
v_isSharedCheck_3684_ = !lean_is_exclusive(v___x_3673_);
if (v_isSharedCheck_3684_ == 0)
{
v___x_3679_ = v___x_3673_;
v_isShared_3680_ = v_isSharedCheck_3684_;
goto v_resetjp_3678_;
}
else
{
lean_inc(v_a_3677_);
lean_dec(v___x_3673_);
v___x_3679_ = lean_box(0);
v_isShared_3680_ = v_isSharedCheck_3684_;
goto v_resetjp_3678_;
}
v_resetjp_3678_:
{
lean_object* v___x_3682_; 
if (v_isShared_3680_ == 0)
{
v___x_3682_ = v___x_3679_;
goto v_reusejp_3681_;
}
else
{
lean_object* v_reuseFailAlloc_3683_; 
v_reuseFailAlloc_3683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3683_, 0, v_a_3677_);
v___x_3682_ = v_reuseFailAlloc_3683_;
goto v_reusejp_3681_;
}
v_reusejp_3681_:
{
return v___x_3682_;
}
}
}
}
else
{
lean_dec_ref(v_x_3657_);
lean_dec_ref(v_post_3655_);
lean_dec_ref(v_pre_3654_);
return v___x_3669_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1(lean_object* v___x_3685_, lean_object* v_pre_3686_, lean_object* v_e_3687_, lean_object* v_post_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_){
_start:
{
lean_object* v___y_3694_; lean_object* v___y_3695_; lean_object* v___y_3696_; lean_object* v___y_3697_; lean_object* v___y_3698_; uint8_t v___y_3699_; lean_object* v___y_3700_; uint8_t v___y_3701_; lean_object* v___y_3711_; lean_object* v___y_3712_; lean_object* v___y_3713_; uint8_t v___y_3714_; lean_object* v___y_3715_; uint8_t v___y_3716_; lean_object* v___y_3724_; lean_object* v___y_3725_; uint8_t v___y_3726_; lean_object* v___y_3727_; lean_object* v___y_3728_; uint8_t v___y_3729_; lean_object* v___x_3736_; 
v___x_3736_ = l_Lean_Core_checkSystem(v___x_3685_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3736_) == 0)
{
lean_object* v___x_3737_; 
lean_dec_ref_known(v___x_3736_, 1);
lean_inc_ref(v_pre_3686_);
lean_inc(v___y_3691_);
lean_inc_ref(v___y_3690_);
lean_inc_ref(v_e_3687_);
v___x_3737_ = lean_apply_4(v_pre_3686_, v_e_3687_, v___y_3690_, v___y_3691_, lean_box(0));
if (lean_obj_tag(v___x_3737_) == 0)
{
lean_object* v_a_3738_; lean_object* v___x_3740_; uint8_t v_isShared_3741_; uint8_t v_isSharedCheck_3827_; 
v_a_3738_ = lean_ctor_get(v___x_3737_, 0);
v_isSharedCheck_3827_ = !lean_is_exclusive(v___x_3737_);
if (v_isSharedCheck_3827_ == 0)
{
v___x_3740_ = v___x_3737_;
v_isShared_3741_ = v_isSharedCheck_3827_;
goto v_resetjp_3739_;
}
else
{
lean_inc(v_a_3738_);
lean_dec(v___x_3737_);
v___x_3740_ = lean_box(0);
v_isShared_3741_ = v_isSharedCheck_3827_;
goto v_resetjp_3739_;
}
v_resetjp_3739_:
{
lean_object* v___y_3743_; 
switch(lean_obj_tag(v_a_3738_))
{
case 0:
{
lean_object* v_e_3817_; lean_object* v___x_3819_; 
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_e_3687_);
lean_dec_ref(v_pre_3686_);
v_e_3817_ = lean_ctor_get(v_a_3738_, 0);
lean_inc_ref(v_e_3817_);
lean_dec_ref_known(v_a_3738_, 1);
if (v_isShared_3741_ == 0)
{
lean_ctor_set(v___x_3740_, 0, v_e_3817_);
v___x_3819_ = v___x_3740_;
goto v_reusejp_3818_;
}
else
{
lean_object* v_reuseFailAlloc_3820_; 
v_reuseFailAlloc_3820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3820_, 0, v_e_3817_);
v___x_3819_ = v_reuseFailAlloc_3820_;
goto v_reusejp_3818_;
}
v_reusejp_3818_:
{
return v___x_3819_;
}
}
case 1:
{
lean_object* v_e_3821_; lean_object* v___x_3822_; 
lean_del_object(v___x_3740_);
lean_dec_ref(v_e_3687_);
v_e_3821_ = lean_ctor_get(v_a_3738_, 0);
lean_inc_ref(v_e_3821_);
lean_dec_ref_known(v_a_3738_, 1);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3822_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_e_3821_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3822_) == 0)
{
lean_object* v_a_3823_; lean_object* v___x_3824_; 
v_a_3823_ = lean_ctor_get(v___x_3822_, 0);
lean_inc(v_a_3823_);
lean_dec_ref_known(v___x_3822_, 1);
v___x_3824_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v_a_3823_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3824_;
}
else
{
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3822_;
}
}
default: 
{
lean_object* v_e_x3f_3825_; 
lean_del_object(v___x_3740_);
v_e_x3f_3825_ = lean_ctor_get(v_a_3738_, 0);
lean_inc(v_e_x3f_3825_);
lean_dec_ref_known(v_a_3738_, 1);
if (lean_obj_tag(v_e_x3f_3825_) == 0)
{
v___y_3743_ = v_e_3687_;
goto v___jp_3742_;
}
else
{
lean_object* v_val_3826_; 
lean_dec_ref(v_e_3687_);
v_val_3826_ = lean_ctor_get(v_e_x3f_3825_, 0);
lean_inc(v_val_3826_);
lean_dec_ref_known(v_e_x3f_3825_, 1);
v___y_3743_ = v_val_3826_;
goto v___jp_3742_;
}
}
}
v___jp_3742_:
{
switch(lean_obj_tag(v___y_3743_))
{
case 7:
{
lean_object* v_binderName_3744_; lean_object* v_binderType_3745_; lean_object* v_body_3746_; uint8_t v_binderInfo_3747_; lean_object* v___x_3748_; 
v_binderName_3744_ = lean_ctor_get(v___y_3743_, 0);
lean_inc(v_binderName_3744_);
v_binderType_3745_ = lean_ctor_get(v___y_3743_, 1);
v_body_3746_ = lean_ctor_get(v___y_3743_, 2);
v_binderInfo_3747_ = lean_ctor_get_uint8(v___y_3743_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_3745_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3748_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_binderType_3745_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3748_) == 0)
{
lean_object* v_a_3749_; lean_object* v___x_3750_; 
v_a_3749_ = lean_ctor_get(v___x_3748_, 0);
lean_inc(v_a_3749_);
lean_dec_ref_known(v___x_3748_, 1);
lean_inc_ref(v_body_3746_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3750_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_body_3746_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3750_) == 0)
{
lean_object* v_a_3751_; size_t v___x_3752_; size_t v___x_3753_; uint8_t v___x_3754_; 
v_a_3751_ = lean_ctor_get(v___x_3750_, 0);
lean_inc(v_a_3751_);
lean_dec_ref_known(v___x_3750_, 1);
v___x_3752_ = lean_ptr_addr(v_binderType_3745_);
v___x_3753_ = lean_ptr_addr(v_a_3749_);
v___x_3754_ = lean_usize_dec_eq(v___x_3752_, v___x_3753_);
if (v___x_3754_ == 0)
{
v___y_3724_ = v_a_3751_;
v___y_3725_ = v___y_3743_;
v___y_3726_ = v_binderInfo_3747_;
v___y_3727_ = v_binderName_3744_;
v___y_3728_ = v_a_3749_;
v___y_3729_ = v___x_3754_;
goto v___jp_3723_;
}
else
{
size_t v___x_3755_; size_t v___x_3756_; uint8_t v___x_3757_; 
v___x_3755_ = lean_ptr_addr(v_body_3746_);
v___x_3756_ = lean_ptr_addr(v_a_3751_);
v___x_3757_ = lean_usize_dec_eq(v___x_3755_, v___x_3756_);
v___y_3724_ = v_a_3751_;
v___y_3725_ = v___y_3743_;
v___y_3726_ = v_binderInfo_3747_;
v___y_3727_ = v_binderName_3744_;
v___y_3728_ = v_a_3749_;
v___y_3729_ = v___x_3757_;
goto v___jp_3723_;
}
}
else
{
lean_dec(v_a_3749_);
lean_dec_ref_known(v___y_3743_, 3);
lean_dec(v_binderName_3744_);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3750_;
}
}
else
{
lean_dec(v_binderName_3744_);
lean_dec_ref_known(v___y_3743_, 3);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3748_;
}
}
case 6:
{
lean_object* v_binderName_3758_; lean_object* v_binderType_3759_; lean_object* v_body_3760_; uint8_t v_binderInfo_3761_; lean_object* v___x_3762_; 
v_binderName_3758_ = lean_ctor_get(v___y_3743_, 0);
lean_inc(v_binderName_3758_);
v_binderType_3759_ = lean_ctor_get(v___y_3743_, 1);
v_body_3760_ = lean_ctor_get(v___y_3743_, 2);
v_binderInfo_3761_ = lean_ctor_get_uint8(v___y_3743_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_3759_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3762_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_binderType_3759_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3762_) == 0)
{
lean_object* v_a_3763_; lean_object* v___x_3764_; 
v_a_3763_ = lean_ctor_get(v___x_3762_, 0);
lean_inc(v_a_3763_);
lean_dec_ref_known(v___x_3762_, 1);
lean_inc_ref(v_body_3760_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3764_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_body_3760_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3764_) == 0)
{
lean_object* v_a_3765_; size_t v___x_3766_; size_t v___x_3767_; uint8_t v___x_3768_; 
v_a_3765_ = lean_ctor_get(v___x_3764_, 0);
lean_inc(v_a_3765_);
lean_dec_ref_known(v___x_3764_, 1);
v___x_3766_ = lean_ptr_addr(v_binderType_3759_);
v___x_3767_ = lean_ptr_addr(v_a_3763_);
v___x_3768_ = lean_usize_dec_eq(v___x_3766_, v___x_3767_);
if (v___x_3768_ == 0)
{
v___y_3711_ = v_a_3765_;
v___y_3712_ = v___y_3743_;
v___y_3713_ = v_a_3763_;
v___y_3714_ = v_binderInfo_3761_;
v___y_3715_ = v_binderName_3758_;
v___y_3716_ = v___x_3768_;
goto v___jp_3710_;
}
else
{
size_t v___x_3769_; size_t v___x_3770_; uint8_t v___x_3771_; 
v___x_3769_ = lean_ptr_addr(v_body_3760_);
v___x_3770_ = lean_ptr_addr(v_a_3765_);
v___x_3771_ = lean_usize_dec_eq(v___x_3769_, v___x_3770_);
v___y_3711_ = v_a_3765_;
v___y_3712_ = v___y_3743_;
v___y_3713_ = v_a_3763_;
v___y_3714_ = v_binderInfo_3761_;
v___y_3715_ = v_binderName_3758_;
v___y_3716_ = v___x_3771_;
goto v___jp_3710_;
}
}
else
{
lean_dec(v_a_3763_);
lean_dec_ref_known(v___y_3743_, 3);
lean_dec(v_binderName_3758_);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3764_;
}
}
else
{
lean_dec_ref_known(v___y_3743_, 3);
lean_dec(v_binderName_3758_);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3762_;
}
}
case 8:
{
lean_object* v_declName_3772_; lean_object* v_type_3773_; lean_object* v_value_3774_; lean_object* v_body_3775_; uint8_t v_nondep_3776_; lean_object* v___x_3777_; 
v_declName_3772_ = lean_ctor_get(v___y_3743_, 0);
lean_inc(v_declName_3772_);
v_type_3773_ = lean_ctor_get(v___y_3743_, 1);
v_value_3774_ = lean_ctor_get(v___y_3743_, 2);
v_body_3775_ = lean_ctor_get(v___y_3743_, 3);
lean_inc_ref(v_body_3775_);
v_nondep_3776_ = lean_ctor_get_uint8(v___y_3743_, sizeof(void*)*4 + 8);
lean_inc_ref(v_type_3773_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3777_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_type_3773_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3777_) == 0)
{
lean_object* v_a_3778_; lean_object* v___x_3779_; 
v_a_3778_ = lean_ctor_get(v___x_3777_, 0);
lean_inc(v_a_3778_);
lean_dec_ref_known(v___x_3777_, 1);
lean_inc_ref(v_value_3774_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3779_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_value_3774_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3779_) == 0)
{
lean_object* v_a_3780_; lean_object* v___x_3781_; 
v_a_3780_ = lean_ctor_get(v___x_3779_, 0);
lean_inc(v_a_3780_);
lean_dec_ref_known(v___x_3779_, 1);
lean_inc_ref(v_body_3775_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3781_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_body_3775_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3781_) == 0)
{
lean_object* v_a_3782_; size_t v___x_3783_; size_t v___x_3784_; uint8_t v___x_3785_; 
v_a_3782_ = lean_ctor_get(v___x_3781_, 0);
lean_inc(v_a_3782_);
lean_dec_ref_known(v___x_3781_, 1);
v___x_3783_ = lean_ptr_addr(v_type_3773_);
v___x_3784_ = lean_ptr_addr(v_a_3778_);
v___x_3785_ = lean_usize_dec_eq(v___x_3783_, v___x_3784_);
if (v___x_3785_ == 0)
{
v___y_3694_ = v_body_3775_;
v___y_3695_ = v_a_3780_;
v___y_3696_ = v_a_3782_;
v___y_3697_ = v_declName_3772_;
v___y_3698_ = v___y_3743_;
v___y_3699_ = v_nondep_3776_;
v___y_3700_ = v_a_3778_;
v___y_3701_ = v___x_3785_;
goto v___jp_3693_;
}
else
{
size_t v___x_3786_; size_t v___x_3787_; uint8_t v___x_3788_; 
v___x_3786_ = lean_ptr_addr(v_value_3774_);
v___x_3787_ = lean_ptr_addr(v_a_3780_);
v___x_3788_ = lean_usize_dec_eq(v___x_3786_, v___x_3787_);
v___y_3694_ = v_body_3775_;
v___y_3695_ = v_a_3780_;
v___y_3696_ = v_a_3782_;
v___y_3697_ = v_declName_3772_;
v___y_3698_ = v___y_3743_;
v___y_3699_ = v_nondep_3776_;
v___y_3700_ = v_a_3778_;
v___y_3701_ = v___x_3788_;
goto v___jp_3693_;
}
}
else
{
lean_dec(v_a_3780_);
lean_dec(v_a_3778_);
lean_dec_ref(v_body_3775_);
lean_dec(v_declName_3772_);
lean_dec_ref_known(v___y_3743_, 4);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3781_;
}
}
else
{
lean_dec(v_a_3778_);
lean_dec_ref(v_body_3775_);
lean_dec(v_declName_3772_);
lean_dec_ref_known(v___y_3743_, 4);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3779_;
}
}
else
{
lean_dec_ref(v_body_3775_);
lean_dec_ref_known(v___y_3743_, 4);
lean_dec(v_declName_3772_);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3777_;
}
}
case 5:
{
lean_object* v_dummy_3789_; lean_object* v_nargs_3790_; lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v___x_3793_; lean_object* v___x_3794_; 
v_dummy_3789_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0, &lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_unfoldConsts___closed__0);
v_nargs_3790_ = l_Lean_Expr_getAppNumArgs(v___y_3743_);
lean_inc(v_nargs_3790_);
v___x_3791_ = lean_mk_array(v_nargs_3790_, v_dummy_3789_);
v___x_3792_ = lean_unsigned_to_nat(1u);
v___x_3793_ = lean_nat_sub(v_nargs_3790_, v___x_3792_);
lean_dec(v_nargs_3790_);
v___x_3794_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3(v_pre_3686_, v_post_3688_, v___y_3743_, v___x_3791_, v___x_3793_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3794_;
}
case 10:
{
lean_object* v_data_3795_; lean_object* v_expr_3796_; lean_object* v___x_3797_; 
v_data_3795_ = lean_ctor_get(v___y_3743_, 0);
v_expr_3796_ = lean_ctor_get(v___y_3743_, 1);
lean_inc_ref(v_expr_3796_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3797_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_expr_3796_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3797_) == 0)
{
lean_object* v_a_3798_; size_t v___x_3799_; size_t v___x_3800_; uint8_t v___x_3801_; 
v_a_3798_ = lean_ctor_get(v___x_3797_, 0);
lean_inc(v_a_3798_);
lean_dec_ref_known(v___x_3797_, 1);
v___x_3799_ = lean_ptr_addr(v_expr_3796_);
v___x_3800_ = lean_ptr_addr(v_a_3798_);
v___x_3801_ = lean_usize_dec_eq(v___x_3799_, v___x_3800_);
if (v___x_3801_ == 0)
{
lean_object* v___x_3802_; lean_object* v___x_3803_; 
lean_inc(v_data_3795_);
lean_dec_ref_known(v___y_3743_, 2);
v___x_3802_ = l_Lean_Expr_mdata___override(v_data_3795_, v_a_3798_);
v___x_3803_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3802_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3803_;
}
else
{
lean_object* v___x_3804_; 
lean_dec(v_a_3798_);
v___x_3804_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3743_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3804_;
}
}
else
{
lean_dec_ref_known(v___y_3743_, 2);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3797_;
}
}
case 11:
{
lean_object* v_typeName_3805_; lean_object* v_idx_3806_; lean_object* v_struct_3807_; lean_object* v___x_3808_; 
v_typeName_3805_ = lean_ctor_get(v___y_3743_, 0);
v_idx_3806_ = lean_ctor_get(v___y_3743_, 1);
v_struct_3807_ = lean_ctor_get(v___y_3743_, 2);
lean_inc_ref(v_struct_3807_);
lean_inc_ref(v_post_3688_);
lean_inc_ref(v_pre_3686_);
v___x_3808_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3686_, v_post_3688_, v_struct_3807_, v___y_3689_, v___y_3690_, v___y_3691_);
if (lean_obj_tag(v___x_3808_) == 0)
{
lean_object* v_a_3809_; size_t v___x_3810_; size_t v___x_3811_; uint8_t v___x_3812_; 
v_a_3809_ = lean_ctor_get(v___x_3808_, 0);
lean_inc(v_a_3809_);
lean_dec_ref_known(v___x_3808_, 1);
v___x_3810_ = lean_ptr_addr(v_struct_3807_);
v___x_3811_ = lean_ptr_addr(v_a_3809_);
v___x_3812_ = lean_usize_dec_eq(v___x_3810_, v___x_3811_);
if (v___x_3812_ == 0)
{
lean_object* v___x_3813_; lean_object* v___x_3814_; 
lean_inc(v_idx_3806_);
lean_inc(v_typeName_3805_);
lean_dec_ref_known(v___y_3743_, 3);
v___x_3813_ = l_Lean_Expr_proj___override(v_typeName_3805_, v_idx_3806_, v_a_3809_);
v___x_3814_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3813_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3814_;
}
else
{
lean_object* v___x_3815_; 
lean_dec(v_a_3809_);
v___x_3815_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3743_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3815_;
}
}
else
{
lean_dec_ref_known(v___y_3743_, 3);
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_pre_3686_);
return v___x_3808_;
}
}
default: 
{
lean_object* v___x_3816_; 
v___x_3816_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3743_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3816_;
}
}
}
}
}
else
{
lean_object* v_a_3828_; lean_object* v___x_3830_; uint8_t v_isShared_3831_; uint8_t v_isSharedCheck_3835_; 
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_e_3687_);
lean_dec_ref(v_pre_3686_);
v_a_3828_ = lean_ctor_get(v___x_3737_, 0);
v_isSharedCheck_3835_ = !lean_is_exclusive(v___x_3737_);
if (v_isSharedCheck_3835_ == 0)
{
v___x_3830_ = v___x_3737_;
v_isShared_3831_ = v_isSharedCheck_3835_;
goto v_resetjp_3829_;
}
else
{
lean_inc(v_a_3828_);
lean_dec(v___x_3737_);
v___x_3830_ = lean_box(0);
v_isShared_3831_ = v_isSharedCheck_3835_;
goto v_resetjp_3829_;
}
v_resetjp_3829_:
{
lean_object* v___x_3833_; 
if (v_isShared_3831_ == 0)
{
v___x_3833_ = v___x_3830_;
goto v_reusejp_3832_;
}
else
{
lean_object* v_reuseFailAlloc_3834_; 
v_reuseFailAlloc_3834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3834_, 0, v_a_3828_);
v___x_3833_ = v_reuseFailAlloc_3834_;
goto v_reusejp_3832_;
}
v_reusejp_3832_:
{
return v___x_3833_;
}
}
}
}
else
{
lean_object* v_a_3836_; lean_object* v___x_3838_; uint8_t v_isShared_3839_; uint8_t v_isSharedCheck_3843_; 
lean_dec_ref(v_post_3688_);
lean_dec_ref(v_e_3687_);
lean_dec_ref(v_pre_3686_);
v_a_3836_ = lean_ctor_get(v___x_3736_, 0);
v_isSharedCheck_3843_ = !lean_is_exclusive(v___x_3736_);
if (v_isSharedCheck_3843_ == 0)
{
v___x_3838_ = v___x_3736_;
v_isShared_3839_ = v_isSharedCheck_3843_;
goto v_resetjp_3837_;
}
else
{
lean_inc(v_a_3836_);
lean_dec(v___x_3736_);
v___x_3838_ = lean_box(0);
v_isShared_3839_ = v_isSharedCheck_3843_;
goto v_resetjp_3837_;
}
v_resetjp_3837_:
{
lean_object* v___x_3841_; 
if (v_isShared_3839_ == 0)
{
v___x_3841_ = v___x_3838_;
goto v_reusejp_3840_;
}
else
{
lean_object* v_reuseFailAlloc_3842_; 
v_reuseFailAlloc_3842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3842_, 0, v_a_3836_);
v___x_3841_ = v_reuseFailAlloc_3842_;
goto v_reusejp_3840_;
}
v_reusejp_3840_:
{
return v___x_3841_;
}
}
}
v___jp_3693_:
{
if (v___y_3701_ == 0)
{
lean_object* v___x_3702_; lean_object* v___x_3703_; 
lean_dec_ref(v___y_3698_);
lean_dec_ref(v___y_3694_);
v___x_3702_ = l_Lean_Expr_letE___override(v___y_3697_, v___y_3700_, v___y_3695_, v___y_3696_, v___y_3699_);
v___x_3703_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3702_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3703_;
}
else
{
size_t v___x_3704_; size_t v___x_3705_; uint8_t v___x_3706_; 
v___x_3704_ = lean_ptr_addr(v___y_3694_);
lean_dec_ref(v___y_3694_);
v___x_3705_ = lean_ptr_addr(v___y_3696_);
v___x_3706_ = lean_usize_dec_eq(v___x_3704_, v___x_3705_);
if (v___x_3706_ == 0)
{
lean_object* v___x_3707_; lean_object* v___x_3708_; 
lean_dec_ref(v___y_3698_);
v___x_3707_ = l_Lean_Expr_letE___override(v___y_3697_, v___y_3700_, v___y_3695_, v___y_3696_, v___y_3699_);
v___x_3708_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3707_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3708_;
}
else
{
lean_object* v___x_3709_; 
lean_dec_ref(v___y_3700_);
lean_dec(v___y_3697_);
lean_dec_ref(v___y_3696_);
lean_dec_ref(v___y_3695_);
v___x_3709_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3698_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3709_;
}
}
}
v___jp_3710_:
{
if (v___y_3716_ == 0)
{
lean_object* v___x_3717_; lean_object* v___x_3718_; 
lean_dec_ref(v___y_3712_);
v___x_3717_ = l_Lean_Expr_lam___override(v___y_3715_, v___y_3713_, v___y_3711_, v___y_3714_);
v___x_3718_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3717_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3718_;
}
else
{
uint8_t v___x_3719_; 
v___x_3719_ = l_Lean_instBEqBinderInfo_beq(v___y_3714_, v___y_3714_);
if (v___x_3719_ == 0)
{
lean_object* v___x_3720_; lean_object* v___x_3721_; 
lean_dec_ref(v___y_3712_);
v___x_3720_ = l_Lean_Expr_lam___override(v___y_3715_, v___y_3713_, v___y_3711_, v___y_3714_);
v___x_3721_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3720_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3721_;
}
else
{
lean_object* v___x_3722_; 
lean_dec(v___y_3715_);
lean_dec_ref(v___y_3713_);
lean_dec_ref(v___y_3711_);
v___x_3722_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3712_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3722_;
}
}
}
v___jp_3723_:
{
if (v___y_3729_ == 0)
{
lean_object* v___x_3730_; lean_object* v___x_3731_; 
lean_dec_ref(v___y_3725_);
v___x_3730_ = l_Lean_Expr_forallE___override(v___y_3727_, v___y_3728_, v___y_3724_, v___y_3726_);
v___x_3731_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3730_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3731_;
}
else
{
uint8_t v___x_3732_; 
v___x_3732_ = l_Lean_instBEqBinderInfo_beq(v___y_3726_, v___y_3726_);
if (v___x_3732_ == 0)
{
lean_object* v___x_3733_; lean_object* v___x_3734_; 
lean_dec_ref(v___y_3725_);
v___x_3733_ = l_Lean_Expr_forallE___override(v___y_3727_, v___y_3728_, v___y_3724_, v___y_3726_);
v___x_3734_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___x_3733_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3734_;
}
else
{
lean_object* v___x_3735_; 
lean_dec_ref(v___y_3728_);
lean_dec(v___y_3727_);
lean_dec_ref(v___y_3724_);
v___x_3735_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3686_, v_post_3688_, v___y_3725_, v___y_3689_, v___y_3690_, v___y_3691_);
return v___x_3735_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1___boxed(lean_object* v___x_3844_, lean_object* v_pre_3845_, lean_object* v_e_3846_, lean_object* v_post_3847_, lean_object* v___y_3848_, lean_object* v___y_3849_, lean_object* v___y_3850_, lean_object* v___y_3851_){
_start:
{
lean_object* v_res_3852_; 
v_res_3852_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1(v___x_3844_, v_pre_3845_, v_e_3846_, v_post_3847_, v___y_3848_, v___y_3849_, v___y_3850_);
lean_dec(v___y_3850_);
lean_dec_ref(v___y_3849_);
lean_dec(v___y_3848_);
return v_res_3852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(lean_object* v_pre_3853_, lean_object* v_post_3854_, lean_object* v_e_3855_, lean_object* v_a_3856_, lean_object* v___y_3857_, lean_object* v___y_3858_){
_start:
{
lean_object* v___x_3860_; lean_object* v___x_3861_; 
lean_inc(v_a_3856_);
v___x_3860_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_3860_, 0, lean_box(0));
lean_closure_set(v___x_3860_, 1, lean_box(0));
lean_closure_set(v___x_3860_, 2, v_a_3856_);
v___x_3861_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0(lean_box(0), v___x_3860_, v___y_3857_, v___y_3858_);
if (lean_obj_tag(v___x_3861_) == 0)
{
lean_object* v_a_3862_; lean_object* v___x_3864_; uint8_t v_isShared_3865_; uint8_t v_isSharedCheck_3893_; 
v_a_3862_ = lean_ctor_get(v___x_3861_, 0);
v_isSharedCheck_3893_ = !lean_is_exclusive(v___x_3861_);
if (v_isSharedCheck_3893_ == 0)
{
v___x_3864_ = v___x_3861_;
v_isShared_3865_ = v_isSharedCheck_3893_;
goto v_resetjp_3863_;
}
else
{
lean_inc(v_a_3862_);
lean_dec(v___x_3861_);
v___x_3864_ = lean_box(0);
v_isShared_3865_ = v_isSharedCheck_3893_;
goto v_resetjp_3863_;
}
v_resetjp_3863_:
{
lean_object* v___x_3866_; 
v___x_3866_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4_spec__8___redArg(v_a_3862_, v_e_3855_);
lean_dec(v_a_3862_);
if (lean_obj_tag(v___x_3866_) == 0)
{
lean_object* v___x_3867_; lean_object* v___f_3868_; lean_object* v___x_3869_; 
lean_del_object(v___x_3864_);
v___x_3867_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___closed__0));
lean_inc_ref(v_e_3855_);
v___f_3868_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__1___boxed), 8, 4);
lean_closure_set(v___f_3868_, 0, v___x_3867_);
lean_closure_set(v___f_3868_, 1, v_pre_3853_);
lean_closure_set(v___f_3868_, 2, v_e_3855_);
lean_closure_set(v___f_3868_, 3, v_post_3854_);
v___x_3869_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg(v___f_3868_, v_a_3856_, v___y_3857_, v___y_3858_);
if (lean_obj_tag(v___x_3869_) == 0)
{
lean_object* v_a_3870_; lean_object* v___f_3871_; lean_object* v___x_3872_; 
v_a_3870_ = lean_ctor_get(v___x_3869_, 0);
lean_inc_n(v_a_3870_, 2);
lean_dec_ref_known(v___x_3869_, 1);
lean_inc(v_a_3856_);
v___f_3871_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3_spec__4___lam__2___boxed), 4, 3);
lean_closure_set(v___f_3871_, 0, v_a_3856_);
lean_closure_set(v___f_3871_, 1, v_e_3855_);
lean_closure_set(v___f_3871_, 2, v_a_3870_);
v___x_3872_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___lam__0(lean_box(0), v___f_3871_, v___y_3857_, v___y_3858_);
if (lean_obj_tag(v___x_3872_) == 0)
{
lean_object* v___x_3874_; uint8_t v_isShared_3875_; uint8_t v_isSharedCheck_3879_; 
v_isSharedCheck_3879_ = !lean_is_exclusive(v___x_3872_);
if (v_isSharedCheck_3879_ == 0)
{
lean_object* v_unused_3880_; 
v_unused_3880_ = lean_ctor_get(v___x_3872_, 0);
lean_dec(v_unused_3880_);
v___x_3874_ = v___x_3872_;
v_isShared_3875_ = v_isSharedCheck_3879_;
goto v_resetjp_3873_;
}
else
{
lean_dec(v___x_3872_);
v___x_3874_ = lean_box(0);
v_isShared_3875_ = v_isSharedCheck_3879_;
goto v_resetjp_3873_;
}
v_resetjp_3873_:
{
lean_object* v___x_3877_; 
if (v_isShared_3875_ == 0)
{
lean_ctor_set(v___x_3874_, 0, v_a_3870_);
v___x_3877_ = v___x_3874_;
goto v_reusejp_3876_;
}
else
{
lean_object* v_reuseFailAlloc_3878_; 
v_reuseFailAlloc_3878_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3878_, 0, v_a_3870_);
v___x_3877_ = v_reuseFailAlloc_3878_;
goto v_reusejp_3876_;
}
v_reusejp_3876_:
{
return v___x_3877_;
}
}
}
else
{
lean_object* v_a_3881_; lean_object* v___x_3883_; uint8_t v_isShared_3884_; uint8_t v_isSharedCheck_3888_; 
lean_dec(v_a_3870_);
v_a_3881_ = lean_ctor_get(v___x_3872_, 0);
v_isSharedCheck_3888_ = !lean_is_exclusive(v___x_3872_);
if (v_isSharedCheck_3888_ == 0)
{
v___x_3883_ = v___x_3872_;
v_isShared_3884_ = v_isSharedCheck_3888_;
goto v_resetjp_3882_;
}
else
{
lean_inc(v_a_3881_);
lean_dec(v___x_3872_);
v___x_3883_ = lean_box(0);
v_isShared_3884_ = v_isSharedCheck_3888_;
goto v_resetjp_3882_;
}
v_resetjp_3882_:
{
lean_object* v___x_3886_; 
if (v_isShared_3884_ == 0)
{
v___x_3886_ = v___x_3883_;
goto v_reusejp_3885_;
}
else
{
lean_object* v_reuseFailAlloc_3887_; 
v_reuseFailAlloc_3887_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3887_, 0, v_a_3881_);
v___x_3886_ = v_reuseFailAlloc_3887_;
goto v_reusejp_3885_;
}
v_reusejp_3885_:
{
return v___x_3886_;
}
}
}
}
else
{
lean_dec_ref(v_e_3855_);
return v___x_3869_;
}
}
else
{
lean_object* v_val_3889_; lean_object* v___x_3891_; 
lean_dec_ref(v_e_3855_);
lean_dec_ref(v_post_3854_);
lean_dec_ref(v_pre_3853_);
v_val_3889_ = lean_ctor_get(v___x_3866_, 0);
lean_inc(v_val_3889_);
lean_dec_ref_known(v___x_3866_, 1);
if (v_isShared_3865_ == 0)
{
lean_ctor_set(v___x_3864_, 0, v_val_3889_);
v___x_3891_ = v___x_3864_;
goto v_reusejp_3890_;
}
else
{
lean_object* v_reuseFailAlloc_3892_; 
v_reuseFailAlloc_3892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3892_, 0, v_val_3889_);
v___x_3891_ = v_reuseFailAlloc_3892_;
goto v_reusejp_3890_;
}
v_reusejp_3890_:
{
return v___x_3891_;
}
}
}
}
else
{
lean_object* v_a_3894_; lean_object* v___x_3896_; uint8_t v_isShared_3897_; uint8_t v_isSharedCheck_3901_; 
lean_dec_ref(v_e_3855_);
lean_dec_ref(v_post_3854_);
lean_dec_ref(v_pre_3853_);
v_a_3894_ = lean_ctor_get(v___x_3861_, 0);
v_isSharedCheck_3901_ = !lean_is_exclusive(v___x_3861_);
if (v_isSharedCheck_3901_ == 0)
{
v___x_3896_ = v___x_3861_;
v_isShared_3897_ = v_isSharedCheck_3901_;
goto v_resetjp_3895_;
}
else
{
lean_inc(v_a_3894_);
lean_dec(v___x_3861_);
v___x_3896_ = lean_box(0);
v_isShared_3897_ = v_isSharedCheck_3901_;
goto v_resetjp_3895_;
}
v_resetjp_3895_:
{
lean_object* v___x_3899_; 
if (v_isShared_3897_ == 0)
{
v___x_3899_ = v___x_3896_;
goto v_reusejp_3898_;
}
else
{
lean_object* v_reuseFailAlloc_3900_; 
v_reuseFailAlloc_3900_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3900_, 0, v_a_3894_);
v___x_3899_ = v_reuseFailAlloc_3900_;
goto v_reusejp_3898_;
}
v_reusejp_3898_:
{
return v___x_3899_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(lean_object* v_pre_3902_, lean_object* v_post_3903_, lean_object* v_e_3904_, lean_object* v_a_3905_, lean_object* v___y_3906_, lean_object* v___y_3907_){
_start:
{
lean_object* v___x_3909_; 
lean_inc_ref(v_post_3903_);
lean_inc(v___y_3907_);
lean_inc_ref(v___y_3906_);
lean_inc_ref(v_e_3904_);
v___x_3909_ = lean_apply_4(v_post_3903_, v_e_3904_, v___y_3906_, v___y_3907_, lean_box(0));
if (lean_obj_tag(v___x_3909_) == 0)
{
lean_object* v_a_3910_; lean_object* v___x_3912_; uint8_t v_isShared_3913_; uint8_t v_isSharedCheck_3928_; 
v_a_3910_ = lean_ctor_get(v___x_3909_, 0);
v_isSharedCheck_3928_ = !lean_is_exclusive(v___x_3909_);
if (v_isSharedCheck_3928_ == 0)
{
v___x_3912_ = v___x_3909_;
v_isShared_3913_ = v_isSharedCheck_3928_;
goto v_resetjp_3911_;
}
else
{
lean_inc(v_a_3910_);
lean_dec(v___x_3909_);
v___x_3912_ = lean_box(0);
v_isShared_3913_ = v_isSharedCheck_3928_;
goto v_resetjp_3911_;
}
v_resetjp_3911_:
{
switch(lean_obj_tag(v_a_3910_))
{
case 0:
{
lean_object* v_e_3914_; lean_object* v___x_3916_; 
lean_dec_ref(v_e_3904_);
lean_dec_ref(v_post_3903_);
lean_dec_ref(v_pre_3902_);
v_e_3914_ = lean_ctor_get(v_a_3910_, 0);
lean_inc_ref(v_e_3914_);
lean_dec_ref_known(v_a_3910_, 1);
if (v_isShared_3913_ == 0)
{
lean_ctor_set(v___x_3912_, 0, v_e_3914_);
v___x_3916_ = v___x_3912_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_3917_; 
v_reuseFailAlloc_3917_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3917_, 0, v_e_3914_);
v___x_3916_ = v_reuseFailAlloc_3917_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
return v___x_3916_;
}
}
case 1:
{
lean_object* v_e_3918_; lean_object* v___x_3919_; 
lean_del_object(v___x_3912_);
lean_dec_ref(v_e_3904_);
v_e_3918_ = lean_ctor_get(v_a_3910_, 0);
lean_inc_ref(v_e_3918_);
lean_dec_ref_known(v_a_3910_, 1);
v___x_3919_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3902_, v_post_3903_, v_e_3918_, v_a_3905_, v___y_3906_, v___y_3907_);
return v___x_3919_;
}
default: 
{
lean_object* v_e_x3f_3920_; 
lean_dec_ref(v_post_3903_);
lean_dec_ref(v_pre_3902_);
v_e_x3f_3920_ = lean_ctor_get(v_a_3910_, 0);
lean_inc(v_e_x3f_3920_);
lean_dec_ref_known(v_a_3910_, 1);
if (lean_obj_tag(v_e_x3f_3920_) == 0)
{
lean_object* v___x_3922_; 
if (v_isShared_3913_ == 0)
{
lean_ctor_set(v___x_3912_, 0, v_e_3904_);
v___x_3922_ = v___x_3912_;
goto v_reusejp_3921_;
}
else
{
lean_object* v_reuseFailAlloc_3923_; 
v_reuseFailAlloc_3923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3923_, 0, v_e_3904_);
v___x_3922_ = v_reuseFailAlloc_3923_;
goto v_reusejp_3921_;
}
v_reusejp_3921_:
{
return v___x_3922_;
}
}
else
{
lean_object* v_val_3924_; lean_object* v___x_3926_; 
lean_dec_ref(v_e_3904_);
v_val_3924_ = lean_ctor_get(v_e_x3f_3920_, 0);
lean_inc(v_val_3924_);
lean_dec_ref_known(v_e_x3f_3920_, 1);
if (v_isShared_3913_ == 0)
{
lean_ctor_set(v___x_3912_, 0, v_val_3924_);
v___x_3926_ = v___x_3912_;
goto v_reusejp_3925_;
}
else
{
lean_object* v_reuseFailAlloc_3927_; 
v_reuseFailAlloc_3927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3927_, 0, v_val_3924_);
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
}
}
else
{
lean_object* v_a_3929_; lean_object* v___x_3931_; uint8_t v_isShared_3932_; uint8_t v_isSharedCheck_3936_; 
lean_dec_ref(v_e_3904_);
lean_dec_ref(v_post_3903_);
lean_dec_ref(v_pre_3902_);
v_a_3929_ = lean_ctor_get(v___x_3909_, 0);
v_isSharedCheck_3936_ = !lean_is_exclusive(v___x_3909_);
if (v_isSharedCheck_3936_ == 0)
{
v___x_3931_ = v___x_3909_;
v_isShared_3932_ = v_isSharedCheck_3936_;
goto v_resetjp_3930_;
}
else
{
lean_inc(v_a_3929_);
lean_dec(v___x_3909_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2___boxed(lean_object* v_pre_3937_, lean_object* v_post_3938_, lean_object* v_e_3939_, lean_object* v_a_3940_, lean_object* v___y_3941_, lean_object* v___y_3942_, lean_object* v___y_3943_){
_start:
{
lean_object* v_res_3944_; 
v_res_3944_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__2(v_pre_3937_, v_post_3938_, v_e_3939_, v_a_3940_, v___y_3941_, v___y_3942_);
lean_dec(v___y_3942_);
lean_dec_ref(v___y_3941_);
lean_dec(v_a_3940_);
return v_res_3944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1___boxed(lean_object* v_pre_3945_, lean_object* v_post_3946_, lean_object* v_sz_3947_, lean_object* v_i_3948_, lean_object* v_bs_3949_, lean_object* v___y_3950_, lean_object* v___y_3951_, lean_object* v___y_3952_, lean_object* v___y_3953_){
_start:
{
size_t v_sz_boxed_3954_; size_t v_i_boxed_3955_; lean_object* v_res_3956_; 
v_sz_boxed_3954_ = lean_unbox_usize(v_sz_3947_);
lean_dec(v_sz_3947_);
v_i_boxed_3955_ = lean_unbox_usize(v_i_3948_);
lean_dec(v_i_3948_);
v_res_3956_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__1(v_pre_3945_, v_post_3946_, v_sz_boxed_3954_, v_i_boxed_3955_, v_bs_3949_, v___y_3950_, v___y_3951_, v___y_3952_);
lean_dec(v___y_3952_);
lean_dec_ref(v___y_3951_);
lean_dec(v___y_3950_);
return v_res_3956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3___boxed(lean_object* v_pre_3957_, lean_object* v_post_3958_, lean_object* v_x_3959_, lean_object* v_x_3960_, lean_object* v_x_3961_, lean_object* v___y_3962_, lean_object* v___y_3963_, lean_object* v___y_3964_, lean_object* v___y_3965_){
_start:
{
lean_object* v_res_3966_; 
v_res_3966_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__3(v_pre_3957_, v_post_3958_, v_x_3959_, v_x_3960_, v_x_3961_, v___y_3962_, v___y_3963_, v___y_3964_);
lean_dec(v___y_3964_);
lean_dec_ref(v___y_3963_);
lean_dec(v___y_3962_);
return v_res_3966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0___boxed(lean_object* v_pre_3967_, lean_object* v_post_3968_, lean_object* v_e_3969_, lean_object* v_a_3970_, lean_object* v___y_3971_, lean_object* v___y_3972_, lean_object* v___y_3973_){
_start:
{
lean_object* v_res_3974_; 
v_res_3974_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3967_, v_post_3968_, v_e_3969_, v_a_3970_, v___y_3971_, v___y_3972_);
lean_dec(v___y_3972_);
lean_dec_ref(v___y_3971_);
lean_dec(v_a_3970_);
return v_res_3974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0(lean_object* v_00_u03b1_3975_, lean_object* v_x_3976_, lean_object* v___y_3977_, lean_object* v___y_3978_){
_start:
{
lean_object* v___x_3980_; lean_object* v___x_3981_; 
v___x_3980_ = lean_apply_1(v_x_3976_, lean_box(0));
v___x_3981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3981_, 0, v___x_3980_);
return v___x_3981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0___boxed(lean_object* v_00_u03b1_3982_, lean_object* v_x_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_){
_start:
{
lean_object* v_res_3987_; 
v_res_3987_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0(v_00_u03b1_3982_, v_x_3983_, v___y_3984_, v___y_3985_);
lean_dec(v___y_3985_);
lean_dec_ref(v___y_3984_);
return v_res_3987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0(lean_object* v_input_3988_, lean_object* v_pre_3989_, lean_object* v_post_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_){
_start:
{
lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v_a_3996_; lean_object* v___x_3997_; 
v___x_3994_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insertBoundaries_spec__3___closed__2);
v___x_3995_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0(lean_box(0), v___x_3994_, v___y_3991_, v___y_3992_);
v_a_3996_ = lean_ctor_get(v___x_3995_, 0);
lean_inc(v_a_3996_);
lean_dec_ref(v___x_3995_);
v___x_3997_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0(v_pre_3989_, v_post_3990_, v_input_3988_, v_a_3996_, v___y_3991_, v___y_3992_);
if (lean_obj_tag(v___x_3997_) == 0)
{
lean_object* v_a_3998_; lean_object* v___x_3999_; lean_object* v___x_4000_; lean_object* v___x_4002_; uint8_t v_isShared_4003_; uint8_t v_isSharedCheck_4007_; 
v_a_3998_ = lean_ctor_get(v___x_3997_, 0);
lean_inc(v_a_3998_);
lean_dec_ref_known(v___x_3997_, 1);
v___x_3999_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_3999_, 0, lean_box(0));
lean_closure_set(v___x_3999_, 1, lean_box(0));
lean_closure_set(v___x_3999_, 2, v_a_3996_);
v___x_4000_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___lam__0(lean_box(0), v___x_3999_, v___y_3991_, v___y_3992_);
v_isSharedCheck_4007_ = !lean_is_exclusive(v___x_4000_);
if (v_isSharedCheck_4007_ == 0)
{
lean_object* v_unused_4008_; 
v_unused_4008_ = lean_ctor_get(v___x_4000_, 0);
lean_dec(v_unused_4008_);
v___x_4002_ = v___x_4000_;
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
else
{
lean_dec(v___x_4000_);
v___x_4002_ = lean_box(0);
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
v_resetjp_4001_:
{
lean_object* v___x_4005_; 
if (v_isShared_4003_ == 0)
{
lean_ctor_set(v___x_4002_, 0, v_a_3998_);
v___x_4005_ = v___x_4002_;
goto v_reusejp_4004_;
}
else
{
lean_object* v_reuseFailAlloc_4006_; 
v_reuseFailAlloc_4006_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4006_, 0, v_a_3998_);
v___x_4005_ = v_reuseFailAlloc_4006_;
goto v_reusejp_4004_;
}
v_reusejp_4004_:
{
return v___x_4005_;
}
}
}
else
{
lean_dec(v_a_3996_);
return v___x_3997_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0___boxed(lean_object* v_input_4009_, lean_object* v_pre_4010_, lean_object* v_post_4011_, lean_object* v___y_4012_, lean_object* v___y_4013_, lean_object* v___y_4014_){
_start:
{
lean_object* v_res_4015_; 
v_res_4015_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0(v_input_4009_, v_pre_4010_, v_post_4011_, v___y_4012_, v___y_4013_);
lean_dec(v___y_4013_);
lean_dec_ref(v___y_4012_);
return v_res_4015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions(lean_object* v_e_4017_, lean_object* v_b_4018_, lean_object* v_a_4019_, lean_object* v_a_4020_){
_start:
{
lean_object* v___f_4022_; lean_object* v___f_4023_; lean_object* v___x_4024_; 
v___f_4022_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___lam__0___boxed), 5, 1);
lean_closure_set(v___f_4022_, 0, v_b_4018_);
v___f_4023_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___closed__0));
v___x_4024_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0(v_e_4017_, v___f_4022_, v___f_4023_, v_a_4019_, v_a_4020_);
return v___x_4024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions___boxed(lean_object* v_e_4025_, lean_object* v_b_4026_, lean_object* v_a_4027_, lean_object* v_a_4028_, lean_object* v_a_4029_){
_start:
{
lean_object* v_res_4030_; 
v_res_4030_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions(v_e_4025_, v_b_4026_, v_a_4027_, v_a_4028_);
lean_dec(v_a_4028_);
lean_dec_ref(v_a_4027_);
return v_res_4030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5(lean_object* v_00_u03b1_4031_, lean_object* v_ref_4032_, lean_object* v___y_4033_, lean_object* v___y_4034_){
_start:
{
lean_object* v___x_4036_; 
v___x_4036_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___redArg(v_ref_4032_);
return v___x_4036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5___boxed(lean_object* v_00_u03b1_4037_, lean_object* v_ref_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_){
_start:
{
lean_object* v_res_4042_; 
v_res_4042_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__5(v_00_u03b1_4037_, v_ref_4038_, v___y_4039_, v___y_4040_);
lean_dec(v___y_4040_);
lean_dec_ref(v___y_4039_);
return v_res_4042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6(lean_object* v_00_u03b1_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_){
_start:
{
lean_object* v___x_4047_; 
v___x_4047_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___redArg();
return v___x_4047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6___boxed(lean_object* v_00_u03b1_4048_, lean_object* v___y_4049_, lean_object* v___y_4050_, lean_object* v___y_4051_){
_start:
{
lean_object* v_res_4052_; 
v_res_4052_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4_spec__6(v_00_u03b1_4048_, v___y_4049_, v___y_4050_);
lean_dec(v___y_4050_);
lean_dec_ref(v___y_4049_);
return v_res_4052_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4(lean_object* v_00_u03b1_4053_, lean_object* v_x_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_){
_start:
{
lean_object* v___x_4059_; 
v___x_4059_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___redArg(v_x_4054_, v___y_4055_, v___y_4056_, v___y_4057_);
return v___x_4059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4___boxed(lean_object* v_00_u03b1_4060_, lean_object* v_x_4061_, lean_object* v___y_4062_, lean_object* v___y_4063_, lean_object* v___y_4064_, lean_object* v___y_4065_){
_start:
{
lean_object* v_res_4066_; 
v_res_4066_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_unfoldInsertions_spec__0_spec__0_spec__4(v_00_u03b1_4060_, v_x_4061_, v___y_4062_, v___y_4063_, v___y_4064_);
lean_dec(v___y_4064_);
lean_dec_ref(v___y_4063_);
lean_dec(v___y_4062_);
return v_res_4066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorIdx(lean_object* v_x_4067_){
_start:
{
if (lean_obj_tag(v_x_4067_) == 0)
{
lean_object* v___x_4068_; 
v___x_4068_ = lean_unsigned_to_nat(0u);
return v___x_4068_;
}
else
{
lean_object* v___x_4069_; 
v___x_4069_ = lean_unsigned_to_nat(1u);
return v___x_4069_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorIdx___boxed(lean_object* v_x_4070_){
_start:
{
lean_object* v_res_4071_; 
v_res_4071_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorIdx(v_x_4070_);
lean_dec_ref(v_x_4070_);
return v_res_4071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(lean_object* v_t_4072_, lean_object* v_k_4073_){
_start:
{
if (lean_obj_tag(v_t_4072_) == 0)
{
lean_object* v_declName_4074_; lean_object* v_unfold_4075_; lean_object* v___x_4076_; 
v_declName_4074_ = lean_ctor_get(v_t_4072_, 0);
lean_inc(v_declName_4074_);
v_unfold_4075_ = lean_ctor_get(v_t_4072_, 1);
lean_inc(v_unfold_4075_);
lean_dec_ref_known(v_t_4072_, 2);
v___x_4076_ = lean_apply_2(v_k_4073_, v_declName_4074_, v_unfold_4075_);
return v___x_4076_;
}
else
{
lean_object* v_declName_4077_; lean_object* v_unfold_4078_; lean_object* v_refold_4079_; lean_object* v_unfold_x27_4080_; lean_object* v_refold_x27_4081_; lean_object* v___x_4082_; 
v_declName_4077_ = lean_ctor_get(v_t_4072_, 0);
lean_inc(v_declName_4077_);
v_unfold_4078_ = lean_ctor_get(v_t_4072_, 1);
lean_inc(v_unfold_4078_);
v_refold_4079_ = lean_ctor_get(v_t_4072_, 2);
lean_inc(v_refold_4079_);
v_unfold_x27_4080_ = lean_ctor_get(v_t_4072_, 3);
lean_inc(v_unfold_x27_4080_);
v_refold_x27_4081_ = lean_ctor_get(v_t_4072_, 4);
lean_inc(v_refold_x27_4081_);
lean_dec_ref_known(v_t_4072_, 5);
v___x_4082_ = lean_apply_5(v_k_4073_, v_declName_4077_, v_unfold_4078_, v_refold_4079_, v_unfold_x27_4080_, v_refold_x27_4081_);
return v___x_4082_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim(lean_object* v_motive_4083_, lean_object* v_ctorIdx_4084_, lean_object* v_t_4085_, lean_object* v_h_4086_, lean_object* v_k_4087_){
_start:
{
lean_object* v___x_4088_; 
v___x_4088_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(v_t_4085_, v_k_4087_);
return v___x_4088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___boxed(lean_object* v_motive_4089_, lean_object* v_ctorIdx_4090_, lean_object* v_t_4091_, lean_object* v_h_4092_, lean_object* v_k_4093_){
_start:
{
lean_object* v_res_4094_; 
v_res_4094_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim(v_motive_4089_, v_ctorIdx_4090_, v_t_4091_, v_h_4092_, v_k_4093_);
lean_dec(v_ctorIdx_4090_);
return v_res_4094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_unfold_elim___redArg(lean_object* v_t_4095_, lean_object* v_unfold_4096_){
_start:
{
lean_object* v___x_4097_; 
v___x_4097_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(v_t_4095_, v_unfold_4096_);
return v___x_4097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_unfold_elim(lean_object* v_motive_4098_, lean_object* v_t_4099_, lean_object* v_h_4100_, lean_object* v_unfold_4101_){
_start:
{
lean_object* v___x_4102_; 
v___x_4102_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(v_t_4099_, v_unfold_4101_);
return v___x_4102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_cast_elim___redArg(lean_object* v_t_4103_, lean_object* v_cast_4104_){
_start:
{
lean_object* v___x_4105_; 
v___x_4105_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(v_t_4103_, v_cast_4104_);
return v___x_4105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_cast_elim(lean_object* v_motive_4106_, lean_object* v_t_4107_, lean_object* v_h_4108_, lean_object* v_cast_4109_){
_start:
{
lean_object* v___x_4110_; 
v___x_4110_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_UnfoldEntry_ctorElim___redArg(v_t_4107_, v_cast_4109_);
return v___x_4110_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg(lean_object* v_k_4111_, lean_object* v_t_4112_){
_start:
{
if (lean_obj_tag(v_t_4112_) == 0)
{
lean_object* v_k_4113_; lean_object* v_l_4114_; lean_object* v_r_4115_; uint8_t v___x_4116_; 
v_k_4113_ = lean_ctor_get(v_t_4112_, 1);
v_l_4114_ = lean_ctor_get(v_t_4112_, 3);
v_r_4115_ = lean_ctor_get(v_t_4112_, 4);
v___x_4116_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_4111_, v_k_4113_);
switch(v___x_4116_)
{
case 0:
{
v_t_4112_ = v_l_4114_;
goto _start;
}
case 1:
{
uint8_t v___x_4118_; 
v___x_4118_ = 1;
return v___x_4118_;
}
default: 
{
v_t_4112_ = v_r_4115_;
goto _start;
}
}
}
else
{
uint8_t v___x_4120_; 
v___x_4120_ = 0;
return v___x_4120_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg___boxed(lean_object* v_k_4121_, lean_object* v_t_4122_){
_start:
{
uint8_t v_res_4123_; lean_object* v_r_4124_; 
v_res_4123_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg(v_k_4121_, v_t_4122_);
lean_dec(v_t_4122_);
lean_dec(v_k_4121_);
v_r_4124_ = lean_box(v_res_4123_);
return v_r_4124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(lean_object* v_k_4125_, lean_object* v_v_4126_, lean_object* v_t_4127_){
_start:
{
if (lean_obj_tag(v_t_4127_) == 0)
{
lean_object* v_size_4128_; lean_object* v_k_4129_; lean_object* v_v_4130_; lean_object* v_l_4131_; lean_object* v_r_4132_; lean_object* v___x_4134_; uint8_t v_isShared_4135_; uint8_t v_isSharedCheck_4412_; 
v_size_4128_ = lean_ctor_get(v_t_4127_, 0);
v_k_4129_ = lean_ctor_get(v_t_4127_, 1);
v_v_4130_ = lean_ctor_get(v_t_4127_, 2);
v_l_4131_ = lean_ctor_get(v_t_4127_, 3);
v_r_4132_ = lean_ctor_get(v_t_4127_, 4);
v_isSharedCheck_4412_ = !lean_is_exclusive(v_t_4127_);
if (v_isSharedCheck_4412_ == 0)
{
v___x_4134_ = v_t_4127_;
v_isShared_4135_ = v_isSharedCheck_4412_;
goto v_resetjp_4133_;
}
else
{
lean_inc(v_r_4132_);
lean_inc(v_l_4131_);
lean_inc(v_v_4130_);
lean_inc(v_k_4129_);
lean_inc(v_size_4128_);
lean_dec(v_t_4127_);
v___x_4134_ = lean_box(0);
v_isShared_4135_ = v_isSharedCheck_4412_;
goto v_resetjp_4133_;
}
v_resetjp_4133_:
{
uint8_t v___x_4136_; 
v___x_4136_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_4125_, v_k_4129_);
switch(v___x_4136_)
{
case 0:
{
lean_object* v_impl_4137_; lean_object* v___x_4138_; 
lean_dec(v_size_4128_);
v_impl_4137_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(v_k_4125_, v_v_4126_, v_l_4131_);
v___x_4138_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_4132_) == 0)
{
lean_object* v_size_4139_; lean_object* v_size_4140_; lean_object* v_k_4141_; lean_object* v_v_4142_; lean_object* v_l_4143_; lean_object* v_r_4144_; lean_object* v___x_4145_; lean_object* v___x_4146_; uint8_t v___x_4147_; 
v_size_4139_ = lean_ctor_get(v_r_4132_, 0);
v_size_4140_ = lean_ctor_get(v_impl_4137_, 0);
lean_inc(v_size_4140_);
v_k_4141_ = lean_ctor_get(v_impl_4137_, 1);
lean_inc(v_k_4141_);
v_v_4142_ = lean_ctor_get(v_impl_4137_, 2);
lean_inc(v_v_4142_);
v_l_4143_ = lean_ctor_get(v_impl_4137_, 3);
lean_inc(v_l_4143_);
v_r_4144_ = lean_ctor_get(v_impl_4137_, 4);
lean_inc(v_r_4144_);
v___x_4145_ = lean_unsigned_to_nat(3u);
v___x_4146_ = lean_nat_mul(v___x_4145_, v_size_4139_);
v___x_4147_ = lean_nat_dec_lt(v___x_4146_, v_size_4140_);
lean_dec(v___x_4146_);
if (v___x_4147_ == 0)
{
lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4151_; 
lean_dec(v_r_4144_);
lean_dec(v_l_4143_);
lean_dec(v_v_4142_);
lean_dec(v_k_4141_);
v___x_4148_ = lean_nat_add(v___x_4138_, v_size_4140_);
lean_dec(v_size_4140_);
v___x_4149_ = lean_nat_add(v___x_4148_, v_size_4139_);
lean_dec(v___x_4148_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 3, v_impl_4137_);
lean_ctor_set(v___x_4134_, 0, v___x_4149_);
v___x_4151_ = v___x_4134_;
goto v_reusejp_4150_;
}
else
{
lean_object* v_reuseFailAlloc_4152_; 
v_reuseFailAlloc_4152_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4152_, 0, v___x_4149_);
lean_ctor_set(v_reuseFailAlloc_4152_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4152_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4152_, 3, v_impl_4137_);
lean_ctor_set(v_reuseFailAlloc_4152_, 4, v_r_4132_);
v___x_4151_ = v_reuseFailAlloc_4152_;
goto v_reusejp_4150_;
}
v_reusejp_4150_:
{
return v___x_4151_;
}
}
else
{
lean_object* v___x_4154_; uint8_t v_isShared_4155_; uint8_t v_isSharedCheck_4218_; 
v_isSharedCheck_4218_ = !lean_is_exclusive(v_impl_4137_);
if (v_isSharedCheck_4218_ == 0)
{
lean_object* v_unused_4219_; lean_object* v_unused_4220_; lean_object* v_unused_4221_; lean_object* v_unused_4222_; lean_object* v_unused_4223_; 
v_unused_4219_ = lean_ctor_get(v_impl_4137_, 4);
lean_dec(v_unused_4219_);
v_unused_4220_ = lean_ctor_get(v_impl_4137_, 3);
lean_dec(v_unused_4220_);
v_unused_4221_ = lean_ctor_get(v_impl_4137_, 2);
lean_dec(v_unused_4221_);
v_unused_4222_ = lean_ctor_get(v_impl_4137_, 1);
lean_dec(v_unused_4222_);
v_unused_4223_ = lean_ctor_get(v_impl_4137_, 0);
lean_dec(v_unused_4223_);
v___x_4154_ = v_impl_4137_;
v_isShared_4155_ = v_isSharedCheck_4218_;
goto v_resetjp_4153_;
}
else
{
lean_dec(v_impl_4137_);
v___x_4154_ = lean_box(0);
v_isShared_4155_ = v_isSharedCheck_4218_;
goto v_resetjp_4153_;
}
v_resetjp_4153_:
{
lean_object* v_size_4156_; lean_object* v_size_4157_; lean_object* v_k_4158_; lean_object* v_v_4159_; lean_object* v_l_4160_; lean_object* v_r_4161_; lean_object* v___x_4162_; lean_object* v___x_4163_; uint8_t v___x_4164_; 
v_size_4156_ = lean_ctor_get(v_l_4143_, 0);
v_size_4157_ = lean_ctor_get(v_r_4144_, 0);
v_k_4158_ = lean_ctor_get(v_r_4144_, 1);
v_v_4159_ = lean_ctor_get(v_r_4144_, 2);
v_l_4160_ = lean_ctor_get(v_r_4144_, 3);
v_r_4161_ = lean_ctor_get(v_r_4144_, 4);
v___x_4162_ = lean_unsigned_to_nat(2u);
v___x_4163_ = lean_nat_mul(v___x_4162_, v_size_4156_);
v___x_4164_ = lean_nat_dec_lt(v_size_4157_, v___x_4163_);
lean_dec(v___x_4163_);
if (v___x_4164_ == 0)
{
lean_object* v___x_4166_; uint8_t v_isShared_4167_; uint8_t v_isSharedCheck_4193_; 
lean_inc(v_r_4161_);
lean_inc(v_l_4160_);
lean_inc(v_v_4159_);
lean_inc(v_k_4158_);
v_isSharedCheck_4193_ = !lean_is_exclusive(v_r_4144_);
if (v_isSharedCheck_4193_ == 0)
{
lean_object* v_unused_4194_; lean_object* v_unused_4195_; lean_object* v_unused_4196_; lean_object* v_unused_4197_; lean_object* v_unused_4198_; 
v_unused_4194_ = lean_ctor_get(v_r_4144_, 4);
lean_dec(v_unused_4194_);
v_unused_4195_ = lean_ctor_get(v_r_4144_, 3);
lean_dec(v_unused_4195_);
v_unused_4196_ = lean_ctor_get(v_r_4144_, 2);
lean_dec(v_unused_4196_);
v_unused_4197_ = lean_ctor_get(v_r_4144_, 1);
lean_dec(v_unused_4197_);
v_unused_4198_ = lean_ctor_get(v_r_4144_, 0);
lean_dec(v_unused_4198_);
v___x_4166_ = v_r_4144_;
v_isShared_4167_ = v_isSharedCheck_4193_;
goto v_resetjp_4165_;
}
else
{
lean_dec(v_r_4144_);
v___x_4166_ = lean_box(0);
v_isShared_4167_ = v_isSharedCheck_4193_;
goto v_resetjp_4165_;
}
v_resetjp_4165_:
{
lean_object* v___x_4168_; lean_object* v___x_4169_; lean_object* v___y_4171_; lean_object* v___y_4172_; lean_object* v___y_4173_; lean_object* v___x_4181_; lean_object* v___y_4183_; 
v___x_4168_ = lean_nat_add(v___x_4138_, v_size_4140_);
lean_dec(v_size_4140_);
v___x_4169_ = lean_nat_add(v___x_4168_, v_size_4139_);
lean_dec(v___x_4168_);
v___x_4181_ = lean_nat_add(v___x_4138_, v_size_4156_);
if (lean_obj_tag(v_l_4160_) == 0)
{
lean_object* v_size_4191_; 
v_size_4191_ = lean_ctor_get(v_l_4160_, 0);
lean_inc(v_size_4191_);
v___y_4183_ = v_size_4191_;
goto v___jp_4182_;
}
else
{
lean_object* v___x_4192_; 
v___x_4192_ = lean_unsigned_to_nat(0u);
v___y_4183_ = v___x_4192_;
goto v___jp_4182_;
}
v___jp_4170_:
{
lean_object* v___x_4174_; lean_object* v___x_4176_; 
v___x_4174_ = lean_nat_add(v___y_4171_, v___y_4173_);
lean_dec(v___y_4173_);
lean_dec(v___y_4171_);
if (v_isShared_4167_ == 0)
{
lean_ctor_set(v___x_4166_, 4, v_r_4132_);
lean_ctor_set(v___x_4166_, 3, v_r_4161_);
lean_ctor_set(v___x_4166_, 2, v_v_4130_);
lean_ctor_set(v___x_4166_, 1, v_k_4129_);
lean_ctor_set(v___x_4166_, 0, v___x_4174_);
v___x_4176_ = v___x_4166_;
goto v_reusejp_4175_;
}
else
{
lean_object* v_reuseFailAlloc_4180_; 
v_reuseFailAlloc_4180_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4180_, 0, v___x_4174_);
lean_ctor_set(v_reuseFailAlloc_4180_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4180_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4180_, 3, v_r_4161_);
lean_ctor_set(v_reuseFailAlloc_4180_, 4, v_r_4132_);
v___x_4176_ = v_reuseFailAlloc_4180_;
goto v_reusejp_4175_;
}
v_reusejp_4175_:
{
lean_object* v___x_4178_; 
if (v_isShared_4155_ == 0)
{
lean_ctor_set(v___x_4154_, 4, v___x_4176_);
lean_ctor_set(v___x_4154_, 3, v___y_4172_);
lean_ctor_set(v___x_4154_, 2, v_v_4159_);
lean_ctor_set(v___x_4154_, 1, v_k_4158_);
lean_ctor_set(v___x_4154_, 0, v___x_4169_);
v___x_4178_ = v___x_4154_;
goto v_reusejp_4177_;
}
else
{
lean_object* v_reuseFailAlloc_4179_; 
v_reuseFailAlloc_4179_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4179_, 0, v___x_4169_);
lean_ctor_set(v_reuseFailAlloc_4179_, 1, v_k_4158_);
lean_ctor_set(v_reuseFailAlloc_4179_, 2, v_v_4159_);
lean_ctor_set(v_reuseFailAlloc_4179_, 3, v___y_4172_);
lean_ctor_set(v_reuseFailAlloc_4179_, 4, v___x_4176_);
v___x_4178_ = v_reuseFailAlloc_4179_;
goto v_reusejp_4177_;
}
v_reusejp_4177_:
{
return v___x_4178_;
}
}
}
v___jp_4182_:
{
lean_object* v___x_4184_; lean_object* v___x_4186_; 
v___x_4184_ = lean_nat_add(v___x_4181_, v___y_4183_);
lean_dec(v___y_4183_);
lean_dec(v___x_4181_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_l_4160_);
lean_ctor_set(v___x_4134_, 3, v_l_4143_);
lean_ctor_set(v___x_4134_, 2, v_v_4142_);
lean_ctor_set(v___x_4134_, 1, v_k_4141_);
lean_ctor_set(v___x_4134_, 0, v___x_4184_);
v___x_4186_ = v___x_4134_;
goto v_reusejp_4185_;
}
else
{
lean_object* v_reuseFailAlloc_4190_; 
v_reuseFailAlloc_4190_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4190_, 0, v___x_4184_);
lean_ctor_set(v_reuseFailAlloc_4190_, 1, v_k_4141_);
lean_ctor_set(v_reuseFailAlloc_4190_, 2, v_v_4142_);
lean_ctor_set(v_reuseFailAlloc_4190_, 3, v_l_4143_);
lean_ctor_set(v_reuseFailAlloc_4190_, 4, v_l_4160_);
v___x_4186_ = v_reuseFailAlloc_4190_;
goto v_reusejp_4185_;
}
v_reusejp_4185_:
{
lean_object* v___x_4187_; 
v___x_4187_ = lean_nat_add(v___x_4138_, v_size_4139_);
if (lean_obj_tag(v_r_4161_) == 0)
{
lean_object* v_size_4188_; 
v_size_4188_ = lean_ctor_get(v_r_4161_, 0);
lean_inc(v_size_4188_);
v___y_4171_ = v___x_4187_;
v___y_4172_ = v___x_4186_;
v___y_4173_ = v_size_4188_;
goto v___jp_4170_;
}
else
{
lean_object* v___x_4189_; 
v___x_4189_ = lean_unsigned_to_nat(0u);
v___y_4171_ = v___x_4187_;
v___y_4172_ = v___x_4186_;
v___y_4173_ = v___x_4189_;
goto v___jp_4170_;
}
}
}
}
}
else
{
lean_object* v___x_4199_; lean_object* v___x_4200_; lean_object* v___x_4201_; lean_object* v___x_4202_; lean_object* v___x_4204_; 
lean_del_object(v___x_4134_);
v___x_4199_ = lean_nat_add(v___x_4138_, v_size_4140_);
lean_dec(v_size_4140_);
v___x_4200_ = lean_nat_add(v___x_4199_, v_size_4139_);
lean_dec(v___x_4199_);
v___x_4201_ = lean_nat_add(v___x_4138_, v_size_4139_);
v___x_4202_ = lean_nat_add(v___x_4201_, v_size_4157_);
lean_dec(v___x_4201_);
lean_inc_ref(v_r_4132_);
if (v_isShared_4155_ == 0)
{
lean_ctor_set(v___x_4154_, 4, v_r_4132_);
lean_ctor_set(v___x_4154_, 3, v_r_4144_);
lean_ctor_set(v___x_4154_, 2, v_v_4130_);
lean_ctor_set(v___x_4154_, 1, v_k_4129_);
lean_ctor_set(v___x_4154_, 0, v___x_4202_);
v___x_4204_ = v___x_4154_;
goto v_reusejp_4203_;
}
else
{
lean_object* v_reuseFailAlloc_4217_; 
v_reuseFailAlloc_4217_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4217_, 0, v___x_4202_);
lean_ctor_set(v_reuseFailAlloc_4217_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4217_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4217_, 3, v_r_4144_);
lean_ctor_set(v_reuseFailAlloc_4217_, 4, v_r_4132_);
v___x_4204_ = v_reuseFailAlloc_4217_;
goto v_reusejp_4203_;
}
v_reusejp_4203_:
{
lean_object* v___x_4206_; uint8_t v_isShared_4207_; uint8_t v_isSharedCheck_4211_; 
v_isSharedCheck_4211_ = !lean_is_exclusive(v_r_4132_);
if (v_isSharedCheck_4211_ == 0)
{
lean_object* v_unused_4212_; lean_object* v_unused_4213_; lean_object* v_unused_4214_; lean_object* v_unused_4215_; lean_object* v_unused_4216_; 
v_unused_4212_ = lean_ctor_get(v_r_4132_, 4);
lean_dec(v_unused_4212_);
v_unused_4213_ = lean_ctor_get(v_r_4132_, 3);
lean_dec(v_unused_4213_);
v_unused_4214_ = lean_ctor_get(v_r_4132_, 2);
lean_dec(v_unused_4214_);
v_unused_4215_ = lean_ctor_get(v_r_4132_, 1);
lean_dec(v_unused_4215_);
v_unused_4216_ = lean_ctor_get(v_r_4132_, 0);
lean_dec(v_unused_4216_);
v___x_4206_ = v_r_4132_;
v_isShared_4207_ = v_isSharedCheck_4211_;
goto v_resetjp_4205_;
}
else
{
lean_dec(v_r_4132_);
v___x_4206_ = lean_box(0);
v_isShared_4207_ = v_isSharedCheck_4211_;
goto v_resetjp_4205_;
}
v_resetjp_4205_:
{
lean_object* v___x_4209_; 
if (v_isShared_4207_ == 0)
{
lean_ctor_set(v___x_4206_, 4, v___x_4204_);
lean_ctor_set(v___x_4206_, 3, v_l_4143_);
lean_ctor_set(v___x_4206_, 2, v_v_4142_);
lean_ctor_set(v___x_4206_, 1, v_k_4141_);
lean_ctor_set(v___x_4206_, 0, v___x_4200_);
v___x_4209_ = v___x_4206_;
goto v_reusejp_4208_;
}
else
{
lean_object* v_reuseFailAlloc_4210_; 
v_reuseFailAlloc_4210_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4210_, 0, v___x_4200_);
lean_ctor_set(v_reuseFailAlloc_4210_, 1, v_k_4141_);
lean_ctor_set(v_reuseFailAlloc_4210_, 2, v_v_4142_);
lean_ctor_set(v_reuseFailAlloc_4210_, 3, v_l_4143_);
lean_ctor_set(v_reuseFailAlloc_4210_, 4, v___x_4204_);
v___x_4209_ = v_reuseFailAlloc_4210_;
goto v_reusejp_4208_;
}
v_reusejp_4208_:
{
return v___x_4209_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_4224_; 
v_l_4224_ = lean_ctor_get(v_impl_4137_, 3);
lean_inc(v_l_4224_);
if (lean_obj_tag(v_l_4224_) == 0)
{
lean_object* v_r_4225_; lean_object* v_k_4226_; lean_object* v_v_4227_; lean_object* v___x_4229_; uint8_t v_isShared_4230_; uint8_t v_isSharedCheck_4238_; 
v_r_4225_ = lean_ctor_get(v_impl_4137_, 4);
v_k_4226_ = lean_ctor_get(v_impl_4137_, 1);
v_v_4227_ = lean_ctor_get(v_impl_4137_, 2);
v_isSharedCheck_4238_ = !lean_is_exclusive(v_impl_4137_);
if (v_isSharedCheck_4238_ == 0)
{
lean_object* v_unused_4239_; lean_object* v_unused_4240_; 
v_unused_4239_ = lean_ctor_get(v_impl_4137_, 3);
lean_dec(v_unused_4239_);
v_unused_4240_ = lean_ctor_get(v_impl_4137_, 0);
lean_dec(v_unused_4240_);
v___x_4229_ = v_impl_4137_;
v_isShared_4230_ = v_isSharedCheck_4238_;
goto v_resetjp_4228_;
}
else
{
lean_inc(v_r_4225_);
lean_inc(v_v_4227_);
lean_inc(v_k_4226_);
lean_dec(v_impl_4137_);
v___x_4229_ = lean_box(0);
v_isShared_4230_ = v_isSharedCheck_4238_;
goto v_resetjp_4228_;
}
v_resetjp_4228_:
{
lean_object* v___x_4231_; lean_object* v___x_4233_; 
v___x_4231_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_4225_);
if (v_isShared_4230_ == 0)
{
lean_ctor_set(v___x_4229_, 3, v_r_4225_);
lean_ctor_set(v___x_4229_, 2, v_v_4130_);
lean_ctor_set(v___x_4229_, 1, v_k_4129_);
lean_ctor_set(v___x_4229_, 0, v___x_4138_);
v___x_4233_ = v___x_4229_;
goto v_reusejp_4232_;
}
else
{
lean_object* v_reuseFailAlloc_4237_; 
v_reuseFailAlloc_4237_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4237_, 0, v___x_4138_);
lean_ctor_set(v_reuseFailAlloc_4237_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4237_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4237_, 3, v_r_4225_);
lean_ctor_set(v_reuseFailAlloc_4237_, 4, v_r_4225_);
v___x_4233_ = v_reuseFailAlloc_4237_;
goto v_reusejp_4232_;
}
v_reusejp_4232_:
{
lean_object* v___x_4235_; 
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v___x_4233_);
lean_ctor_set(v___x_4134_, 3, v_l_4224_);
lean_ctor_set(v___x_4134_, 2, v_v_4227_);
lean_ctor_set(v___x_4134_, 1, v_k_4226_);
lean_ctor_set(v___x_4134_, 0, v___x_4231_);
v___x_4235_ = v___x_4134_;
goto v_reusejp_4234_;
}
else
{
lean_object* v_reuseFailAlloc_4236_; 
v_reuseFailAlloc_4236_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4236_, 0, v___x_4231_);
lean_ctor_set(v_reuseFailAlloc_4236_, 1, v_k_4226_);
lean_ctor_set(v_reuseFailAlloc_4236_, 2, v_v_4227_);
lean_ctor_set(v_reuseFailAlloc_4236_, 3, v_l_4224_);
lean_ctor_set(v_reuseFailAlloc_4236_, 4, v___x_4233_);
v___x_4235_ = v_reuseFailAlloc_4236_;
goto v_reusejp_4234_;
}
v_reusejp_4234_:
{
return v___x_4235_;
}
}
}
}
else
{
lean_object* v_r_4241_; 
v_r_4241_ = lean_ctor_get(v_impl_4137_, 4);
lean_inc(v_r_4241_);
if (lean_obj_tag(v_r_4241_) == 0)
{
lean_object* v_k_4242_; lean_object* v_v_4243_; lean_object* v___x_4245_; uint8_t v_isShared_4246_; uint8_t v_isSharedCheck_4266_; 
v_k_4242_ = lean_ctor_get(v_impl_4137_, 1);
v_v_4243_ = lean_ctor_get(v_impl_4137_, 2);
v_isSharedCheck_4266_ = !lean_is_exclusive(v_impl_4137_);
if (v_isSharedCheck_4266_ == 0)
{
lean_object* v_unused_4267_; lean_object* v_unused_4268_; lean_object* v_unused_4269_; 
v_unused_4267_ = lean_ctor_get(v_impl_4137_, 4);
lean_dec(v_unused_4267_);
v_unused_4268_ = lean_ctor_get(v_impl_4137_, 3);
lean_dec(v_unused_4268_);
v_unused_4269_ = lean_ctor_get(v_impl_4137_, 0);
lean_dec(v_unused_4269_);
v___x_4245_ = v_impl_4137_;
v_isShared_4246_ = v_isSharedCheck_4266_;
goto v_resetjp_4244_;
}
else
{
lean_inc(v_v_4243_);
lean_inc(v_k_4242_);
lean_dec(v_impl_4137_);
v___x_4245_ = lean_box(0);
v_isShared_4246_ = v_isSharedCheck_4266_;
goto v_resetjp_4244_;
}
v_resetjp_4244_:
{
lean_object* v_k_4247_; lean_object* v_v_4248_; lean_object* v___x_4250_; uint8_t v_isShared_4251_; uint8_t v_isSharedCheck_4262_; 
v_k_4247_ = lean_ctor_get(v_r_4241_, 1);
v_v_4248_ = lean_ctor_get(v_r_4241_, 2);
v_isSharedCheck_4262_ = !lean_is_exclusive(v_r_4241_);
if (v_isSharedCheck_4262_ == 0)
{
lean_object* v_unused_4263_; lean_object* v_unused_4264_; lean_object* v_unused_4265_; 
v_unused_4263_ = lean_ctor_get(v_r_4241_, 4);
lean_dec(v_unused_4263_);
v_unused_4264_ = lean_ctor_get(v_r_4241_, 3);
lean_dec(v_unused_4264_);
v_unused_4265_ = lean_ctor_get(v_r_4241_, 0);
lean_dec(v_unused_4265_);
v___x_4250_ = v_r_4241_;
v_isShared_4251_ = v_isSharedCheck_4262_;
goto v_resetjp_4249_;
}
else
{
lean_inc(v_v_4248_);
lean_inc(v_k_4247_);
lean_dec(v_r_4241_);
v___x_4250_ = lean_box(0);
v_isShared_4251_ = v_isSharedCheck_4262_;
goto v_resetjp_4249_;
}
v_resetjp_4249_:
{
lean_object* v___x_4252_; lean_object* v___x_4254_; 
v___x_4252_ = lean_unsigned_to_nat(3u);
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 4, v_l_4224_);
lean_ctor_set(v___x_4250_, 3, v_l_4224_);
lean_ctor_set(v___x_4250_, 2, v_v_4243_);
lean_ctor_set(v___x_4250_, 1, v_k_4242_);
lean_ctor_set(v___x_4250_, 0, v___x_4138_);
v___x_4254_ = v___x_4250_;
goto v_reusejp_4253_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v___x_4138_);
lean_ctor_set(v_reuseFailAlloc_4261_, 1, v_k_4242_);
lean_ctor_set(v_reuseFailAlloc_4261_, 2, v_v_4243_);
lean_ctor_set(v_reuseFailAlloc_4261_, 3, v_l_4224_);
lean_ctor_set(v_reuseFailAlloc_4261_, 4, v_l_4224_);
v___x_4254_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4253_;
}
v_reusejp_4253_:
{
lean_object* v___x_4256_; 
if (v_isShared_4246_ == 0)
{
lean_ctor_set(v___x_4245_, 4, v_l_4224_);
lean_ctor_set(v___x_4245_, 2, v_v_4130_);
lean_ctor_set(v___x_4245_, 1, v_k_4129_);
lean_ctor_set(v___x_4245_, 0, v___x_4138_);
v___x_4256_ = v___x_4245_;
goto v_reusejp_4255_;
}
else
{
lean_object* v_reuseFailAlloc_4260_; 
v_reuseFailAlloc_4260_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4260_, 0, v___x_4138_);
lean_ctor_set(v_reuseFailAlloc_4260_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4260_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4260_, 3, v_l_4224_);
lean_ctor_set(v_reuseFailAlloc_4260_, 4, v_l_4224_);
v___x_4256_ = v_reuseFailAlloc_4260_;
goto v_reusejp_4255_;
}
v_reusejp_4255_:
{
lean_object* v___x_4258_; 
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v___x_4256_);
lean_ctor_set(v___x_4134_, 3, v___x_4254_);
lean_ctor_set(v___x_4134_, 2, v_v_4248_);
lean_ctor_set(v___x_4134_, 1, v_k_4247_);
lean_ctor_set(v___x_4134_, 0, v___x_4252_);
v___x_4258_ = v___x_4134_;
goto v_reusejp_4257_;
}
else
{
lean_object* v_reuseFailAlloc_4259_; 
v_reuseFailAlloc_4259_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4259_, 0, v___x_4252_);
lean_ctor_set(v_reuseFailAlloc_4259_, 1, v_k_4247_);
lean_ctor_set(v_reuseFailAlloc_4259_, 2, v_v_4248_);
lean_ctor_set(v_reuseFailAlloc_4259_, 3, v___x_4254_);
lean_ctor_set(v_reuseFailAlloc_4259_, 4, v___x_4256_);
v___x_4258_ = v_reuseFailAlloc_4259_;
goto v_reusejp_4257_;
}
v_reusejp_4257_:
{
return v___x_4258_;
}
}
}
}
}
}
else
{
lean_object* v___x_4270_; lean_object* v___x_4272_; 
v___x_4270_ = lean_unsigned_to_nat(2u);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_r_4241_);
lean_ctor_set(v___x_4134_, 3, v_impl_4137_);
lean_ctor_set(v___x_4134_, 0, v___x_4270_);
v___x_4272_ = v___x_4134_;
goto v_reusejp_4271_;
}
else
{
lean_object* v_reuseFailAlloc_4273_; 
v_reuseFailAlloc_4273_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4273_, 0, v___x_4270_);
lean_ctor_set(v_reuseFailAlloc_4273_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4273_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4273_, 3, v_impl_4137_);
lean_ctor_set(v_reuseFailAlloc_4273_, 4, v_r_4241_);
v___x_4272_ = v_reuseFailAlloc_4273_;
goto v_reusejp_4271_;
}
v_reusejp_4271_:
{
return v___x_4272_;
}
}
}
}
}
case 1:
{
lean_object* v___x_4275_; 
lean_dec(v_v_4130_);
lean_dec(v_k_4129_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 2, v_v_4126_);
lean_ctor_set(v___x_4134_, 1, v_k_4125_);
v___x_4275_ = v___x_4134_;
goto v_reusejp_4274_;
}
else
{
lean_object* v_reuseFailAlloc_4276_; 
v_reuseFailAlloc_4276_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4276_, 0, v_size_4128_);
lean_ctor_set(v_reuseFailAlloc_4276_, 1, v_k_4125_);
lean_ctor_set(v_reuseFailAlloc_4276_, 2, v_v_4126_);
lean_ctor_set(v_reuseFailAlloc_4276_, 3, v_l_4131_);
lean_ctor_set(v_reuseFailAlloc_4276_, 4, v_r_4132_);
v___x_4275_ = v_reuseFailAlloc_4276_;
goto v_reusejp_4274_;
}
v_reusejp_4274_:
{
return v___x_4275_;
}
}
default: 
{
lean_object* v_impl_4277_; lean_object* v___x_4278_; 
lean_dec(v_size_4128_);
v_impl_4277_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(v_k_4125_, v_v_4126_, v_r_4132_);
v___x_4278_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_4131_) == 0)
{
lean_object* v_size_4279_; lean_object* v_size_4280_; lean_object* v_k_4281_; lean_object* v_v_4282_; lean_object* v_l_4283_; lean_object* v_r_4284_; lean_object* v___x_4285_; lean_object* v___x_4286_; uint8_t v___x_4287_; 
v_size_4279_ = lean_ctor_get(v_l_4131_, 0);
v_size_4280_ = lean_ctor_get(v_impl_4277_, 0);
lean_inc(v_size_4280_);
v_k_4281_ = lean_ctor_get(v_impl_4277_, 1);
lean_inc(v_k_4281_);
v_v_4282_ = lean_ctor_get(v_impl_4277_, 2);
lean_inc(v_v_4282_);
v_l_4283_ = lean_ctor_get(v_impl_4277_, 3);
lean_inc(v_l_4283_);
v_r_4284_ = lean_ctor_get(v_impl_4277_, 4);
lean_inc(v_r_4284_);
v___x_4285_ = lean_unsigned_to_nat(3u);
v___x_4286_ = lean_nat_mul(v___x_4285_, v_size_4279_);
v___x_4287_ = lean_nat_dec_lt(v___x_4286_, v_size_4280_);
lean_dec(v___x_4286_);
if (v___x_4287_ == 0)
{
lean_object* v___x_4288_; lean_object* v___x_4289_; lean_object* v___x_4291_; 
lean_dec(v_r_4284_);
lean_dec(v_l_4283_);
lean_dec(v_v_4282_);
lean_dec(v_k_4281_);
v___x_4288_ = lean_nat_add(v___x_4278_, v_size_4279_);
v___x_4289_ = lean_nat_add(v___x_4288_, v_size_4280_);
lean_dec(v_size_4280_);
lean_dec(v___x_4288_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_impl_4277_);
lean_ctor_set(v___x_4134_, 0, v___x_4289_);
v___x_4291_ = v___x_4134_;
goto v_reusejp_4290_;
}
else
{
lean_object* v_reuseFailAlloc_4292_; 
v_reuseFailAlloc_4292_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4292_, 0, v___x_4289_);
lean_ctor_set(v_reuseFailAlloc_4292_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4292_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4292_, 3, v_l_4131_);
lean_ctor_set(v_reuseFailAlloc_4292_, 4, v_impl_4277_);
v___x_4291_ = v_reuseFailAlloc_4292_;
goto v_reusejp_4290_;
}
v_reusejp_4290_:
{
return v___x_4291_;
}
}
else
{
lean_object* v___x_4294_; uint8_t v_isShared_4295_; uint8_t v_isSharedCheck_4356_; 
v_isSharedCheck_4356_ = !lean_is_exclusive(v_impl_4277_);
if (v_isSharedCheck_4356_ == 0)
{
lean_object* v_unused_4357_; lean_object* v_unused_4358_; lean_object* v_unused_4359_; lean_object* v_unused_4360_; lean_object* v_unused_4361_; 
v_unused_4357_ = lean_ctor_get(v_impl_4277_, 4);
lean_dec(v_unused_4357_);
v_unused_4358_ = lean_ctor_get(v_impl_4277_, 3);
lean_dec(v_unused_4358_);
v_unused_4359_ = lean_ctor_get(v_impl_4277_, 2);
lean_dec(v_unused_4359_);
v_unused_4360_ = lean_ctor_get(v_impl_4277_, 1);
lean_dec(v_unused_4360_);
v_unused_4361_ = lean_ctor_get(v_impl_4277_, 0);
lean_dec(v_unused_4361_);
v___x_4294_ = v_impl_4277_;
v_isShared_4295_ = v_isSharedCheck_4356_;
goto v_resetjp_4293_;
}
else
{
lean_dec(v_impl_4277_);
v___x_4294_ = lean_box(0);
v_isShared_4295_ = v_isSharedCheck_4356_;
goto v_resetjp_4293_;
}
v_resetjp_4293_:
{
lean_object* v_size_4296_; lean_object* v_k_4297_; lean_object* v_v_4298_; lean_object* v_l_4299_; lean_object* v_r_4300_; lean_object* v_size_4301_; lean_object* v___x_4302_; lean_object* v___x_4303_; uint8_t v___x_4304_; 
v_size_4296_ = lean_ctor_get(v_l_4283_, 0);
v_k_4297_ = lean_ctor_get(v_l_4283_, 1);
v_v_4298_ = lean_ctor_get(v_l_4283_, 2);
v_l_4299_ = lean_ctor_get(v_l_4283_, 3);
v_r_4300_ = lean_ctor_get(v_l_4283_, 4);
v_size_4301_ = lean_ctor_get(v_r_4284_, 0);
v___x_4302_ = lean_unsigned_to_nat(2u);
v___x_4303_ = lean_nat_mul(v___x_4302_, v_size_4301_);
v___x_4304_ = lean_nat_dec_lt(v_size_4296_, v___x_4303_);
lean_dec(v___x_4303_);
if (v___x_4304_ == 0)
{
lean_object* v___x_4306_; uint8_t v_isShared_4307_; uint8_t v_isSharedCheck_4332_; 
lean_inc(v_r_4300_);
lean_inc(v_l_4299_);
lean_inc(v_v_4298_);
lean_inc(v_k_4297_);
v_isSharedCheck_4332_ = !lean_is_exclusive(v_l_4283_);
if (v_isSharedCheck_4332_ == 0)
{
lean_object* v_unused_4333_; lean_object* v_unused_4334_; lean_object* v_unused_4335_; lean_object* v_unused_4336_; lean_object* v_unused_4337_; 
v_unused_4333_ = lean_ctor_get(v_l_4283_, 4);
lean_dec(v_unused_4333_);
v_unused_4334_ = lean_ctor_get(v_l_4283_, 3);
lean_dec(v_unused_4334_);
v_unused_4335_ = lean_ctor_get(v_l_4283_, 2);
lean_dec(v_unused_4335_);
v_unused_4336_ = lean_ctor_get(v_l_4283_, 1);
lean_dec(v_unused_4336_);
v_unused_4337_ = lean_ctor_get(v_l_4283_, 0);
lean_dec(v_unused_4337_);
v___x_4306_ = v_l_4283_;
v_isShared_4307_ = v_isSharedCheck_4332_;
goto v_resetjp_4305_;
}
else
{
lean_dec(v_l_4283_);
v___x_4306_ = lean_box(0);
v_isShared_4307_ = v_isSharedCheck_4332_;
goto v_resetjp_4305_;
}
v_resetjp_4305_:
{
lean_object* v___x_4308_; lean_object* v___x_4309_; lean_object* v___y_4311_; lean_object* v___y_4312_; lean_object* v___y_4313_; lean_object* v___y_4322_; 
v___x_4308_ = lean_nat_add(v___x_4278_, v_size_4279_);
v___x_4309_ = lean_nat_add(v___x_4308_, v_size_4280_);
lean_dec(v_size_4280_);
if (lean_obj_tag(v_l_4299_) == 0)
{
lean_object* v_size_4330_; 
v_size_4330_ = lean_ctor_get(v_l_4299_, 0);
lean_inc(v_size_4330_);
v___y_4322_ = v_size_4330_;
goto v___jp_4321_;
}
else
{
lean_object* v___x_4331_; 
v___x_4331_ = lean_unsigned_to_nat(0u);
v___y_4322_ = v___x_4331_;
goto v___jp_4321_;
}
v___jp_4310_:
{
lean_object* v___x_4314_; lean_object* v___x_4316_; 
v___x_4314_ = lean_nat_add(v___y_4311_, v___y_4313_);
lean_dec(v___y_4313_);
lean_dec(v___y_4311_);
if (v_isShared_4307_ == 0)
{
lean_ctor_set(v___x_4306_, 4, v_r_4284_);
lean_ctor_set(v___x_4306_, 3, v_r_4300_);
lean_ctor_set(v___x_4306_, 2, v_v_4282_);
lean_ctor_set(v___x_4306_, 1, v_k_4281_);
lean_ctor_set(v___x_4306_, 0, v___x_4314_);
v___x_4316_ = v___x_4306_;
goto v_reusejp_4315_;
}
else
{
lean_object* v_reuseFailAlloc_4320_; 
v_reuseFailAlloc_4320_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4320_, 0, v___x_4314_);
lean_ctor_set(v_reuseFailAlloc_4320_, 1, v_k_4281_);
lean_ctor_set(v_reuseFailAlloc_4320_, 2, v_v_4282_);
lean_ctor_set(v_reuseFailAlloc_4320_, 3, v_r_4300_);
lean_ctor_set(v_reuseFailAlloc_4320_, 4, v_r_4284_);
v___x_4316_ = v_reuseFailAlloc_4320_;
goto v_reusejp_4315_;
}
v_reusejp_4315_:
{
lean_object* v___x_4318_; 
if (v_isShared_4295_ == 0)
{
lean_ctor_set(v___x_4294_, 4, v___x_4316_);
lean_ctor_set(v___x_4294_, 3, v___y_4312_);
lean_ctor_set(v___x_4294_, 2, v_v_4298_);
lean_ctor_set(v___x_4294_, 1, v_k_4297_);
lean_ctor_set(v___x_4294_, 0, v___x_4309_);
v___x_4318_ = v___x_4294_;
goto v_reusejp_4317_;
}
else
{
lean_object* v_reuseFailAlloc_4319_; 
v_reuseFailAlloc_4319_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4319_, 0, v___x_4309_);
lean_ctor_set(v_reuseFailAlloc_4319_, 1, v_k_4297_);
lean_ctor_set(v_reuseFailAlloc_4319_, 2, v_v_4298_);
lean_ctor_set(v_reuseFailAlloc_4319_, 3, v___y_4312_);
lean_ctor_set(v_reuseFailAlloc_4319_, 4, v___x_4316_);
v___x_4318_ = v_reuseFailAlloc_4319_;
goto v_reusejp_4317_;
}
v_reusejp_4317_:
{
return v___x_4318_;
}
}
}
v___jp_4321_:
{
lean_object* v___x_4323_; lean_object* v___x_4325_; 
v___x_4323_ = lean_nat_add(v___x_4308_, v___y_4322_);
lean_dec(v___y_4322_);
lean_dec(v___x_4308_);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_l_4299_);
lean_ctor_set(v___x_4134_, 0, v___x_4323_);
v___x_4325_ = v___x_4134_;
goto v_reusejp_4324_;
}
else
{
lean_object* v_reuseFailAlloc_4329_; 
v_reuseFailAlloc_4329_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4329_, 0, v___x_4323_);
lean_ctor_set(v_reuseFailAlloc_4329_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4329_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4329_, 3, v_l_4131_);
lean_ctor_set(v_reuseFailAlloc_4329_, 4, v_l_4299_);
v___x_4325_ = v_reuseFailAlloc_4329_;
goto v_reusejp_4324_;
}
v_reusejp_4324_:
{
lean_object* v___x_4326_; 
v___x_4326_ = lean_nat_add(v___x_4278_, v_size_4301_);
if (lean_obj_tag(v_r_4300_) == 0)
{
lean_object* v_size_4327_; 
v_size_4327_ = lean_ctor_get(v_r_4300_, 0);
lean_inc(v_size_4327_);
v___y_4311_ = v___x_4326_;
v___y_4312_ = v___x_4325_;
v___y_4313_ = v_size_4327_;
goto v___jp_4310_;
}
else
{
lean_object* v___x_4328_; 
v___x_4328_ = lean_unsigned_to_nat(0u);
v___y_4311_ = v___x_4326_;
v___y_4312_ = v___x_4325_;
v___y_4313_ = v___x_4328_;
goto v___jp_4310_;
}
}
}
}
}
else
{
lean_object* v___x_4338_; lean_object* v___x_4339_; lean_object* v___x_4340_; lean_object* v___x_4342_; 
lean_del_object(v___x_4134_);
v___x_4338_ = lean_nat_add(v___x_4278_, v_size_4279_);
v___x_4339_ = lean_nat_add(v___x_4338_, v_size_4280_);
lean_dec(v_size_4280_);
v___x_4340_ = lean_nat_add(v___x_4338_, v_size_4296_);
lean_dec(v___x_4338_);
lean_inc_ref(v_l_4131_);
if (v_isShared_4295_ == 0)
{
lean_ctor_set(v___x_4294_, 4, v_l_4283_);
lean_ctor_set(v___x_4294_, 3, v_l_4131_);
lean_ctor_set(v___x_4294_, 2, v_v_4130_);
lean_ctor_set(v___x_4294_, 1, v_k_4129_);
lean_ctor_set(v___x_4294_, 0, v___x_4340_);
v___x_4342_ = v___x_4294_;
goto v_reusejp_4341_;
}
else
{
lean_object* v_reuseFailAlloc_4355_; 
v_reuseFailAlloc_4355_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4355_, 0, v___x_4340_);
lean_ctor_set(v_reuseFailAlloc_4355_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4355_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4355_, 3, v_l_4131_);
lean_ctor_set(v_reuseFailAlloc_4355_, 4, v_l_4283_);
v___x_4342_ = v_reuseFailAlloc_4355_;
goto v_reusejp_4341_;
}
v_reusejp_4341_:
{
lean_object* v___x_4344_; uint8_t v_isShared_4345_; uint8_t v_isSharedCheck_4349_; 
v_isSharedCheck_4349_ = !lean_is_exclusive(v_l_4131_);
if (v_isSharedCheck_4349_ == 0)
{
lean_object* v_unused_4350_; lean_object* v_unused_4351_; lean_object* v_unused_4352_; lean_object* v_unused_4353_; lean_object* v_unused_4354_; 
v_unused_4350_ = lean_ctor_get(v_l_4131_, 4);
lean_dec(v_unused_4350_);
v_unused_4351_ = lean_ctor_get(v_l_4131_, 3);
lean_dec(v_unused_4351_);
v_unused_4352_ = lean_ctor_get(v_l_4131_, 2);
lean_dec(v_unused_4352_);
v_unused_4353_ = lean_ctor_get(v_l_4131_, 1);
lean_dec(v_unused_4353_);
v_unused_4354_ = lean_ctor_get(v_l_4131_, 0);
lean_dec(v_unused_4354_);
v___x_4344_ = v_l_4131_;
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
else
{
lean_dec(v_l_4131_);
v___x_4344_ = lean_box(0);
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
v_resetjp_4343_:
{
lean_object* v___x_4347_; 
if (v_isShared_4345_ == 0)
{
lean_ctor_set(v___x_4344_, 4, v_r_4284_);
lean_ctor_set(v___x_4344_, 3, v___x_4342_);
lean_ctor_set(v___x_4344_, 2, v_v_4282_);
lean_ctor_set(v___x_4344_, 1, v_k_4281_);
lean_ctor_set(v___x_4344_, 0, v___x_4339_);
v___x_4347_ = v___x_4344_;
goto v_reusejp_4346_;
}
else
{
lean_object* v_reuseFailAlloc_4348_; 
v_reuseFailAlloc_4348_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4348_, 0, v___x_4339_);
lean_ctor_set(v_reuseFailAlloc_4348_, 1, v_k_4281_);
lean_ctor_set(v_reuseFailAlloc_4348_, 2, v_v_4282_);
lean_ctor_set(v_reuseFailAlloc_4348_, 3, v___x_4342_);
lean_ctor_set(v_reuseFailAlloc_4348_, 4, v_r_4284_);
v___x_4347_ = v_reuseFailAlloc_4348_;
goto v_reusejp_4346_;
}
v_reusejp_4346_:
{
return v___x_4347_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_4362_; 
v_l_4362_ = lean_ctor_get(v_impl_4277_, 3);
lean_inc(v_l_4362_);
if (lean_obj_tag(v_l_4362_) == 0)
{
lean_object* v_r_4363_; lean_object* v_k_4364_; lean_object* v_v_4365_; lean_object* v___x_4367_; uint8_t v_isShared_4368_; uint8_t v_isSharedCheck_4388_; 
v_r_4363_ = lean_ctor_get(v_impl_4277_, 4);
v_k_4364_ = lean_ctor_get(v_impl_4277_, 1);
v_v_4365_ = lean_ctor_get(v_impl_4277_, 2);
v_isSharedCheck_4388_ = !lean_is_exclusive(v_impl_4277_);
if (v_isSharedCheck_4388_ == 0)
{
lean_object* v_unused_4389_; lean_object* v_unused_4390_; 
v_unused_4389_ = lean_ctor_get(v_impl_4277_, 3);
lean_dec(v_unused_4389_);
v_unused_4390_ = lean_ctor_get(v_impl_4277_, 0);
lean_dec(v_unused_4390_);
v___x_4367_ = v_impl_4277_;
v_isShared_4368_ = v_isSharedCheck_4388_;
goto v_resetjp_4366_;
}
else
{
lean_inc(v_r_4363_);
lean_inc(v_v_4365_);
lean_inc(v_k_4364_);
lean_dec(v_impl_4277_);
v___x_4367_ = lean_box(0);
v_isShared_4368_ = v_isSharedCheck_4388_;
goto v_resetjp_4366_;
}
v_resetjp_4366_:
{
lean_object* v_k_4369_; lean_object* v_v_4370_; lean_object* v___x_4372_; uint8_t v_isShared_4373_; uint8_t v_isSharedCheck_4384_; 
v_k_4369_ = lean_ctor_get(v_l_4362_, 1);
v_v_4370_ = lean_ctor_get(v_l_4362_, 2);
v_isSharedCheck_4384_ = !lean_is_exclusive(v_l_4362_);
if (v_isSharedCheck_4384_ == 0)
{
lean_object* v_unused_4385_; lean_object* v_unused_4386_; lean_object* v_unused_4387_; 
v_unused_4385_ = lean_ctor_get(v_l_4362_, 4);
lean_dec(v_unused_4385_);
v_unused_4386_ = lean_ctor_get(v_l_4362_, 3);
lean_dec(v_unused_4386_);
v_unused_4387_ = lean_ctor_get(v_l_4362_, 0);
lean_dec(v_unused_4387_);
v___x_4372_ = v_l_4362_;
v_isShared_4373_ = v_isSharedCheck_4384_;
goto v_resetjp_4371_;
}
else
{
lean_inc(v_v_4370_);
lean_inc(v_k_4369_);
lean_dec(v_l_4362_);
v___x_4372_ = lean_box(0);
v_isShared_4373_ = v_isSharedCheck_4384_;
goto v_resetjp_4371_;
}
v_resetjp_4371_:
{
lean_object* v___x_4374_; lean_object* v___x_4376_; 
v___x_4374_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_4363_, 2);
if (v_isShared_4373_ == 0)
{
lean_ctor_set(v___x_4372_, 4, v_r_4363_);
lean_ctor_set(v___x_4372_, 3, v_r_4363_);
lean_ctor_set(v___x_4372_, 2, v_v_4130_);
lean_ctor_set(v___x_4372_, 1, v_k_4129_);
lean_ctor_set(v___x_4372_, 0, v___x_4278_);
v___x_4376_ = v___x_4372_;
goto v_reusejp_4375_;
}
else
{
lean_object* v_reuseFailAlloc_4383_; 
v_reuseFailAlloc_4383_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4383_, 0, v___x_4278_);
lean_ctor_set(v_reuseFailAlloc_4383_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4383_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4383_, 3, v_r_4363_);
lean_ctor_set(v_reuseFailAlloc_4383_, 4, v_r_4363_);
v___x_4376_ = v_reuseFailAlloc_4383_;
goto v_reusejp_4375_;
}
v_reusejp_4375_:
{
lean_object* v___x_4378_; 
lean_inc(v_r_4363_);
if (v_isShared_4368_ == 0)
{
lean_ctor_set(v___x_4367_, 3, v_r_4363_);
lean_ctor_set(v___x_4367_, 0, v___x_4278_);
v___x_4378_ = v___x_4367_;
goto v_reusejp_4377_;
}
else
{
lean_object* v_reuseFailAlloc_4382_; 
v_reuseFailAlloc_4382_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4382_, 0, v___x_4278_);
lean_ctor_set(v_reuseFailAlloc_4382_, 1, v_k_4364_);
lean_ctor_set(v_reuseFailAlloc_4382_, 2, v_v_4365_);
lean_ctor_set(v_reuseFailAlloc_4382_, 3, v_r_4363_);
lean_ctor_set(v_reuseFailAlloc_4382_, 4, v_r_4363_);
v___x_4378_ = v_reuseFailAlloc_4382_;
goto v_reusejp_4377_;
}
v_reusejp_4377_:
{
lean_object* v___x_4380_; 
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v___x_4378_);
lean_ctor_set(v___x_4134_, 3, v___x_4376_);
lean_ctor_set(v___x_4134_, 2, v_v_4370_);
lean_ctor_set(v___x_4134_, 1, v_k_4369_);
lean_ctor_set(v___x_4134_, 0, v___x_4374_);
v___x_4380_ = v___x_4134_;
goto v_reusejp_4379_;
}
else
{
lean_object* v_reuseFailAlloc_4381_; 
v_reuseFailAlloc_4381_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4381_, 0, v___x_4374_);
lean_ctor_set(v_reuseFailAlloc_4381_, 1, v_k_4369_);
lean_ctor_set(v_reuseFailAlloc_4381_, 2, v_v_4370_);
lean_ctor_set(v_reuseFailAlloc_4381_, 3, v___x_4376_);
lean_ctor_set(v_reuseFailAlloc_4381_, 4, v___x_4378_);
v___x_4380_ = v_reuseFailAlloc_4381_;
goto v_reusejp_4379_;
}
v_reusejp_4379_:
{
return v___x_4380_;
}
}
}
}
}
}
else
{
lean_object* v_r_4391_; 
v_r_4391_ = lean_ctor_get(v_impl_4277_, 4);
lean_inc(v_r_4391_);
if (lean_obj_tag(v_r_4391_) == 0)
{
lean_object* v_k_4392_; lean_object* v_v_4393_; lean_object* v___x_4395_; uint8_t v_isShared_4396_; uint8_t v_isSharedCheck_4404_; 
v_k_4392_ = lean_ctor_get(v_impl_4277_, 1);
v_v_4393_ = lean_ctor_get(v_impl_4277_, 2);
v_isSharedCheck_4404_ = !lean_is_exclusive(v_impl_4277_);
if (v_isSharedCheck_4404_ == 0)
{
lean_object* v_unused_4405_; lean_object* v_unused_4406_; lean_object* v_unused_4407_; 
v_unused_4405_ = lean_ctor_get(v_impl_4277_, 4);
lean_dec(v_unused_4405_);
v_unused_4406_ = lean_ctor_get(v_impl_4277_, 3);
lean_dec(v_unused_4406_);
v_unused_4407_ = lean_ctor_get(v_impl_4277_, 0);
lean_dec(v_unused_4407_);
v___x_4395_ = v_impl_4277_;
v_isShared_4396_ = v_isSharedCheck_4404_;
goto v_resetjp_4394_;
}
else
{
lean_inc(v_v_4393_);
lean_inc(v_k_4392_);
lean_dec(v_impl_4277_);
v___x_4395_ = lean_box(0);
v_isShared_4396_ = v_isSharedCheck_4404_;
goto v_resetjp_4394_;
}
v_resetjp_4394_:
{
lean_object* v___x_4397_; lean_object* v___x_4399_; 
v___x_4397_ = lean_unsigned_to_nat(3u);
if (v_isShared_4396_ == 0)
{
lean_ctor_set(v___x_4395_, 4, v_l_4362_);
lean_ctor_set(v___x_4395_, 2, v_v_4130_);
lean_ctor_set(v___x_4395_, 1, v_k_4129_);
lean_ctor_set(v___x_4395_, 0, v___x_4278_);
v___x_4399_ = v___x_4395_;
goto v_reusejp_4398_;
}
else
{
lean_object* v_reuseFailAlloc_4403_; 
v_reuseFailAlloc_4403_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4403_, 0, v___x_4278_);
lean_ctor_set(v_reuseFailAlloc_4403_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4403_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4403_, 3, v_l_4362_);
lean_ctor_set(v_reuseFailAlloc_4403_, 4, v_l_4362_);
v___x_4399_ = v_reuseFailAlloc_4403_;
goto v_reusejp_4398_;
}
v_reusejp_4398_:
{
lean_object* v___x_4401_; 
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_r_4391_);
lean_ctor_set(v___x_4134_, 3, v___x_4399_);
lean_ctor_set(v___x_4134_, 2, v_v_4393_);
lean_ctor_set(v___x_4134_, 1, v_k_4392_);
lean_ctor_set(v___x_4134_, 0, v___x_4397_);
v___x_4401_ = v___x_4134_;
goto v_reusejp_4400_;
}
else
{
lean_object* v_reuseFailAlloc_4402_; 
v_reuseFailAlloc_4402_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4402_, 0, v___x_4397_);
lean_ctor_set(v_reuseFailAlloc_4402_, 1, v_k_4392_);
lean_ctor_set(v_reuseFailAlloc_4402_, 2, v_v_4393_);
lean_ctor_set(v_reuseFailAlloc_4402_, 3, v___x_4399_);
lean_ctor_set(v_reuseFailAlloc_4402_, 4, v_r_4391_);
v___x_4401_ = v_reuseFailAlloc_4402_;
goto v_reusejp_4400_;
}
v_reusejp_4400_:
{
return v___x_4401_;
}
}
}
}
else
{
lean_object* v___x_4408_; lean_object* v___x_4410_; 
v___x_4408_ = lean_unsigned_to_nat(2u);
if (v_isShared_4135_ == 0)
{
lean_ctor_set(v___x_4134_, 4, v_impl_4277_);
lean_ctor_set(v___x_4134_, 3, v_r_4391_);
lean_ctor_set(v___x_4134_, 0, v___x_4408_);
v___x_4410_ = v___x_4134_;
goto v_reusejp_4409_;
}
else
{
lean_object* v_reuseFailAlloc_4411_; 
v_reuseFailAlloc_4411_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4411_, 0, v___x_4408_);
lean_ctor_set(v_reuseFailAlloc_4411_, 1, v_k_4129_);
lean_ctor_set(v_reuseFailAlloc_4411_, 2, v_v_4130_);
lean_ctor_set(v_reuseFailAlloc_4411_, 3, v_r_4391_);
lean_ctor_set(v_reuseFailAlloc_4411_, 4, v_impl_4277_);
v___x_4410_ = v_reuseFailAlloc_4411_;
goto v_reusejp_4409_;
}
v_reusejp_4409_:
{
return v___x_4410_;
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
lean_object* v___x_4413_; lean_object* v___x_4414_; 
v___x_4413_ = lean_unsigned_to_nat(1u);
v___x_4414_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4414_, 0, v___x_4413_);
lean_ctor_set(v___x_4414_, 1, v_k_4125_);
lean_ctor_set(v___x_4414_, 2, v_v_4126_);
lean_ctor_set(v___x_4414_, 3, v_t_4127_);
lean_ctor_set(v___x_4414_, 4, v_t_4127_);
return v___x_4414_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg(lean_object* v_as_x27_4415_, lean_object* v_b_4416_){
_start:
{
if (lean_obj_tag(v_as_x27_4415_) == 0)
{
return v_b_4416_;
}
else
{
lean_object* v_head_4417_; lean_object* v_tail_4418_; uint8_t v___x_4419_; 
v_head_4417_ = lean_ctor_get(v_as_x27_4415_, 0);
v_tail_4418_ = lean_ctor_get(v_as_x27_4415_, 1);
v___x_4419_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg(v_head_4417_, v_b_4416_);
if (v___x_4419_ == 0)
{
lean_object* v___x_4420_; lean_object* v___x_4421_; 
v___x_4420_ = lean_box(0);
lean_inc(v_head_4417_);
v___x_4421_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(v_head_4417_, v___x_4420_, v_b_4416_);
v_as_x27_4415_ = v_tail_4418_;
v_b_4416_ = v___x_4421_;
goto _start;
}
else
{
v_as_x27_4415_ = v_tail_4418_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg___boxed(lean_object* v_as_x27_4424_, lean_object* v_b_4425_){
_start:
{
lean_object* v_res_4426_; 
v_res_4426_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg(v_as_x27_4424_, v_b_4425_);
lean_dec(v_as_x27_4424_);
return v_res_4426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert(lean_object* v_b_4429_, lean_object* v_x_4430_){
_start:
{
if (lean_obj_tag(v_x_4430_) == 0)
{
lean_object* v_declName_4431_; lean_object* v_unfold_4432_; lean_object* v_unfolds_4433_; lean_object* v_casts_4434_; lean_object* v_insertionFuns_4435_; lean_object* v___x_4437_; uint8_t v_isShared_4438_; uint8_t v_isSharedCheck_4451_; 
v_declName_4431_ = lean_ctor_get(v_x_4430_, 0);
lean_inc(v_declName_4431_);
v_unfold_4432_ = lean_ctor_get(v_x_4430_, 1);
lean_inc(v_unfold_4432_);
lean_dec_ref_known(v_x_4430_, 2);
v_unfolds_4433_ = lean_ctor_get(v_b_4429_, 0);
v_casts_4434_ = lean_ctor_get(v_b_4429_, 1);
v_insertionFuns_4435_ = lean_ctor_get(v_b_4429_, 2);
v_isSharedCheck_4451_ = !lean_is_exclusive(v_b_4429_);
if (v_isSharedCheck_4451_ == 0)
{
v___x_4437_ = v_b_4429_;
v_isShared_4438_ = v_isSharedCheck_4451_;
goto v_resetjp_4436_;
}
else
{
lean_inc(v_insertionFuns_4435_);
lean_inc(v_casts_4434_);
lean_inc(v_unfolds_4433_);
lean_dec(v_b_4429_);
v___x_4437_ = lean_box(0);
v_isShared_4438_ = v_isSharedCheck_4451_;
goto v_resetjp_4436_;
}
v_resetjp_4436_:
{
lean_object* v___x_4439_; lean_object* v___x_4440_; lean_object* v___x_4441_; lean_object* v___x_4442_; uint8_t v___x_4443_; uint8_t v___x_4444_; lean_object* v___x_4445_; lean_object* v___x_4446_; lean_object* v___x_4447_; lean_object* v___x_4449_; 
v___x_4439_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert___closed__0));
v___x_4440_ = lean_box(0);
lean_inc(v_unfold_4432_);
v___x_4441_ = l_Lean_mkConst(v_unfold_4432_, v___x_4440_);
v___x_4442_ = lean_unsigned_to_nat(1000u);
v___x_4443_ = 1;
v___x_4444_ = 0;
v___x_4445_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_4445_, 0, v_unfold_4432_);
lean_ctor_set_uint8(v___x_4445_, sizeof(void*)*1, v___x_4443_);
lean_ctor_set_uint8(v___x_4445_, sizeof(void*)*1 + 1, v___x_4444_);
v___x_4446_ = lean_alloc_ctor(0, 5, 4);
lean_ctor_set(v___x_4446_, 0, v___x_4439_);
lean_ctor_set(v___x_4446_, 1, v___x_4439_);
lean_ctor_set(v___x_4446_, 2, v___x_4441_);
lean_ctor_set(v___x_4446_, 3, v___x_4442_);
lean_ctor_set(v___x_4446_, 4, v___x_4445_);
lean_ctor_set_uint8(v___x_4446_, sizeof(void*)*5, v___x_4443_);
lean_ctor_set_uint8(v___x_4446_, sizeof(void*)*5 + 1, v___x_4444_);
lean_ctor_set_uint8(v___x_4446_, sizeof(void*)*5 + 2, v___x_4444_);
lean_ctor_set_uint8(v___x_4446_, sizeof(void*)*5 + 3, v___x_4444_);
v___x_4447_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_declName_4431_, v___x_4446_, v_unfolds_4433_);
if (v_isShared_4438_ == 0)
{
lean_ctor_set(v___x_4437_, 0, v___x_4447_);
v___x_4449_ = v___x_4437_;
goto v_reusejp_4448_;
}
else
{
lean_object* v_reuseFailAlloc_4450_; 
v_reuseFailAlloc_4450_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4450_, 0, v___x_4447_);
lean_ctor_set(v_reuseFailAlloc_4450_, 1, v_casts_4434_);
lean_ctor_set(v_reuseFailAlloc_4450_, 2, v_insertionFuns_4435_);
v___x_4449_ = v_reuseFailAlloc_4450_;
goto v_reusejp_4448_;
}
v_reusejp_4448_:
{
return v___x_4449_;
}
}
}
else
{
lean_object* v_declName_4452_; lean_object* v_unfold_4453_; lean_object* v_refold_4454_; lean_object* v_unfold_x27_4455_; lean_object* v_refold_x27_4456_; lean_object* v_unfolds_4457_; lean_object* v_casts_4458_; lean_object* v_insertionFuns_4459_; lean_object* v___x_4461_; uint8_t v_isShared_4462_; uint8_t v_isSharedCheck_4474_; 
v_declName_4452_ = lean_ctor_get(v_x_4430_, 0);
lean_inc(v_declName_4452_);
v_unfold_4453_ = lean_ctor_get(v_x_4430_, 1);
lean_inc(v_unfold_4453_);
v_refold_4454_ = lean_ctor_get(v_x_4430_, 2);
lean_inc(v_refold_4454_);
v_unfold_x27_4455_ = lean_ctor_get(v_x_4430_, 3);
lean_inc(v_unfold_x27_4455_);
v_refold_x27_4456_ = lean_ctor_get(v_x_4430_, 4);
lean_inc(v_refold_x27_4456_);
lean_dec_ref_known(v_x_4430_, 5);
v_unfolds_4457_ = lean_ctor_get(v_b_4429_, 0);
v_casts_4458_ = lean_ctor_get(v_b_4429_, 1);
v_insertionFuns_4459_ = lean_ctor_get(v_b_4429_, 2);
v_isSharedCheck_4474_ = !lean_is_exclusive(v_b_4429_);
if (v_isSharedCheck_4474_ == 0)
{
v___x_4461_ = v_b_4429_;
v_isShared_4462_ = v_isSharedCheck_4474_;
goto v_resetjp_4460_;
}
else
{
lean_inc(v_insertionFuns_4459_);
lean_inc(v_casts_4458_);
lean_inc(v_unfolds_4457_);
lean_dec(v_b_4429_);
v___x_4461_ = lean_box(0);
v_isShared_4462_ = v_isSharedCheck_4474_;
goto v_resetjp_4460_;
}
v_resetjp_4460_:
{
lean_object* v___x_4463_; lean_object* v___x_4464_; lean_object* v___x_4465_; lean_object* v___x_4466_; lean_object* v___x_4467_; lean_object* v___x_4468_; lean_object* v___x_4469_; lean_object* v___x_4470_; lean_object* v___x_4472_; 
lean_inc(v_refold_4454_);
lean_inc(v_unfold_4453_);
v___x_4463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4463_, 0, v_unfold_4453_);
lean_ctor_set(v___x_4463_, 1, v_refold_4454_);
v___x_4464_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_declName_4452_, v___x_4463_, v_casts_4458_);
v___x_4465_ = lean_box(0);
v___x_4466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4466_, 0, v_refold_x27_4456_);
lean_ctor_set(v___x_4466_, 1, v___x_4465_);
v___x_4467_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4467_, 0, v_unfold_x27_4455_);
lean_ctor_set(v___x_4467_, 1, v___x_4466_);
v___x_4468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4468_, 0, v_refold_4454_);
lean_ctor_set(v___x_4468_, 1, v___x_4467_);
v___x_4469_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4469_, 0, v_unfold_4453_);
lean_ctor_set(v___x_4469_, 1, v___x_4468_);
v___x_4470_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg(v___x_4469_, v_insertionFuns_4459_);
lean_dec_ref_known(v___x_4469_, 2);
if (v_isShared_4462_ == 0)
{
lean_ctor_set(v___x_4461_, 2, v___x_4470_);
lean_ctor_set(v___x_4461_, 1, v___x_4464_);
v___x_4472_ = v___x_4461_;
goto v_reusejp_4471_;
}
else
{
lean_object* v_reuseFailAlloc_4473_; 
v_reuseFailAlloc_4473_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4473_, 0, v_unfolds_4457_);
lean_ctor_set(v_reuseFailAlloc_4473_, 1, v___x_4464_);
lean_ctor_set(v_reuseFailAlloc_4473_, 2, v___x_4470_);
v___x_4472_ = v_reuseFailAlloc_4473_;
goto v_reusejp_4471_;
}
v_reusejp_4471_:
{
return v___x_4472_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0(lean_object* v_00_u03b2_4475_, lean_object* v_k_4476_, lean_object* v_t_4477_){
_start:
{
uint8_t v___x_4478_; 
v___x_4478_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___redArg(v_k_4476_, v_t_4477_);
return v___x_4478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0___boxed(lean_object* v_00_u03b2_4479_, lean_object* v_k_4480_, lean_object* v_t_4481_){
_start:
{
uint8_t v_res_4482_; lean_object* v_r_4483_; 
v_res_4482_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__0(v_00_u03b2_4479_, v_k_4480_, v_t_4481_);
lean_dec(v_t_4481_);
lean_dec(v_k_4480_);
v_r_4483_ = lean_box(v_res_4482_);
return v_r_4483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1(lean_object* v_00_u03b2_4484_, lean_object* v_k_4485_, lean_object* v_v_4486_, lean_object* v_t_4487_, lean_object* v_hl_4488_){
_start:
{
lean_object* v___x_4489_; 
v___x_4489_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__1___redArg(v_k_4485_, v_v_4486_, v_t_4487_);
return v___x_4489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2(lean_object* v_as_4490_, lean_object* v_as_x27_4491_, lean_object* v_b_4492_, lean_object* v_a_4493_){
_start:
{
lean_object* v___x_4494_; 
v___x_4494_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___redArg(v_as_x27_4491_, v_b_4492_);
return v___x_4494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2___boxed(lean_object* v_as_4495_, lean_object* v_as_x27_4496_, lean_object* v_b_4497_, lean_object* v_a_4498_){
_start:
{
lean_object* v_res_4499_; 
v_res_4499_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert_spec__2(v_as_4495_, v_as_x27_4496_, v_b_4497_, v_a_4498_);
lean_dec(v_as_x27_4496_);
lean_dec(v_as_4495_);
return v_res_4499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0(lean_object* v_as_4500_, size_t v_i_4501_, size_t v_stop_4502_, lean_object* v_b_4503_){
_start:
{
uint8_t v___x_4504_; 
v___x_4504_ = lean_usize_dec_eq(v_i_4501_, v_stop_4502_);
if (v___x_4504_ == 0)
{
lean_object* v___x_4505_; lean_object* v___x_4506_; size_t v___x_4507_; size_t v___x_4508_; 
v___x_4505_ = lean_array_uget_borrowed(v_as_4500_, v_i_4501_);
lean_inc(v___x_4505_);
v___x_4506_ = lp_mathlib___private_Mathlib_Tactic_Translate_UnfoldBoundary_0__Mathlib_Tactic_UnfoldBoundary_UnfoldBoundaries_insert(v_b_4503_, v___x_4505_);
v___x_4507_ = ((size_t)1ULL);
v___x_4508_ = lean_usize_add(v_i_4501_, v___x_4507_);
v_i_4501_ = v___x_4508_;
v_b_4503_ = v___x_4506_;
goto _start;
}
else
{
return v_b_4503_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0___boxed(lean_object* v_as_4510_, lean_object* v_i_4511_, lean_object* v_stop_4512_, lean_object* v_b_4513_){
_start:
{
size_t v_i_boxed_4514_; size_t v_stop_boxed_4515_; lean_object* v_res_4516_; 
v_i_boxed_4514_ = lean_unbox_usize(v_i_4511_);
lean_dec(v_i_4511_);
v_stop_boxed_4515_ = lean_unbox_usize(v_stop_4512_);
lean_dec(v_stop_4512_);
v_res_4516_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0(v_as_4510_, v_i_boxed_4514_, v_stop_boxed_4515_, v_b_4513_);
lean_dec_ref(v_as_4510_);
return v_res_4516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1(lean_object* v_as_4517_, size_t v_i_4518_, size_t v_stop_4519_, lean_object* v_b_4520_){
_start:
{
lean_object* v___y_4522_; uint8_t v___x_4526_; 
v___x_4526_ = lean_usize_dec_eq(v_i_4518_, v_stop_4519_);
if (v___x_4526_ == 0)
{
lean_object* v___x_4527_; lean_object* v___x_4528_; lean_object* v___x_4529_; uint8_t v___x_4530_; 
v___x_4527_ = lean_array_uget_borrowed(v_as_4517_, v_i_4518_);
v___x_4528_ = lean_unsigned_to_nat(0u);
v___x_4529_ = lean_array_get_size(v___x_4527_);
v___x_4530_ = lean_nat_dec_lt(v___x_4528_, v___x_4529_);
if (v___x_4530_ == 0)
{
v___y_4522_ = v_b_4520_;
goto v___jp_4521_;
}
else
{
uint8_t v___x_4531_; 
v___x_4531_ = lean_nat_dec_le(v___x_4529_, v___x_4529_);
if (v___x_4531_ == 0)
{
if (v___x_4530_ == 0)
{
v___y_4522_ = v_b_4520_;
goto v___jp_4521_;
}
else
{
size_t v___x_4532_; size_t v___x_4533_; lean_object* v___x_4534_; 
v___x_4532_ = ((size_t)0ULL);
v___x_4533_ = lean_usize_of_nat(v___x_4529_);
v___x_4534_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0(v___x_4527_, v___x_4532_, v___x_4533_, v_b_4520_);
v___y_4522_ = v___x_4534_;
goto v___jp_4521_;
}
}
else
{
size_t v___x_4535_; size_t v___x_4536_; lean_object* v___x_4537_; 
v___x_4535_ = ((size_t)0ULL);
v___x_4536_ = lean_usize_of_nat(v___x_4529_);
v___x_4537_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__0(v___x_4527_, v___x_4535_, v___x_4536_, v_b_4520_);
v___y_4522_ = v___x_4537_;
goto v___jp_4521_;
}
}
}
else
{
return v_b_4520_;
}
v___jp_4521_:
{
size_t v___x_4523_; size_t v___x_4524_; 
v___x_4523_ = ((size_t)1ULL);
v___x_4524_ = lean_usize_add(v_i_4518_, v___x_4523_);
v_i_4518_ = v___x_4524_;
v_b_4520_ = v___y_4522_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1___boxed(lean_object* v_as_4538_, lean_object* v_i_4539_, lean_object* v_stop_4540_, lean_object* v_b_4541_){
_start:
{
size_t v_i_boxed_4542_; size_t v_stop_boxed_4543_; lean_object* v_res_4544_; 
v_i_boxed_4542_ = lean_unbox_usize(v_i_4539_);
lean_dec(v_i_4539_);
v_stop_boxed_4543_ = lean_unbox_usize(v_stop_4540_);
lean_dec(v_stop_4540_);
v_res_4544_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1(v_as_4538_, v_i_boxed_4542_, v_stop_boxed_4543_, v_b_4541_);
lean_dec_ref(v_as_4538_);
return v_res_4544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0(lean_object* v_as_4545_){
_start:
{
lean_object* v___x_4546_; lean_object* v___x_4547_; lean_object* v___x_4548_; uint8_t v___x_4549_; 
v___x_4546_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0, &lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default___closed__0);
v___x_4547_ = lean_unsigned_to_nat(0u);
v___x_4548_ = lean_array_get_size(v_as_4545_);
v___x_4549_ = lean_nat_dec_lt(v___x_4547_, v___x_4548_);
if (v___x_4549_ == 0)
{
return v___x_4546_;
}
else
{
uint8_t v___x_4550_; 
v___x_4550_ = lean_nat_dec_le(v___x_4548_, v___x_4548_);
if (v___x_4550_ == 0)
{
if (v___x_4549_ == 0)
{
return v___x_4546_;
}
else
{
size_t v___x_4551_; size_t v___x_4552_; lean_object* v___x_4553_; 
v___x_4551_ = ((size_t)0ULL);
v___x_4552_ = lean_usize_of_nat(v___x_4548_);
v___x_4553_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1(v_as_4545_, v___x_4551_, v___x_4552_, v___x_4546_);
return v___x_4553_;
}
}
else
{
size_t v___x_4554_; size_t v___x_4555_; lean_object* v___x_4556_; 
v___x_4554_ = ((size_t)0ULL);
v___x_4555_ = lean_usize_of_nat(v___x_4548_);
v___x_4556_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt_spec__1(v_as_4545_, v___x_4554_, v___x_4555_, v___x_4546_);
return v___x_4556_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0___boxed(lean_object* v_as_4557_){
_start:
{
lean_object* v_res_4558_; 
v_res_4558_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__0(v_as_4557_);
lean_dec_ref(v_as_4557_);
return v_res_4558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___lam__1(lean_object* v_es_4559_){
_start:
{
lean_object* v___x_4560_; 
v___x_4560_ = lean_array_mk(v_es_4559_);
return v___x_4560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt(){
_start:
{
lean_object* v___x_4581_; lean_object* v___x_4582_; 
v___x_4581_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___closed__8));
v___x_4582_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_4581_);
return v___x_4582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt___boxed(lean_object* v_a_4583_){
_start:
{
lean_object* v_res_4584_; 
v_res_4584_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt();
return v_res_4584_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(uint8_t builtin) {
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
res = runtime_initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default = _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries_default);
lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries = _init_lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_UnfoldBoundary_instInhabitedUnfoldBoundaries);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_UnfoldBoundary(builtin);
}
#ifdef __cplusplus
}
#endif
