// Lean compiler output
// Module: Mathlib.Tactic.GRewrite.Core
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Rewrite public import Mathlib.Tactic.GCongr.Core public import Lean.Meta.Tactic.Rewrite meta import Mathlib.Tactic.GCongr.Core
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
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_throwFunctionExpected___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshTypeMVar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_st_ref_take(lean_object*);
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_updateRel(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Meta_getMVarsNoDelayed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_postprocessAppMVars(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_tactic_skipAssignedInstances;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_getRel(lean_object*);
lean_object* l_Lean_Expr_toHeadIndex(lean_object*);
lean_object* l_Lean_Expr_headNumArgs(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_applyRflOrId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_local_ctx_num_indices(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_collectFVars(lean_object*, lean_object*);
uint8_t l_Lean_LocalContext_contains(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_addDecl(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isClass_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_dischargeSide(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_applyGCongrLemma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_getCongrAppFnArgs(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_findGCongrLemmas_x3f_x27___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_relImpRelLemma(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_GCongr_forwardExt;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_applySymm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqHeadIndex_beq(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_gcongrForwardDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_gcongrDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_GCongrM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_gcongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_gcongrForward(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_kabstract(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GCongr_mkHoleAnnotation(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasBinderNameHint(lean_object*);
lean_object* l_Lean_Expr_resolveBinderNameHint(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Meta_check(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_rewrite(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getForallArity(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "grewrite"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 57, 211, 158, 182, 81, 204, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "could not discharge "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_a"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 106, 112, 29, 6, 211, 214, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_GCongr_gcongrDischarger___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "rewritten expression is not type correct:"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "\nError: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 176, .m_capacity = 176, .m_length = 175, .m_data = "\n\nPossible solutions: use grewrite's 'occs' configuration option to limit which occurrences are rewritten, or specify what the rewritten expression should be and use 'gcongr'."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "did not find instance of the pattern in the target expression"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__0_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__0_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__0_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__0_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 101, 182, 48, 118, 97, 45, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__2_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__2_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__2_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__3_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__2_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__3_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__3_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__5_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__3_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__5_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__5_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__7_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__5_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__7_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__7_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "GRewrite"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__9_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__7_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(123, 145, 139, 17, 247, 134, 158, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__9_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__9_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__10_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__10_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__10_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__11_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__9_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__10_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(19, 134, 141, 46, 35, 18, 49, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__11_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__11_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__12_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__11_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(254, 222, 179, 30, 147, 122, 244, 129)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__12_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__12_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__13_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__12_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(247, 69, 245, 97, 197, 208, 142, 163)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__13_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__13_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__14_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__13_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 90, 156, 95, 180, 134, 16, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__14_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__14_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__15_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__14_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(126, 39, 174, 144, 1, 236, 134, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__15_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__15_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__16_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__16_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__16_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__17_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__15_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__16_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(179, 254, 97, 181, 39, 121, 177, 106)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__17_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__17_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__18_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__18_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__18_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__19_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__17_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__18_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 70, 111, 180, 158, 4, 209, 243)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__19_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__19_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__20_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__19_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(31, 213, 216, 70, 125, 38, 28, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__20_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__20_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__21_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__20_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(46, 64, 102, 192, 65, 39, 184, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__21_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__21_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__22_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__21_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__8_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(182, 160, 8, 4, 120, 39, 183, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__22_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__22_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__23_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__22_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__10_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(146, 45, 97, 186, 197, 75, 61, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__23_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__23_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__25_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__25_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__25_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__27_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__27_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__27_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_noMatch_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_noMatch_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matched_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matched_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matchedOutOfScope_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matchedOutOfScope_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__2_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "GCongr"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__3_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__4_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "exactRefl"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "rewriting with `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0;
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "grw: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = " is a dependent relation"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_Implies"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(41, 107, 183, 187, 41, 214, 66, 241)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0(uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "applying `gcongr` lemma "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = " could not be closed with `rfl`:\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15(uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17(lean_object*, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "cached: no rewrite"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2___boxed(lean_object**);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " of relation `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "→"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "visiting `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` in the "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "RHS"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "LHS"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "internal `grewrite` error: invalid `gcongr` goal "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1(lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Did not find a rewrite with"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "\nin the target expression"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "\n\nUse the command `set_option trace.Meta.grewrite true` to inspect this."};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2;
static const lean_closure_object lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_GCongr_gcongrForwardDischarger___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = " is not a valid relation"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "pattern is a metavariable"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\nfrom relation"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = " is not a relation"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__9_value),LEAN_SCALAR_PTR_LITERAL(146, 109, 21, 40, 70, 113, 251, 6)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__11_value),LEAN_SCALAR_PTR_LITERAL(183, 66, 254, 161, 210, 133, 94, 78)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__13_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__14 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__8_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__15 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__15_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "apply_rw"};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__16 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__16_value),LEAN_SCALAR_PTR_LITERAL(124, 98, 115, 205, 35, 237, 57, 208)}};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__17 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__17_value;
static const lean_string_object lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "invalid implication "};
static const lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__18 = (const lean_object*)&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__2));
v___x_6_ = l_Lean_stringToMessageData(v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__4));
v___x_9_ = l_Lean_stringToMessageData(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain(lean_object* v_hrel_10_, lean_object* v_goal_11_, lean_object* v_a_12_, lean_object* v_a_13_, lean_object* v_a_14_, lean_object* v_a_15_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = lean_unsigned_to_nat(1u);
v___x_18_ = lean_mk_empty_array_with_capacity(v___x_17_);
lean_inc_ref(v_hrel_10_);
v___x_19_ = lean_array_push(v___x_18_, v_hrel_10_);
lean_inc(v_goal_11_);
v___x_20_ = lp_mathlib_Lean_MVarId_gcongrForward(v___x_19_, v_goal_11_, v_a_12_, v_a_13_, v_a_14_, v_a_15_);
lean_dec_ref(v___x_19_);
if (lean_obj_tag(v___x_20_) == 0)
{
lean_object* v_a_21_; uint8_t v___x_22_; 
v_a_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc(v_a_21_);
v___x_22_ = lean_unbox(v_a_21_);
lean_dec(v_a_21_);
if (v___x_22_ == 0)
{
lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_58_; 
v_isSharedCheck_58_ = !lean_is_exclusive(v___x_20_);
if (v_isSharedCheck_58_ == 0)
{
lean_object* v_unused_59_; 
v_unused_59_ = lean_ctor_get(v___x_20_, 0);
lean_dec(v_unused_59_);
v___x_24_ = v___x_20_;
v_isShared_25_ = v_isSharedCheck_58_;
goto v_resetjp_23_;
}
else
{
lean_dec(v___x_20_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_58_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_26_; 
lean_inc(v_goal_11_);
v___x_26_ = l_Lean_MVarId_getType(v_goal_11_, v_a_12_, v_a_13_, v_a_14_, v_a_15_);
if (lean_obj_tag(v___x_26_) == 0)
{
lean_object* v_a_27_; lean_object* v___x_28_; 
v_a_27_ = lean_ctor_get(v___x_26_, 0);
lean_inc(v_a_27_);
lean_dec_ref_known(v___x_26_, 1);
lean_inc(v_a_15_);
lean_inc_ref(v_a_14_);
lean_inc(v_a_13_);
lean_inc_ref(v_a_12_);
v___x_28_ = lean_infer_type(v_hrel_10_, v_a_12_, v_a_13_, v_a_14_, v_a_15_);
if (lean_obj_tag(v___x_28_) == 0)
{
lean_object* v_a_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_39_; 
v_a_29_ = lean_ctor_get(v___x_28_, 0);
lean_inc(v_a_29_);
lean_dec_ref_known(v___x_28_, 1);
v___x_30_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1));
v___x_31_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__3);
v___x_32_ = l_Lean_MessageData_ofExpr(v_a_27_);
v___x_33_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_31_);
lean_ctor_set(v___x_33_, 1, v___x_32_);
v___x_34_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__5);
v___x_35_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_35_, 0, v___x_33_);
lean_ctor_set(v___x_35_, 1, v___x_34_);
v___x_36_ = l_Lean_MessageData_ofExpr(v_a_29_);
v___x_37_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_37_, 0, v___x_35_);
lean_ctor_set(v___x_37_, 1, v___x_36_);
if (v_isShared_25_ == 0)
{
lean_ctor_set_tag(v___x_24_, 1);
lean_ctor_set(v___x_24_, 0, v___x_37_);
v___x_39_ = v___x_24_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v___x_37_);
v___x_39_ = v_reuseFailAlloc_41_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
lean_object* v___x_40_; 
v___x_40_ = l_Lean_Meta_throwTacticEx___redArg(v___x_30_, v_goal_11_, v___x_39_, v_a_12_, v_a_13_, v_a_14_, v_a_15_);
return v___x_40_;
}
}
else
{
lean_object* v_a_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_49_; 
lean_dec(v_a_27_);
lean_del_object(v___x_24_);
lean_dec(v_goal_11_);
v_a_42_ = lean_ctor_get(v___x_28_, 0);
v_isSharedCheck_49_ = !lean_is_exclusive(v___x_28_);
if (v_isSharedCheck_49_ == 0)
{
v___x_44_ = v___x_28_;
v_isShared_45_ = v_isSharedCheck_49_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_a_42_);
lean_dec(v___x_28_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_49_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___x_47_; 
if (v_isShared_45_ == 0)
{
v___x_47_ = v___x_44_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v_a_42_);
v___x_47_ = v_reuseFailAlloc_48_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
return v___x_47_;
}
}
}
}
else
{
lean_object* v_a_50_; lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_57_; 
lean_del_object(v___x_24_);
lean_dec(v_goal_11_);
lean_dec_ref(v_hrel_10_);
v_a_50_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_57_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_57_ == 0)
{
v___x_52_ = v___x_26_;
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
else
{
lean_inc(v_a_50_);
lean_dec(v___x_26_);
v___x_52_ = lean_box(0);
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
v_resetjp_51_:
{
lean_object* v___x_55_; 
if (v_isShared_53_ == 0)
{
v___x_55_ = v___x_52_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v_a_50_);
v___x_55_ = v_reuseFailAlloc_56_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
return v___x_55_;
}
}
}
}
}
else
{
lean_dec(v_goal_11_);
lean_dec_ref(v_hrel_10_);
return v___x_20_;
}
}
else
{
lean_dec(v_goal_11_);
lean_dec_ref(v_hrel_10_);
return v___x_20_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___boxed(lean_object* v_hrel_60_, lean_object* v_goal_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain(v_hrel_60_, v_goal_61_, v_a_62_, v_a_63_, v_a_64_, v_a_65_);
lean_dec(v_a_65_);
lean_dec_ref(v_a_64_);
lean_dec(v_a_63_);
lean_dec_ref(v_a_62_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(lean_object* v_e_68_, lean_object* v___y_69_){
_start:
{
uint8_t v___x_71_; 
v___x_71_ = l_Lean_Expr_hasMVar(v_e_68_);
if (v___x_71_ == 0)
{
lean_object* v___x_72_; 
v___x_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_72_, 0, v_e_68_);
return v___x_72_;
}
else
{
lean_object* v___x_73_; lean_object* v_mctx_74_; lean_object* v___x_75_; lean_object* v_fst_76_; lean_object* v_snd_77_; lean_object* v___x_78_; lean_object* v_cache_79_; lean_object* v_zetaDeltaFVarIds_80_; lean_object* v_postponed_81_; lean_object* v_diag_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_91_; 
v___x_73_ = lean_st_ref_get(v___y_69_);
v_mctx_74_ = lean_ctor_get(v___x_73_, 0);
lean_inc_ref(v_mctx_74_);
lean_dec(v___x_73_);
v___x_75_ = l_Lean_instantiateMVarsCore(v_mctx_74_, v_e_68_);
v_fst_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_fst_76_);
v_snd_77_ = lean_ctor_get(v___x_75_, 1);
lean_inc(v_snd_77_);
lean_dec_ref(v___x_75_);
v___x_78_ = lean_st_ref_take(v___y_69_);
v_cache_79_ = lean_ctor_get(v___x_78_, 1);
v_zetaDeltaFVarIds_80_ = lean_ctor_get(v___x_78_, 2);
v_postponed_81_ = lean_ctor_get(v___x_78_, 3);
v_diag_82_ = lean_ctor_get(v___x_78_, 4);
v_isSharedCheck_91_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_91_ == 0)
{
lean_object* v_unused_92_; 
v_unused_92_ = lean_ctor_get(v___x_78_, 0);
lean_dec(v_unused_92_);
v___x_84_ = v___x_78_;
v_isShared_85_ = v_isSharedCheck_91_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_diag_82_);
lean_inc(v_postponed_81_);
lean_inc(v_zetaDeltaFVarIds_80_);
lean_inc(v_cache_79_);
lean_dec(v___x_78_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_91_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 0, v_snd_77_);
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v_snd_77_);
lean_ctor_set(v_reuseFailAlloc_90_, 1, v_cache_79_);
lean_ctor_set(v_reuseFailAlloc_90_, 2, v_zetaDeltaFVarIds_80_);
lean_ctor_set(v_reuseFailAlloc_90_, 3, v_postponed_81_);
lean_ctor_set(v_reuseFailAlloc_90_, 4, v_diag_82_);
v___x_87_ = v_reuseFailAlloc_90_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = lean_st_ref_set(v___y_69_, v___x_87_);
v___x_89_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_89_, 0, v_fst_76_);
return v___x_89_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg___boxed(lean_object* v_e_93_, lean_object* v___y_94_, lean_object* v___y_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(v_e_93_, v___y_94_);
lean_dec(v___y_94_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0(lean_object* v_e_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(v_e_97_, v___y_99_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___boxed(lean_object* v_e_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0(v_e_104_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec(v___y_106_);
lean_dec_ref(v___y_105_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0(lean_object* v_e_u2081_114_, lean_object* v_e_u2082_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; lean_object* v___x_118_; 
v___x_116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1));
v___x_117_ = 0;
v___x_118_ = l_Lean_Expr_forallE___override(v___x_116_, v_e_u2081_114_, v_e_u2082_115_, v___x_117_);
return v___x_118_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__1));
v___x_122_ = l_Lean_stringToMessageData(v___x_121_);
return v___x_122_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__3));
v___x_125_ = l_Lean_stringToMessageData(v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__5));
v___x_128_ = l_Lean_stringToMessageData(v___x_127_);
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__7));
v___x_131_ = l_Lean_stringToMessageData(v___x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract(lean_object* v_goal_132_, lean_object* v_e_133_, lean_object* v_hrel_134_, lean_object* v_pattern_135_, lean_object* v_replacement_136_, uint8_t v_forwardImp_137_, lean_object* v_config_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v___y_145_; lean_object* v___y_146_; lean_object* v___y_147_; lean_object* v___y_148_; lean_object* v___y_149_; lean_object* v___y_150_; lean_object* v_toConfig_207_; uint8_t v_transparency_208_; uint8_t v_offsetCnstrs_209_; lean_object* v_occs_210_; lean_object* v___x_211_; uint8_t v_foApprox_212_; uint8_t v_ctxApprox_213_; uint8_t v_quasiPatternApprox_214_; uint8_t v_constApprox_215_; uint8_t v_isDefEqStuckEx_216_; uint8_t v_unificationHints_217_; uint8_t v_proofIrrelevance_218_; uint8_t v_assignSyntheticOpaque_219_; uint8_t v_etaStruct_220_; uint8_t v_univApprox_221_; uint8_t v_iota_222_; uint8_t v_beta_223_; uint8_t v_proj_224_; uint8_t v_zeta_225_; uint8_t v_zetaDelta_226_; uint8_t v_zetaUnused_227_; uint8_t v_zetaHave_228_; uint8_t v_canUnfoldPredicateConfig_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_345_; 
v_toConfig_207_ = lean_ctor_get(v_config_138_, 0);
lean_inc_ref(v_toConfig_207_);
lean_dec_ref(v_config_138_);
v_transparency_208_ = lean_ctor_get_uint8(v_toConfig_207_, sizeof(void*)*1);
v_offsetCnstrs_209_ = lean_ctor_get_uint8(v_toConfig_207_, sizeof(void*)*1 + 1);
v_occs_210_ = lean_ctor_get(v_toConfig_207_, 0);
lean_inc(v_occs_210_);
lean_dec_ref(v_toConfig_207_);
v___x_211_ = l_Lean_Meta_Context_config(v_a_139_);
v_foApprox_212_ = lean_ctor_get_uint8(v___x_211_, 0);
v_ctxApprox_213_ = lean_ctor_get_uint8(v___x_211_, 1);
v_quasiPatternApprox_214_ = lean_ctor_get_uint8(v___x_211_, 2);
v_constApprox_215_ = lean_ctor_get_uint8(v___x_211_, 3);
v_isDefEqStuckEx_216_ = lean_ctor_get_uint8(v___x_211_, 4);
v_unificationHints_217_ = lean_ctor_get_uint8(v___x_211_, 5);
v_proofIrrelevance_218_ = lean_ctor_get_uint8(v___x_211_, 6);
v_assignSyntheticOpaque_219_ = lean_ctor_get_uint8(v___x_211_, 7);
v_etaStruct_220_ = lean_ctor_get_uint8(v___x_211_, 10);
v_univApprox_221_ = lean_ctor_get_uint8(v___x_211_, 11);
v_iota_222_ = lean_ctor_get_uint8(v___x_211_, 12);
v_beta_223_ = lean_ctor_get_uint8(v___x_211_, 13);
v_proj_224_ = lean_ctor_get_uint8(v___x_211_, 14);
v_zeta_225_ = lean_ctor_get_uint8(v___x_211_, 15);
v_zetaDelta_226_ = lean_ctor_get_uint8(v___x_211_, 16);
v_zetaUnused_227_ = lean_ctor_get_uint8(v___x_211_, 17);
v_zetaHave_228_ = lean_ctor_get_uint8(v___x_211_, 18);
v_canUnfoldPredicateConfig_229_ = lean_ctor_get_uint8(v___x_211_, 19);
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_345_ == 0)
{
v___x_231_ = v___x_211_;
v_isShared_232_ = v_isSharedCheck_345_;
goto v_resetjp_230_;
}
else
{
lean_dec(v___x_211_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_345_;
goto v_resetjp_230_;
}
v___jp_144_:
{
lean_object* v___x_151_; uint8_t v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_151_, 0, v___y_150_);
v___x_152_ = 0;
v___x_153_ = lean_box(0);
v___x_154_ = l_Lean_Meta_mkFreshExprMVar(v___x_151_, v___x_152_, v___x_153_, v___y_148_, v___y_149_, v___y_146_, v___y_145_);
if (lean_obj_tag(v___x_154_) == 0)
{
lean_object* v_a_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v_a_155_ = lean_ctor_get(v___x_154_, 0);
lean_inc(v_a_155_);
lean_dec_ref_known(v___x_154_, 1);
v___x_156_ = l_Lean_Expr_mvarId_x21(v_a_155_);
v___x_157_ = lean_box(v_forwardImp_137_);
v___x_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
v___x_159_ = lean_unsigned_to_nat(1000000u);
v___x_160_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_gcongr___boxed), 10, 3);
lean_closure_set(v___x_160_, 0, v___x_156_);
lean_closure_set(v___x_160_, 1, v___x_158_);
lean_closure_set(v___x_160_, 2, v___x_159_);
v___x_161_ = lean_box(0);
v___x_162_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___boxed), 7, 1);
lean_closure_set(v___x_162_, 0, v_hrel_134_);
v___x_163_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__0));
v___x_164_ = lp_mathlib_Mathlib_Tactic_GCongr_GCongrM_run___redArg(v___x_160_, v___x_161_, v___x_162_, v___x_163_, v___y_148_, v___y_149_, v___y_146_, v___y_145_);
if (lean_obj_tag(v___x_164_) == 0)
{
lean_object* v_a_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_190_; 
v_a_165_ = lean_ctor_get(v___x_164_, 0);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_164_);
if (v_isSharedCheck_190_ == 0)
{
v___x_167_ = v___x_164_;
v_isShared_168_ = v_isSharedCheck_190_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_a_165_);
lean_dec(v___x_164_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_190_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v_snd_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_188_; 
v_snd_169_ = lean_ctor_get(v_a_165_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_a_165_);
if (v_isSharedCheck_188_ == 0)
{
lean_object* v_unused_189_; 
v_unused_189_ = lean_ctor_get(v_a_165_, 0);
lean_dec(v_unused_189_);
v___x_171_ = v_a_165_;
v_isShared_172_ = v_isSharedCheck_188_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_snd_169_);
lean_dec(v_a_165_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_188_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v_newGoals_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_186_; 
v_newGoals_173_ = lean_ctor_get(v_snd_169_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v_snd_169_);
if (v_isSharedCheck_186_ == 0)
{
lean_object* v_unused_187_; 
v_unused_187_ = lean_ctor_get(v_snd_169_, 1);
lean_dec(v_unused_187_);
v___x_175_ = v_snd_169_;
v_isShared_176_ = v_isSharedCheck_186_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_newGoals_173_);
lean_dec(v_snd_169_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_186_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_178_; 
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 1, v_newGoals_173_);
lean_ctor_set(v___x_171_, 0, v_a_155_);
v___x_178_ = v___x_171_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v_a_155_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_newGoals_173_);
v___x_178_ = v_reuseFailAlloc_185_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
lean_object* v___x_180_; 
if (v_isShared_176_ == 0)
{
lean_ctor_set(v___x_175_, 1, v___x_178_);
lean_ctor_set(v___x_175_, 0, v___y_147_);
v___x_180_ = v___x_175_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v___y_147_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v___x_178_);
v___x_180_ = v_reuseFailAlloc_184_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
lean_object* v___x_182_; 
if (v_isShared_168_ == 0)
{
lean_ctor_set(v___x_167_, 0, v___x_180_);
v___x_182_ = v___x_167_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v___x_180_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_198_; 
lean_dec(v_a_155_);
lean_dec_ref(v___y_147_);
v_a_191_ = lean_ctor_get(v___x_164_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_164_);
if (v_isSharedCheck_198_ == 0)
{
v___x_193_ = v___x_164_;
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_164_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_a_191_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
else
{
lean_object* v_a_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_206_; 
lean_dec_ref(v___y_147_);
lean_dec_ref(v_hrel_134_);
v_a_199_ = lean_ctor_get(v___x_154_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_206_ == 0)
{
v___x_201_ = v___x_154_;
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_a_199_);
lean_dec(v___x_154_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_204_; 
if (v_isShared_202_ == 0)
{
v___x_204_ = v___x_201_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_a_199_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
}
v_resetjp_230_:
{
uint8_t v_trackZetaDelta_233_; lean_object* v_zetaDeltaSet_234_; lean_object* v_lctx_235_; lean_object* v_localInstances_236_; lean_object* v_defEqCtx_x3f_237_; lean_object* v_synthPendingDepth_238_; lean_object* v_customCanUnfoldPredicate_x3f_239_; uint8_t v_univApprox_240_; uint8_t v_inTypeClassResolution_241_; uint8_t v_cacheInferType_242_; lean_object* v___x_244_; 
v_trackZetaDelta_233_ = lean_ctor_get_uint8(v_a_139_, sizeof(void*)*7);
v_zetaDeltaSet_234_ = lean_ctor_get(v_a_139_, 1);
v_lctx_235_ = lean_ctor_get(v_a_139_, 2);
v_localInstances_236_ = lean_ctor_get(v_a_139_, 3);
v_defEqCtx_x3f_237_ = lean_ctor_get(v_a_139_, 4);
v_synthPendingDepth_238_ = lean_ctor_get(v_a_139_, 5);
v_customCanUnfoldPredicate_x3f_239_ = lean_ctor_get(v_a_139_, 6);
v_univApprox_240_ = lean_ctor_get_uint8(v_a_139_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_241_ = lean_ctor_get_uint8(v_a_139_, sizeof(void*)*7 + 2);
v_cacheInferType_242_ = lean_ctor_get_uint8(v_a_139_, sizeof(void*)*7 + 3);
if (v_isShared_232_ == 0)
{
v___x_244_ = v___x_231_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 0, v_foApprox_212_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 1, v_ctxApprox_213_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 2, v_quasiPatternApprox_214_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 3, v_constApprox_215_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 4, v_isDefEqStuckEx_216_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 5, v_unificationHints_217_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 6, v_proofIrrelevance_218_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 7, v_assignSyntheticOpaque_219_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 10, v_etaStruct_220_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 11, v_univApprox_221_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 12, v_iota_222_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 13, v_beta_223_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 14, v_proj_224_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 15, v_zeta_225_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 16, v_zetaDelta_226_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 17, v_zetaUnused_227_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 18, v_zetaHave_228_);
lean_ctor_set_uint8(v_reuseFailAlloc_344_, 19, v_canUnfoldPredicateConfig_229_);
v___x_244_ = v_reuseFailAlloc_344_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
uint64_t v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
lean_ctor_set_uint8(v___x_244_, 8, v_offsetCnstrs_209_);
lean_ctor_set_uint8(v___x_244_, 9, v_transparency_208_);
v___x_245_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_244_);
v___x_246_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_246_, 0, v___x_244_);
lean_ctor_set_uint64(v___x_246_, sizeof(void*)*1, v___x_245_);
lean_inc(v_customCanUnfoldPredicate_x3f_239_);
lean_inc(v_synthPendingDepth_238_);
lean_inc(v_defEqCtx_x3f_237_);
lean_inc_ref(v_localInstances_236_);
lean_inc_ref(v_lctx_235_);
lean_inc(v_zetaDeltaSet_234_);
v___x_247_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set(v___x_247_, 1, v_zetaDeltaSet_234_);
lean_ctor_set(v___x_247_, 2, v_lctx_235_);
lean_ctor_set(v___x_247_, 3, v_localInstances_236_);
lean_ctor_set(v___x_247_, 4, v_defEqCtx_x3f_237_);
lean_ctor_set(v___x_247_, 5, v_synthPendingDepth_238_);
lean_ctor_set(v___x_247_, 6, v_customCanUnfoldPredicate_x3f_239_);
lean_ctor_set_uint8(v___x_247_, sizeof(void*)*7, v_trackZetaDelta_233_);
lean_ctor_set_uint8(v___x_247_, sizeof(void*)*7 + 1, v_univApprox_240_);
lean_ctor_set_uint8(v___x_247_, sizeof(void*)*7 + 2, v_inTypeClassResolution_241_);
lean_ctor_set_uint8(v___x_247_, sizeof(void*)*7 + 3, v_cacheInferType_242_);
lean_inc_ref(v_pattern_135_);
v___x_248_ = l_Lean_Meta_kabstract(v_e_133_, v_pattern_135_, v_occs_210_, v___x_247_, v_a_140_, v_a_141_, v_a_142_);
lean_dec_ref_known(v___x_247_, 7);
if (lean_obj_tag(v___x_248_) == 0)
{
lean_object* v_a_249_; lean_object* v_eNew_251_; lean_object* v___y_252_; lean_object* v___y_253_; lean_object* v___y_254_; lean_object* v___y_255_; lean_object* v___y_261_; lean_object* v___y_262_; lean_object* v___y_263_; lean_object* v___y_264_; lean_object* v___y_265_; lean_object* v___y_266_; lean_object* v___y_287_; lean_object* v___y_288_; lean_object* v___y_289_; lean_object* v___y_290_; lean_object* v___y_291_; lean_object* v___y_292_; lean_object* v___y_293_; uint8_t v___y_294_; lean_object* v___y_309_; lean_object* v___y_310_; lean_object* v___y_311_; lean_object* v___y_312_; uint8_t v___x_321_; 
v_a_249_ = lean_ctor_get(v___x_248_, 0);
lean_inc(v_a_249_);
lean_dec_ref_known(v___x_248_, 1);
v___x_321_ = l_Lean_Expr_hasLooseBVars(v_a_249_);
if (v___x_321_ == 0)
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_322_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1));
v___x_323_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__8);
lean_inc_ref(v_pattern_135_);
v___x_324_ = l_Lean_indentExpr(v_pattern_135_);
v___x_325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_323_);
lean_ctor_set(v___x_325_, 1, v___x_324_);
v___x_326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
lean_inc(v_goal_132_);
v___x_327_ = l_Lean_Meta_throwTacticEx___redArg(v___x_322_, v_goal_132_, v___x_326_, v_a_139_, v_a_140_, v_a_141_, v_a_142_);
if (lean_obj_tag(v___x_327_) == 0)
{
lean_dec_ref_known(v___x_327_, 1);
v___y_309_ = v_a_139_;
v___y_310_ = v_a_140_;
v___y_311_ = v_a_141_;
v___y_312_ = v_a_142_;
goto v___jp_308_;
}
else
{
lean_object* v_a_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_335_; 
lean_dec(v_a_249_);
lean_dec_ref(v_pattern_135_);
lean_dec_ref(v_hrel_134_);
lean_dec(v_goal_132_);
v_a_328_ = lean_ctor_get(v___x_327_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_327_);
if (v_isSharedCheck_335_ == 0)
{
v___x_330_ = v___x_327_;
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_a_328_);
lean_dec(v___x_327_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_a_328_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
else
{
v___y_309_ = v_a_139_;
v___y_310_ = v_a_140_;
v___y_311_ = v_a_141_;
v___y_312_ = v_a_142_;
goto v___jp_308_;
}
v___jp_250_:
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = lp_mathlib_Mathlib_Tactic_GCongr_mkHoleAnnotation(v_pattern_135_);
v___x_257_ = lean_expr_instantiate1(v_a_249_, v___x_256_);
lean_dec_ref(v___x_256_);
lean_dec(v_a_249_);
if (v_forwardImp_137_ == 0)
{
lean_object* v___x_258_; 
lean_inc_ref(v_eNew_251_);
v___x_258_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0(v_eNew_251_, v___x_257_);
v___y_145_ = v___y_255_;
v___y_146_ = v___y_254_;
v___y_147_ = v_eNew_251_;
v___y_148_ = v___y_252_;
v___y_149_ = v___y_253_;
v___y_150_ = v___x_258_;
goto v___jp_144_;
}
else
{
lean_object* v___x_259_; 
lean_inc_ref(v_eNew_251_);
v___x_259_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0(v___x_257_, v_eNew_251_);
v___y_145_ = v___y_255_;
v___y_146_ = v___y_254_;
v___y_147_ = v_eNew_251_;
v___y_148_ = v___y_252_;
v___y_149_ = v___y_253_;
v___y_150_ = v___x_259_;
goto v___jp_144_;
}
}
v___jp_260_:
{
if (lean_obj_tag(v___y_266_) == 0)
{
uint8_t v___x_267_; 
lean_dec_ref_known(v___y_266_, 1);
v___x_267_ = l_Lean_Expr_hasBinderNameHint(v_replacement_136_);
if (v___x_267_ == 0)
{
v_eNew_251_ = v___y_265_;
v___y_252_ = v___y_261_;
v___y_253_ = v___y_262_;
v___y_254_ = v___y_264_;
v___y_255_ = v___y_263_;
goto v___jp_250_;
}
else
{
lean_object* v___x_268_; 
v___x_268_ = l_Lean_Expr_resolveBinderNameHint(v___y_265_, v___y_264_, v___y_263_);
if (lean_obj_tag(v___x_268_) == 0)
{
lean_object* v_a_269_; 
v_a_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_a_269_);
lean_dec_ref_known(v___x_268_, 1);
v_eNew_251_ = v_a_269_;
v___y_252_ = v___y_261_;
v___y_253_ = v___y_262_;
v___y_254_ = v___y_264_;
v___y_255_ = v___y_263_;
goto v___jp_250_;
}
else
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
lean_dec(v_a_249_);
lean_dec_ref(v_pattern_135_);
lean_dec_ref(v_hrel_134_);
v_a_270_ = lean_ctor_get(v___x_268_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v___x_268_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_268_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_275_; 
if (v_isShared_273_ == 0)
{
v___x_275_ = v___x_272_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_a_270_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
}
}
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
lean_dec_ref(v___y_265_);
lean_dec(v_a_249_);
lean_dec_ref(v_pattern_135_);
lean_dec_ref(v_hrel_134_);
v_a_278_ = lean_ctor_get(v___y_266_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___y_266_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___y_266_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v___y_266_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_283_; 
if (v_isShared_281_ == 0)
{
v___x_283_ = v___x_280_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_a_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
}
v___jp_286_:
{
if (v___y_294_ == 0)
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
lean_dec_ref(v___y_290_);
v___x_295_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1));
v___x_296_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__2);
lean_inc_ref(v___y_293_);
v___x_297_ = l_Lean_MessageData_ofExpr(v___y_293_);
v___x_298_ = l_Lean_indentD(v___x_297_);
v___x_299_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_296_);
lean_ctor_set(v___x_299_, 1, v___x_298_);
v___x_300_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__4);
v___x_301_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_299_);
lean_ctor_set(v___x_301_, 1, v___x_300_);
v___x_302_ = l_Lean_Exception_toMessageData(v___y_289_);
v___x_303_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_303_, 0, v___x_301_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
v___x_304_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__6);
v___x_305_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_303_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
v___x_307_ = l_Lean_Meta_throwTacticEx___redArg(v___x_295_, v_goal_132_, v___x_306_, v___y_287_, v___y_288_, v___y_292_, v___y_291_);
v___y_261_ = v___y_287_;
v___y_262_ = v___y_288_;
v___y_263_ = v___y_291_;
v___y_264_ = v___y_292_;
v___y_265_ = v___y_293_;
v___y_266_ = v___x_307_;
goto v___jp_260_;
}
else
{
lean_dec_ref(v___y_289_);
lean_dec(v_goal_132_);
v___y_261_ = v___y_287_;
v___y_262_ = v___y_288_;
v___y_263_ = v___y_291_;
v___y_264_ = v___y_292_;
v___y_265_ = v___y_293_;
v___y_266_ = v___y_290_;
goto v___jp_260_;
}
}
v___jp_308_:
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v_a_315_; uint8_t v___x_316_; lean_object* v___x_317_; 
v___x_313_ = lean_expr_instantiate1(v_a_249_, v_replacement_136_);
v___x_314_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(v___x_313_, v___y_310_);
v_a_315_ = lean_ctor_get(v___x_314_, 0);
lean_inc_n(v_a_315_, 2);
lean_dec_ref(v___x_314_);
v___x_316_ = 0;
v___x_317_ = l_Lean_Meta_check(v_a_315_, v___x_316_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
if (lean_obj_tag(v___x_317_) == 0)
{
lean_dec(v_goal_132_);
v___y_261_ = v___y_309_;
v___y_262_ = v___y_310_;
v___y_263_ = v___y_312_;
v___y_264_ = v___y_311_;
v___y_265_ = v_a_315_;
v___y_266_ = v___x_317_;
goto v___jp_260_;
}
else
{
lean_object* v_a_318_; uint8_t v___x_319_; 
v_a_318_ = lean_ctor_get(v___x_317_, 0);
lean_inc(v_a_318_);
v___x_319_ = l_Lean_Exception_isInterrupt(v_a_318_);
if (v___x_319_ == 0)
{
uint8_t v___x_320_; 
lean_inc(v_a_318_);
v___x_320_ = l_Lean_Exception_isRuntime(v_a_318_);
v___y_287_ = v___y_309_;
v___y_288_ = v___y_310_;
v___y_289_ = v_a_318_;
v___y_290_ = v___x_317_;
v___y_291_ = v___y_312_;
v___y_292_ = v___y_311_;
v___y_293_ = v_a_315_;
v___y_294_ = v___x_320_;
goto v___jp_286_;
}
else
{
v___y_287_ = v___y_309_;
v___y_288_ = v___y_310_;
v___y_289_ = v_a_318_;
v___y_290_ = v___x_317_;
v___y_291_ = v___y_312_;
v___y_292_ = v___y_311_;
v___y_293_ = v_a_315_;
v___y_294_ = v___x_319_;
goto v___jp_286_;
}
}
}
}
else
{
lean_object* v_a_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_343_; 
lean_dec_ref(v_pattern_135_);
lean_dec_ref(v_hrel_134_);
lean_dec(v_goal_132_);
v_a_336_ = lean_ctor_get(v___x_248_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v___x_248_);
if (v_isSharedCheck_343_ == 0)
{
v___x_338_ = v___x_248_;
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_a_336_);
lean_dec(v___x_248_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_343_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v___x_341_; 
if (v_isShared_339_ == 0)
{
v___x_341_ = v___x_338_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v_a_336_);
v___x_341_ = v_reuseFailAlloc_342_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
return v___x_341_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___boxed(lean_object* v_goal_346_, lean_object* v_e_347_, lean_object* v_hrel_348_, lean_object* v_pattern_349_, lean_object* v_replacement_350_, lean_object* v_forwardImp_351_, lean_object* v_config_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_){
_start:
{
uint8_t v_forwardImp_boxed_358_; lean_object* v_res_359_; 
v_forwardImp_boxed_358_ = lean_unbox(v_forwardImp_351_);
v_res_359_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract(v_goal_346_, v_e_347_, v_hrel_348_, v_pattern_349_, v_replacement_350_, v_forwardImp_boxed_358_, v_config_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_);
lean_dec(v_a_356_);
lean_dec_ref(v_a_355_);
lean_dec(v_a_354_);
lean_dec_ref(v_a_353_);
lean_dec_ref(v_replacement_350_);
return v_res_359_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_416_ = lean_unsigned_to_nat(3959275054u);
v___x_417_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__23_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_418_ = l_Lean_Name_num___override(v___x_417_, v___x_416_);
return v___x_418_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_420_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__25_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_421_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__24_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_);
v___x_422_ = l_Lean_Name_str___override(v___x_421_, v___x_420_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___x_424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__27_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_425_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__26_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_);
v___x_426_ = l_Lean_Name_str___override(v___x_425_, v___x_424_);
return v___x_426_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_427_ = lean_unsigned_to_nat(2u);
v___x_428_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__28_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_);
v___x_429_ = l_Lean_Name_num___override(v___x_428_, v___x_427_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_431_; uint8_t v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_431_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_432_ = 0;
v___x_433_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__29_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_);
v___x_434_ = l_Lean_registerTraceClass(v___x_431_, v___x_432_, v___x_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2____boxed(lean_object* v_a_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_();
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorIdx(lean_object* v_x_437_){
_start:
{
switch(lean_obj_tag(v_x_437_))
{
case 0:
{
lean_object* v___x_438_; 
v___x_438_ = lean_unsigned_to_nat(0u);
return v___x_438_;
}
case 1:
{
lean_object* v___x_439_; 
v___x_439_ = lean_unsigned_to_nat(1u);
return v___x_439_;
}
default: 
{
lean_object* v___x_440_; 
v___x_440_ = lean_unsigned_to_nat(2u);
return v___x_440_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorIdx___boxed(lean_object* v_x_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorIdx(v_x_441_);
lean_dec(v_x_441_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(lean_object* v_t_443_, lean_object* v_k_444_){
_start:
{
if (lean_obj_tag(v_t_443_) == 2)
{
lean_object* v_lctx_445_; lean_object* v___x_446_; 
v_lctx_445_ = lean_ctor_get(v_t_443_, 0);
lean_inc_ref(v_lctx_445_);
lean_dec_ref_known(v_t_443_, 1);
v___x_446_ = lean_apply_1(v_k_444_, v_lctx_445_);
return v___x_446_;
}
else
{
lean_dec(v_t_443_);
return v_k_444_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim(lean_object* v_motive_447_, lean_object* v_ctorIdx_448_, lean_object* v_t_449_, lean_object* v_h_450_, lean_object* v_k_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_449_, v_k_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___boxed(lean_object* v_motive_453_, lean_object* v_ctorIdx_454_, lean_object* v_t_455_, lean_object* v_h_456_, lean_object* v_k_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim(v_motive_453_, v_ctorIdx_454_, v_t_455_, v_h_456_, v_k_457_);
lean_dec(v_ctorIdx_454_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_noMatch_elim___redArg(lean_object* v_t_459_, lean_object* v_noMatch_460_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_459_, v_noMatch_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_noMatch_elim(lean_object* v_motive_462_, lean_object* v_t_463_, lean_object* v_h_464_, lean_object* v_noMatch_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_463_, v_noMatch_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matched_elim___redArg(lean_object* v_t_467_, lean_object* v_matched_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_467_, v_matched_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matched_elim(lean_object* v_motive_470_, lean_object* v_t_471_, lean_object* v_h_472_, lean_object* v_matched_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_471_, v_matched_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matchedOutOfScope_elim___redArg(lean_object* v_t_475_, lean_object* v_matchedOutOfScope_476_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_475_, v_matchedOutOfScope_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_matchedOutOfScope_elim(lean_object* v_motive_478_, lean_object* v_t_479_, lean_object* v_h_480_, lean_object* v_matchedOutOfScope_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_Progress_ctorElim___redArg(v_t_479_, v_matchedOutOfScope_481_);
return v___x_482_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_483_ = lean_unsigned_to_nat(32u);
v___x_484_ = lean_mk_empty_array_with_capacity(v___x_483_);
v___x_485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_485_, 0, v___x_484_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1(void){
_start:
{
size_t v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_486_ = ((size_t)5ULL);
v___x_487_ = lean_unsigned_to_nat(0u);
v___x_488_ = lean_unsigned_to_nat(32u);
v___x_489_ = lean_mk_empty_array_with_capacity(v___x_488_);
v___x_490_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__0);
v___x_491_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_491_, 0, v___x_490_);
lean_ctor_set(v___x_491_, 1, v___x_489_);
lean_ctor_set(v___x_491_, 2, v___x_487_);
lean_ctor_set(v___x_491_, 3, v___x_487_);
lean_ctor_set_usize(v___x_491_, 4, v___x_486_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg(lean_object* v___y_492_){
_start:
{
lean_object* v___x_494_; lean_object* v_traceState_495_; lean_object* v_traces_496_; lean_object* v___x_497_; lean_object* v_traceState_498_; lean_object* v_env_499_; lean_object* v_nextMacroScope_500_; lean_object* v_ngen_501_; lean_object* v_auxDeclNGen_502_; lean_object* v_cache_503_; lean_object* v_messages_504_; lean_object* v_infoState_505_; lean_object* v_snapshotTasks_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_525_; 
v___x_494_ = lean_st_ref_get(v___y_492_);
v_traceState_495_ = lean_ctor_get(v___x_494_, 4);
lean_inc_ref(v_traceState_495_);
lean_dec(v___x_494_);
v_traces_496_ = lean_ctor_get(v_traceState_495_, 0);
lean_inc_ref(v_traces_496_);
lean_dec_ref(v_traceState_495_);
v___x_497_ = lean_st_ref_take(v___y_492_);
v_traceState_498_ = lean_ctor_get(v___x_497_, 4);
v_env_499_ = lean_ctor_get(v___x_497_, 0);
v_nextMacroScope_500_ = lean_ctor_get(v___x_497_, 1);
v_ngen_501_ = lean_ctor_get(v___x_497_, 2);
v_auxDeclNGen_502_ = lean_ctor_get(v___x_497_, 3);
v_cache_503_ = lean_ctor_get(v___x_497_, 5);
v_messages_504_ = lean_ctor_get(v___x_497_, 6);
v_infoState_505_ = lean_ctor_get(v___x_497_, 7);
v_snapshotTasks_506_ = lean_ctor_get(v___x_497_, 8);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_497_);
if (v_isSharedCheck_525_ == 0)
{
v___x_508_ = v___x_497_;
v_isShared_509_ = v_isSharedCheck_525_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_snapshotTasks_506_);
lean_inc(v_infoState_505_);
lean_inc(v_messages_504_);
lean_inc(v_cache_503_);
lean_inc(v_traceState_498_);
lean_inc(v_auxDeclNGen_502_);
lean_inc(v_ngen_501_);
lean_inc(v_nextMacroScope_500_);
lean_inc(v_env_499_);
lean_dec(v___x_497_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_525_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
uint64_t v_tid_510_; lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_523_; 
v_tid_510_ = lean_ctor_get_uint64(v_traceState_498_, sizeof(void*)*1);
v_isSharedCheck_523_ = !lean_is_exclusive(v_traceState_498_);
if (v_isSharedCheck_523_ == 0)
{
lean_object* v_unused_524_; 
v_unused_524_ = lean_ctor_get(v_traceState_498_, 0);
lean_dec(v_unused_524_);
v___x_512_ = v_traceState_498_;
v_isShared_513_ = v_isSharedCheck_523_;
goto v_resetjp_511_;
}
else
{
lean_dec(v_traceState_498_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_523_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_514_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1);
if (v_isShared_513_ == 0)
{
lean_ctor_set(v___x_512_, 0, v___x_514_);
v___x_516_ = v___x_512_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_514_);
lean_ctor_set_uint64(v_reuseFailAlloc_522_, sizeof(void*)*1, v_tid_510_);
v___x_516_ = v_reuseFailAlloc_522_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
lean_object* v___x_518_; 
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 4, v___x_516_);
v___x_518_ = v___x_508_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v_env_499_);
lean_ctor_set(v_reuseFailAlloc_521_, 1, v_nextMacroScope_500_);
lean_ctor_set(v_reuseFailAlloc_521_, 2, v_ngen_501_);
lean_ctor_set(v_reuseFailAlloc_521_, 3, v_auxDeclNGen_502_);
lean_ctor_set(v_reuseFailAlloc_521_, 4, v___x_516_);
lean_ctor_set(v_reuseFailAlloc_521_, 5, v_cache_503_);
lean_ctor_set(v_reuseFailAlloc_521_, 6, v_messages_504_);
lean_ctor_set(v_reuseFailAlloc_521_, 7, v_infoState_505_);
lean_ctor_set(v_reuseFailAlloc_521_, 8, v_snapshotTasks_506_);
v___x_518_ = v_reuseFailAlloc_521_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = lean_st_ref_set(v___y_492_, v___x_518_);
v___x_520_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_520_, 0, v_traces_496_);
return v___x_520_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___boxed(lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg(v___y_526_);
lean_dec(v___y_526_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2(lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg(v___y_532_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___boxed(lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2(v___y_535_, v___y_536_, v___y_537_, v___y_538_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
lean_dec(v___y_536_);
lean_dec_ref(v___y_535_);
return v_res_540_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(lean_object* v_opts_541_, lean_object* v_opt_542_){
_start:
{
lean_object* v_name_543_; lean_object* v_defValue_544_; lean_object* v_map_545_; lean_object* v___x_546_; 
v_name_543_ = lean_ctor_get(v_opt_542_, 0);
v_defValue_544_ = lean_ctor_get(v_opt_542_, 1);
v_map_545_ = lean_ctor_get(v_opts_541_, 0);
v___x_546_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_545_, v_name_543_);
if (lean_obj_tag(v___x_546_) == 0)
{
uint8_t v___x_547_; 
v___x_547_ = lean_unbox(v_defValue_544_);
return v___x_547_;
}
else
{
lean_object* v_val_548_; 
v_val_548_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_val_548_);
lean_dec_ref_known(v___x_546_, 1);
if (lean_obj_tag(v_val_548_) == 1)
{
uint8_t v_v_549_; 
v_v_549_ = lean_ctor_get_uint8(v_val_548_, 0);
lean_dec_ref_known(v_val_548_, 0);
return v_v_549_;
}
else
{
uint8_t v___x_550_; 
lean_dec(v_val_548_);
v___x_550_ = lean_unbox(v_defValue_544_);
return v___x_550_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3___boxed(lean_object* v_opts_551_, lean_object* v_opt_552_){
_start:
{
uint8_t v_res_553_; lean_object* v_r_554_; 
v_res_553_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_551_, v_opt_552_);
lean_dec_ref(v_opt_552_);
lean_dec_ref(v_opts_551_);
v_r_554_ = lean_box(v_res_553_);
return v_r_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12___redArg(lean_object* v_x_555_, lean_object* v_x_556_, lean_object* v_x_557_, lean_object* v_x_558_){
_start:
{
lean_object* v_ks_559_; lean_object* v_vs_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_584_; 
v_ks_559_ = lean_ctor_get(v_x_555_, 0);
v_vs_560_ = lean_ctor_get(v_x_555_, 1);
v_isSharedCheck_584_ = !lean_is_exclusive(v_x_555_);
if (v_isSharedCheck_584_ == 0)
{
v___x_562_ = v_x_555_;
v_isShared_563_ = v_isSharedCheck_584_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_vs_560_);
lean_inc(v_ks_559_);
lean_dec(v_x_555_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_584_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v___x_564_; uint8_t v___x_565_; 
v___x_564_ = lean_array_get_size(v_ks_559_);
v___x_565_ = lean_nat_dec_lt(v_x_556_, v___x_564_);
if (v___x_565_ == 0)
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_569_; 
lean_dec(v_x_556_);
v___x_566_ = lean_array_push(v_ks_559_, v_x_557_);
v___x_567_ = lean_array_push(v_vs_560_, v_x_558_);
if (v_isShared_563_ == 0)
{
lean_ctor_set(v___x_562_, 1, v___x_567_);
lean_ctor_set(v___x_562_, 0, v___x_566_);
v___x_569_ = v___x_562_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
else
{
lean_object* v_k_x27_571_; uint8_t v___x_572_; 
v_k_x27_571_ = lean_array_fget_borrowed(v_ks_559_, v_x_556_);
v___x_572_ = l_Lean_instBEqMVarId_beq(v_x_557_, v_k_x27_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_574_; 
if (v_isShared_563_ == 0)
{
v___x_574_ = v___x_562_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_ks_559_);
lean_ctor_set(v_reuseFailAlloc_578_, 1, v_vs_560_);
v___x_574_ = v_reuseFailAlloc_578_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_575_ = lean_unsigned_to_nat(1u);
v___x_576_ = lean_nat_add(v_x_556_, v___x_575_);
lean_dec(v_x_556_);
v_x_555_ = v___x_574_;
v_x_556_ = v___x_576_;
goto _start;
}
}
else
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_582_; 
v___x_579_ = lean_array_fset(v_ks_559_, v_x_556_, v_x_557_);
v___x_580_ = lean_array_fset(v_vs_560_, v_x_556_, v_x_558_);
lean_dec(v_x_556_);
if (v_isShared_563_ == 0)
{
lean_ctor_set(v___x_562_, 1, v___x_580_);
lean_ctor_set(v___x_562_, 0, v___x_579_);
v___x_582_ = v___x_562_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v___x_579_);
lean_ctor_set(v_reuseFailAlloc_583_, 1, v___x_580_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9___redArg(lean_object* v_n_585_, lean_object* v_k_586_, lean_object* v_v_587_){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_588_ = lean_unsigned_to_nat(0u);
v___x_589_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12___redArg(v_n_585_, v___x_588_, v_k_586_, v_v_587_);
return v___x_589_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_590_; 
v___x_590_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(lean_object* v_x_591_, size_t v_x_592_, size_t v_x_593_, lean_object* v_x_594_, lean_object* v_x_595_){
_start:
{
if (lean_obj_tag(v_x_591_) == 0)
{
lean_object* v_es_596_; size_t v___x_597_; size_t v___x_598_; lean_object* v_j_599_; lean_object* v___x_600_; uint8_t v___x_601_; 
v_es_596_ = lean_ctor_get(v_x_591_, 0);
v___x_597_ = ((size_t)31ULL);
v___x_598_ = lean_usize_land(v_x_592_, v___x_597_);
v_j_599_ = lean_usize_to_nat(v___x_598_);
v___x_600_ = lean_array_get_size(v_es_596_);
v___x_601_ = lean_nat_dec_lt(v_j_599_, v___x_600_);
if (v___x_601_ == 0)
{
lean_dec(v_j_599_);
lean_dec(v_x_595_);
lean_dec(v_x_594_);
return v_x_591_;
}
else
{
lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_640_; 
lean_inc_ref(v_es_596_);
v_isSharedCheck_640_ = !lean_is_exclusive(v_x_591_);
if (v_isSharedCheck_640_ == 0)
{
lean_object* v_unused_641_; 
v_unused_641_ = lean_ctor_get(v_x_591_, 0);
lean_dec(v_unused_641_);
v___x_603_ = v_x_591_;
v_isShared_604_ = v_isSharedCheck_640_;
goto v_resetjp_602_;
}
else
{
lean_dec(v_x_591_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_640_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v_v_605_; lean_object* v___x_606_; lean_object* v_xs_x27_607_; lean_object* v___y_609_; 
v_v_605_ = lean_array_fget(v_es_596_, v_j_599_);
v___x_606_ = lean_box(0);
v_xs_x27_607_ = lean_array_fset(v_es_596_, v_j_599_, v___x_606_);
switch(lean_obj_tag(v_v_605_))
{
case 0:
{
lean_object* v_key_614_; lean_object* v_val_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_625_; 
v_key_614_ = lean_ctor_get(v_v_605_, 0);
v_val_615_ = lean_ctor_get(v_v_605_, 1);
v_isSharedCheck_625_ = !lean_is_exclusive(v_v_605_);
if (v_isSharedCheck_625_ == 0)
{
v___x_617_ = v_v_605_;
v_isShared_618_ = v_isSharedCheck_625_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_val_615_);
lean_inc(v_key_614_);
lean_dec(v_v_605_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_625_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
uint8_t v___x_619_; 
v___x_619_ = l_Lean_instBEqMVarId_beq(v_x_594_, v_key_614_);
if (v___x_619_ == 0)
{
lean_object* v___x_620_; lean_object* v___x_621_; 
lean_del_object(v___x_617_);
v___x_620_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_614_, v_val_615_, v_x_594_, v_x_595_);
v___x_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_621_, 0, v___x_620_);
v___y_609_ = v___x_621_;
goto v___jp_608_;
}
else
{
lean_object* v___x_623_; 
lean_dec(v_val_615_);
lean_dec(v_key_614_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 1, v_x_595_);
lean_ctor_set(v___x_617_, 0, v_x_594_);
v___x_623_ = v___x_617_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_x_594_);
lean_ctor_set(v_reuseFailAlloc_624_, 1, v_x_595_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
v___y_609_ = v___x_623_;
goto v___jp_608_;
}
}
}
}
case 1:
{
lean_object* v_node_626_; lean_object* v___x_628_; uint8_t v_isShared_629_; uint8_t v_isSharedCheck_638_; 
v_node_626_ = lean_ctor_get(v_v_605_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v_v_605_);
if (v_isSharedCheck_638_ == 0)
{
v___x_628_ = v_v_605_;
v_isShared_629_ = v_isSharedCheck_638_;
goto v_resetjp_627_;
}
else
{
lean_inc(v_node_626_);
lean_dec(v_v_605_);
v___x_628_ = lean_box(0);
v_isShared_629_ = v_isSharedCheck_638_;
goto v_resetjp_627_;
}
v_resetjp_627_:
{
size_t v___x_630_; size_t v___x_631_; size_t v___x_632_; size_t v___x_633_; lean_object* v___x_634_; lean_object* v___x_636_; 
v___x_630_ = ((size_t)5ULL);
v___x_631_ = lean_usize_shift_right(v_x_592_, v___x_630_);
v___x_632_ = ((size_t)1ULL);
v___x_633_ = lean_usize_add(v_x_593_, v___x_632_);
v___x_634_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(v_node_626_, v___x_631_, v___x_633_, v_x_594_, v_x_595_);
if (v_isShared_629_ == 0)
{
lean_ctor_set(v___x_628_, 0, v___x_634_);
v___x_636_ = v___x_628_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v___x_634_);
v___x_636_ = v_reuseFailAlloc_637_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
v___y_609_ = v___x_636_;
goto v___jp_608_;
}
}
}
default: 
{
lean_object* v___x_639_; 
v___x_639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_639_, 0, v_x_594_);
lean_ctor_set(v___x_639_, 1, v_x_595_);
v___y_609_ = v___x_639_;
goto v___jp_608_;
}
}
v___jp_608_:
{
lean_object* v___x_610_; lean_object* v___x_612_; 
v___x_610_ = lean_array_fset(v_xs_x27_607_, v_j_599_, v___y_609_);
lean_dec(v_j_599_);
if (v_isShared_604_ == 0)
{
lean_ctor_set(v___x_603_, 0, v___x_610_);
v___x_612_ = v___x_603_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_610_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
}
else
{
lean_object* v_ks_642_; lean_object* v_vs_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_663_; 
v_ks_642_ = lean_ctor_get(v_x_591_, 0);
v_vs_643_ = lean_ctor_get(v_x_591_, 1);
v_isSharedCheck_663_ = !lean_is_exclusive(v_x_591_);
if (v_isSharedCheck_663_ == 0)
{
v___x_645_ = v_x_591_;
v_isShared_646_ = v_isSharedCheck_663_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_vs_643_);
lean_inc(v_ks_642_);
lean_dec(v_x_591_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_663_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_648_; 
if (v_isShared_646_ == 0)
{
v___x_648_ = v___x_645_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_ks_642_);
lean_ctor_set(v_reuseFailAlloc_662_, 1, v_vs_643_);
v___x_648_ = v_reuseFailAlloc_662_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
lean_object* v_newNode_649_; uint8_t v___y_651_; size_t v___x_657_; uint8_t v___x_658_; 
v_newNode_649_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9___redArg(v___x_648_, v_x_594_, v_x_595_);
v___x_657_ = ((size_t)7ULL);
v___x_658_ = lean_usize_dec_le(v___x_657_, v_x_593_);
if (v___x_658_ == 0)
{
lean_object* v___x_659_; lean_object* v___x_660_; uint8_t v___x_661_; 
v___x_659_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_649_);
v___x_660_ = lean_unsigned_to_nat(4u);
v___x_661_ = lean_nat_dec_lt(v___x_659_, v___x_660_);
lean_dec(v___x_659_);
v___y_651_ = v___x_661_;
goto v___jp_650_;
}
else
{
v___y_651_ = v___x_658_;
goto v___jp_650_;
}
v___jp_650_:
{
if (v___y_651_ == 0)
{
lean_object* v_ks_652_; lean_object* v_vs_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v_ks_652_ = lean_ctor_get(v_newNode_649_, 0);
lean_inc_ref(v_ks_652_);
v_vs_653_ = lean_ctor_get(v_newNode_649_, 1);
lean_inc_ref(v_vs_653_);
lean_dec_ref(v_newNode_649_);
v___x_654_ = lean_unsigned_to_nat(0u);
v___x_655_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___closed__0);
v___x_656_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg(v_x_593_, v_ks_652_, v_vs_653_, v___x_654_, v___x_655_);
lean_dec_ref(v_vs_653_);
lean_dec_ref(v_ks_652_);
return v___x_656_;
}
else
{
return v_newNode_649_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg(size_t v_depth_664_, lean_object* v_keys_665_, lean_object* v_vals_666_, lean_object* v_i_667_, lean_object* v_entries_668_){
_start:
{
lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = lean_array_get_size(v_keys_665_);
v___x_670_ = lean_nat_dec_lt(v_i_667_, v___x_669_);
if (v___x_670_ == 0)
{
lean_dec(v_i_667_);
return v_entries_668_;
}
else
{
lean_object* v_k_671_; lean_object* v_v_672_; uint64_t v___x_673_; size_t v_h_674_; size_t v___x_675_; lean_object* v___x_676_; size_t v___x_677_; size_t v___x_678_; size_t v___x_679_; size_t v_h_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v_k_671_ = lean_array_fget_borrowed(v_keys_665_, v_i_667_);
v_v_672_ = lean_array_fget_borrowed(v_vals_666_, v_i_667_);
v___x_673_ = l_Lean_instHashableMVarId_hash(v_k_671_);
v_h_674_ = lean_uint64_to_usize(v___x_673_);
v___x_675_ = ((size_t)5ULL);
v___x_676_ = lean_unsigned_to_nat(1u);
v___x_677_ = ((size_t)1ULL);
v___x_678_ = lean_usize_sub(v_depth_664_, v___x_677_);
v___x_679_ = lean_usize_mul(v___x_675_, v___x_678_);
v_h_680_ = lean_usize_shift_right(v_h_674_, v___x_679_);
v___x_681_ = lean_nat_add(v_i_667_, v___x_676_);
lean_dec(v_i_667_);
lean_inc(v_v_672_);
lean_inc(v_k_671_);
v___x_682_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(v_entries_668_, v_h_680_, v_depth_664_, v_k_671_, v_v_672_);
v_i_667_ = v___x_681_;
v_entries_668_ = v___x_682_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg___boxed(lean_object* v_depth_684_, lean_object* v_keys_685_, lean_object* v_vals_686_, lean_object* v_i_687_, lean_object* v_entries_688_){
_start:
{
size_t v_depth_boxed_689_; lean_object* v_res_690_; 
v_depth_boxed_689_ = lean_unbox_usize(v_depth_684_);
lean_dec(v_depth_684_);
v_res_690_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg(v_depth_boxed_689_, v_keys_685_, v_vals_686_, v_i_687_, v_entries_688_);
lean_dec_ref(v_vals_686_);
lean_dec_ref(v_keys_685_);
return v_res_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_x_691_, lean_object* v_x_692_, lean_object* v_x_693_, lean_object* v_x_694_, lean_object* v_x_695_){
_start:
{
size_t v_x_14739__boxed_696_; size_t v_x_14740__boxed_697_; lean_object* v_res_698_; 
v_x_14739__boxed_696_ = lean_unbox_usize(v_x_692_);
lean_dec(v_x_692_);
v_x_14740__boxed_697_ = lean_unbox_usize(v_x_693_);
lean_dec(v_x_693_);
v_res_698_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(v_x_691_, v_x_14739__boxed_696_, v_x_14740__boxed_697_, v_x_694_, v_x_695_);
return v_res_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(lean_object* v_x_699_, lean_object* v_x_700_, lean_object* v_x_701_){
_start:
{
uint64_t v___x_702_; size_t v___x_703_; size_t v___x_704_; lean_object* v___x_705_; 
v___x_702_ = l_Lean_instHashableMVarId_hash(v_x_700_);
v___x_703_ = lean_uint64_to_usize(v___x_702_);
v___x_704_ = ((size_t)1ULL);
v___x_705_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(v_x_699_, v___x_703_, v___x_704_, v_x_700_, v_x_701_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg(lean_object* v_mvarId_706_, lean_object* v_val_707_, lean_object* v___y_708_){
_start:
{
lean_object* v___x_710_; lean_object* v_mctx_711_; lean_object* v_cache_712_; lean_object* v_zetaDeltaFVarIds_713_; lean_object* v_postponed_714_; lean_object* v_diag_715_; lean_object* v___x_717_; uint8_t v_isShared_718_; uint8_t v_isSharedCheck_743_; 
v___x_710_ = lean_st_ref_take(v___y_708_);
v_mctx_711_ = lean_ctor_get(v___x_710_, 0);
v_cache_712_ = lean_ctor_get(v___x_710_, 1);
v_zetaDeltaFVarIds_713_ = lean_ctor_get(v___x_710_, 2);
v_postponed_714_ = lean_ctor_get(v___x_710_, 3);
v_diag_715_ = lean_ctor_get(v___x_710_, 4);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_710_);
if (v_isSharedCheck_743_ == 0)
{
v___x_717_ = v___x_710_;
v_isShared_718_ = v_isSharedCheck_743_;
goto v_resetjp_716_;
}
else
{
lean_inc(v_diag_715_);
lean_inc(v_postponed_714_);
lean_inc(v_zetaDeltaFVarIds_713_);
lean_inc(v_cache_712_);
lean_inc(v_mctx_711_);
lean_dec(v___x_710_);
v___x_717_ = lean_box(0);
v_isShared_718_ = v_isSharedCheck_743_;
goto v_resetjp_716_;
}
v_resetjp_716_:
{
lean_object* v_depth_719_; lean_object* v_levelAssignDepth_720_; lean_object* v_lmvarCounter_721_; lean_object* v_mvarCounter_722_; lean_object* v_lDecls_723_; lean_object* v_decls_724_; lean_object* v_userNames_725_; lean_object* v_lAssignment_726_; lean_object* v_eAssignment_727_; lean_object* v_dAssignment_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_742_; 
v_depth_719_ = lean_ctor_get(v_mctx_711_, 0);
v_levelAssignDepth_720_ = lean_ctor_get(v_mctx_711_, 1);
v_lmvarCounter_721_ = lean_ctor_get(v_mctx_711_, 2);
v_mvarCounter_722_ = lean_ctor_get(v_mctx_711_, 3);
v_lDecls_723_ = lean_ctor_get(v_mctx_711_, 4);
v_decls_724_ = lean_ctor_get(v_mctx_711_, 5);
v_userNames_725_ = lean_ctor_get(v_mctx_711_, 6);
v_lAssignment_726_ = lean_ctor_get(v_mctx_711_, 7);
v_eAssignment_727_ = lean_ctor_get(v_mctx_711_, 8);
v_dAssignment_728_ = lean_ctor_get(v_mctx_711_, 9);
v_isSharedCheck_742_ = !lean_is_exclusive(v_mctx_711_);
if (v_isSharedCheck_742_ == 0)
{
v___x_730_ = v_mctx_711_;
v_isShared_731_ = v_isSharedCheck_742_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_dAssignment_728_);
lean_inc(v_eAssignment_727_);
lean_inc(v_lAssignment_726_);
lean_inc(v_userNames_725_);
lean_inc(v_decls_724_);
lean_inc(v_lDecls_723_);
lean_inc(v_mvarCounter_722_);
lean_inc(v_lmvarCounter_721_);
lean_inc(v_levelAssignDepth_720_);
lean_inc(v_depth_719_);
lean_dec(v_mctx_711_);
v___x_730_ = lean_box(0);
v_isShared_731_ = v_isSharedCheck_742_;
goto v_resetjp_729_;
}
v_resetjp_729_:
{
lean_object* v___x_732_; lean_object* v___x_734_; 
v___x_732_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(v_eAssignment_727_, v_mvarId_706_, v_val_707_);
if (v_isShared_731_ == 0)
{
lean_ctor_set(v___x_730_, 8, v___x_732_);
v___x_734_ = v___x_730_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v_depth_719_);
lean_ctor_set(v_reuseFailAlloc_741_, 1, v_levelAssignDepth_720_);
lean_ctor_set(v_reuseFailAlloc_741_, 2, v_lmvarCounter_721_);
lean_ctor_set(v_reuseFailAlloc_741_, 3, v_mvarCounter_722_);
lean_ctor_set(v_reuseFailAlloc_741_, 4, v_lDecls_723_);
lean_ctor_set(v_reuseFailAlloc_741_, 5, v_decls_724_);
lean_ctor_set(v_reuseFailAlloc_741_, 6, v_userNames_725_);
lean_ctor_set(v_reuseFailAlloc_741_, 7, v_lAssignment_726_);
lean_ctor_set(v_reuseFailAlloc_741_, 8, v___x_732_);
lean_ctor_set(v_reuseFailAlloc_741_, 9, v_dAssignment_728_);
v___x_734_ = v_reuseFailAlloc_741_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
lean_object* v___x_736_; 
if (v_isShared_718_ == 0)
{
lean_ctor_set(v___x_717_, 0, v___x_734_);
v___x_736_ = v___x_717_;
goto v_reusejp_735_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v___x_734_);
lean_ctor_set(v_reuseFailAlloc_740_, 1, v_cache_712_);
lean_ctor_set(v_reuseFailAlloc_740_, 2, v_zetaDeltaFVarIds_713_);
lean_ctor_set(v_reuseFailAlloc_740_, 3, v_postponed_714_);
lean_ctor_set(v_reuseFailAlloc_740_, 4, v_diag_715_);
v___x_736_ = v_reuseFailAlloc_740_;
goto v_reusejp_735_;
}
v_reusejp_735_:
{
lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_737_ = lean_st_ref_set(v___y_708_, v___x_736_);
v___x_738_ = lean_box(0);
v___x_739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
return v___x_739_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg___boxed(lean_object* v_mvarId_744_, lean_object* v_val_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v_res_748_; 
v_res_748_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg(v_mvarId_744_, v_val_745_, v___y_746_);
lean_dec(v___y_746_);
return v_res_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg(lean_object* v_snd_761_, lean_object* v_goal_762_, lean_object* v___x_763_, lean_object* v_as_x27_764_, lean_object* v_b_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_){
_start:
{
if (lean_obj_tag(v_as_x27_764_) == 0)
{
lean_object* v___x_771_; 
lean_dec_ref(v___x_763_);
lean_dec(v_goal_762_);
lean_dec_ref(v_snd_761_);
v___x_771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_771_, 0, v_b_765_);
return v___x_771_;
}
else
{
lean_object* v_head_772_; lean_object* v_tail_773_; lean_object* v_fst_774_; lean_object* v_snd_775_; lean_object* v___x_776_; lean_object* v___y_778_; uint8_t v___y_779_; 
lean_dec_ref(v_b_765_);
v_head_772_ = lean_ctor_get(v_as_x27_764_, 0);
v_tail_773_ = lean_ctor_get(v_as_x27_764_, 1);
v_fst_774_ = lean_ctor_get(v_head_772_, 0);
v_snd_775_ = lean_ctor_get(v_head_772_, 1);
v___x_776_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__0));
if (lean_obj_tag(v_fst_774_) == 1)
{
lean_object* v_pre_810_; 
v_pre_810_ = lean_ctor_get(v_fst_774_, 0);
if (lean_obj_tag(v_pre_810_) == 1)
{
lean_object* v_pre_811_; 
v_pre_811_ = lean_ctor_get(v_pre_810_, 0);
if (lean_obj_tag(v_pre_811_) == 1)
{
lean_object* v_pre_812_; 
v_pre_812_ = lean_ctor_get(v_pre_811_, 0);
if (lean_obj_tag(v_pre_812_) == 1)
{
lean_object* v_pre_813_; 
v_pre_813_ = lean_ctor_get(v_pre_812_, 0);
if (lean_obj_tag(v_pre_813_) == 0)
{
lean_object* v_str_814_; lean_object* v_str_815_; lean_object* v_str_816_; lean_object* v_str_817_; lean_object* v___x_818_; uint8_t v___x_819_; 
v_str_814_ = lean_ctor_get(v_fst_774_, 1);
v_str_815_ = lean_ctor_get(v_pre_810_, 1);
v_str_816_ = lean_ctor_get(v_pre_811_, 1);
v_str_817_ = lean_ctor_get(v_pre_812_, 1);
v___x_818_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__4_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_819_ = lean_string_dec_eq(v_str_817_, v___x_818_);
if (v___x_819_ == 0)
{
goto v___jp_796_;
}
else
{
lean_object* v___x_820_; uint8_t v___x_821_; 
v___x_820_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__6_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_821_ = lean_string_dec_eq(v_str_816_, v___x_820_);
if (v___x_821_ == 0)
{
goto v___jp_796_;
}
else
{
lean_object* v___x_822_; uint8_t v___x_823_; 
v___x_822_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__3));
v___x_823_ = lean_string_dec_eq(v_str_815_, v___x_822_);
if (v___x_823_ == 0)
{
goto v___jp_796_;
}
else
{
lean_object* v___x_824_; uint8_t v___x_825_; 
v___x_824_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__4));
v___x_825_ = lean_string_dec_eq(v_str_814_, v___x_824_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; uint8_t v___x_827_; 
v___x_826_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__5));
v___x_827_ = lean_string_dec_eq(v_str_814_, v___x_826_);
if (v___x_827_ == 0)
{
goto v___jp_796_;
}
else
{
v_as_x27_764_ = v_tail_773_;
v_b_765_ = v___x_776_;
goto _start;
}
}
else
{
v_as_x27_764_ = v_tail_773_;
v_b_765_ = v___x_776_;
goto _start;
}
}
}
}
}
else
{
goto v___jp_796_;
}
}
else
{
goto v___jp_796_;
}
}
else
{
goto v___jp_796_;
}
}
else
{
goto v___jp_796_;
}
}
else
{
goto v___jp_796_;
}
v___jp_777_:
{
if (v___y_779_ == 0)
{
lean_object* v___x_780_; lean_object* v_cache_781_; lean_object* v_zetaDeltaFVarIds_782_; lean_object* v_postponed_783_; lean_object* v_diag_784_; lean_object* v___x_786_; uint8_t v_isShared_787_; uint8_t v_isSharedCheck_793_; 
lean_dec_ref(v___y_778_);
v___x_780_ = lean_st_ref_take(v___y_767_);
v_cache_781_ = lean_ctor_get(v___x_780_, 1);
v_zetaDeltaFVarIds_782_ = lean_ctor_get(v___x_780_, 2);
v_postponed_783_ = lean_ctor_get(v___x_780_, 3);
v_diag_784_ = lean_ctor_get(v___x_780_, 4);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_780_);
if (v_isSharedCheck_793_ == 0)
{
lean_object* v_unused_794_; 
v_unused_794_ = lean_ctor_get(v___x_780_, 0);
lean_dec(v_unused_794_);
v___x_786_ = v___x_780_;
v_isShared_787_ = v_isSharedCheck_793_;
goto v_resetjp_785_;
}
else
{
lean_inc(v_diag_784_);
lean_inc(v_postponed_783_);
lean_inc(v_zetaDeltaFVarIds_782_);
lean_inc(v_cache_781_);
lean_dec(v___x_780_);
v___x_786_ = lean_box(0);
v_isShared_787_ = v_isSharedCheck_793_;
goto v_resetjp_785_;
}
v_resetjp_785_:
{
lean_object* v___x_789_; 
lean_inc_ref(v___x_763_);
if (v_isShared_787_ == 0)
{
lean_ctor_set(v___x_786_, 0, v___x_763_);
v___x_789_ = v___x_786_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v___x_763_);
lean_ctor_set(v_reuseFailAlloc_792_, 1, v_cache_781_);
lean_ctor_set(v_reuseFailAlloc_792_, 2, v_zetaDeltaFVarIds_782_);
lean_ctor_set(v_reuseFailAlloc_792_, 3, v_postponed_783_);
lean_ctor_set(v_reuseFailAlloc_792_, 4, v_diag_784_);
v___x_789_ = v_reuseFailAlloc_792_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
lean_object* v___x_790_; 
v___x_790_ = lean_st_ref_set(v___y_767_, v___x_789_);
v_as_x27_764_ = v_tail_773_;
v_b_765_ = v___x_776_;
goto _start;
}
}
}
else
{
lean_object* v___x_795_; 
lean_dec_ref(v___x_763_);
lean_dec(v_goal_762_);
lean_dec_ref(v_snd_761_);
v___x_795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_795_, 0, v___y_778_);
return v___x_795_;
}
}
v___jp_796_:
{
lean_object* v___x_797_; 
lean_inc(v_snd_775_);
lean_inc(v___y_769_);
lean_inc_ref(v___y_768_);
lean_inc(v___y_767_);
lean_inc_ref(v___y_766_);
lean_inc(v_goal_762_);
lean_inc_ref(v_snd_761_);
v___x_797_ = lean_apply_7(v_snd_775_, v_snd_761_, v_goal_762_, v___y_766_, v___y_767_, v___y_768_, v___y_769_, lean_box(0));
if (lean_obj_tag(v___x_797_) == 0)
{
lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_805_; 
lean_dec_ref(v___x_763_);
lean_dec(v_goal_762_);
lean_dec_ref(v_snd_761_);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_805_ == 0)
{
lean_object* v_unused_806_; 
v_unused_806_ = lean_ctor_get(v___x_797_, 0);
lean_dec(v_unused_806_);
v___x_799_ = v___x_797_;
v_isShared_800_ = v_isSharedCheck_805_;
goto v_resetjp_798_;
}
else
{
lean_dec(v___x_797_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_805_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_801_; lean_object* v___x_803_; 
v___x_801_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__2));
if (v_isShared_800_ == 0)
{
lean_ctor_set(v___x_799_, 0, v___x_801_);
v___x_803_ = v___x_799_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v___x_801_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
else
{
lean_object* v_a_807_; uint8_t v___x_808_; 
v_a_807_ = lean_ctor_get(v___x_797_, 0);
lean_inc(v_a_807_);
lean_dec_ref_known(v___x_797_, 1);
v___x_808_ = l_Lean_Exception_isInterrupt(v_a_807_);
if (v___x_808_ == 0)
{
uint8_t v___x_809_; 
lean_inc(v_a_807_);
v___x_809_ = l_Lean_Exception_isRuntime(v_a_807_);
v___y_778_ = v_a_807_;
v___y_779_ = v___x_809_;
goto v___jp_777_;
}
else
{
v___y_778_ = v_a_807_;
v___y_779_ = v___x_808_;
goto v___jp_777_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___boxed(lean_object* v_snd_830_, lean_object* v_goal_831_, lean_object* v___x_832_, lean_object* v_as_x27_833_, lean_object* v_b_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg(v_snd_830_, v_goal_831_, v___x_832_, v_as_x27_833_, v_b_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
lean_dec(v_as_x27_833_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0(lean_object* v_config_843_, lean_object* v_goal_844_, lean_object* v_____x_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_){
_start:
{
lean_object* v_fst_851_; lean_object* v_snd_852_; lean_object* v___x_853_; lean_object* v_toConfig_854_; uint8_t v_foApprox_855_; uint8_t v_ctxApprox_856_; uint8_t v_quasiPatternApprox_857_; uint8_t v_constApprox_858_; uint8_t v_isDefEqStuckEx_859_; uint8_t v_unificationHints_860_; uint8_t v_proofIrrelevance_861_; uint8_t v_assignSyntheticOpaque_862_; uint8_t v_etaStruct_863_; uint8_t v_univApprox_864_; uint8_t v_iota_865_; uint8_t v_beta_866_; uint8_t v_proj_867_; uint8_t v_zeta_868_; uint8_t v_zetaDelta_869_; uint8_t v_zetaUnused_870_; uint8_t v_zetaHave_871_; uint8_t v_canUnfoldPredicateConfig_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_950_; 
v_fst_851_ = lean_ctor_get(v_____x_845_, 0);
lean_inc(v_fst_851_);
v_snd_852_ = lean_ctor_get(v_____x_845_, 1);
lean_inc(v_snd_852_);
lean_dec_ref(v_____x_845_);
v___x_853_ = l_Lean_Meta_Context_config(v___y_846_);
v_toConfig_854_ = lean_ctor_get(v_config_843_, 0);
v_foApprox_855_ = lean_ctor_get_uint8(v___x_853_, 0);
v_ctxApprox_856_ = lean_ctor_get_uint8(v___x_853_, 1);
v_quasiPatternApprox_857_ = lean_ctor_get_uint8(v___x_853_, 2);
v_constApprox_858_ = lean_ctor_get_uint8(v___x_853_, 3);
v_isDefEqStuckEx_859_ = lean_ctor_get_uint8(v___x_853_, 4);
v_unificationHints_860_ = lean_ctor_get_uint8(v___x_853_, 5);
v_proofIrrelevance_861_ = lean_ctor_get_uint8(v___x_853_, 6);
v_assignSyntheticOpaque_862_ = lean_ctor_get_uint8(v___x_853_, 7);
v_etaStruct_863_ = lean_ctor_get_uint8(v___x_853_, 10);
v_univApprox_864_ = lean_ctor_get_uint8(v___x_853_, 11);
v_iota_865_ = lean_ctor_get_uint8(v___x_853_, 12);
v_beta_866_ = lean_ctor_get_uint8(v___x_853_, 13);
v_proj_867_ = lean_ctor_get_uint8(v___x_853_, 14);
v_zeta_868_ = lean_ctor_get_uint8(v___x_853_, 15);
v_zetaDelta_869_ = lean_ctor_get_uint8(v___x_853_, 16);
v_zetaUnused_870_ = lean_ctor_get_uint8(v___x_853_, 17);
v_zetaHave_871_ = lean_ctor_get_uint8(v___x_853_, 18);
v_canUnfoldPredicateConfig_872_ = lean_ctor_get_uint8(v___x_853_, 19);
v_isSharedCheck_950_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_950_ == 0)
{
v___x_874_ = v___x_853_;
v_isShared_875_ = v_isSharedCheck_950_;
goto v_resetjp_873_;
}
else
{
lean_dec(v___x_853_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_950_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
uint8_t v_transparency_876_; uint8_t v_offsetCnstrs_877_; uint8_t v_trackZetaDelta_878_; lean_object* v_zetaDeltaSet_879_; lean_object* v_lctx_880_; lean_object* v_localInstances_881_; lean_object* v_defEqCtx_x3f_882_; lean_object* v_synthPendingDepth_883_; lean_object* v_customCanUnfoldPredicate_x3f_884_; uint8_t v_univApprox_885_; uint8_t v_inTypeClassResolution_886_; uint8_t v_cacheInferType_887_; lean_object* v___x_889_; 
v_transparency_876_ = lean_ctor_get_uint8(v_toConfig_854_, sizeof(void*)*1);
v_offsetCnstrs_877_ = lean_ctor_get_uint8(v_toConfig_854_, sizeof(void*)*1 + 1);
v_trackZetaDelta_878_ = lean_ctor_get_uint8(v___y_846_, sizeof(void*)*7);
v_zetaDeltaSet_879_ = lean_ctor_get(v___y_846_, 1);
v_lctx_880_ = lean_ctor_get(v___y_846_, 2);
v_localInstances_881_ = lean_ctor_get(v___y_846_, 3);
v_defEqCtx_x3f_882_ = lean_ctor_get(v___y_846_, 4);
v_synthPendingDepth_883_ = lean_ctor_get(v___y_846_, 5);
v_customCanUnfoldPredicate_x3f_884_ = lean_ctor_get(v___y_846_, 6);
v_univApprox_885_ = lean_ctor_get_uint8(v___y_846_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_886_ = lean_ctor_get_uint8(v___y_846_, sizeof(void*)*7 + 2);
v_cacheInferType_887_ = lean_ctor_get_uint8(v___y_846_, sizeof(void*)*7 + 3);
if (v_isShared_875_ == 0)
{
v___x_889_ = v___x_874_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 0, v_foApprox_855_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 1, v_ctxApprox_856_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 2, v_quasiPatternApprox_857_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 3, v_constApprox_858_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 4, v_isDefEqStuckEx_859_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 5, v_unificationHints_860_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 6, v_proofIrrelevance_861_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 7, v_assignSyntheticOpaque_862_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 10, v_etaStruct_863_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 11, v_univApprox_864_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 12, v_iota_865_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 13, v_beta_866_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 14, v_proj_867_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 15, v_zeta_868_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 16, v_zetaDelta_869_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 17, v_zetaUnused_870_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 18, v_zetaHave_871_);
lean_ctor_set_uint8(v_reuseFailAlloc_949_, 19, v_canUnfoldPredicateConfig_872_);
v___x_889_ = v_reuseFailAlloc_949_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
uint64_t v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; 
lean_ctor_set_uint8(v___x_889_, 8, v_offsetCnstrs_877_);
lean_ctor_set_uint8(v___x_889_, 9, v_transparency_876_);
v___x_890_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_889_);
v___x_891_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_891_, 0, v___x_889_);
lean_ctor_set_uint64(v___x_891_, sizeof(void*)*1, v___x_890_);
lean_inc(v_customCanUnfoldPredicate_x3f_884_);
lean_inc(v_synthPendingDepth_883_);
lean_inc(v_defEqCtx_x3f_882_);
lean_inc_ref(v_localInstances_881_);
lean_inc_ref(v_lctx_880_);
lean_inc(v_zetaDeltaSet_879_);
v___x_892_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_892_, 0, v___x_891_);
lean_ctor_set(v___x_892_, 1, v_zetaDeltaSet_879_);
lean_ctor_set(v___x_892_, 2, v_lctx_880_);
lean_ctor_set(v___x_892_, 3, v_localInstances_881_);
lean_ctor_set(v___x_892_, 4, v_defEqCtx_x3f_882_);
lean_ctor_set(v___x_892_, 5, v_synthPendingDepth_883_);
lean_ctor_set(v___x_892_, 6, v_customCanUnfoldPredicate_x3f_884_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*7, v_trackZetaDelta_878_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*7 + 1, v_univApprox_885_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*7 + 2, v_inTypeClassResolution_886_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*7 + 3, v_cacheInferType_887_);
lean_inc(v_goal_844_);
v___x_893_ = l_Lean_MVarId_getType(v_goal_844_, v___x_892_, v___y_847_, v___y_848_, v___y_849_);
if (lean_obj_tag(v___x_893_) == 0)
{
lean_object* v_a_894_; lean_object* v___x_895_; 
v_a_894_ = lean_ctor_get(v___x_893_, 0);
lean_inc(v_a_894_);
lean_dec_ref_known(v___x_893_, 1);
v___x_895_ = l_Lean_Meta_isExprDefEq(v_a_894_, v_fst_851_, v___x_892_, v___y_847_, v___y_848_, v___y_849_);
if (lean_obj_tag(v___x_895_) == 0)
{
lean_object* v_a_896_; uint8_t v___x_897_; 
v_a_896_ = lean_ctor_get(v___x_895_, 0);
lean_inc(v_a_896_);
lean_dec_ref_known(v___x_895_, 1);
v___x_897_ = lean_unbox(v_a_896_);
if (v___x_897_ == 0)
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v_mctx_900_; lean_object* v_env_901_; lean_object* v___x_902_; lean_object* v_toEnvExtension_903_; lean_object* v_asyncMode_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v_snd_908_; lean_object* v___x_909_; lean_object* v___x_910_; 
v___x_898_ = lean_st_ref_get(v___y_847_);
v___x_899_ = lean_st_ref_get(v___y_849_);
v_mctx_900_ = lean_ctor_get(v___x_898_, 0);
lean_inc_ref(v_mctx_900_);
lean_dec(v___x_898_);
v_env_901_ = lean_ctor_get(v___x_899_, 0);
lean_inc_ref(v_env_901_);
lean_dec(v___x_899_);
v___x_902_ = lp_mathlib_Mathlib_Tactic_GCongr_forwardExt;
v_toEnvExtension_903_ = lean_ctor_get(v___x_902_, 0);
v_asyncMode_904_ = lean_ctor_get(v_toEnvExtension_903_, 2);
v___x_905_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___closed__0));
v___x_906_ = lean_box(0);
v___x_907_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_905_, v___x_902_, v_env_901_, v_asyncMode_904_, v___x_906_);
v_snd_908_ = lean_ctor_get(v___x_907_, 1);
lean_inc(v_snd_908_);
lean_dec(v___x_907_);
v___x_909_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg___closed__0));
v___x_910_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg(v_snd_852_, v_goal_844_, v_mctx_900_, v_snd_908_, v___x_909_, v___x_892_, v___y_847_, v___y_848_, v___y_849_);
lean_dec_ref_known(v___x_892_, 7);
lean_dec(v_snd_908_);
if (lean_obj_tag(v___x_910_) == 0)
{
lean_object* v_a_911_; lean_object* v___x_913_; uint8_t v_isShared_914_; uint8_t v_isSharedCheck_923_; 
v_a_911_ = lean_ctor_get(v___x_910_, 0);
v_isSharedCheck_923_ = !lean_is_exclusive(v___x_910_);
if (v_isSharedCheck_923_ == 0)
{
v___x_913_ = v___x_910_;
v_isShared_914_ = v_isSharedCheck_923_;
goto v_resetjp_912_;
}
else
{
lean_inc(v_a_911_);
lean_dec(v___x_910_);
v___x_913_ = lean_box(0);
v_isShared_914_ = v_isSharedCheck_923_;
goto v_resetjp_912_;
}
v_resetjp_912_:
{
lean_object* v_fst_915_; 
v_fst_915_ = lean_ctor_get(v_a_911_, 0);
lean_inc(v_fst_915_);
lean_dec(v_a_911_);
if (lean_obj_tag(v_fst_915_) == 0)
{
lean_object* v___x_917_; 
if (v_isShared_914_ == 0)
{
lean_ctor_set(v___x_913_, 0, v_a_896_);
v___x_917_ = v___x_913_;
goto v_reusejp_916_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v_a_896_);
v___x_917_ = v_reuseFailAlloc_918_;
goto v_reusejp_916_;
}
v_reusejp_916_:
{
return v___x_917_;
}
}
else
{
lean_object* v_val_919_; lean_object* v___x_921_; 
lean_dec(v_a_896_);
v_val_919_ = lean_ctor_get(v_fst_915_, 0);
lean_inc(v_val_919_);
lean_dec_ref_known(v_fst_915_, 1);
if (v_isShared_914_ == 0)
{
lean_ctor_set(v___x_913_, 0, v_val_919_);
v___x_921_ = v___x_913_;
goto v_reusejp_920_;
}
else
{
lean_object* v_reuseFailAlloc_922_; 
v_reuseFailAlloc_922_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_922_, 0, v_val_919_);
v___x_921_ = v_reuseFailAlloc_922_;
goto v_reusejp_920_;
}
v_reusejp_920_:
{
return v___x_921_;
}
}
}
}
else
{
lean_object* v_a_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_931_; 
lean_dec(v_a_896_);
v_a_924_ = lean_ctor_get(v___x_910_, 0);
v_isSharedCheck_931_ = !lean_is_exclusive(v___x_910_);
if (v_isSharedCheck_931_ == 0)
{
v___x_926_ = v___x_910_;
v_isShared_927_ = v_isSharedCheck_931_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_a_924_);
lean_dec(v___x_910_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_931_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___x_929_; 
if (v_isShared_927_ == 0)
{
v___x_929_ = v___x_926_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_a_924_);
v___x_929_ = v_reuseFailAlloc_930_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
return v___x_929_;
}
}
}
}
else
{
lean_object* v___x_932_; lean_object* v___x_934_; uint8_t v_isShared_935_; uint8_t v_isSharedCheck_939_; 
lean_dec_ref_known(v___x_892_, 7);
v___x_932_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg(v_goal_844_, v_snd_852_, v___y_847_);
v_isSharedCheck_939_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_939_ == 0)
{
lean_object* v_unused_940_; 
v_unused_940_ = lean_ctor_get(v___x_932_, 0);
lean_dec(v_unused_940_);
v___x_934_ = v___x_932_;
v_isShared_935_ = v_isSharedCheck_939_;
goto v_resetjp_933_;
}
else
{
lean_dec(v___x_932_);
v___x_934_ = lean_box(0);
v_isShared_935_ = v_isSharedCheck_939_;
goto v_resetjp_933_;
}
v_resetjp_933_:
{
lean_object* v___x_937_; 
if (v_isShared_935_ == 0)
{
lean_ctor_set(v___x_934_, 0, v_a_896_);
v___x_937_ = v___x_934_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v_a_896_);
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
else
{
lean_dec_ref_known(v___x_892_, 7);
lean_dec(v_snd_852_);
lean_dec(v_goal_844_);
return v___x_895_;
}
}
else
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_948_; 
lean_dec_ref_known(v___x_892_, 7);
lean_dec(v_snd_852_);
lean_dec(v_fst_851_);
lean_dec(v_goal_844_);
v_a_941_ = lean_ctor_get(v___x_893_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_893_);
if (v_isSharedCheck_948_ == 0)
{
v___x_943_ = v___x_893_;
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_893_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_946_; 
if (v_isShared_944_ == 0)
{
v___x_946_ = v___x_943_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_a_941_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___boxed(lean_object* v_config_951_, lean_object* v_goal_952_, lean_object* v_____x_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0(v_config_951_, v_goal_952_, v_____x_953_, v___y_954_, v___y_955_, v___y_956_, v___y_957_);
lean_dec(v___y_957_);
lean_dec_ref(v___y_956_);
lean_dec(v___y_955_);
lean_dec_ref(v___y_954_);
lean_dec_ref(v_config_951_);
return v_res_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(uint8_t v_symm_960_, lean_object* v_lem_961_, lean_object* v___f_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_){
_start:
{
if (v_symm_960_ == 0)
{
lean_object* v_proof_968_; lean_object* v_type_969_; lean_object* v___x_970_; lean_object* v___x_971_; 
v_proof_968_ = lean_ctor_get(v_lem_961_, 0);
lean_inc_ref(v_proof_968_);
v_type_969_ = lean_ctor_get(v_lem_961_, 1);
lean_inc_ref(v_type_969_);
lean_dec_ref(v_lem_961_);
v___x_970_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_970_, 0, v_type_969_);
lean_ctor_set(v___x_970_, 1, v_proof_968_);
lean_inc(v___y_966_);
lean_inc_ref(v___y_965_);
lean_inc(v___y_964_);
lean_inc_ref(v___y_963_);
v___x_971_ = lean_apply_6(v___f_962_, v___x_970_, v___y_963_, v___y_964_, v___y_965_, v___y_966_, lean_box(0));
return v___x_971_;
}
else
{
lean_object* v_proof_972_; lean_object* v___x_973_; 
v_proof_972_ = lean_ctor_get(v_lem_961_, 0);
lean_inc_ref(v_proof_972_);
lean_dec_ref(v_lem_961_);
v___x_973_ = l_Lean_Expr_applySymm(v_proof_972_, v___y_963_, v___y_964_, v___y_965_, v___y_966_);
if (lean_obj_tag(v___x_973_) == 0)
{
lean_object* v_a_974_; lean_object* v___x_975_; 
v_a_974_ = lean_ctor_get(v___x_973_, 0);
lean_inc_n(v_a_974_, 2);
lean_dec_ref_known(v___x_973_, 1);
lean_inc(v___y_966_);
lean_inc_ref(v___y_965_);
lean_inc(v___y_964_);
lean_inc_ref(v___y_963_);
v___x_975_ = lean_infer_type(v_a_974_, v___y_963_, v___y_964_, v___y_965_, v___y_966_);
if (lean_obj_tag(v___x_975_) == 0)
{
lean_object* v_a_976_; lean_object* v___x_977_; lean_object* v___x_978_; 
v_a_976_ = lean_ctor_get(v___x_975_, 0);
lean_inc(v_a_976_);
lean_dec_ref_known(v___x_975_, 1);
v___x_977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_977_, 0, v_a_976_);
lean_ctor_set(v___x_977_, 1, v_a_974_);
lean_inc(v___y_966_);
lean_inc_ref(v___y_965_);
lean_inc(v___y_964_);
lean_inc_ref(v___y_963_);
v___x_978_ = lean_apply_6(v___f_962_, v___x_977_, v___y_963_, v___y_964_, v___y_965_, v___y_966_, lean_box(0));
return v___x_978_;
}
else
{
lean_object* v_a_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_986_; 
lean_dec(v_a_974_);
lean_dec_ref(v___f_962_);
v_a_979_ = lean_ctor_get(v___x_975_, 0);
v_isSharedCheck_986_ = !lean_is_exclusive(v___x_975_);
if (v_isSharedCheck_986_ == 0)
{
v___x_981_ = v___x_975_;
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_a_979_);
lean_dec(v___x_975_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v___x_984_; 
if (v_isShared_982_ == 0)
{
v___x_984_ = v___x_981_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_985_; 
v_reuseFailAlloc_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_985_, 0, v_a_979_);
v___x_984_ = v_reuseFailAlloc_985_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
return v___x_984_;
}
}
}
}
else
{
lean_object* v_a_987_; lean_object* v___x_989_; uint8_t v_isShared_990_; uint8_t v_isSharedCheck_1002_; 
lean_dec_ref(v___f_962_);
v_a_987_ = lean_ctor_get(v___x_973_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_973_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_989_ = v___x_973_;
v_isShared_990_ = v_isSharedCheck_1002_;
goto v_resetjp_988_;
}
else
{
lean_inc(v_a_987_);
lean_dec(v___x_973_);
v___x_989_ = lean_box(0);
v_isShared_990_ = v_isSharedCheck_1002_;
goto v_resetjp_988_;
}
v_resetjp_988_:
{
uint8_t v___y_992_; uint8_t v___x_1000_; 
v___x_1000_ = l_Lean_Exception_isInterrupt(v_a_987_);
if (v___x_1000_ == 0)
{
uint8_t v___x_1001_; 
lean_inc(v_a_987_);
v___x_1001_ = l_Lean_Exception_isRuntime(v_a_987_);
v___y_992_ = v___x_1001_;
goto v___jp_991_;
}
else
{
v___y_992_ = v___x_1000_;
goto v___jp_991_;
}
v___jp_991_:
{
if (v___y_992_ == 0)
{
lean_object* v___x_993_; lean_object* v___x_995_; 
lean_dec(v_a_987_);
v___x_993_ = lean_box(v___y_992_);
if (v_isShared_990_ == 0)
{
lean_ctor_set_tag(v___x_989_, 0);
lean_ctor_set(v___x_989_, 0, v___x_993_);
v___x_995_ = v___x_989_;
goto v_reusejp_994_;
}
else
{
lean_object* v_reuseFailAlloc_996_; 
v_reuseFailAlloc_996_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_996_, 0, v___x_993_);
v___x_995_ = v_reuseFailAlloc_996_;
goto v_reusejp_994_;
}
v_reusejp_994_:
{
return v___x_995_;
}
}
else
{
lean_object* v___x_998_; 
if (v_isShared_990_ == 0)
{
v___x_998_ = v___x_989_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v_a_987_);
v___x_998_ = v_reuseFailAlloc_999_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
return v___x_998_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1___boxed(lean_object* v_symm_1003_, lean_object* v_lem_1004_, lean_object* v___f_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
uint8_t v_symm_boxed_1011_; lean_object* v_res_1012_; 
v_symm_boxed_1011_ = lean_unbox(v_symm_1003_);
v_res_1012_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(v_symm_boxed_1011_, v_lem_1004_, v___f_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
lean_dec(v___y_1007_);
lean_dec_ref(v___y_1006_);
return v_res_1012_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1014_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__0));
v___x_1015_ = l_Lean_stringToMessageData(v___x_1014_);
return v___x_1015_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3(void){
_start:
{
lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1017_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__2));
v___x_1018_ = l_Lean_stringToMessageData(v___x_1017_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2(lean_object* v_lem_1019_, lean_object* v_x_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_){
_start:
{
lean_object* v_proof_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; 
v_proof_1026_ = lean_ctor_get(v_lem_1019_, 0);
lean_inc_ref(v_proof_1026_);
lean_dec_ref(v_lem_1019_);
v___x_1027_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__1);
v___x_1028_ = l_Lean_MessageData_ofExpr(v_proof_1026_);
v___x_1029_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1029_, 0, v___x_1027_);
lean_ctor_set(v___x_1029_, 1, v___x_1028_);
v___x_1030_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3);
v___x_1031_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1029_);
lean_ctor_set(v___x_1031_, 1, v___x_1030_);
v___x_1032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1031_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___boxed(lean_object* v_lem_1033_, lean_object* v_x_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_){
_start:
{
lean_object* v_res_1040_; 
v_res_1040_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2(v_lem_1033_, v_x_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_);
lean_dec(v___y_1038_);
lean_dec_ref(v___y_1037_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
lean_dec_ref(v_x_1034_);
return v_res_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(lean_object* v_x_1041_){
_start:
{
if (lean_obj_tag(v_x_1041_) == 0)
{
lean_object* v_a_1043_; lean_object* v___x_1045_; uint8_t v_isShared_1046_; uint8_t v_isSharedCheck_1050_; 
v_a_1043_ = lean_ctor_get(v_x_1041_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v_x_1041_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_1045_ = v_x_1041_;
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
else
{
lean_inc(v_a_1043_);
lean_dec(v_x_1041_);
v___x_1045_ = lean_box(0);
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
v_resetjp_1044_:
{
lean_object* v___x_1048_; 
if (v_isShared_1046_ == 0)
{
lean_ctor_set_tag(v___x_1045_, 1);
v___x_1048_ = v___x_1045_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v_a_1043_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
}
else
{
lean_object* v_a_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1058_; 
v_a_1051_ = lean_ctor_get(v_x_1041_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v_x_1041_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1053_ = v_x_1041_;
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_a_1051_);
lean_dec(v_x_1041_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1056_; 
if (v_isShared_1054_ == 0)
{
lean_ctor_set_tag(v___x_1053_, 0);
v___x_1056_ = v___x_1053_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_a_1051_);
v___x_1056_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
return v___x_1056_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg___boxed(lean_object* v_x_1059_, lean_object* v___y_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(v_x_1059_);
return v_res_1061_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7(lean_object* v_e_1062_){
_start:
{
if (lean_obj_tag(v_e_1062_) == 0)
{
uint8_t v___x_1063_; 
v___x_1063_ = 2;
return v___x_1063_;
}
else
{
lean_object* v_a_1064_; uint8_t v___x_1065_; 
v_a_1064_ = lean_ctor_get(v_e_1062_, 0);
v___x_1065_ = lean_unbox(v_a_1064_);
if (v___x_1065_ == 0)
{
uint8_t v___x_1066_; 
v___x_1066_ = 1;
return v___x_1066_;
}
else
{
uint8_t v___x_1067_; 
v___x_1067_ = 0;
return v___x_1067_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7___boxed(lean_object* v_e_1068_){
_start:
{
uint8_t v_res_1069_; lean_object* v_r_1070_; 
v_res_1069_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7(v_e_1068_);
lean_dec_ref(v_e_1068_);
v_r_1070_ = lean_box(v_res_1069_);
return v_r_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(lean_object* v_opts_1071_, lean_object* v_opt_1072_){
_start:
{
lean_object* v_name_1073_; lean_object* v_defValue_1074_; lean_object* v_map_1075_; lean_object* v___x_1076_; 
v_name_1073_ = lean_ctor_get(v_opt_1072_, 0);
v_defValue_1074_ = lean_ctor_get(v_opt_1072_, 1);
v_map_1075_ = lean_ctor_get(v_opts_1071_, 0);
v___x_1076_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1075_, v_name_1073_);
if (lean_obj_tag(v___x_1076_) == 0)
{
lean_inc(v_defValue_1074_);
return v_defValue_1074_;
}
else
{
lean_object* v_val_1077_; 
v_val_1077_ = lean_ctor_get(v___x_1076_, 0);
lean_inc(v_val_1077_);
lean_dec_ref_known(v___x_1076_, 1);
if (lean_obj_tag(v_val_1077_) == 3)
{
lean_object* v_v_1078_; 
v_v_1078_ = lean_ctor_get(v_val_1077_, 0);
lean_inc(v_v_1078_);
lean_dec_ref_known(v_val_1077_, 1);
return v_v_1078_;
}
else
{
lean_dec(v_val_1077_);
lean_inc(v_defValue_1074_);
return v_defValue_1074_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8___boxed(lean_object* v_opts_1079_, lean_object* v_opt_1080_){
_start:
{
lean_object* v_res_1081_; 
v_res_1081_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_1079_, v_opt_1080_);
lean_dec_ref(v_opt_1080_);
lean_dec_ref(v_opts_1079_);
return v_res_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(lean_object* v_msgData_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_){
_start:
{
lean_object* v___x_1088_; lean_object* v_env_1089_; lean_object* v___x_1090_; lean_object* v_mctx_1091_; lean_object* v_lctx_1092_; lean_object* v_options_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; 
v___x_1088_ = lean_st_ref_get(v___y_1086_);
v_env_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc_ref(v_env_1089_);
lean_dec(v___x_1088_);
v___x_1090_ = lean_st_ref_get(v___y_1084_);
v_mctx_1091_ = lean_ctor_get(v___x_1090_, 0);
lean_inc_ref(v_mctx_1091_);
lean_dec(v___x_1090_);
v_lctx_1092_ = lean_ctor_get(v___y_1083_, 2);
v_options_1093_ = lean_ctor_get(v___y_1085_, 2);
lean_inc_ref(v_options_1093_);
lean_inc_ref(v_lctx_1092_);
v___x_1094_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1094_, 0, v_env_1089_);
lean_ctor_set(v___x_1094_, 1, v_mctx_1091_);
lean_ctor_set(v___x_1094_, 2, v_lctx_1092_);
lean_ctor_set(v___x_1094_, 3, v_options_1093_);
v___x_1095_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1095_, 0, v___x_1094_);
lean_ctor_set(v___x_1095_, 1, v_msgData_1082_);
v___x_1096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1096_, 0, v___x_1095_);
return v___x_1096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8___boxed(lean_object* v_msgData_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msgData_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7(size_t v_sz_1104_, size_t v_i_1105_, lean_object* v_bs_1106_){
_start:
{
uint8_t v___x_1107_; 
v___x_1107_ = lean_usize_dec_lt(v_i_1105_, v_sz_1104_);
if (v___x_1107_ == 0)
{
return v_bs_1106_;
}
else
{
lean_object* v_v_1108_; lean_object* v_msg_1109_; lean_object* v___x_1110_; lean_object* v_bs_x27_1111_; size_t v___x_1112_; size_t v___x_1113_; lean_object* v___x_1114_; 
v_v_1108_ = lean_array_uget_borrowed(v_bs_1106_, v_i_1105_);
v_msg_1109_ = lean_ctor_get(v_v_1108_, 1);
lean_inc_ref(v_msg_1109_);
v___x_1110_ = lean_unsigned_to_nat(0u);
v_bs_x27_1111_ = lean_array_uset(v_bs_1106_, v_i_1105_, v___x_1110_);
v___x_1112_ = ((size_t)1ULL);
v___x_1113_ = lean_usize_add(v_i_1105_, v___x_1112_);
v___x_1114_ = lean_array_uset(v_bs_x27_1111_, v_i_1105_, v_msg_1109_);
v_i_1105_ = v___x_1113_;
v_bs_1106_ = v___x_1114_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7___boxed(lean_object* v_sz_1116_, lean_object* v_i_1117_, lean_object* v_bs_1118_){
_start:
{
size_t v_sz_boxed_1119_; size_t v_i_boxed_1120_; lean_object* v_res_1121_; 
v_sz_boxed_1119_ = lean_unbox_usize(v_sz_1116_);
lean_dec(v_sz_1116_);
v_i_boxed_1120_ = lean_unbox_usize(v_i_1117_);
lean_dec(v_i_1117_);
v_res_1121_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7(v_sz_boxed_1119_, v_i_boxed_1120_, v_bs_1118_);
return v_res_1121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5(lean_object* v_oldTraces_1122_, lean_object* v_data_1123_, lean_object* v_ref_1124_, lean_object* v_msg_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
lean_object* v_fileName_1131_; lean_object* v_fileMap_1132_; lean_object* v_options_1133_; lean_object* v_currRecDepth_1134_; lean_object* v_maxRecDepth_1135_; lean_object* v_ref_1136_; lean_object* v_currNamespace_1137_; lean_object* v_openDecls_1138_; lean_object* v_initHeartbeats_1139_; lean_object* v_maxHeartbeats_1140_; lean_object* v_quotContext_1141_; lean_object* v_currMacroScope_1142_; uint8_t v_diag_1143_; lean_object* v_cancelTk_x3f_1144_; uint8_t v_suppressElabErrors_1145_; lean_object* v_inheritedTraceOptions_1146_; lean_object* v___x_1147_; lean_object* v_traceState_1148_; lean_object* v_traces_1149_; lean_object* v_ref_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; size_t v_sz_1153_; size_t v___x_1154_; lean_object* v___x_1155_; lean_object* v_msg_1156_; lean_object* v___x_1157_; lean_object* v_a_1158_; lean_object* v___x_1160_; uint8_t v_isShared_1161_; uint8_t v_isSharedCheck_1195_; 
v_fileName_1131_ = lean_ctor_get(v___y_1128_, 0);
v_fileMap_1132_ = lean_ctor_get(v___y_1128_, 1);
v_options_1133_ = lean_ctor_get(v___y_1128_, 2);
v_currRecDepth_1134_ = lean_ctor_get(v___y_1128_, 3);
v_maxRecDepth_1135_ = lean_ctor_get(v___y_1128_, 4);
v_ref_1136_ = lean_ctor_get(v___y_1128_, 5);
v_currNamespace_1137_ = lean_ctor_get(v___y_1128_, 6);
v_openDecls_1138_ = lean_ctor_get(v___y_1128_, 7);
v_initHeartbeats_1139_ = lean_ctor_get(v___y_1128_, 8);
v_maxHeartbeats_1140_ = lean_ctor_get(v___y_1128_, 9);
v_quotContext_1141_ = lean_ctor_get(v___y_1128_, 10);
v_currMacroScope_1142_ = lean_ctor_get(v___y_1128_, 11);
v_diag_1143_ = lean_ctor_get_uint8(v___y_1128_, sizeof(void*)*14);
v_cancelTk_x3f_1144_ = lean_ctor_get(v___y_1128_, 12);
v_suppressElabErrors_1145_ = lean_ctor_get_uint8(v___y_1128_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1146_ = lean_ctor_get(v___y_1128_, 13);
v___x_1147_ = lean_st_ref_get(v___y_1129_);
v_traceState_1148_ = lean_ctor_get(v___x_1147_, 4);
lean_inc_ref(v_traceState_1148_);
lean_dec(v___x_1147_);
v_traces_1149_ = lean_ctor_get(v_traceState_1148_, 0);
lean_inc_ref(v_traces_1149_);
lean_dec_ref(v_traceState_1148_);
v_ref_1150_ = l_Lean_replaceRef(v_ref_1124_, v_ref_1136_);
lean_inc_ref(v_inheritedTraceOptions_1146_);
lean_inc(v_cancelTk_x3f_1144_);
lean_inc(v_currMacroScope_1142_);
lean_inc(v_quotContext_1141_);
lean_inc(v_maxHeartbeats_1140_);
lean_inc(v_initHeartbeats_1139_);
lean_inc(v_openDecls_1138_);
lean_inc(v_currNamespace_1137_);
lean_inc(v_maxRecDepth_1135_);
lean_inc(v_currRecDepth_1134_);
lean_inc_ref(v_options_1133_);
lean_inc_ref(v_fileMap_1132_);
lean_inc_ref(v_fileName_1131_);
v___x_1151_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1151_, 0, v_fileName_1131_);
lean_ctor_set(v___x_1151_, 1, v_fileMap_1132_);
lean_ctor_set(v___x_1151_, 2, v_options_1133_);
lean_ctor_set(v___x_1151_, 3, v_currRecDepth_1134_);
lean_ctor_set(v___x_1151_, 4, v_maxRecDepth_1135_);
lean_ctor_set(v___x_1151_, 5, v_ref_1150_);
lean_ctor_set(v___x_1151_, 6, v_currNamespace_1137_);
lean_ctor_set(v___x_1151_, 7, v_openDecls_1138_);
lean_ctor_set(v___x_1151_, 8, v_initHeartbeats_1139_);
lean_ctor_set(v___x_1151_, 9, v_maxHeartbeats_1140_);
lean_ctor_set(v___x_1151_, 10, v_quotContext_1141_);
lean_ctor_set(v___x_1151_, 11, v_currMacroScope_1142_);
lean_ctor_set(v___x_1151_, 12, v_cancelTk_x3f_1144_);
lean_ctor_set(v___x_1151_, 13, v_inheritedTraceOptions_1146_);
lean_ctor_set_uint8(v___x_1151_, sizeof(void*)*14, v_diag_1143_);
lean_ctor_set_uint8(v___x_1151_, sizeof(void*)*14 + 1, v_suppressElabErrors_1145_);
v___x_1152_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1149_);
lean_dec_ref(v_traces_1149_);
v_sz_1153_ = lean_array_size(v___x_1152_);
v___x_1154_ = ((size_t)0ULL);
v___x_1155_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7(v_sz_1153_, v___x_1154_, v___x_1152_);
v_msg_1156_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1156_, 0, v_data_1123_);
lean_ctor_set(v_msg_1156_, 1, v_msg_1125_);
lean_ctor_set(v_msg_1156_, 2, v___x_1155_);
v___x_1157_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msg_1156_, v___y_1126_, v___y_1127_, v___x_1151_, v___y_1129_);
lean_dec_ref_known(v___x_1151_, 14);
v_a_1158_ = lean_ctor_get(v___x_1157_, 0);
v_isSharedCheck_1195_ = !lean_is_exclusive(v___x_1157_);
if (v_isSharedCheck_1195_ == 0)
{
v___x_1160_ = v___x_1157_;
v_isShared_1161_ = v_isSharedCheck_1195_;
goto v_resetjp_1159_;
}
else
{
lean_inc(v_a_1158_);
lean_dec(v___x_1157_);
v___x_1160_ = lean_box(0);
v_isShared_1161_ = v_isSharedCheck_1195_;
goto v_resetjp_1159_;
}
v_resetjp_1159_:
{
lean_object* v___x_1162_; lean_object* v_traceState_1163_; lean_object* v_env_1164_; lean_object* v_nextMacroScope_1165_; lean_object* v_ngen_1166_; lean_object* v_auxDeclNGen_1167_; lean_object* v_cache_1168_; lean_object* v_messages_1169_; lean_object* v_infoState_1170_; lean_object* v_snapshotTasks_1171_; lean_object* v___x_1173_; uint8_t v_isShared_1174_; uint8_t v_isSharedCheck_1194_; 
v___x_1162_ = lean_st_ref_take(v___y_1129_);
v_traceState_1163_ = lean_ctor_get(v___x_1162_, 4);
v_env_1164_ = lean_ctor_get(v___x_1162_, 0);
v_nextMacroScope_1165_ = lean_ctor_get(v___x_1162_, 1);
v_ngen_1166_ = lean_ctor_get(v___x_1162_, 2);
v_auxDeclNGen_1167_ = lean_ctor_get(v___x_1162_, 3);
v_cache_1168_ = lean_ctor_get(v___x_1162_, 5);
v_messages_1169_ = lean_ctor_get(v___x_1162_, 6);
v_infoState_1170_ = lean_ctor_get(v___x_1162_, 7);
v_snapshotTasks_1171_ = lean_ctor_get(v___x_1162_, 8);
v_isSharedCheck_1194_ = !lean_is_exclusive(v___x_1162_);
if (v_isSharedCheck_1194_ == 0)
{
v___x_1173_ = v___x_1162_;
v_isShared_1174_ = v_isSharedCheck_1194_;
goto v_resetjp_1172_;
}
else
{
lean_inc(v_snapshotTasks_1171_);
lean_inc(v_infoState_1170_);
lean_inc(v_messages_1169_);
lean_inc(v_cache_1168_);
lean_inc(v_traceState_1163_);
lean_inc(v_auxDeclNGen_1167_);
lean_inc(v_ngen_1166_);
lean_inc(v_nextMacroScope_1165_);
lean_inc(v_env_1164_);
lean_dec(v___x_1162_);
v___x_1173_ = lean_box(0);
v_isShared_1174_ = v_isSharedCheck_1194_;
goto v_resetjp_1172_;
}
v_resetjp_1172_:
{
uint64_t v_tid_1175_; lean_object* v___x_1177_; uint8_t v_isShared_1178_; uint8_t v_isSharedCheck_1192_; 
v_tid_1175_ = lean_ctor_get_uint64(v_traceState_1163_, sizeof(void*)*1);
v_isSharedCheck_1192_ = !lean_is_exclusive(v_traceState_1163_);
if (v_isSharedCheck_1192_ == 0)
{
lean_object* v_unused_1193_; 
v_unused_1193_ = lean_ctor_get(v_traceState_1163_, 0);
lean_dec(v_unused_1193_);
v___x_1177_ = v_traceState_1163_;
v_isShared_1178_ = v_isSharedCheck_1192_;
goto v_resetjp_1176_;
}
else
{
lean_dec(v_traceState_1163_);
v___x_1177_ = lean_box(0);
v_isShared_1178_ = v_isSharedCheck_1192_;
goto v_resetjp_1176_;
}
v_resetjp_1176_:
{
lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1182_; 
v___x_1179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1179_, 0, v_ref_1124_);
lean_ctor_set(v___x_1179_, 1, v_a_1158_);
v___x_1180_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1122_, v___x_1179_);
if (v_isShared_1178_ == 0)
{
lean_ctor_set(v___x_1177_, 0, v___x_1180_);
v___x_1182_ = v___x_1177_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v___x_1180_);
lean_ctor_set_uint64(v_reuseFailAlloc_1191_, sizeof(void*)*1, v_tid_1175_);
v___x_1182_ = v_reuseFailAlloc_1191_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
lean_object* v___x_1184_; 
if (v_isShared_1174_ == 0)
{
lean_ctor_set(v___x_1173_, 4, v___x_1182_);
v___x_1184_ = v___x_1173_;
goto v_reusejp_1183_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_env_1164_);
lean_ctor_set(v_reuseFailAlloc_1190_, 1, v_nextMacroScope_1165_);
lean_ctor_set(v_reuseFailAlloc_1190_, 2, v_ngen_1166_);
lean_ctor_set(v_reuseFailAlloc_1190_, 3, v_auxDeclNGen_1167_);
lean_ctor_set(v_reuseFailAlloc_1190_, 4, v___x_1182_);
lean_ctor_set(v_reuseFailAlloc_1190_, 5, v_cache_1168_);
lean_ctor_set(v_reuseFailAlloc_1190_, 6, v_messages_1169_);
lean_ctor_set(v_reuseFailAlloc_1190_, 7, v_infoState_1170_);
lean_ctor_set(v_reuseFailAlloc_1190_, 8, v_snapshotTasks_1171_);
v___x_1184_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1183_;
}
v_reusejp_1183_:
{
lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1188_; 
v___x_1185_ = lean_st_ref_set(v___y_1129_, v___x_1184_);
v___x_1186_ = lean_box(0);
if (v_isShared_1161_ == 0)
{
lean_ctor_set(v___x_1160_, 0, v___x_1186_);
v___x_1188_ = v___x_1160_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v___x_1186_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5___boxed(lean_object* v_oldTraces_1196_, lean_object* v_data_1197_, lean_object* v_ref_1198_, lean_object* v_msg_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_){
_start:
{
lean_object* v_res_1205_; 
v_res_1205_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5(v_oldTraces_1196_, v_data_1197_, v_ref_1198_, v_msg_1199_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
lean_dec(v___y_1201_);
lean_dec_ref(v___y_1200_);
return v_res_1205_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0(void){
_start:
{
lean_object* v___x_1206_; double v___x_1207_; 
v___x_1206_ = lean_unsigned_to_nat(0u);
v___x_1207_ = lean_float_of_nat(v___x_1206_);
return v___x_1207_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2(void){
_start:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; 
v___x_1209_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__1));
v___x_1210_ = l_Lean_stringToMessageData(v___x_1209_);
return v___x_1210_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3(void){
_start:
{
lean_object* v___x_1211_; double v___x_1212_; 
v___x_1211_ = lean_unsigned_to_nat(1000u);
v___x_1212_ = lean_float_of_nat(v___x_1211_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4(lean_object* v_cls_1213_, uint8_t v_collapsed_1214_, lean_object* v_tag_1215_, lean_object* v_opts_1216_, uint8_t v_clsEnabled_1217_, lean_object* v_oldTraces_1218_, lean_object* v_msg_1219_, lean_object* v_resStartStop_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_){
_start:
{
lean_object* v_fst_1226_; lean_object* v_snd_1227_; lean_object* v___y_1229_; lean_object* v___y_1230_; lean_object* v_data_1231_; lean_object* v_fst_1242_; lean_object* v_snd_1243_; lean_object* v___x_1244_; uint8_t v___x_1245_; lean_object* v___y_1247_; lean_object* v_a_1248_; uint8_t v___y_1263_; double v___y_1294_; 
v_fst_1226_ = lean_ctor_get(v_resStartStop_1220_, 0);
lean_inc(v_fst_1226_);
v_snd_1227_ = lean_ctor_get(v_resStartStop_1220_, 1);
lean_inc(v_snd_1227_);
lean_dec_ref(v_resStartStop_1220_);
v_fst_1242_ = lean_ctor_get(v_snd_1227_, 0);
lean_inc(v_fst_1242_);
v_snd_1243_ = lean_ctor_get(v_snd_1227_, 1);
lean_inc(v_snd_1243_);
lean_dec(v_snd_1227_);
v___x_1244_ = l_Lean_trace_profiler;
v___x_1245_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_1216_, v___x_1244_);
if (v___x_1245_ == 0)
{
v___y_1263_ = v___x_1245_;
goto v___jp_1262_;
}
else
{
lean_object* v___x_1299_; uint8_t v___x_1300_; 
v___x_1299_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1300_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_1216_, v___x_1299_);
if (v___x_1300_ == 0)
{
lean_object* v___x_1301_; lean_object* v___x_1302_; double v___x_1303_; double v___x_1304_; double v___x_1305_; 
v___x_1301_ = l_Lean_trace_profiler_threshold;
v___x_1302_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_1216_, v___x_1301_);
v___x_1303_ = lean_float_of_nat(v___x_1302_);
v___x_1304_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3);
v___x_1305_ = lean_float_div(v___x_1303_, v___x_1304_);
v___y_1294_ = v___x_1305_;
goto v___jp_1293_;
}
else
{
lean_object* v___x_1306_; lean_object* v___x_1307_; double v___x_1308_; 
v___x_1306_ = l_Lean_trace_profiler_threshold;
v___x_1307_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_1216_, v___x_1306_);
v___x_1308_ = lean_float_of_nat(v___x_1307_);
v___y_1294_ = v___x_1308_;
goto v___jp_1293_;
}
}
v___jp_1228_:
{
lean_object* v___x_1232_; 
lean_inc(v___y_1230_);
v___x_1232_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5(v_oldTraces_1218_, v_data_1231_, v___y_1230_, v___y_1229_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
if (lean_obj_tag(v___x_1232_) == 0)
{
lean_object* v___x_1233_; 
lean_dec_ref_known(v___x_1232_, 1);
v___x_1233_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(v_fst_1226_);
return v___x_1233_;
}
else
{
lean_object* v_a_1234_; lean_object* v___x_1236_; uint8_t v_isShared_1237_; uint8_t v_isSharedCheck_1241_; 
lean_dec(v_fst_1226_);
v_a_1234_ = lean_ctor_get(v___x_1232_, 0);
v_isSharedCheck_1241_ = !lean_is_exclusive(v___x_1232_);
if (v_isSharedCheck_1241_ == 0)
{
v___x_1236_ = v___x_1232_;
v_isShared_1237_ = v_isSharedCheck_1241_;
goto v_resetjp_1235_;
}
else
{
lean_inc(v_a_1234_);
lean_dec(v___x_1232_);
v___x_1236_ = lean_box(0);
v_isShared_1237_ = v_isSharedCheck_1241_;
goto v_resetjp_1235_;
}
v_resetjp_1235_:
{
lean_object* v___x_1239_; 
if (v_isShared_1237_ == 0)
{
v___x_1239_ = v___x_1236_;
goto v_reusejp_1238_;
}
else
{
lean_object* v_reuseFailAlloc_1240_; 
v_reuseFailAlloc_1240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1240_, 0, v_a_1234_);
v___x_1239_ = v_reuseFailAlloc_1240_;
goto v_reusejp_1238_;
}
v_reusejp_1238_:
{
return v___x_1239_;
}
}
}
}
v___jp_1246_:
{
uint8_t v_result_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; double v___x_1252_; lean_object* v_data_1253_; 
v_result_1249_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7(v_fst_1226_);
v___x_1250_ = lean_box(v_result_1249_);
v___x_1251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1251_, 0, v___x_1250_);
v___x_1252_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0);
lean_inc_ref(v_tag_1215_);
lean_inc_ref(v___x_1251_);
lean_inc(v_cls_1213_);
v_data_1253_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1253_, 0, v_cls_1213_);
lean_ctor_set(v_data_1253_, 1, v___x_1251_);
lean_ctor_set(v_data_1253_, 2, v_tag_1215_);
lean_ctor_set_float(v_data_1253_, sizeof(void*)*3, v___x_1252_);
lean_ctor_set_float(v_data_1253_, sizeof(void*)*3 + 8, v___x_1252_);
lean_ctor_set_uint8(v_data_1253_, sizeof(void*)*3 + 16, v_collapsed_1214_);
if (v___x_1245_ == 0)
{
lean_dec_ref_known(v___x_1251_, 1);
lean_dec(v_snd_1243_);
lean_dec(v_fst_1242_);
lean_dec_ref(v_tag_1215_);
lean_dec(v_cls_1213_);
v___y_1229_ = v_a_1248_;
v___y_1230_ = v___y_1247_;
v_data_1231_ = v_data_1253_;
goto v___jp_1228_;
}
else
{
lean_object* v_data_1254_; double v___x_1255_; double v___x_1256_; 
lean_dec_ref_known(v_data_1253_, 3);
v_data_1254_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1254_, 0, v_cls_1213_);
lean_ctor_set(v_data_1254_, 1, v___x_1251_);
lean_ctor_set(v_data_1254_, 2, v_tag_1215_);
v___x_1255_ = lean_unbox_float(v_fst_1242_);
lean_dec(v_fst_1242_);
lean_ctor_set_float(v_data_1254_, sizeof(void*)*3, v___x_1255_);
v___x_1256_ = lean_unbox_float(v_snd_1243_);
lean_dec(v_snd_1243_);
lean_ctor_set_float(v_data_1254_, sizeof(void*)*3 + 8, v___x_1256_);
lean_ctor_set_uint8(v_data_1254_, sizeof(void*)*3 + 16, v_collapsed_1214_);
v___y_1229_ = v_a_1248_;
v___y_1230_ = v___y_1247_;
v_data_1231_ = v_data_1254_;
goto v___jp_1228_;
}
}
v___jp_1257_:
{
lean_object* v_ref_1258_; lean_object* v___x_1259_; 
v_ref_1258_ = lean_ctor_get(v___y_1223_, 5);
lean_inc(v___y_1224_);
lean_inc_ref(v___y_1223_);
lean_inc(v___y_1222_);
lean_inc_ref(v___y_1221_);
lean_inc(v_fst_1226_);
v___x_1259_ = lean_apply_6(v_msg_1219_, v_fst_1226_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_, lean_box(0));
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_object* v_a_1260_; 
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_a_1260_);
lean_dec_ref_known(v___x_1259_, 1);
v___y_1247_ = v_ref_1258_;
v_a_1248_ = v_a_1260_;
goto v___jp_1246_;
}
else
{
lean_object* v___x_1261_; 
lean_dec_ref_known(v___x_1259_, 1);
v___x_1261_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2);
v___y_1247_ = v_ref_1258_;
v_a_1248_ = v___x_1261_;
goto v___jp_1246_;
}
}
v___jp_1262_:
{
if (v_clsEnabled_1217_ == 0)
{
if (v___y_1263_ == 0)
{
lean_object* v___x_1264_; lean_object* v_traceState_1265_; lean_object* v_env_1266_; lean_object* v_nextMacroScope_1267_; lean_object* v_ngen_1268_; lean_object* v_auxDeclNGen_1269_; lean_object* v_cache_1270_; lean_object* v_messages_1271_; lean_object* v_infoState_1272_; lean_object* v_snapshotTasks_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1292_; 
lean_dec(v_snd_1243_);
lean_dec(v_fst_1242_);
lean_dec_ref(v_msg_1219_);
lean_dec_ref(v_tag_1215_);
lean_dec(v_cls_1213_);
v___x_1264_ = lean_st_ref_take(v___y_1224_);
v_traceState_1265_ = lean_ctor_get(v___x_1264_, 4);
v_env_1266_ = lean_ctor_get(v___x_1264_, 0);
v_nextMacroScope_1267_ = lean_ctor_get(v___x_1264_, 1);
v_ngen_1268_ = lean_ctor_get(v___x_1264_, 2);
v_auxDeclNGen_1269_ = lean_ctor_get(v___x_1264_, 3);
v_cache_1270_ = lean_ctor_get(v___x_1264_, 5);
v_messages_1271_ = lean_ctor_get(v___x_1264_, 6);
v_infoState_1272_ = lean_ctor_get(v___x_1264_, 7);
v_snapshotTasks_1273_ = lean_ctor_get(v___x_1264_, 8);
v_isSharedCheck_1292_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1292_ == 0)
{
v___x_1275_ = v___x_1264_;
v_isShared_1276_ = v_isSharedCheck_1292_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_snapshotTasks_1273_);
lean_inc(v_infoState_1272_);
lean_inc(v_messages_1271_);
lean_inc(v_cache_1270_);
lean_inc(v_traceState_1265_);
lean_inc(v_auxDeclNGen_1269_);
lean_inc(v_ngen_1268_);
lean_inc(v_nextMacroScope_1267_);
lean_inc(v_env_1266_);
lean_dec(v___x_1264_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1292_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
uint64_t v_tid_1277_; lean_object* v_traces_1278_; lean_object* v___x_1280_; uint8_t v_isShared_1281_; uint8_t v_isSharedCheck_1291_; 
v_tid_1277_ = lean_ctor_get_uint64(v_traceState_1265_, sizeof(void*)*1);
v_traces_1278_ = lean_ctor_get(v_traceState_1265_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v_traceState_1265_);
if (v_isSharedCheck_1291_ == 0)
{
v___x_1280_ = v_traceState_1265_;
v_isShared_1281_ = v_isSharedCheck_1291_;
goto v_resetjp_1279_;
}
else
{
lean_inc(v_traces_1278_);
lean_dec(v_traceState_1265_);
v___x_1280_ = lean_box(0);
v_isShared_1281_ = v_isSharedCheck_1291_;
goto v_resetjp_1279_;
}
v_resetjp_1279_:
{
lean_object* v___x_1282_; lean_object* v___x_1284_; 
v___x_1282_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1218_, v_traces_1278_);
lean_dec_ref(v_traces_1278_);
if (v_isShared_1281_ == 0)
{
lean_ctor_set(v___x_1280_, 0, v___x_1282_);
v___x_1284_ = v___x_1280_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1282_);
lean_ctor_set_uint64(v_reuseFailAlloc_1290_, sizeof(void*)*1, v_tid_1277_);
v___x_1284_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
lean_object* v___x_1286_; 
if (v_isShared_1276_ == 0)
{
lean_ctor_set(v___x_1275_, 4, v___x_1284_);
v___x_1286_ = v___x_1275_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v_env_1266_);
lean_ctor_set(v_reuseFailAlloc_1289_, 1, v_nextMacroScope_1267_);
lean_ctor_set(v_reuseFailAlloc_1289_, 2, v_ngen_1268_);
lean_ctor_set(v_reuseFailAlloc_1289_, 3, v_auxDeclNGen_1269_);
lean_ctor_set(v_reuseFailAlloc_1289_, 4, v___x_1284_);
lean_ctor_set(v_reuseFailAlloc_1289_, 5, v_cache_1270_);
lean_ctor_set(v_reuseFailAlloc_1289_, 6, v_messages_1271_);
lean_ctor_set(v_reuseFailAlloc_1289_, 7, v_infoState_1272_);
lean_ctor_set(v_reuseFailAlloc_1289_, 8, v_snapshotTasks_1273_);
v___x_1286_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1287_ = lean_st_ref_set(v___y_1224_, v___x_1286_);
v___x_1288_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(v_fst_1226_);
return v___x_1288_;
}
}
}
}
}
else
{
goto v___jp_1257_;
}
}
else
{
goto v___jp_1257_;
}
}
v___jp_1293_:
{
double v___x_1295_; double v___x_1296_; double v___x_1297_; uint8_t v___x_1298_; 
v___x_1295_ = lean_unbox_float(v_snd_1243_);
v___x_1296_ = lean_unbox_float(v_fst_1242_);
v___x_1297_ = lean_float_sub(v___x_1295_, v___x_1296_);
v___x_1298_ = lean_float_decLt(v___y_1294_, v___x_1297_);
v___y_1263_ = v___x_1298_;
goto v___jp_1262_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___boxed(lean_object* v_cls_1309_, lean_object* v_collapsed_1310_, lean_object* v_tag_1311_, lean_object* v_opts_1312_, lean_object* v_clsEnabled_1313_, lean_object* v_oldTraces_1314_, lean_object* v_msg_1315_, lean_object* v_resStartStop_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
uint8_t v_collapsed_boxed_1322_; uint8_t v_clsEnabled_boxed_1323_; lean_object* v_res_1324_; 
v_collapsed_boxed_1322_ = lean_unbox(v_collapsed_1310_);
v_clsEnabled_boxed_1323_ = lean_unbox(v_clsEnabled_1313_);
v_res_1324_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4(v_cls_1309_, v_collapsed_boxed_1322_, v_tag_1311_, v_opts_1312_, v_clsEnabled_boxed_1323_, v_oldTraces_1314_, v_msg_1315_, v_resStartStop_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec_ref(v_opts_1312_);
return v_res_1324_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3(void){
_start:
{
lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; 
v___x_1329_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_1330_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__2));
v___x_1331_ = l_Lean_Name_append(v___x_1330_, v___x_1329_);
return v___x_1331_;
}
}
static double _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4(void){
_start:
{
lean_object* v___x_1332_; double v___x_1333_; 
v___x_1332_ = lean_unsigned_to_nat(1000000000u);
v___x_1333_ = lean_float_of_nat(v___x_1332_);
return v___x_1333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(lean_object* v_lem_1334_, lean_object* v_goal_1335_, uint8_t v_symm_1336_, lean_object* v_config_1337_, lean_object* v_a_1338_, lean_object* v_a_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_){
_start:
{
lean_object* v_options_1343_; lean_object* v_inheritedTraceOptions_1344_; uint8_t v_hasTrace_1345_; lean_object* v___f_1346_; 
v_options_1343_ = lean_ctor_get(v_a_1340_, 2);
v_inheritedTraceOptions_1344_ = lean_ctor_get(v_a_1340_, 13);
v_hasTrace_1345_ = lean_ctor_get_uint8(v_options_1343_, sizeof(void*)*1);
v___f_1346_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1346_, 0, v_config_1337_);
lean_closure_set(v___f_1346_, 1, v_goal_1335_);
if (v_hasTrace_1345_ == 0)
{
lean_object* v___x_1347_; 
v___x_1347_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(v_symm_1336_, v_lem_1334_, v___f_1346_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
return v___x_1347_;
}
else
{
lean_object* v___f_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; uint8_t v___x_1352_; lean_object* v___y_1354_; lean_object* v___y_1355_; lean_object* v_a_1356_; lean_object* v___y_1369_; lean_object* v___y_1370_; lean_object* v_a_1371_; 
lean_inc_ref(v_lem_1334_);
v___f_1348_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___boxed), 7, 1);
lean_closure_set(v___f_1348_, 0, v_lem_1334_);
v___x_1349_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_1350_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0));
v___x_1351_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_1352_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1344_, v_options_1343_, v___x_1351_);
if (v___x_1352_ == 0)
{
lean_object* v___x_1421_; uint8_t v___x_1422_; 
v___x_1421_ = l_Lean_trace_profiler;
v___x_1422_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_1343_, v___x_1421_);
if (v___x_1422_ == 0)
{
lean_object* v___x_1423_; 
lean_dec_ref(v___f_1348_);
v___x_1423_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(v_symm_1336_, v_lem_1334_, v___f_1346_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
return v___x_1423_;
}
else
{
goto v___jp_1380_;
}
}
else
{
goto v___jp_1380_;
}
v___jp_1353_:
{
lean_object* v___x_1357_; double v___x_1358_; double v___x_1359_; double v___x_1360_; double v___x_1361_; double v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1357_ = lean_io_mono_nanos_now();
v___x_1358_ = lean_float_of_nat(v___y_1355_);
v___x_1359_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4);
v___x_1360_ = lean_float_div(v___x_1358_, v___x_1359_);
v___x_1361_ = lean_float_of_nat(v___x_1357_);
v___x_1362_ = lean_float_div(v___x_1361_, v___x_1359_);
v___x_1363_ = lean_box_float(v___x_1360_);
v___x_1364_ = lean_box_float(v___x_1362_);
v___x_1365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1365_, 0, v___x_1363_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_a_1356_);
lean_ctor_set(v___x_1366_, 1, v___x_1365_);
v___x_1367_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4(v___x_1349_, v_hasTrace_1345_, v___x_1350_, v_options_1343_, v___x_1352_, v___y_1354_, v___f_1348_, v___x_1366_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
return v___x_1367_;
}
v___jp_1368_:
{
lean_object* v___x_1372_; double v___x_1373_; double v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; 
v___x_1372_ = lean_io_get_num_heartbeats();
v___x_1373_ = lean_float_of_nat(v___y_1370_);
v___x_1374_ = lean_float_of_nat(v___x_1372_);
v___x_1375_ = lean_box_float(v___x_1373_);
v___x_1376_ = lean_box_float(v___x_1374_);
v___x_1377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1377_, 0, v___x_1375_);
lean_ctor_set(v___x_1377_, 1, v___x_1376_);
v___x_1378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1378_, 0, v_a_1371_);
lean_ctor_set(v___x_1378_, 1, v___x_1377_);
v___x_1379_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4(v___x_1349_, v_hasTrace_1345_, v___x_1350_, v_options_1343_, v___x_1352_, v___y_1369_, v___f_1348_, v___x_1378_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
return v___x_1379_;
}
v___jp_1380_:
{
lean_object* v___x_1381_; lean_object* v_a_1382_; lean_object* v___x_1383_; uint8_t v___x_1384_; 
v___x_1381_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg(v_a_1341_);
v_a_1382_ = lean_ctor_get(v___x_1381_, 0);
lean_inc(v_a_1382_);
lean_dec_ref(v___x_1381_);
v___x_1383_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1384_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_1343_, v___x_1383_);
if (v___x_1384_ == 0)
{
lean_object* v___x_1385_; lean_object* v___x_1386_; 
v___x_1385_ = lean_io_mono_nanos_now();
v___x_1386_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(v_symm_1336_, v_lem_1334_, v___f_1346_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
if (lean_obj_tag(v___x_1386_) == 0)
{
lean_object* v_a_1387_; lean_object* v___x_1389_; uint8_t v_isShared_1390_; uint8_t v_isSharedCheck_1394_; 
v_a_1387_ = lean_ctor_get(v___x_1386_, 0);
v_isSharedCheck_1394_ = !lean_is_exclusive(v___x_1386_);
if (v_isSharedCheck_1394_ == 0)
{
v___x_1389_ = v___x_1386_;
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
else
{
lean_inc(v_a_1387_);
lean_dec(v___x_1386_);
v___x_1389_ = lean_box(0);
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
v_resetjp_1388_:
{
lean_object* v___x_1392_; 
if (v_isShared_1390_ == 0)
{
lean_ctor_set_tag(v___x_1389_, 1);
v___x_1392_ = v___x_1389_;
goto v_reusejp_1391_;
}
else
{
lean_object* v_reuseFailAlloc_1393_; 
v_reuseFailAlloc_1393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1393_, 0, v_a_1387_);
v___x_1392_ = v_reuseFailAlloc_1393_;
goto v_reusejp_1391_;
}
v_reusejp_1391_:
{
v___y_1354_ = v_a_1382_;
v___y_1355_ = v___x_1385_;
v_a_1356_ = v___x_1392_;
goto v___jp_1353_;
}
}
}
else
{
lean_object* v_a_1395_; lean_object* v___x_1397_; uint8_t v_isShared_1398_; uint8_t v_isSharedCheck_1402_; 
v_a_1395_ = lean_ctor_get(v___x_1386_, 0);
v_isSharedCheck_1402_ = !lean_is_exclusive(v___x_1386_);
if (v_isSharedCheck_1402_ == 0)
{
v___x_1397_ = v___x_1386_;
v_isShared_1398_ = v_isSharedCheck_1402_;
goto v_resetjp_1396_;
}
else
{
lean_inc(v_a_1395_);
lean_dec(v___x_1386_);
v___x_1397_ = lean_box(0);
v_isShared_1398_ = v_isSharedCheck_1402_;
goto v_resetjp_1396_;
}
v_resetjp_1396_:
{
lean_object* v___x_1400_; 
if (v_isShared_1398_ == 0)
{
lean_ctor_set_tag(v___x_1397_, 0);
v___x_1400_ = v___x_1397_;
goto v_reusejp_1399_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v_a_1395_);
v___x_1400_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1399_;
}
v_reusejp_1399_:
{
v___y_1354_ = v_a_1382_;
v___y_1355_ = v___x_1385_;
v_a_1356_ = v___x_1400_;
goto v___jp_1353_;
}
}
}
}
else
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = lean_io_get_num_heartbeats();
v___x_1404_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__1(v_symm_1336_, v_lem_1334_, v___f_1346_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_);
if (lean_obj_tag(v___x_1404_) == 0)
{
lean_object* v_a_1405_; lean_object* v___x_1407_; uint8_t v_isShared_1408_; uint8_t v_isSharedCheck_1412_; 
v_a_1405_ = lean_ctor_get(v___x_1404_, 0);
v_isSharedCheck_1412_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1412_ == 0)
{
v___x_1407_ = v___x_1404_;
v_isShared_1408_ = v_isSharedCheck_1412_;
goto v_resetjp_1406_;
}
else
{
lean_inc(v_a_1405_);
lean_dec(v___x_1404_);
v___x_1407_ = lean_box(0);
v_isShared_1408_ = v_isSharedCheck_1412_;
goto v_resetjp_1406_;
}
v_resetjp_1406_:
{
lean_object* v___x_1410_; 
if (v_isShared_1408_ == 0)
{
lean_ctor_set_tag(v___x_1407_, 1);
v___x_1410_ = v___x_1407_;
goto v_reusejp_1409_;
}
else
{
lean_object* v_reuseFailAlloc_1411_; 
v_reuseFailAlloc_1411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1411_, 0, v_a_1405_);
v___x_1410_ = v_reuseFailAlloc_1411_;
goto v_reusejp_1409_;
}
v_reusejp_1409_:
{
v___y_1369_ = v_a_1382_;
v___y_1370_ = v___x_1403_;
v_a_1371_ = v___x_1410_;
goto v___jp_1368_;
}
}
}
else
{
lean_object* v_a_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1420_; 
v_a_1413_ = lean_ctor_get(v___x_1404_, 0);
v_isSharedCheck_1420_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1420_ == 0)
{
v___x_1415_ = v___x_1404_;
v_isShared_1416_ = v_isSharedCheck_1420_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_a_1413_);
lean_dec(v___x_1404_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1420_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
lean_object* v___x_1418_; 
if (v_isShared_1416_ == 0)
{
lean_ctor_set_tag(v___x_1415_, 0);
v___x_1418_ = v___x_1415_;
goto v_reusejp_1417_;
}
else
{
lean_object* v_reuseFailAlloc_1419_; 
v_reuseFailAlloc_1419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1419_, 0, v_a_1413_);
v___x_1418_ = v_reuseFailAlloc_1419_;
goto v_reusejp_1417_;
}
v_reusejp_1417_:
{
v___y_1369_ = v_a_1382_;
v___y_1370_ = v___x_1403_;
v_a_1371_ = v___x_1418_;
goto v___jp_1368_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___boxed(lean_object* v_lem_1424_, lean_object* v_goal_1425_, lean_object* v_symm_1426_, lean_object* v_config_1427_, lean_object* v_a_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_){
_start:
{
uint8_t v_symm_boxed_1433_; lean_object* v_res_1434_; 
v_symm_boxed_1433_ = lean_unbox(v_symm_1426_);
v_res_1434_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(v_lem_1424_, v_goal_1425_, v_symm_boxed_1433_, v_config_1427_, v_a_1428_, v_a_1429_, v_a_1430_, v_a_1431_);
lean_dec(v_a_1431_);
lean_dec_ref(v_a_1430_);
lean_dec(v_a_1429_);
lean_dec_ref(v_a_1428_);
return v_res_1434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0(lean_object* v_snd_1435_, lean_object* v_goal_1436_, lean_object* v___x_1437_, lean_object* v_as_1438_, lean_object* v_as_x27_1439_, lean_object* v_b_1440_, lean_object* v_a_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_){
_start:
{
lean_object* v___x_1447_; 
v___x_1447_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___redArg(v_snd_1435_, v_goal_1436_, v___x_1437_, v_as_x27_1439_, v_b_1440_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_);
return v___x_1447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0___boxed(lean_object* v_snd_1448_, lean_object* v_goal_1449_, lean_object* v___x_1450_, lean_object* v_as_1451_, lean_object* v_as_x27_1452_, lean_object* v_b_1453_, lean_object* v_a_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_){
_start:
{
lean_object* v_res_1460_; 
v_res_1460_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__0(v_snd_1448_, v_goal_1449_, v___x_1450_, v_as_1451_, v_as_x27_1452_, v_b_1453_, v_a_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_);
lean_dec(v___y_1458_);
lean_dec_ref(v___y_1457_);
lean_dec(v___y_1456_);
lean_dec_ref(v___y_1455_);
lean_dec(v_as_x27_1452_);
lean_dec(v_as_1451_);
return v_res_1460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1(lean_object* v_mvarId_1461_, lean_object* v_val_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
lean_object* v___x_1468_; 
v___x_1468_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___redArg(v_mvarId_1461_, v_val_1462_, v___y_1464_);
return v___x_1468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1___boxed(lean_object* v_mvarId_1469_, lean_object* v_val_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_){
_start:
{
lean_object* v_res_1476_; 
v_res_1476_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1(v_mvarId_1469_, v_val_1470_, v___y_1471_, v___y_1472_, v___y_1473_, v___y_1474_);
lean_dec(v___y_1474_);
lean_dec_ref(v___y_1473_);
lean_dec(v___y_1472_);
lean_dec_ref(v___y_1471_);
return v_res_1476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6(lean_object* v_00_u03b1_1477_, lean_object* v_x_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
lean_object* v___x_1484_; 
v___x_1484_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___redArg(v_x_1478_);
return v___x_1484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6___boxed(lean_object* v_00_u03b1_1485_, lean_object* v_x_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__6(v_00_u03b1_1485_, v_x_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
lean_dec(v___y_1488_);
lean_dec_ref(v___y_1487_);
return v_res_1492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1(lean_object* v_00_u03b2_1493_, lean_object* v_x_1494_, lean_object* v_x_1495_, lean_object* v_x_1496_){
_start:
{
lean_object* v___x_1497_; 
v___x_1497_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(v_x_1494_, v_x_1495_, v_x_1496_);
return v___x_1497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4(lean_object* v_00_u03b2_1498_, lean_object* v_x_1499_, size_t v_x_1500_, size_t v_x_1501_, lean_object* v_x_1502_, lean_object* v_x_1503_){
_start:
{
lean_object* v___x_1504_; 
v___x_1504_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___redArg(v_x_1499_, v_x_1500_, v_x_1501_, v_x_1502_, v_x_1503_);
return v___x_1504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4___boxed(lean_object* v_00_u03b2_1505_, lean_object* v_x_1506_, lean_object* v_x_1507_, lean_object* v_x_1508_, lean_object* v_x_1509_, lean_object* v_x_1510_){
_start:
{
size_t v_x_16021__boxed_1511_; size_t v_x_16022__boxed_1512_; lean_object* v_res_1513_; 
v_x_16021__boxed_1511_ = lean_unbox_usize(v_x_1507_);
lean_dec(v_x_1507_);
v_x_16022__boxed_1512_ = lean_unbox_usize(v_x_1508_);
lean_dec(v_x_1508_);
v_res_1513_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4(v_00_u03b2_1505_, v_x_1506_, v_x_16021__boxed_1511_, v_x_16022__boxed_1512_, v_x_1509_, v_x_1510_);
return v_res_1513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9(lean_object* v_00_u03b2_1514_, lean_object* v_n_1515_, lean_object* v_k_1516_, lean_object* v_v_1517_){
_start:
{
lean_object* v___x_1518_; 
v___x_1518_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9___redArg(v_n_1515_, v_k_1516_, v_v_1517_);
return v___x_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10(lean_object* v_00_u03b2_1519_, size_t v_depth_1520_, lean_object* v_keys_1521_, lean_object* v_vals_1522_, lean_object* v_heq_1523_, lean_object* v_i_1524_, lean_object* v_entries_1525_){
_start:
{
lean_object* v___x_1526_; 
v___x_1526_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___redArg(v_depth_1520_, v_keys_1521_, v_vals_1522_, v_i_1524_, v_entries_1525_);
return v___x_1526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10___boxed(lean_object* v_00_u03b2_1527_, lean_object* v_depth_1528_, lean_object* v_keys_1529_, lean_object* v_vals_1530_, lean_object* v_heq_1531_, lean_object* v_i_1532_, lean_object* v_entries_1533_){
_start:
{
size_t v_depth_boxed_1534_; lean_object* v_res_1535_; 
v_depth_boxed_1534_ = lean_unbox_usize(v_depth_1528_);
lean_dec(v_depth_1528_);
v_res_1535_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__10(v_00_u03b2_1527_, v_depth_boxed_1534_, v_keys_1529_, v_vals_1530_, v_heq_1531_, v_i_1532_, v_entries_1533_);
lean_dec_ref(v_vals_1530_);
lean_dec_ref(v_keys_1529_);
return v_res_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12(lean_object* v_00_u03b2_1536_, lean_object* v_x_1537_, lean_object* v_x_1538_, lean_object* v_x_1539_, lean_object* v_x_1540_){
_start:
{
lean_object* v___x_1541_; 
v___x_1541_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1_spec__4_spec__9_spec__12___redArg(v_x_1537_, v_x_1538_, v_x_1539_, v_x_1540_);
return v___x_1541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg(lean_object* v_msg_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_){
_start:
{
lean_object* v_ref_1548_; lean_object* v___x_1549_; lean_object* v_a_1550_; lean_object* v___x_1552_; uint8_t v_isShared_1553_; uint8_t v_isSharedCheck_1558_; 
v_ref_1548_ = lean_ctor_get(v___y_1545_, 5);
v___x_1549_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msg_1542_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_);
v_a_1550_ = lean_ctor_get(v___x_1549_, 0);
v_isSharedCheck_1558_ = !lean_is_exclusive(v___x_1549_);
if (v_isSharedCheck_1558_ == 0)
{
v___x_1552_ = v___x_1549_;
v_isShared_1553_ = v_isSharedCheck_1558_;
goto v_resetjp_1551_;
}
else
{
lean_inc(v_a_1550_);
lean_dec(v___x_1549_);
v___x_1552_ = lean_box(0);
v_isShared_1553_ = v_isSharedCheck_1558_;
goto v_resetjp_1551_;
}
v_resetjp_1551_:
{
lean_object* v___x_1554_; lean_object* v___x_1556_; 
lean_inc(v_ref_1548_);
v___x_1554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1554_, 0, v_ref_1548_);
lean_ctor_set(v___x_1554_, 1, v_a_1550_);
if (v_isShared_1553_ == 0)
{
lean_ctor_set_tag(v___x_1552_, 1);
lean_ctor_set(v___x_1552_, 0, v___x_1554_);
v___x_1556_ = v___x_1552_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1557_; 
v_reuseFailAlloc_1557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1557_, 0, v___x_1554_);
v___x_1556_ = v_reuseFailAlloc_1557_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
return v___x_1556_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg___boxed(lean_object* v_msg_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
lean_object* v_res_1565_; 
v_res_1565_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg(v_msg_1559_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1562_);
lean_dec(v___y_1561_);
lean_dec_ref(v___y_1560_);
return v_res_1565_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1(void){
_start:
{
lean_object* v___x_1567_; lean_object* v___x_1568_; 
v___x_1567_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__0));
v___x_1568_ = l_Lean_stringToMessageData(v___x_1567_);
return v___x_1568_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3(void){
_start:
{
lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___x_1570_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__2));
v___x_1571_ = l_Lean_stringToMessageData(v___x_1570_);
return v___x_1571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(lean_object* v_rel_x3f_1572_, lean_object* v_e_1573_, uint8_t v_forward_1574_, lean_object* v_a_1575_, lean_object* v_a_1576_, lean_object* v_a_1577_, lean_object* v_a_1578_){
_start:
{
if (lean_obj_tag(v_rel_x3f_1572_) == 1)
{
lean_object* v_val_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1699_; 
v_val_1580_ = lean_ctor_get(v_rel_x3f_1572_, 0);
v_isSharedCheck_1699_ = !lean_is_exclusive(v_rel_x3f_1572_);
if (v_isSharedCheck_1699_ == 0)
{
v___x_1582_ = v_rel_x3f_1572_;
v_isShared_1583_ = v_isSharedCheck_1699_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_val_1580_);
lean_dec(v_rel_x3f_1572_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1699_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
lean_object* v___x_1584_; 
lean_inc(v_a_1578_);
lean_inc_ref(v_a_1577_);
lean_inc(v_a_1576_);
lean_inc_ref(v_a_1575_);
lean_inc(v_val_1580_);
v___x_1584_ = lean_infer_type(v_val_1580_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
if (lean_obj_tag(v___x_1584_) == 0)
{
lean_object* v_a_1585_; lean_object* v___x_1586_; 
v_a_1585_ = lean_ctor_get(v___x_1584_, 0);
lean_inc(v_a_1585_);
lean_dec_ref_known(v___x_1584_, 1);
lean_inc(v_a_1578_);
lean_inc_ref(v_a_1577_);
lean_inc(v_a_1576_);
lean_inc_ref(v_a_1575_);
v___x_1586_ = lean_whnf(v_a_1585_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
if (lean_obj_tag(v___x_1586_) == 0)
{
lean_object* v_a_1587_; 
v_a_1587_ = lean_ctor_get(v___x_1586_, 0);
lean_inc(v_a_1587_);
lean_dec_ref_known(v___x_1586_, 1);
if (lean_obj_tag(v_a_1587_) == 7)
{
lean_object* v_body_1588_; 
v_body_1588_ = lean_ctor_get(v_a_1587_, 2);
lean_inc_ref(v_body_1588_);
if (lean_obj_tag(v_body_1588_) == 7)
{
lean_object* v_binderType_1589_; lean_object* v_binderType_1590_; lean_object* v___y_1592_; lean_object* v___y_1593_; lean_object* v___y_1594_; lean_object* v___y_1595_; uint8_t v___x_1666_; 
v_binderType_1589_ = lean_ctor_get(v_a_1587_, 1);
lean_inc_ref(v_binderType_1589_);
lean_dec_ref_known(v_a_1587_, 3);
v_binderType_1590_ = lean_ctor_get(v_body_1588_, 1);
lean_inc_ref(v_binderType_1590_);
lean_dec_ref_known(v_body_1588_, 3);
v___x_1666_ = l_Lean_Expr_hasLooseBVars(v_binderType_1590_);
if (v___x_1666_ == 0)
{
v___y_1592_ = v_a_1575_;
v___y_1593_ = v_a_1576_;
v___y_1594_ = v_a_1577_;
v___y_1595_ = v_a_1578_;
goto v___jp_1591_;
}
else
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v_a_1673_; lean_object* v___x_1675_; uint8_t v_isShared_1676_; uint8_t v_isSharedCheck_1680_; 
lean_dec_ref(v_binderType_1590_);
lean_dec_ref(v_binderType_1589_);
lean_del_object(v___x_1582_);
lean_dec_ref(v_e_1573_);
v___x_1667_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__1);
v___x_1668_ = l_Lean_MessageData_ofExpr(v_val_1580_);
v___x_1669_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1669_, 0, v___x_1667_);
lean_ctor_set(v___x_1669_, 1, v___x_1668_);
v___x_1670_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___closed__3);
v___x_1671_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1671_, 0, v___x_1669_);
lean_ctor_set(v___x_1671_, 1, v___x_1670_);
v___x_1672_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg(v___x_1671_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
v_a_1673_ = lean_ctor_get(v___x_1672_, 0);
v_isSharedCheck_1680_ = !lean_is_exclusive(v___x_1672_);
if (v_isSharedCheck_1680_ == 0)
{
v___x_1675_ = v___x_1672_;
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
else
{
lean_inc(v_a_1673_);
lean_dec(v___x_1672_);
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
v___jp_1591_:
{
if (v_forward_1574_ == 0)
{
lean_object* v___x_1597_; 
lean_dec_ref(v_binderType_1590_);
if (v_isShared_1583_ == 0)
{
lean_ctor_set(v___x_1582_, 0, v_binderType_1589_);
v___x_1597_ = v___x_1582_;
goto v_reusejp_1596_;
}
else
{
lean_object* v_reuseFailAlloc_1630_; 
v_reuseFailAlloc_1630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1630_, 0, v_binderType_1589_);
v___x_1597_ = v_reuseFailAlloc_1630_;
goto v_reusejp_1596_;
}
v_reusejp_1596_:
{
uint8_t v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; 
v___x_1598_ = 0;
v___x_1599_ = lean_box(0);
v___x_1600_ = l_Lean_Meta_mkFreshExprMVar(v___x_1597_, v___x_1598_, v___x_1599_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1600_) == 0)
{
lean_object* v_a_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; 
v_a_1601_ = lean_ctor_get(v___x_1600_, 0);
lean_inc_n(v_a_1601_, 2);
lean_dec_ref_known(v___x_1600_, 1);
v___x_1602_ = l_Lean_mkAppB(v_val_1580_, v_a_1601_, v_e_1573_);
v___x_1603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1603_, 0, v___x_1602_);
v___x_1604_ = l_Lean_Meta_mkFreshExprMVar(v___x_1603_, v___x_1598_, v___x_1599_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1604_) == 0)
{
lean_object* v_a_1605_; lean_object* v___x_1607_; uint8_t v_isShared_1608_; uint8_t v_isSharedCheck_1613_; 
v_a_1605_ = lean_ctor_get(v___x_1604_, 0);
v_isSharedCheck_1613_ = !lean_is_exclusive(v___x_1604_);
if (v_isSharedCheck_1613_ == 0)
{
v___x_1607_ = v___x_1604_;
v_isShared_1608_ = v_isSharedCheck_1613_;
goto v_resetjp_1606_;
}
else
{
lean_inc(v_a_1605_);
lean_dec(v___x_1604_);
v___x_1607_ = lean_box(0);
v_isShared_1608_ = v_isSharedCheck_1613_;
goto v_resetjp_1606_;
}
v_resetjp_1606_:
{
lean_object* v___x_1609_; lean_object* v___x_1611_; 
v___x_1609_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1609_, 0, v_a_1601_);
lean_ctor_set(v___x_1609_, 1, v_a_1605_);
if (v_isShared_1608_ == 0)
{
lean_ctor_set(v___x_1607_, 0, v___x_1609_);
v___x_1611_ = v___x_1607_;
goto v_reusejp_1610_;
}
else
{
lean_object* v_reuseFailAlloc_1612_; 
v_reuseFailAlloc_1612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1612_, 0, v___x_1609_);
v___x_1611_ = v_reuseFailAlloc_1612_;
goto v_reusejp_1610_;
}
v_reusejp_1610_:
{
return v___x_1611_;
}
}
}
else
{
lean_object* v_a_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1621_; 
lean_dec(v_a_1601_);
v_a_1614_ = lean_ctor_get(v___x_1604_, 0);
v_isSharedCheck_1621_ = !lean_is_exclusive(v___x_1604_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1616_ = v___x_1604_;
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_a_1614_);
lean_dec(v___x_1604_);
v___x_1616_ = lean_box(0);
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
v_resetjp_1615_:
{
lean_object* v___x_1619_; 
if (v_isShared_1617_ == 0)
{
v___x_1619_ = v___x_1616_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_a_1614_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
return v___x_1619_;
}
}
}
}
else
{
lean_object* v_a_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1629_; 
lean_dec(v_val_1580_);
lean_dec_ref(v_e_1573_);
v_a_1622_ = lean_ctor_get(v___x_1600_, 0);
v_isSharedCheck_1629_ = !lean_is_exclusive(v___x_1600_);
if (v_isSharedCheck_1629_ == 0)
{
v___x_1624_ = v___x_1600_;
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_a_1622_);
lean_dec(v___x_1600_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1627_; 
if (v_isShared_1625_ == 0)
{
v___x_1627_ = v___x_1624_;
goto v_reusejp_1626_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v_a_1622_);
v___x_1627_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1626_;
}
v_reusejp_1626_:
{
return v___x_1627_;
}
}
}
}
}
else
{
lean_object* v___x_1632_; 
lean_dec_ref(v_binderType_1589_);
if (v_isShared_1583_ == 0)
{
lean_ctor_set(v___x_1582_, 0, v_binderType_1590_);
v___x_1632_ = v___x_1582_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1665_; 
v_reuseFailAlloc_1665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1665_, 0, v_binderType_1590_);
v___x_1632_ = v_reuseFailAlloc_1665_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
uint8_t v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; 
v___x_1633_ = 0;
v___x_1634_ = lean_box(0);
v___x_1635_ = l_Lean_Meta_mkFreshExprMVar(v___x_1632_, v___x_1633_, v___x_1634_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1635_) == 0)
{
lean_object* v_a_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; 
v_a_1636_ = lean_ctor_get(v___x_1635_, 0);
lean_inc_n(v_a_1636_, 2);
lean_dec_ref_known(v___x_1635_, 1);
v___x_1637_ = l_Lean_mkAppB(v_val_1580_, v_e_1573_, v_a_1636_);
v___x_1638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1637_);
v___x_1639_ = l_Lean_Meta_mkFreshExprMVar(v___x_1638_, v___x_1633_, v___x_1634_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1639_) == 0)
{
lean_object* v_a_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_1648_; 
v_a_1640_ = lean_ctor_get(v___x_1639_, 0);
v_isSharedCheck_1648_ = !lean_is_exclusive(v___x_1639_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1642_ = v___x_1639_;
v_isShared_1643_ = v_isSharedCheck_1648_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_a_1640_);
lean_dec(v___x_1639_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_1648_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
lean_object* v___x_1644_; lean_object* v___x_1646_; 
v___x_1644_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1644_, 0, v_a_1636_);
lean_ctor_set(v___x_1644_, 1, v_a_1640_);
if (v_isShared_1643_ == 0)
{
lean_ctor_set(v___x_1642_, 0, v___x_1644_);
v___x_1646_ = v___x_1642_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v___x_1644_);
v___x_1646_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
return v___x_1646_;
}
}
}
else
{
lean_object* v_a_1649_; lean_object* v___x_1651_; uint8_t v_isShared_1652_; uint8_t v_isSharedCheck_1656_; 
lean_dec(v_a_1636_);
v_a_1649_ = lean_ctor_get(v___x_1639_, 0);
v_isSharedCheck_1656_ = !lean_is_exclusive(v___x_1639_);
if (v_isSharedCheck_1656_ == 0)
{
v___x_1651_ = v___x_1639_;
v_isShared_1652_ = v_isSharedCheck_1656_;
goto v_resetjp_1650_;
}
else
{
lean_inc(v_a_1649_);
lean_dec(v___x_1639_);
v___x_1651_ = lean_box(0);
v_isShared_1652_ = v_isSharedCheck_1656_;
goto v_resetjp_1650_;
}
v_resetjp_1650_:
{
lean_object* v___x_1654_; 
if (v_isShared_1652_ == 0)
{
v___x_1654_ = v___x_1651_;
goto v_reusejp_1653_;
}
else
{
lean_object* v_reuseFailAlloc_1655_; 
v_reuseFailAlloc_1655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1655_, 0, v_a_1649_);
v___x_1654_ = v_reuseFailAlloc_1655_;
goto v_reusejp_1653_;
}
v_reusejp_1653_:
{
return v___x_1654_;
}
}
}
}
else
{
lean_object* v_a_1657_; lean_object* v___x_1659_; uint8_t v_isShared_1660_; uint8_t v_isSharedCheck_1664_; 
lean_dec(v_val_1580_);
lean_dec_ref(v_e_1573_);
v_a_1657_ = lean_ctor_get(v___x_1635_, 0);
v_isSharedCheck_1664_ = !lean_is_exclusive(v___x_1635_);
if (v_isSharedCheck_1664_ == 0)
{
v___x_1659_ = v___x_1635_;
v_isShared_1660_ = v_isSharedCheck_1664_;
goto v_resetjp_1658_;
}
else
{
lean_inc(v_a_1657_);
lean_dec(v___x_1635_);
v___x_1659_ = lean_box(0);
v_isShared_1660_ = v_isSharedCheck_1664_;
goto v_resetjp_1658_;
}
v_resetjp_1658_:
{
lean_object* v___x_1662_; 
if (v_isShared_1660_ == 0)
{
v___x_1662_ = v___x_1659_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1663_; 
v_reuseFailAlloc_1663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1663_, 0, v_a_1657_);
v___x_1662_ = v_reuseFailAlloc_1663_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
return v___x_1662_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1681_; 
lean_dec_ref(v_body_1588_);
lean_dec_ref_known(v_a_1587_, 3);
lean_del_object(v___x_1582_);
lean_dec_ref(v_e_1573_);
v___x_1681_ = l_Lean_Meta_throwFunctionExpected___redArg(v_val_1580_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
return v___x_1681_;
}
}
else
{
lean_object* v___x_1682_; 
lean_dec(v_a_1587_);
lean_del_object(v___x_1582_);
lean_dec_ref(v_e_1573_);
v___x_1682_ = l_Lean_Meta_throwFunctionExpected___redArg(v_val_1580_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
return v___x_1682_;
}
}
else
{
lean_object* v_a_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1690_; 
lean_del_object(v___x_1582_);
lean_dec(v_val_1580_);
lean_dec_ref(v_e_1573_);
v_a_1683_ = lean_ctor_get(v___x_1586_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v___x_1586_);
if (v_isSharedCheck_1690_ == 0)
{
v___x_1685_ = v___x_1586_;
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_a_1683_);
lean_dec(v___x_1586_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
lean_object* v___x_1688_; 
if (v_isShared_1686_ == 0)
{
v___x_1688_ = v___x_1685_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_a_1683_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
else
{
lean_object* v_a_1691_; lean_object* v___x_1693_; uint8_t v_isShared_1694_; uint8_t v_isSharedCheck_1698_; 
lean_del_object(v___x_1582_);
lean_dec(v_val_1580_);
lean_dec_ref(v_e_1573_);
v_a_1691_ = lean_ctor_get(v___x_1584_, 0);
v_isSharedCheck_1698_ = !lean_is_exclusive(v___x_1584_);
if (v_isSharedCheck_1698_ == 0)
{
v___x_1693_ = v___x_1584_;
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
else
{
lean_inc(v_a_1691_);
lean_dec(v___x_1584_);
v___x_1693_ = lean_box(0);
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
v_resetjp_1692_:
{
lean_object* v___x_1696_; 
if (v_isShared_1694_ == 0)
{
v___x_1696_ = v___x_1693_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1697_; 
v_reuseFailAlloc_1697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1697_, 0, v_a_1691_);
v___x_1696_ = v_reuseFailAlloc_1697_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
return v___x_1696_;
}
}
}
}
}
else
{
uint8_t v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; 
lean_dec(v_rel_x3f_1572_);
v___x_1700_ = 0;
v___x_1701_ = lean_box(0);
v___x_1702_ = l_Lean_Meta_mkFreshTypeMVar(v___x_1700_, v___x_1701_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
if (lean_obj_tag(v___x_1702_) == 0)
{
lean_object* v_a_1703_; lean_object* v___y_1705_; 
v_a_1703_ = lean_ctor_get(v___x_1702_, 0);
lean_inc(v_a_1703_);
lean_dec_ref_known(v___x_1702_, 1);
if (v_forward_1574_ == 0)
{
lean_object* v___x_1725_; uint8_t v___x_1726_; lean_object* v___x_1727_; 
v___x_1725_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1));
v___x_1726_ = 0;
lean_inc(v_a_1703_);
v___x_1727_ = l_Lean_Expr_forallE___override(v___x_1725_, v_a_1703_, v_e_1573_, v___x_1726_);
v___y_1705_ = v___x_1727_;
goto v___jp_1704_;
}
else
{
lean_object* v___x_1728_; uint8_t v___x_1729_; lean_object* v___x_1730_; 
v___x_1728_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___lam__0___closed__1));
v___x_1729_ = 0;
lean_inc(v_a_1703_);
v___x_1730_ = l_Lean_Expr_forallE___override(v___x_1728_, v_e_1573_, v_a_1703_, v___x_1729_);
v___y_1705_ = v___x_1730_;
goto v___jp_1704_;
}
v___jp_1704_:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; 
v___x_1706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1706_, 0, v___y_1705_);
v___x_1707_ = l_Lean_Meta_mkFreshExprMVar(v___x_1706_, v___x_1700_, v___x_1701_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; lean_object* v___x_1710_; uint8_t v_isShared_1711_; uint8_t v_isSharedCheck_1716_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1716_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1716_ == 0)
{
v___x_1710_ = v___x_1707_;
v_isShared_1711_ = v_isSharedCheck_1716_;
goto v_resetjp_1709_;
}
else
{
lean_inc(v_a_1708_);
lean_dec(v___x_1707_);
v___x_1710_ = lean_box(0);
v_isShared_1711_ = v_isSharedCheck_1716_;
goto v_resetjp_1709_;
}
v_resetjp_1709_:
{
lean_object* v___x_1712_; lean_object* v___x_1714_; 
v___x_1712_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1712_, 0, v_a_1703_);
lean_ctor_set(v___x_1712_, 1, v_a_1708_);
if (v_isShared_1711_ == 0)
{
lean_ctor_set(v___x_1710_, 0, v___x_1712_);
v___x_1714_ = v___x_1710_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v___x_1712_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
}
else
{
lean_object* v_a_1717_; lean_object* v___x_1719_; uint8_t v_isShared_1720_; uint8_t v_isSharedCheck_1724_; 
lean_dec(v_a_1703_);
v_a_1717_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1724_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1724_ == 0)
{
v___x_1719_ = v___x_1707_;
v_isShared_1720_ = v_isSharedCheck_1724_;
goto v_resetjp_1718_;
}
else
{
lean_inc(v_a_1717_);
lean_dec(v___x_1707_);
v___x_1719_ = lean_box(0);
v_isShared_1720_ = v_isSharedCheck_1724_;
goto v_resetjp_1718_;
}
v_resetjp_1718_:
{
lean_object* v___x_1722_; 
if (v_isShared_1720_ == 0)
{
v___x_1722_ = v___x_1719_;
goto v_reusejp_1721_;
}
else
{
lean_object* v_reuseFailAlloc_1723_; 
v_reuseFailAlloc_1723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1723_, 0, v_a_1717_);
v___x_1722_ = v_reuseFailAlloc_1723_;
goto v_reusejp_1721_;
}
v_reusejp_1721_:
{
return v___x_1722_;
}
}
}
}
}
else
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1738_; 
lean_dec_ref(v_e_1573_);
v_a_1731_ = lean_ctor_get(v___x_1702_, 0);
v_isSharedCheck_1738_ = !lean_is_exclusive(v___x_1702_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1733_ = v___x_1702_;
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1702_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1736_; 
if (v_isShared_1734_ == 0)
{
v___x_1736_ = v___x_1733_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_a_1731_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal___boxed(lean_object* v_rel_x3f_1739_, lean_object* v_e_1740_, lean_object* v_forward_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_){
_start:
{
uint8_t v_forward_boxed_1747_; lean_object* v_res_1748_; 
v_forward_boxed_1747_ = lean_unbox(v_forward_1741_);
v_res_1748_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(v_rel_x3f_1739_, v_e_1740_, v_forward_boxed_1747_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_);
lean_dec(v_a_1745_);
lean_dec_ref(v_a_1744_);
lean_dec(v_a_1743_);
lean_dec_ref(v_a_1742_);
return v_res_1748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0(lean_object* v_00_u03b1_1749_, lean_object* v_msg_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_){
_start:
{
lean_object* v___x_1756_; 
v___x_1756_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___redArg(v_msg_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0___boxed(lean_object* v_00_u03b1_1757_, lean_object* v_msg_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v_res_1764_; 
v_res_1764_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal_spec__0(v_00_u03b1_1757_, v_msg_1758_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27(lean_object* v_e_1768_){
_start:
{
switch(lean_obj_tag(v_e_1768_))
{
case 5:
{
lean_object* v_fn_1769_; 
v_fn_1769_ = lean_ctor_get(v_e_1768_, 0);
if (lean_obj_tag(v_fn_1769_) == 5)
{
lean_object* v_arg_1770_; lean_object* v_fn_1771_; lean_object* v_arg_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; 
v_arg_1770_ = lean_ctor_get(v_e_1768_, 1);
v_fn_1771_ = lean_ctor_get(v_fn_1769_, 0);
v_arg_1772_ = lean_ctor_get(v_fn_1769_, 1);
v___x_1773_ = l_Lean_Expr_getAppFn(v_fn_1771_);
v___x_1774_ = l_Lean_Expr_constName_x3f(v___x_1773_);
lean_dec_ref(v___x_1773_);
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v___x_1775_; 
v___x_1775_ = lean_box(0);
return v___x_1775_;
}
else
{
lean_object* v_val_1776_; lean_object* v___x_1778_; uint8_t v_isShared_1779_; uint8_t v_isSharedCheck_1787_; 
v_val_1776_ = lean_ctor_get(v___x_1774_, 0);
v_isSharedCheck_1787_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1778_ = v___x_1774_;
v_isShared_1779_ = v_isSharedCheck_1787_;
goto v_resetjp_1777_;
}
else
{
lean_inc(v_val_1776_);
lean_dec(v___x_1774_);
v___x_1778_ = lean_box(0);
v_isShared_1779_ = v_isSharedCheck_1787_;
goto v_resetjp_1777_;
}
v_resetjp_1777_:
{
lean_object* v___x_1781_; 
lean_inc_ref(v_fn_1771_);
if (v_isShared_1779_ == 0)
{
lean_ctor_set(v___x_1778_, 0, v_fn_1771_);
v___x_1781_ = v___x_1778_;
goto v_reusejp_1780_;
}
else
{
lean_object* v_reuseFailAlloc_1786_; 
v_reuseFailAlloc_1786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1786_, 0, v_fn_1771_);
v___x_1781_ = v_reuseFailAlloc_1786_;
goto v_reusejp_1780_;
}
v_reusejp_1780_:
{
lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; 
lean_inc_ref(v_arg_1770_);
lean_inc_ref(v_arg_1772_);
v___x_1782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1782_, 0, v_arg_1772_);
lean_ctor_set(v___x_1782_, 1, v_arg_1770_);
v___x_1783_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1783_, 0, v___x_1781_);
lean_ctor_set(v___x_1783_, 1, v___x_1782_);
v___x_1784_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1784_, 0, v_val_1776_);
lean_ctor_set(v___x_1784_, 1, v___x_1783_);
v___x_1785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1785_, 0, v___x_1784_);
return v___x_1785_;
}
}
}
}
else
{
lean_object* v___x_1788_; 
v___x_1788_ = lean_box(0);
return v___x_1788_;
}
}
case 7:
{
lean_object* v_binderType_1789_; lean_object* v_body_1790_; uint8_t v___x_1791_; 
v_binderType_1789_ = lean_ctor_get(v_e_1768_, 1);
v_body_1790_ = lean_ctor_get(v_e_1768_, 2);
v___x_1791_ = l_Lean_Expr_hasLooseBVars(v_body_1790_);
if (v___x_1791_ == 0)
{
lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; 
v___x_1792_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1));
v___x_1793_ = lean_box(0);
lean_inc_ref(v_body_1790_);
lean_inc_ref(v_binderType_1789_);
v___x_1794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1794_, 0, v_binderType_1789_);
lean_ctor_set(v___x_1794_, 1, v_body_1790_);
v___x_1795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1795_, 0, v___x_1793_);
lean_ctor_set(v___x_1795_, 1, v___x_1794_);
v___x_1796_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1796_, 0, v___x_1792_);
lean_ctor_set(v___x_1796_, 1, v___x_1795_);
v___x_1797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1797_, 0, v___x_1796_);
return v___x_1797_;
}
else
{
lean_object* v___x_1798_; 
v___x_1798_ = lean_box(0);
return v___x_1798_;
}
}
default: 
{
lean_object* v___x_1799_; 
v___x_1799_ = lean_box(0);
return v___x_1799_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___boxed(lean_object* v_e_1800_){
_start:
{
lean_object* v_res_1801_; 
v_res_1801_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27(v_e_1800_);
lean_dec_ref(v_e_1800_);
return v_res_1801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(lean_object* v_mvarId_1802_, lean_object* v_val_1803_, lean_object* v___y_1804_){
_start:
{
lean_object* v___x_1806_; lean_object* v_mctx_1807_; lean_object* v_cache_1808_; lean_object* v_zetaDeltaFVarIds_1809_; lean_object* v_postponed_1810_; lean_object* v_diag_1811_; lean_object* v___x_1813_; uint8_t v_isShared_1814_; uint8_t v_isSharedCheck_1839_; 
v___x_1806_ = lean_st_ref_take(v___y_1804_);
v_mctx_1807_ = lean_ctor_get(v___x_1806_, 0);
v_cache_1808_ = lean_ctor_get(v___x_1806_, 1);
v_zetaDeltaFVarIds_1809_ = lean_ctor_get(v___x_1806_, 2);
v_postponed_1810_ = lean_ctor_get(v___x_1806_, 3);
v_diag_1811_ = lean_ctor_get(v___x_1806_, 4);
v_isSharedCheck_1839_ = !lean_is_exclusive(v___x_1806_);
if (v_isSharedCheck_1839_ == 0)
{
v___x_1813_ = v___x_1806_;
v_isShared_1814_ = v_isSharedCheck_1839_;
goto v_resetjp_1812_;
}
else
{
lean_inc(v_diag_1811_);
lean_inc(v_postponed_1810_);
lean_inc(v_zetaDeltaFVarIds_1809_);
lean_inc(v_cache_1808_);
lean_inc(v_mctx_1807_);
lean_dec(v___x_1806_);
v___x_1813_ = lean_box(0);
v_isShared_1814_ = v_isSharedCheck_1839_;
goto v_resetjp_1812_;
}
v_resetjp_1812_:
{
lean_object* v_depth_1815_; lean_object* v_levelAssignDepth_1816_; lean_object* v_lmvarCounter_1817_; lean_object* v_mvarCounter_1818_; lean_object* v_lDecls_1819_; lean_object* v_decls_1820_; lean_object* v_userNames_1821_; lean_object* v_lAssignment_1822_; lean_object* v_eAssignment_1823_; lean_object* v_dAssignment_1824_; lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_1838_; 
v_depth_1815_ = lean_ctor_get(v_mctx_1807_, 0);
v_levelAssignDepth_1816_ = lean_ctor_get(v_mctx_1807_, 1);
v_lmvarCounter_1817_ = lean_ctor_get(v_mctx_1807_, 2);
v_mvarCounter_1818_ = lean_ctor_get(v_mctx_1807_, 3);
v_lDecls_1819_ = lean_ctor_get(v_mctx_1807_, 4);
v_decls_1820_ = lean_ctor_get(v_mctx_1807_, 5);
v_userNames_1821_ = lean_ctor_get(v_mctx_1807_, 6);
v_lAssignment_1822_ = lean_ctor_get(v_mctx_1807_, 7);
v_eAssignment_1823_ = lean_ctor_get(v_mctx_1807_, 8);
v_dAssignment_1824_ = lean_ctor_get(v_mctx_1807_, 9);
v_isSharedCheck_1838_ = !lean_is_exclusive(v_mctx_1807_);
if (v_isSharedCheck_1838_ == 0)
{
v___x_1826_ = v_mctx_1807_;
v_isShared_1827_ = v_isSharedCheck_1838_;
goto v_resetjp_1825_;
}
else
{
lean_inc(v_dAssignment_1824_);
lean_inc(v_eAssignment_1823_);
lean_inc(v_lAssignment_1822_);
lean_inc(v_userNames_1821_);
lean_inc(v_decls_1820_);
lean_inc(v_lDecls_1819_);
lean_inc(v_mvarCounter_1818_);
lean_inc(v_lmvarCounter_1817_);
lean_inc(v_levelAssignDepth_1816_);
lean_inc(v_depth_1815_);
lean_dec(v_mctx_1807_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_1838_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1828_; lean_object* v___x_1830_; 
v___x_1828_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(v_eAssignment_1823_, v_mvarId_1802_, v_val_1803_);
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 8, v___x_1828_);
v___x_1830_ = v___x_1826_;
goto v_reusejp_1829_;
}
else
{
lean_object* v_reuseFailAlloc_1837_; 
v_reuseFailAlloc_1837_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1837_, 0, v_depth_1815_);
lean_ctor_set(v_reuseFailAlloc_1837_, 1, v_levelAssignDepth_1816_);
lean_ctor_set(v_reuseFailAlloc_1837_, 2, v_lmvarCounter_1817_);
lean_ctor_set(v_reuseFailAlloc_1837_, 3, v_mvarCounter_1818_);
lean_ctor_set(v_reuseFailAlloc_1837_, 4, v_lDecls_1819_);
lean_ctor_set(v_reuseFailAlloc_1837_, 5, v_decls_1820_);
lean_ctor_set(v_reuseFailAlloc_1837_, 6, v_userNames_1821_);
lean_ctor_set(v_reuseFailAlloc_1837_, 7, v_lAssignment_1822_);
lean_ctor_set(v_reuseFailAlloc_1837_, 8, v___x_1828_);
lean_ctor_set(v_reuseFailAlloc_1837_, 9, v_dAssignment_1824_);
v___x_1830_ = v_reuseFailAlloc_1837_;
goto v_reusejp_1829_;
}
v_reusejp_1829_:
{
lean_object* v___x_1832_; 
if (v_isShared_1814_ == 0)
{
lean_ctor_set(v___x_1813_, 0, v___x_1830_);
v___x_1832_ = v___x_1813_;
goto v_reusejp_1831_;
}
else
{
lean_object* v_reuseFailAlloc_1836_; 
v_reuseFailAlloc_1836_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1836_, 0, v___x_1830_);
lean_ctor_set(v_reuseFailAlloc_1836_, 1, v_cache_1808_);
lean_ctor_set(v_reuseFailAlloc_1836_, 2, v_zetaDeltaFVarIds_1809_);
lean_ctor_set(v_reuseFailAlloc_1836_, 3, v_postponed_1810_);
lean_ctor_set(v_reuseFailAlloc_1836_, 4, v_diag_1811_);
v___x_1832_ = v_reuseFailAlloc_1836_;
goto v_reusejp_1831_;
}
v_reusejp_1831_:
{
lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; 
v___x_1833_ = lean_st_ref_set(v___y_1804_, v___x_1832_);
v___x_1834_ = lean_box(0);
v___x_1835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1834_);
return v___x_1835_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg___boxed(lean_object* v_mvarId_1840_, lean_object* v_val_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_){
_start:
{
lean_object* v_res_1844_; 
v_res_1844_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_mvarId_1840_, v_val_1841_, v___y_1842_);
lean_dec(v___y_1842_);
return v_res_1844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1(lean_object* v_fst_1845_, lean_object* v_goal_1846_, lean_object* v_snd_1847_, lean_object* v_x_1848_, lean_object* v_x_1849_, lean_object* v_x_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
if (lean_obj_tag(v_x_1848_) == 5)
{
lean_object* v_fn_1860_; lean_object* v_arg_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; 
v_fn_1860_ = lean_ctor_get(v_x_1848_, 0);
lean_inc_ref(v_fn_1860_);
v_arg_1861_ = lean_ctor_get(v_x_1848_, 1);
lean_inc_ref(v_arg_1861_);
lean_dec_ref_known(v_x_1848_, 2);
v___x_1862_ = lean_array_set(v_x_1849_, v_x_1850_, v_arg_1861_);
v___x_1863_ = lean_unsigned_to_nat(1u);
v___x_1864_ = lean_nat_sub(v_x_1850_, v___x_1863_);
lean_dec(v_x_1850_);
v_x_1848_ = v_fn_1860_;
v_x_1849_ = v___x_1862_;
v_x_1850_ = v___x_1864_;
goto _start;
}
else
{
uint8_t v___x_1866_; uint8_t v___x_1867_; uint8_t v___x_1868_; lean_object* v___x_1869_; 
lean_dec(v_x_1850_);
v___x_1866_ = 0;
v___x_1867_ = 1;
v___x_1868_ = 1;
v___x_1869_ = l_Lean_Meta_mkLambdaFVars(v_x_1849_, v_fst_1845_, v___x_1866_, v___x_1867_, v___x_1866_, v___x_1867_, v___x_1868_, v___y_1855_, v___y_1856_, v___y_1857_, v___y_1858_);
lean_dec_ref(v_x_1849_);
if (lean_obj_tag(v___x_1869_) == 0)
{
lean_object* v_a_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1875_; uint8_t v_isShared_1876_; uint8_t v_isSharedCheck_1881_; 
v_a_1870_ = lean_ctor_get(v___x_1869_, 0);
lean_inc(v_a_1870_);
lean_dec_ref_known(v___x_1869_, 1);
v___x_1871_ = l_Lean_Expr_mvarId_x21(v_x_1848_);
lean_dec_ref(v_x_1848_);
v___x_1872_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v___x_1871_, v_a_1870_, v___y_1856_);
lean_dec_ref(v___x_1872_);
v___x_1873_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_goal_1846_, v_snd_1847_, v___y_1856_);
v_isSharedCheck_1881_ = !lean_is_exclusive(v___x_1873_);
if (v_isSharedCheck_1881_ == 0)
{
lean_object* v_unused_1882_; 
v_unused_1882_ = lean_ctor_get(v___x_1873_, 0);
lean_dec(v_unused_1882_);
v___x_1875_ = v___x_1873_;
v_isShared_1876_ = v_isSharedCheck_1881_;
goto v_resetjp_1874_;
}
else
{
lean_dec(v___x_1873_);
v___x_1875_ = lean_box(0);
v_isShared_1876_ = v_isSharedCheck_1881_;
goto v_resetjp_1874_;
}
v_resetjp_1874_:
{
lean_object* v___x_1877_; lean_object* v___x_1879_; 
v___x_1877_ = lean_box(v___x_1867_);
if (v_isShared_1876_ == 0)
{
lean_ctor_set(v___x_1875_, 0, v___x_1877_);
v___x_1879_ = v___x_1875_;
goto v_reusejp_1878_;
}
else
{
lean_object* v_reuseFailAlloc_1880_; 
v_reuseFailAlloc_1880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1880_, 0, v___x_1877_);
v___x_1879_ = v_reuseFailAlloc_1880_;
goto v_reusejp_1878_;
}
v_reusejp_1878_:
{
return v___x_1879_;
}
}
}
else
{
lean_object* v_a_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1890_; 
lean_dec_ref(v_x_1848_);
lean_dec_ref(v_snd_1847_);
lean_dec(v_goal_1846_);
v_a_1883_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1890_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1890_ == 0)
{
v___x_1885_ = v___x_1869_;
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_a_1883_);
lean_dec(v___x_1869_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
lean_object* v___x_1888_; 
if (v_isShared_1886_ == 0)
{
v___x_1888_ = v___x_1885_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1889_; 
v_reuseFailAlloc_1889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1889_, 0, v_a_1883_);
v___x_1888_ = v_reuseFailAlloc_1889_;
goto v_reusejp_1887_;
}
v_reusejp_1887_:
{
return v___x_1888_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1___boxed(lean_object* v_fst_1891_, lean_object* v_goal_1892_, lean_object* v_snd_1893_, lean_object* v_x_1894_, lean_object* v_x_1895_, lean_object* v_x_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_){
_start:
{
lean_object* v_res_1906_; 
v_res_1906_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1(v_fst_1891_, v_goal_1892_, v_snd_1893_, v_x_1894_, v_x_1895_, v_x_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
lean_dec(v___y_1902_);
lean_dec_ref(v___y_1901_);
lean_dec(v___y_1900_);
lean_dec_ref(v___y_1899_);
lean_dec(v___y_1898_);
lean_dec_ref(v___y_1897_);
return v_res_1906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(lean_object* v_e_1907_, lean_object* v___y_1908_){
_start:
{
uint8_t v___x_1910_; 
v___x_1910_ = l_Lean_Expr_hasMVar(v_e_1907_);
if (v___x_1910_ == 0)
{
lean_object* v___x_1911_; 
v___x_1911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1911_, 0, v_e_1907_);
return v___x_1911_;
}
else
{
lean_object* v___x_1912_; lean_object* v_mctx_1913_; lean_object* v___x_1914_; lean_object* v_fst_1915_; lean_object* v_snd_1916_; lean_object* v___x_1917_; lean_object* v_cache_1918_; lean_object* v_zetaDeltaFVarIds_1919_; lean_object* v_postponed_1920_; lean_object* v_diag_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1930_; 
v___x_1912_ = lean_st_ref_get(v___y_1908_);
v_mctx_1913_ = lean_ctor_get(v___x_1912_, 0);
lean_inc_ref(v_mctx_1913_);
lean_dec(v___x_1912_);
v___x_1914_ = l_Lean_instantiateMVarsCore(v_mctx_1913_, v_e_1907_);
v_fst_1915_ = lean_ctor_get(v___x_1914_, 0);
lean_inc(v_fst_1915_);
v_snd_1916_ = lean_ctor_get(v___x_1914_, 1);
lean_inc(v_snd_1916_);
lean_dec_ref(v___x_1914_);
v___x_1917_ = lean_st_ref_take(v___y_1908_);
v_cache_1918_ = lean_ctor_get(v___x_1917_, 1);
v_zetaDeltaFVarIds_1919_ = lean_ctor_get(v___x_1917_, 2);
v_postponed_1920_ = lean_ctor_get(v___x_1917_, 3);
v_diag_1921_ = lean_ctor_get(v___x_1917_, 4);
v_isSharedCheck_1930_ = !lean_is_exclusive(v___x_1917_);
if (v_isSharedCheck_1930_ == 0)
{
lean_object* v_unused_1931_; 
v_unused_1931_ = lean_ctor_get(v___x_1917_, 0);
lean_dec(v_unused_1931_);
v___x_1923_ = v___x_1917_;
v_isShared_1924_ = v_isSharedCheck_1930_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_diag_1921_);
lean_inc(v_postponed_1920_);
lean_inc(v_zetaDeltaFVarIds_1919_);
lean_inc(v_cache_1918_);
lean_dec(v___x_1917_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1930_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___x_1926_; 
if (v_isShared_1924_ == 0)
{
lean_ctor_set(v___x_1923_, 0, v_snd_1916_);
v___x_1926_ = v___x_1923_;
goto v_reusejp_1925_;
}
else
{
lean_object* v_reuseFailAlloc_1929_; 
v_reuseFailAlloc_1929_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1929_, 0, v_snd_1916_);
lean_ctor_set(v_reuseFailAlloc_1929_, 1, v_cache_1918_);
lean_ctor_set(v_reuseFailAlloc_1929_, 2, v_zetaDeltaFVarIds_1919_);
lean_ctor_set(v_reuseFailAlloc_1929_, 3, v_postponed_1920_);
lean_ctor_set(v_reuseFailAlloc_1929_, 4, v_diag_1921_);
v___x_1926_ = v_reuseFailAlloc_1929_;
goto v_reusejp_1925_;
}
v_reusejp_1925_:
{
lean_object* v___x_1927_; lean_object* v___x_1928_; 
v___x_1927_ = lean_st_ref_set(v___y_1908_, v___x_1926_);
v___x_1928_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1928_, 0, v_fst_1915_);
return v___x_1928_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg___boxed(lean_object* v_e_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_){
_start:
{
lean_object* v_res_1935_; 
v_res_1935_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_1932_, v___y_1933_);
lean_dec(v___y_1933_);
return v_res_1935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16(uint8_t v___x_1936_, lean_object* v_as_1937_, size_t v_sz_1938_, size_t v_i_1939_, lean_object* v_b_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
lean_object* v_a_1951_; uint8_t v___x_1955_; 
v___x_1955_ = lean_usize_dec_lt(v_i_1939_, v_sz_1938_);
if (v___x_1955_ == 0)
{
lean_object* v___x_1956_; 
v___x_1956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1956_, 0, v_b_1940_);
return v___x_1956_;
}
else
{
lean_object* v_a_1957_; lean_object* v___x_1958_; 
v_a_1957_ = lean_array_uget_borrowed(v_as_1937_, v_i_1939_);
lean_inc(v_a_1957_);
v___x_1958_ = l_Lean_MVarId_getType(v_a_1957_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
if (lean_obj_tag(v___x_1958_) == 0)
{
lean_object* v_a_1959_; lean_object* v___x_1960_; 
v_a_1959_ = lean_ctor_get(v___x_1958_, 0);
lean_inc_n(v_a_1959_, 2);
lean_dec_ref_known(v___x_1958_, 1);
v___x_1960_ = l_Lean_Meta_isClass_x3f(v_a_1959_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
if (lean_obj_tag(v___x_1960_) == 0)
{
lean_object* v_a_1961_; lean_object* v___x_1962_; 
v_a_1961_ = lean_ctor_get(v___x_1960_, 0);
lean_inc(v_a_1961_);
lean_dec_ref_known(v___x_1960_, 1);
v___x_1962_ = lean_box(0);
if (lean_obj_tag(v_a_1961_) == 0)
{
if (v___x_1936_ == 0)
{
lean_object* v___x_1977_; 
lean_dec(v_a_1959_);
lean_inc(v_a_1957_);
v___x_1977_ = lp_mathlib_Mathlib_Tactic_GCongr_dischargeSide(v_a_1957_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
if (lean_obj_tag(v___x_1977_) == 0)
{
lean_dec_ref_known(v___x_1977_, 1);
v_a_1951_ = v___x_1962_;
goto v___jp_1950_;
}
else
{
return v___x_1977_;
}
}
else
{
goto v___jp_1963_;
}
}
else
{
lean_dec_ref_known(v_a_1961_, 1);
goto v___jp_1963_;
}
v___jp_1963_:
{
lean_object* v___x_1964_; lean_object* v___x_1965_; 
v___x_1964_ = lean_box(0);
v___x_1965_ = l_Lean_Meta_synthInstance_x3f(v_a_1959_, v___x_1964_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
if (lean_obj_tag(v___x_1965_) == 0)
{
lean_object* v_a_1966_; 
v_a_1966_ = lean_ctor_get(v___x_1965_, 0);
lean_inc(v_a_1966_);
lean_dec_ref_known(v___x_1965_, 1);
if (lean_obj_tag(v_a_1966_) == 1)
{
lean_object* v_val_1967_; lean_object* v___x_1968_; 
v_val_1967_ = lean_ctor_get(v_a_1966_, 0);
lean_inc(v_val_1967_);
lean_dec_ref_known(v_a_1966_, 1);
lean_inc(v_a_1957_);
v___x_1968_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_a_1957_, v_val_1967_, v___y_1946_);
if (lean_obj_tag(v___x_1968_) == 0)
{
lean_dec_ref_known(v___x_1968_, 1);
v_a_1951_ = v___x_1962_;
goto v___jp_1950_;
}
else
{
return v___x_1968_;
}
}
else
{
lean_dec(v_a_1966_);
v_a_1951_ = v___x_1962_;
goto v___jp_1950_;
}
}
else
{
lean_object* v_a_1969_; lean_object* v___x_1971_; uint8_t v_isShared_1972_; uint8_t v_isSharedCheck_1976_; 
v_a_1969_ = lean_ctor_get(v___x_1965_, 0);
v_isSharedCheck_1976_ = !lean_is_exclusive(v___x_1965_);
if (v_isSharedCheck_1976_ == 0)
{
v___x_1971_ = v___x_1965_;
v_isShared_1972_ = v_isSharedCheck_1976_;
goto v_resetjp_1970_;
}
else
{
lean_inc(v_a_1969_);
lean_dec(v___x_1965_);
v___x_1971_ = lean_box(0);
v_isShared_1972_ = v_isSharedCheck_1976_;
goto v_resetjp_1970_;
}
v_resetjp_1970_:
{
lean_object* v___x_1974_; 
if (v_isShared_1972_ == 0)
{
v___x_1974_ = v___x_1971_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_1975_; 
v_reuseFailAlloc_1975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1975_, 0, v_a_1969_);
v___x_1974_ = v_reuseFailAlloc_1975_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
return v___x_1974_;
}
}
}
}
}
else
{
lean_object* v_a_1978_; lean_object* v___x_1980_; uint8_t v_isShared_1981_; uint8_t v_isSharedCheck_1985_; 
lean_dec(v_a_1959_);
v_a_1978_ = lean_ctor_get(v___x_1960_, 0);
v_isSharedCheck_1985_ = !lean_is_exclusive(v___x_1960_);
if (v_isSharedCheck_1985_ == 0)
{
v___x_1980_ = v___x_1960_;
v_isShared_1981_ = v_isSharedCheck_1985_;
goto v_resetjp_1979_;
}
else
{
lean_inc(v_a_1978_);
lean_dec(v___x_1960_);
v___x_1980_ = lean_box(0);
v_isShared_1981_ = v_isSharedCheck_1985_;
goto v_resetjp_1979_;
}
v_resetjp_1979_:
{
lean_object* v___x_1983_; 
if (v_isShared_1981_ == 0)
{
v___x_1983_ = v___x_1980_;
goto v_reusejp_1982_;
}
else
{
lean_object* v_reuseFailAlloc_1984_; 
v_reuseFailAlloc_1984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1984_, 0, v_a_1978_);
v___x_1983_ = v_reuseFailAlloc_1984_;
goto v_reusejp_1982_;
}
v_reusejp_1982_:
{
return v___x_1983_;
}
}
}
}
else
{
lean_object* v_a_1986_; lean_object* v___x_1988_; uint8_t v_isShared_1989_; uint8_t v_isSharedCheck_1993_; 
v_a_1986_ = lean_ctor_get(v___x_1958_, 0);
v_isSharedCheck_1993_ = !lean_is_exclusive(v___x_1958_);
if (v_isSharedCheck_1993_ == 0)
{
v___x_1988_ = v___x_1958_;
v_isShared_1989_ = v_isSharedCheck_1993_;
goto v_resetjp_1987_;
}
else
{
lean_inc(v_a_1986_);
lean_dec(v___x_1958_);
v___x_1988_ = lean_box(0);
v_isShared_1989_ = v_isSharedCheck_1993_;
goto v_resetjp_1987_;
}
v_resetjp_1987_:
{
lean_object* v___x_1991_; 
if (v_isShared_1989_ == 0)
{
v___x_1991_ = v___x_1988_;
goto v_reusejp_1990_;
}
else
{
lean_object* v_reuseFailAlloc_1992_; 
v_reuseFailAlloc_1992_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1992_, 0, v_a_1986_);
v___x_1991_ = v_reuseFailAlloc_1992_;
goto v_reusejp_1990_;
}
v_reusejp_1990_:
{
return v___x_1991_;
}
}
}
}
v___jp_1950_:
{
size_t v___x_1952_; size_t v___x_1953_; 
v___x_1952_ = ((size_t)1ULL);
v___x_1953_ = lean_usize_add(v_i_1939_, v___x_1952_);
v_i_1939_ = v___x_1953_;
v_b_1940_ = v_a_1951_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16___boxed(lean_object* v___x_1994_, lean_object* v_as_1995_, lean_object* v_sz_1996_, lean_object* v_i_1997_, lean_object* v_b_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
uint8_t v___x_416464__boxed_2008_; size_t v_sz_boxed_2009_; size_t v_i_boxed_2010_; lean_object* v_res_2011_; 
v___x_416464__boxed_2008_ = lean_unbox(v___x_1994_);
v_sz_boxed_2009_ = lean_unbox_usize(v_sz_1996_);
lean_dec(v_sz_1996_);
v_i_boxed_2010_ = lean_unbox_usize(v_i_1997_);
lean_dec(v_i_1997_);
v_res_2011_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16(v___x_416464__boxed_2008_, v_as_1995_, v_sz_boxed_2009_, v_i_boxed_2010_, v_b_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_, v___y_2006_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___y_2003_);
lean_dec(v___y_2002_);
lean_dec_ref(v___y_2001_);
lean_dec(v___y_2000_);
lean_dec_ref(v___y_1999_);
lean_dec_ref(v_as_1995_);
return v_res_2011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23(lean_object* v_as_2012_, size_t v_i_2013_, size_t v_stop_2014_, lean_object* v_b_2015_){
_start:
{
uint8_t v___x_2016_; 
v___x_2016_ = lean_usize_dec_eq(v_i_2013_, v_stop_2014_);
if (v___x_2016_ == 0)
{
lean_object* v___x_2017_; lean_object* v___x_2018_; size_t v___x_2019_; size_t v___x_2020_; 
v___x_2017_ = lean_array_uget_borrowed(v_as_2012_, v_i_2013_);
lean_inc(v___x_2017_);
v___x_2018_ = l_Lean_LocalContext_addDecl(v_b_2015_, v___x_2017_);
v___x_2019_ = ((size_t)1ULL);
v___x_2020_ = lean_usize_add(v_i_2013_, v___x_2019_);
v_i_2013_ = v___x_2020_;
v_b_2015_ = v___x_2018_;
goto _start;
}
else
{
return v_b_2015_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23___boxed(lean_object* v_as_2022_, lean_object* v_i_2023_, lean_object* v_stop_2024_, lean_object* v_b_2025_){
_start:
{
size_t v_i_boxed_2026_; size_t v_stop_boxed_2027_; lean_object* v_res_2028_; 
v_i_boxed_2026_ = lean_unbox_usize(v_i_2023_);
lean_dec(v_i_2023_);
v_stop_boxed_2027_ = lean_unbox_usize(v_stop_2024_);
lean_dec(v_stop_2024_);
v_res_2028_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23(v_as_2022_, v_i_boxed_2026_, v_stop_boxed_2027_, v_b_2025_);
lean_dec_ref(v_as_2022_);
return v_res_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24(lean_object* v___x_2029_, lean_object* v_as_2030_, size_t v_i_2031_, size_t v_stop_2032_, lean_object* v_b_2033_){
_start:
{
uint8_t v___x_2034_; 
v___x_2034_ = lean_usize_dec_eq(v_i_2031_, v_stop_2032_);
if (v___x_2034_ == 0)
{
lean_object* v___x_2035_; lean_object* v_fst_2036_; lean_object* v_snd_2037_; lean_object* v___y_2039_; lean_object* v___x_2087_; lean_object* v___x_2088_; uint8_t v___x_2089_; 
v___x_2035_ = lean_array_uget_borrowed(v_as_2030_, v_i_2031_);
v_fst_2036_ = lean_ctor_get(v___x_2035_, 0);
v_snd_2037_ = lean_ctor_get(v___x_2035_, 1);
v___x_2087_ = lean_unsigned_to_nat(0u);
v___x_2088_ = lean_array_get_size(v_snd_2037_);
v___x_2089_ = lean_nat_dec_lt(v___x_2087_, v___x_2088_);
if (v___x_2089_ == 0)
{
lean_inc_ref(v___x_2029_);
v___y_2039_ = v___x_2029_;
goto v___jp_2038_;
}
else
{
uint8_t v___x_2090_; 
v___x_2090_ = lean_nat_dec_le(v___x_2088_, v___x_2088_);
if (v___x_2090_ == 0)
{
if (v___x_2089_ == 0)
{
lean_inc_ref(v___x_2029_);
v___y_2039_ = v___x_2029_;
goto v___jp_2038_;
}
else
{
size_t v___x_2091_; size_t v___x_2092_; lean_object* v___x_2093_; 
v___x_2091_ = ((size_t)0ULL);
v___x_2092_ = lean_usize_of_nat(v___x_2088_);
lean_inc_ref(v___x_2029_);
v___x_2093_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23(v_snd_2037_, v___x_2091_, v___x_2092_, v___x_2029_);
v___y_2039_ = v___x_2093_;
goto v___jp_2038_;
}
}
else
{
size_t v___x_2094_; size_t v___x_2095_; lean_object* v___x_2096_; 
v___x_2094_ = ((size_t)0ULL);
v___x_2095_ = lean_usize_of_nat(v___x_2088_);
lean_inc_ref(v___x_2029_);
v___x_2096_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__23(v_snd_2037_, v___x_2094_, v___x_2095_, v___x_2029_);
v___y_2039_ = v___x_2096_;
goto v___jp_2038_;
}
}
v___jp_2038_:
{
lean_object* v_depth_2040_; lean_object* v_levelAssignDepth_2041_; lean_object* v_lmvarCounter_2042_; lean_object* v_mvarCounter_2043_; lean_object* v_lDecls_2044_; lean_object* v_decls_2045_; lean_object* v_userNames_2046_; lean_object* v_lAssignment_2047_; lean_object* v_eAssignment_2048_; lean_object* v_dAssignment_2049_; lean_object* v___x_2050_; lean_object* v___x_2052_; uint8_t v_isShared_2053_; uint8_t v_isSharedCheck_2076_; 
v_depth_2040_ = lean_ctor_get(v_b_2033_, 0);
lean_inc(v_depth_2040_);
v_levelAssignDepth_2041_ = lean_ctor_get(v_b_2033_, 1);
lean_inc(v_levelAssignDepth_2041_);
v_lmvarCounter_2042_ = lean_ctor_get(v_b_2033_, 2);
lean_inc(v_lmvarCounter_2042_);
v_mvarCounter_2043_ = lean_ctor_get(v_b_2033_, 3);
lean_inc(v_mvarCounter_2043_);
v_lDecls_2044_ = lean_ctor_get(v_b_2033_, 4);
lean_inc_ref(v_lDecls_2044_);
v_decls_2045_ = lean_ctor_get(v_b_2033_, 5);
lean_inc_ref(v_decls_2045_);
v_userNames_2046_ = lean_ctor_get(v_b_2033_, 6);
lean_inc_ref(v_userNames_2046_);
v_lAssignment_2047_ = lean_ctor_get(v_b_2033_, 7);
lean_inc_ref(v_lAssignment_2047_);
v_eAssignment_2048_ = lean_ctor_get(v_b_2033_, 8);
lean_inc_ref(v_eAssignment_2048_);
v_dAssignment_2049_ = lean_ctor_get(v_b_2033_, 9);
lean_inc_ref(v_dAssignment_2049_);
lean_inc(v_fst_2036_);
v___x_2050_ = l_Lean_MetavarContext_getDecl(v_b_2033_, v_fst_2036_);
v_isSharedCheck_2076_ = !lean_is_exclusive(v_b_2033_);
if (v_isSharedCheck_2076_ == 0)
{
lean_object* v_unused_2077_; lean_object* v_unused_2078_; lean_object* v_unused_2079_; lean_object* v_unused_2080_; lean_object* v_unused_2081_; lean_object* v_unused_2082_; lean_object* v_unused_2083_; lean_object* v_unused_2084_; lean_object* v_unused_2085_; lean_object* v_unused_2086_; 
v_unused_2077_ = lean_ctor_get(v_b_2033_, 9);
lean_dec(v_unused_2077_);
v_unused_2078_ = lean_ctor_get(v_b_2033_, 8);
lean_dec(v_unused_2078_);
v_unused_2079_ = lean_ctor_get(v_b_2033_, 7);
lean_dec(v_unused_2079_);
v_unused_2080_ = lean_ctor_get(v_b_2033_, 6);
lean_dec(v_unused_2080_);
v_unused_2081_ = lean_ctor_get(v_b_2033_, 5);
lean_dec(v_unused_2081_);
v_unused_2082_ = lean_ctor_get(v_b_2033_, 4);
lean_dec(v_unused_2082_);
v_unused_2083_ = lean_ctor_get(v_b_2033_, 3);
lean_dec(v_unused_2083_);
v_unused_2084_ = lean_ctor_get(v_b_2033_, 2);
lean_dec(v_unused_2084_);
v_unused_2085_ = lean_ctor_get(v_b_2033_, 1);
lean_dec(v_unused_2085_);
v_unused_2086_ = lean_ctor_get(v_b_2033_, 0);
lean_dec(v_unused_2086_);
v___x_2052_ = v_b_2033_;
v_isShared_2053_ = v_isSharedCheck_2076_;
goto v_resetjp_2051_;
}
else
{
lean_dec(v_b_2033_);
v___x_2052_ = lean_box(0);
v_isShared_2053_ = v_isSharedCheck_2076_;
goto v_resetjp_2051_;
}
v_resetjp_2051_:
{
lean_object* v_userName_2054_; lean_object* v_type_2055_; lean_object* v_depth_2056_; lean_object* v_localInstances_2057_; uint8_t v_kind_2058_; lean_object* v_numScopeArgs_2059_; lean_object* v_index_2060_; lean_object* v___x_2062_; uint8_t v_isShared_2063_; uint8_t v_isSharedCheck_2074_; 
v_userName_2054_ = lean_ctor_get(v___x_2050_, 0);
v_type_2055_ = lean_ctor_get(v___x_2050_, 2);
v_depth_2056_ = lean_ctor_get(v___x_2050_, 3);
v_localInstances_2057_ = lean_ctor_get(v___x_2050_, 4);
v_kind_2058_ = lean_ctor_get_uint8(v___x_2050_, sizeof(void*)*7);
v_numScopeArgs_2059_ = lean_ctor_get(v___x_2050_, 5);
v_index_2060_ = lean_ctor_get(v___x_2050_, 6);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2050_);
if (v_isSharedCheck_2074_ == 0)
{
lean_object* v_unused_2075_; 
v_unused_2075_ = lean_ctor_get(v___x_2050_, 1);
lean_dec(v_unused_2075_);
v___x_2062_ = v___x_2050_;
v_isShared_2063_ = v_isSharedCheck_2074_;
goto v_resetjp_2061_;
}
else
{
lean_inc(v_index_2060_);
lean_inc(v_numScopeArgs_2059_);
lean_inc(v_localInstances_2057_);
lean_inc(v_depth_2056_);
lean_inc(v_type_2055_);
lean_inc(v_userName_2054_);
lean_dec(v___x_2050_);
v___x_2062_ = lean_box(0);
v_isShared_2063_ = v_isSharedCheck_2074_;
goto v_resetjp_2061_;
}
v_resetjp_2061_:
{
lean_object* v___x_2065_; 
if (v_isShared_2063_ == 0)
{
lean_ctor_set(v___x_2062_, 1, v___y_2039_);
v___x_2065_ = v___x_2062_;
goto v_reusejp_2064_;
}
else
{
lean_object* v_reuseFailAlloc_2073_; 
v_reuseFailAlloc_2073_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_2073_, 0, v_userName_2054_);
lean_ctor_set(v_reuseFailAlloc_2073_, 1, v___y_2039_);
lean_ctor_set(v_reuseFailAlloc_2073_, 2, v_type_2055_);
lean_ctor_set(v_reuseFailAlloc_2073_, 3, v_depth_2056_);
lean_ctor_set(v_reuseFailAlloc_2073_, 4, v_localInstances_2057_);
lean_ctor_set(v_reuseFailAlloc_2073_, 5, v_numScopeArgs_2059_);
lean_ctor_set(v_reuseFailAlloc_2073_, 6, v_index_2060_);
lean_ctor_set_uint8(v_reuseFailAlloc_2073_, sizeof(void*)*7, v_kind_2058_);
v___x_2065_ = v_reuseFailAlloc_2073_;
goto v_reusejp_2064_;
}
v_reusejp_2064_:
{
lean_object* v___x_2066_; lean_object* v___x_2068_; 
lean_inc(v_fst_2036_);
v___x_2066_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__1_spec__1___redArg(v_decls_2045_, v_fst_2036_, v___x_2065_);
if (v_isShared_2053_ == 0)
{
lean_ctor_set(v___x_2052_, 5, v___x_2066_);
v___x_2068_ = v___x_2052_;
goto v_reusejp_2067_;
}
else
{
lean_object* v_reuseFailAlloc_2072_; 
v_reuseFailAlloc_2072_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2072_, 0, v_depth_2040_);
lean_ctor_set(v_reuseFailAlloc_2072_, 1, v_levelAssignDepth_2041_);
lean_ctor_set(v_reuseFailAlloc_2072_, 2, v_lmvarCounter_2042_);
lean_ctor_set(v_reuseFailAlloc_2072_, 3, v_mvarCounter_2043_);
lean_ctor_set(v_reuseFailAlloc_2072_, 4, v_lDecls_2044_);
lean_ctor_set(v_reuseFailAlloc_2072_, 5, v___x_2066_);
lean_ctor_set(v_reuseFailAlloc_2072_, 6, v_userNames_2046_);
lean_ctor_set(v_reuseFailAlloc_2072_, 7, v_lAssignment_2047_);
lean_ctor_set(v_reuseFailAlloc_2072_, 8, v_eAssignment_2048_);
lean_ctor_set(v_reuseFailAlloc_2072_, 9, v_dAssignment_2049_);
v___x_2068_ = v_reuseFailAlloc_2072_;
goto v_reusejp_2067_;
}
v_reusejp_2067_:
{
size_t v___x_2069_; size_t v___x_2070_; 
v___x_2069_ = ((size_t)1ULL);
v___x_2070_ = lean_usize_add(v_i_2031_, v___x_2069_);
v_i_2031_ = v___x_2070_;
v_b_2033_ = v___x_2068_;
goto _start;
}
}
}
}
}
}
else
{
lean_dec_ref(v___x_2029_);
return v_b_2033_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24___boxed(lean_object* v___x_2097_, lean_object* v_as_2098_, lean_object* v_i_2099_, lean_object* v_stop_2100_, lean_object* v_b_2101_){
_start:
{
size_t v_i_boxed_2102_; size_t v_stop_boxed_2103_; lean_object* v_res_2104_; 
v_i_boxed_2102_ = lean_unbox_usize(v_i_2099_);
lean_dec(v_i_2099_);
v_stop_boxed_2103_ = lean_unbox_usize(v_stop_2100_);
lean_dec(v_stop_2100_);
v_res_2104_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24(v___x_2097_, v_as_2098_, v_i_boxed_2102_, v_stop_boxed_2103_, v_b_2101_);
lean_dec_ref(v_as_2098_);
return v_res_2104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0(uint8_t v___y_2105_, uint8_t v___x_2106_, uint8_t v_____do__lift_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_){
_start:
{
if (v_____do__lift_2107_ == 0)
{
lean_object* v___x_2117_; lean_object* v___x_2118_; 
v___x_2117_ = lean_box(v___y_2105_);
v___x_2118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2118_, 0, v___x_2117_);
return v___x_2118_;
}
else
{
lean_object* v___x_2119_; lean_object* v___x_2120_; 
v___x_2119_ = lean_box(v___x_2106_);
v___x_2120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2120_, 0, v___x_2119_);
return v___x_2120_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0___boxed(lean_object* v___y_2121_, lean_object* v___x_2122_, lean_object* v_____do__lift_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_){
_start:
{
uint8_t v___y_416692__boxed_2133_; uint8_t v___x_416693__boxed_2134_; uint8_t v_____do__lift_416694__boxed_2135_; lean_object* v_res_2136_; 
v___y_416692__boxed_2133_ = lean_unbox(v___y_2121_);
v___x_416693__boxed_2134_ = lean_unbox(v___x_2122_);
v_____do__lift_416694__boxed_2135_ = lean_unbox(v_____do__lift_2123_);
v_res_2136_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0(v___y_416692__boxed_2133_, v___x_416693__boxed_2134_, v_____do__lift_416694__boxed_2135_, v___y_2124_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_, v___y_2131_);
lean_dec(v___y_2131_);
lean_dec_ref(v___y_2130_);
lean_dec(v___y_2129_);
lean_dec_ref(v___y_2128_);
lean_dec(v___y_2127_);
lean_dec_ref(v___y_2126_);
lean_dec(v___y_2125_);
lean_dec_ref(v___y_2124_);
return v_res_2136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg(lean_object* v_mvarId_2137_, lean_object* v___y_2138_){
_start:
{
lean_object* v___x_2140_; lean_object* v_mctx_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; 
v___x_2140_ = lean_st_ref_get(v___y_2138_);
v_mctx_2141_ = lean_ctor_get(v___x_2140_, 0);
lean_inc_ref(v_mctx_2141_);
lean_dec(v___x_2140_);
v___x_2142_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_2141_, v_mvarId_2137_);
lean_dec_ref(v_mctx_2141_);
v___x_2143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2143_, 0, v___x_2142_);
return v___x_2143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg___boxed(lean_object* v_mvarId_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_){
_start:
{
lean_object* v_res_2147_; 
v_res_2147_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg(v_mvarId_2144_, v___y_2145_);
lean_dec(v___y_2145_);
lean_dec(v_mvarId_2144_);
return v_res_2147_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21(lean_object* v___x_2148_, uint8_t v___y_2149_, lean_object* v___x_2150_, lean_object* v___x_2151_, lean_object* v_as_2152_, size_t v_i_2153_, size_t v_stop_2154_){
_start:
{
uint8_t v___x_2155_; 
v___x_2155_ = lean_usize_dec_eq(v_i_2153_, v_stop_2154_);
if (v___x_2155_ == 0)
{
uint8_t v___x_2156_; uint8_t v___y_2158_; lean_object* v___x_2162_; uint8_t v___x_2163_; 
v___x_2156_ = 1;
v___x_2162_ = lean_array_uget_borrowed(v_as_2152_, v_i_2153_);
v___x_2163_ = l_Lean_LocalContext_contains(v___x_2148_, v___x_2162_);
if (v___x_2163_ == 0)
{
v___y_2158_ = v___y_2149_;
goto v___jp_2157_;
}
else
{
uint8_t v___x_2164_; 
v___x_2164_ = lean_nat_dec_eq(v___x_2150_, v___x_2151_);
v___y_2158_ = v___x_2164_;
goto v___jp_2157_;
}
v___jp_2157_:
{
if (v___y_2158_ == 0)
{
size_t v___x_2159_; size_t v___x_2160_; 
v___x_2159_ = ((size_t)1ULL);
v___x_2160_ = lean_usize_add(v_i_2153_, v___x_2159_);
v_i_2153_ = v___x_2160_;
goto _start;
}
else
{
return v___x_2156_;
}
}
}
else
{
uint8_t v___x_2165_; 
v___x_2165_ = 0;
return v___x_2165_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21___boxed(lean_object* v___x_2166_, lean_object* v___y_2167_, lean_object* v___x_2168_, lean_object* v___x_2169_, lean_object* v_as_2170_, lean_object* v_i_2171_, lean_object* v_stop_2172_){
_start:
{
uint8_t v___y_416749__boxed_2173_; size_t v_i_boxed_2174_; size_t v_stop_boxed_2175_; uint8_t v_res_2176_; lean_object* v_r_2177_; 
v___y_416749__boxed_2173_ = lean_unbox(v___y_2167_);
v_i_boxed_2174_ = lean_unbox_usize(v_i_2171_);
lean_dec(v_i_2171_);
v_stop_boxed_2175_ = lean_unbox_usize(v_stop_2172_);
lean_dec(v_stop_2172_);
v_res_2176_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21(v___x_2166_, v___y_416749__boxed_2173_, v___x_2168_, v___x_2169_, v_as_2170_, v_i_boxed_2174_, v_stop_boxed_2175_);
lean_dec_ref(v_as_2170_);
lean_dec(v___x_2169_);
lean_dec(v___x_2168_);
lean_dec_ref(v___x_2166_);
v_r_2177_ = lean_box(v_res_2176_);
return v_r_2177_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0(void){
_start:
{
lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; 
v___x_2178_ = lean_box(0);
v___x_2179_ = lean_unsigned_to_nat(16u);
v___x_2180_ = lean_mk_array(v___x_2179_, v___x_2178_);
return v___x_2180_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1(void){
_start:
{
lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; 
v___x_2181_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__0);
v___x_2182_ = lean_unsigned_to_nat(0u);
v___x_2183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___x_2182_);
lean_ctor_set(v___x_2183_, 1, v___x_2181_);
return v___x_2183_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3(void){
_start:
{
lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; 
v___x_2186_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__2));
v___x_2187_ = lean_box(1);
v___x_2188_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__1);
v___x_2189_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2189_, 0, v___x_2188_);
lean_ctor_set(v___x_2189_, 1, v___x_2187_);
lean_ctor_set(v___x_2189_, 2, v___x_2186_);
return v___x_2189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22(lean_object* v___x_2190_, lean_object* v___x_2191_, lean_object* v___x_2192_, uint8_t v___y_2193_, lean_object* v_as_2194_, size_t v_i_2195_, size_t v_stop_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_){
_start:
{
uint8_t v___x_2206_; 
v___x_2206_ = lean_usize_dec_eq(v_i_2195_, v_stop_2196_);
if (v___x_2206_ == 0)
{
lean_object* v___x_2207_; lean_object* v_fst_2208_; lean_object* v___x_2209_; 
v___x_2207_ = lean_array_uget_borrowed(v_as_2194_, v_i_2195_);
v_fst_2208_ = lean_ctor_get(v___x_2207_, 0);
v___x_2209_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg(v_fst_2208_, v___y_2202_);
if (lean_obj_tag(v___x_2209_) == 0)
{
lean_object* v_a_2210_; lean_object* v___x_2212_; uint8_t v_isShared_2213_; uint8_t v_isSharedCheck_2235_; 
v_a_2210_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2235_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2235_ == 0)
{
v___x_2212_ = v___x_2209_;
v_isShared_2213_ = v_isSharedCheck_2235_;
goto v_resetjp_2211_;
}
else
{
lean_inc(v_a_2210_);
lean_dec(v___x_2209_);
v___x_2212_ = lean_box(0);
v_isShared_2213_ = v_isSharedCheck_2235_;
goto v_resetjp_2211_;
}
v_resetjp_2211_:
{
uint8_t v___x_2214_; uint8_t v_a_2216_; 
v___x_2214_ = 1;
if (lean_obj_tag(v_a_2210_) == 1)
{
lean_object* v_val_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v_fvarIds_2228_; uint8_t v___x_2229_; lean_object* v___x_2230_; uint8_t v___x_2231_; 
v_val_2224_ = lean_ctor_get(v_a_2210_, 0);
lean_inc(v_val_2224_);
lean_dec_ref_known(v_a_2210_, 1);
v___x_2225_ = lean_unsigned_to_nat(0u);
v___x_2226_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__3);
v___x_2227_ = l_Lean_collectFVars(v___x_2226_, v_val_2224_);
v_fvarIds_2228_ = lean_ctor_get(v___x_2227_, 2);
lean_inc_ref(v_fvarIds_2228_);
lean_dec_ref(v___x_2227_);
v___x_2229_ = lean_nat_dec_eq(v___x_2190_, v___x_2191_);
v___x_2230_ = lean_array_get_size(v_fvarIds_2228_);
v___x_2231_ = lean_nat_dec_lt(v___x_2225_, v___x_2230_);
if (v___x_2231_ == 0)
{
lean_dec_ref(v_fvarIds_2228_);
v_a_2216_ = v___x_2229_;
goto v___jp_2215_;
}
else
{
if (v___x_2231_ == 0)
{
lean_dec_ref(v_fvarIds_2228_);
v_a_2216_ = v___x_2229_;
goto v___jp_2215_;
}
else
{
size_t v___x_2232_; size_t v___x_2233_; uint8_t v___x_2234_; 
v___x_2232_ = ((size_t)0ULL);
v___x_2233_ = lean_usize_of_nat(v___x_2230_);
v___x_2234_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__21(v___x_2192_, v___y_2193_, v___x_2190_, v___x_2191_, v_fvarIds_2228_, v___x_2232_, v___x_2233_);
lean_dec_ref(v_fvarIds_2228_);
if (v___x_2234_ == 0)
{
v_a_2216_ = v___x_2229_;
goto v___jp_2215_;
}
else
{
v_a_2216_ = v___y_2193_;
goto v___jp_2215_;
}
}
}
}
else
{
lean_dec(v_a_2210_);
v_a_2216_ = v___y_2193_;
goto v___jp_2215_;
}
v___jp_2215_:
{
if (v_a_2216_ == 0)
{
size_t v___x_2217_; size_t v___x_2218_; 
lean_del_object(v___x_2212_);
v___x_2217_ = ((size_t)1ULL);
v___x_2218_ = lean_usize_add(v_i_2195_, v___x_2217_);
v_i_2195_ = v___x_2218_;
goto _start;
}
else
{
lean_object* v___x_2220_; lean_object* v___x_2222_; 
v___x_2220_ = lean_box(v___x_2214_);
if (v_isShared_2213_ == 0)
{
lean_ctor_set(v___x_2212_, 0, v___x_2220_);
v___x_2222_ = v___x_2212_;
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
}
}
}
else
{
lean_object* v_a_2236_; lean_object* v___x_2238_; uint8_t v_isShared_2239_; uint8_t v_isSharedCheck_2243_; 
v_a_2236_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2243_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2243_ == 0)
{
v___x_2238_ = v___x_2209_;
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
else
{
lean_inc(v_a_2236_);
lean_dec(v___x_2209_);
v___x_2238_ = lean_box(0);
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
v_resetjp_2237_:
{
lean_object* v___x_2241_; 
if (v_isShared_2239_ == 0)
{
v___x_2241_ = v___x_2238_;
goto v_reusejp_2240_;
}
else
{
lean_object* v_reuseFailAlloc_2242_; 
v_reuseFailAlloc_2242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2242_, 0, v_a_2236_);
v___x_2241_ = v_reuseFailAlloc_2242_;
goto v_reusejp_2240_;
}
v_reusejp_2240_:
{
return v___x_2241_;
}
}
}
}
else
{
uint8_t v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; 
v___x_2244_ = 0;
v___x_2245_ = lean_box(v___x_2244_);
v___x_2246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2246_, 0, v___x_2245_);
return v___x_2246_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___boxed(lean_object* v___x_2247_, lean_object* v___x_2248_, lean_object* v___x_2249_, lean_object* v___y_2250_, lean_object* v_as_2251_, lean_object* v_i_2252_, lean_object* v_stop_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_){
_start:
{
uint8_t v___y_416809__boxed_2263_; size_t v_i_boxed_2264_; size_t v_stop_boxed_2265_; lean_object* v_res_2266_; 
v___y_416809__boxed_2263_ = lean_unbox(v___y_2250_);
v_i_boxed_2264_ = lean_unbox_usize(v_i_2252_);
lean_dec(v_i_2252_);
v_stop_boxed_2265_ = lean_unbox_usize(v_stop_2253_);
lean_dec(v_stop_2253_);
v_res_2266_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22(v___x_2247_, v___x_2248_, v___x_2249_, v___y_416809__boxed_2263_, v_as_2251_, v_i_boxed_2264_, v_stop_boxed_2265_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_, v___y_2261_);
lean_dec(v___y_2261_);
lean_dec_ref(v___y_2260_);
lean_dec(v___y_2259_);
lean_dec_ref(v___y_2258_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
lean_dec(v___y_2255_);
lean_dec_ref(v___y_2254_);
lean_dec_ref(v_as_2251_);
lean_dec_ref(v___x_2249_);
lean_dec(v___x_2248_);
lean_dec(v___x_2247_);
return v_res_2266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0(lean_object* v_x_2267_, lean_object* v___y_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_){
_start:
{
lean_object* v___x_2277_; 
lean_inc(v___y_2271_);
lean_inc_ref(v___y_2270_);
lean_inc(v___y_2269_);
lean_inc_ref(v___y_2268_);
v___x_2277_ = lean_apply_9(v_x_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, lean_box(0));
return v___x_2277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0___boxed(lean_object* v_x_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_){
_start:
{
lean_object* v_res_2288_; 
v_res_2288_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0(v_x_2278_, v___y_2279_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_, v___y_2284_, v___y_2285_, v___y_2286_);
lean_dec(v___y_2282_);
lean_dec_ref(v___y_2281_);
lean_dec(v___y_2280_);
lean_dec_ref(v___y_2279_);
return v_res_2288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg(lean_object* v_mvarId_2289_, lean_object* v_x_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_, lean_object* v___y_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_){
_start:
{
lean_object* v___f_2300_; lean_object* v___x_2301_; 
lean_inc(v___y_2294_);
lean_inc_ref(v___y_2293_);
lean_inc(v___y_2292_);
lean_inc_ref(v___y_2291_);
v___f_2300_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_2300_, 0, v_x_2290_);
lean_closure_set(v___f_2300_, 1, v___y_2291_);
lean_closure_set(v___f_2300_, 2, v___y_2292_);
lean_closure_set(v___f_2300_, 3, v___y_2293_);
lean_closure_set(v___f_2300_, 4, v___y_2294_);
v___x_2301_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2289_, v___f_2300_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_);
if (lean_obj_tag(v___x_2301_) == 0)
{
return v___x_2301_;
}
else
{
lean_object* v_a_2302_; lean_object* v___x_2304_; uint8_t v_isShared_2305_; uint8_t v_isSharedCheck_2309_; 
v_a_2302_ = lean_ctor_get(v___x_2301_, 0);
v_isSharedCheck_2309_ = !lean_is_exclusive(v___x_2301_);
if (v_isSharedCheck_2309_ == 0)
{
v___x_2304_ = v___x_2301_;
v_isShared_2305_ = v_isSharedCheck_2309_;
goto v_resetjp_2303_;
}
else
{
lean_inc(v_a_2302_);
lean_dec(v___x_2301_);
v___x_2304_ = lean_box(0);
v_isShared_2305_ = v_isSharedCheck_2309_;
goto v_resetjp_2303_;
}
v_resetjp_2303_:
{
lean_object* v___x_2307_; 
if (v_isShared_2305_ == 0)
{
v___x_2307_ = v___x_2304_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2308_; 
v_reuseFailAlloc_2308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2308_, 0, v_a_2302_);
v___x_2307_ = v_reuseFailAlloc_2308_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
return v___x_2307_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg___boxed(lean_object* v_mvarId_2310_, lean_object* v_x_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_){
_start:
{
lean_object* v_res_2321_; 
v_res_2321_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg(v_mvarId_2310_, v_x_2311_, v___y_2312_, v___y_2313_, v___y_2314_, v___y_2315_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
lean_dec(v___y_2319_);
lean_dec_ref(v___y_2318_);
lean_dec(v___y_2317_);
lean_dec_ref(v___y_2316_);
lean_dec(v___y_2315_);
lean_dec_ref(v___y_2314_);
lean_dec(v___y_2313_);
lean_dec_ref(v___y_2312_);
return v_res_2321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(lean_object* v_cls_2324_, lean_object* v_msg_2325_, lean_object* v___y_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_){
_start:
{
lean_object* v_ref_2331_; lean_object* v___x_2332_; lean_object* v_a_2333_; lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2377_; 
v_ref_2331_ = lean_ctor_get(v___y_2328_, 5);
v___x_2332_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msg_2325_, v___y_2326_, v___y_2327_, v___y_2328_, v___y_2329_);
v_a_2333_ = lean_ctor_get(v___x_2332_, 0);
v_isSharedCheck_2377_ = !lean_is_exclusive(v___x_2332_);
if (v_isSharedCheck_2377_ == 0)
{
v___x_2335_ = v___x_2332_;
v_isShared_2336_ = v_isSharedCheck_2377_;
goto v_resetjp_2334_;
}
else
{
lean_inc(v_a_2333_);
lean_dec(v___x_2332_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2377_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
lean_object* v___x_2337_; lean_object* v_traceState_2338_; lean_object* v_env_2339_; lean_object* v_nextMacroScope_2340_; lean_object* v_ngen_2341_; lean_object* v_auxDeclNGen_2342_; lean_object* v_cache_2343_; lean_object* v_messages_2344_; lean_object* v_infoState_2345_; lean_object* v_snapshotTasks_2346_; lean_object* v___x_2348_; uint8_t v_isShared_2349_; uint8_t v_isSharedCheck_2376_; 
v___x_2337_ = lean_st_ref_take(v___y_2329_);
v_traceState_2338_ = lean_ctor_get(v___x_2337_, 4);
v_env_2339_ = lean_ctor_get(v___x_2337_, 0);
v_nextMacroScope_2340_ = lean_ctor_get(v___x_2337_, 1);
v_ngen_2341_ = lean_ctor_get(v___x_2337_, 2);
v_auxDeclNGen_2342_ = lean_ctor_get(v___x_2337_, 3);
v_cache_2343_ = lean_ctor_get(v___x_2337_, 5);
v_messages_2344_ = lean_ctor_get(v___x_2337_, 6);
v_infoState_2345_ = lean_ctor_get(v___x_2337_, 7);
v_snapshotTasks_2346_ = lean_ctor_get(v___x_2337_, 8);
v_isSharedCheck_2376_ = !lean_is_exclusive(v___x_2337_);
if (v_isSharedCheck_2376_ == 0)
{
v___x_2348_ = v___x_2337_;
v_isShared_2349_ = v_isSharedCheck_2376_;
goto v_resetjp_2347_;
}
else
{
lean_inc(v_snapshotTasks_2346_);
lean_inc(v_infoState_2345_);
lean_inc(v_messages_2344_);
lean_inc(v_cache_2343_);
lean_inc(v_traceState_2338_);
lean_inc(v_auxDeclNGen_2342_);
lean_inc(v_ngen_2341_);
lean_inc(v_nextMacroScope_2340_);
lean_inc(v_env_2339_);
lean_dec(v___x_2337_);
v___x_2348_ = lean_box(0);
v_isShared_2349_ = v_isSharedCheck_2376_;
goto v_resetjp_2347_;
}
v_resetjp_2347_:
{
uint64_t v_tid_2350_; lean_object* v_traces_2351_; lean_object* v___x_2353_; uint8_t v_isShared_2354_; uint8_t v_isSharedCheck_2375_; 
v_tid_2350_ = lean_ctor_get_uint64(v_traceState_2338_, sizeof(void*)*1);
v_traces_2351_ = lean_ctor_get(v_traceState_2338_, 0);
v_isSharedCheck_2375_ = !lean_is_exclusive(v_traceState_2338_);
if (v_isSharedCheck_2375_ == 0)
{
v___x_2353_ = v_traceState_2338_;
v_isShared_2354_ = v_isSharedCheck_2375_;
goto v_resetjp_2352_;
}
else
{
lean_inc(v_traces_2351_);
lean_dec(v_traceState_2338_);
v___x_2353_ = lean_box(0);
v_isShared_2354_ = v_isSharedCheck_2375_;
goto v_resetjp_2352_;
}
v_resetjp_2352_:
{
lean_object* v___x_2355_; double v___x_2356_; uint8_t v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2365_; 
v___x_2355_ = lean_box(0);
v___x_2356_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0);
v___x_2357_ = 0;
v___x_2358_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0));
v___x_2359_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2359_, 0, v_cls_2324_);
lean_ctor_set(v___x_2359_, 1, v___x_2355_);
lean_ctor_set(v___x_2359_, 2, v___x_2358_);
lean_ctor_set_float(v___x_2359_, sizeof(void*)*3, v___x_2356_);
lean_ctor_set_float(v___x_2359_, sizeof(void*)*3 + 8, v___x_2356_);
lean_ctor_set_uint8(v___x_2359_, sizeof(void*)*3 + 16, v___x_2357_);
v___x_2360_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___closed__0));
v___x_2361_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2361_, 0, v___x_2359_);
lean_ctor_set(v___x_2361_, 1, v_a_2333_);
lean_ctor_set(v___x_2361_, 2, v___x_2360_);
lean_inc(v_ref_2331_);
v___x_2362_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2362_, 0, v_ref_2331_);
lean_ctor_set(v___x_2362_, 1, v___x_2361_);
v___x_2363_ = l_Lean_PersistentArray_push___redArg(v_traces_2351_, v___x_2362_);
if (v_isShared_2354_ == 0)
{
lean_ctor_set(v___x_2353_, 0, v___x_2363_);
v___x_2365_ = v___x_2353_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2374_; 
v_reuseFailAlloc_2374_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2374_, 0, v___x_2363_);
lean_ctor_set_uint64(v_reuseFailAlloc_2374_, sizeof(void*)*1, v_tid_2350_);
v___x_2365_ = v_reuseFailAlloc_2374_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
lean_object* v___x_2367_; 
if (v_isShared_2349_ == 0)
{
lean_ctor_set(v___x_2348_, 4, v___x_2365_);
v___x_2367_ = v___x_2348_;
goto v_reusejp_2366_;
}
else
{
lean_object* v_reuseFailAlloc_2373_; 
v_reuseFailAlloc_2373_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2373_, 0, v_env_2339_);
lean_ctor_set(v_reuseFailAlloc_2373_, 1, v_nextMacroScope_2340_);
lean_ctor_set(v_reuseFailAlloc_2373_, 2, v_ngen_2341_);
lean_ctor_set(v_reuseFailAlloc_2373_, 3, v_auxDeclNGen_2342_);
lean_ctor_set(v_reuseFailAlloc_2373_, 4, v___x_2365_);
lean_ctor_set(v_reuseFailAlloc_2373_, 5, v_cache_2343_);
lean_ctor_set(v_reuseFailAlloc_2373_, 6, v_messages_2344_);
lean_ctor_set(v_reuseFailAlloc_2373_, 7, v_infoState_2345_);
lean_ctor_set(v_reuseFailAlloc_2373_, 8, v_snapshotTasks_2346_);
v___x_2367_ = v_reuseFailAlloc_2373_;
goto v_reusejp_2366_;
}
v_reusejp_2366_:
{
lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2371_; 
v___x_2368_ = lean_st_ref_set(v___y_2329_, v___x_2367_);
v___x_2369_ = lean_box(0);
if (v_isShared_2336_ == 0)
{
lean_ctor_set(v___x_2335_, 0, v___x_2369_);
v___x_2371_ = v___x_2335_;
goto v_reusejp_2370_;
}
else
{
lean_object* v_reuseFailAlloc_2372_; 
v_reuseFailAlloc_2372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2372_, 0, v___x_2369_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg___boxed(lean_object* v_cls_2378_, lean_object* v_msg_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_){
_start:
{
lean_object* v_res_2385_; 
v_res_2385_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_2378_, v_msg_2379_, v___y_2380_, v___y_2381_, v___y_2382_, v___y_2383_);
lean_dec(v___y_2383_);
lean_dec_ref(v___y_2382_);
lean_dec(v___y_2381_);
lean_dec_ref(v___y_2380_);
return v_res_2385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(uint8_t v___x_2386_, lean_object* v_____r_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_, lean_object* v___y_2393_, lean_object* v___y_2394_, lean_object* v___y_2395_){
_start:
{
lean_object* v___x_2397_; lean_object* v___x_2398_; lean_object* v___x_2399_; 
v___x_2397_ = lean_box(v___x_2386_);
v___x_2398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2398_, 0, v___x_2397_);
v___x_2399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2399_, 0, v___x_2398_);
return v___x_2399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0___boxed(lean_object* v___x_2400_, lean_object* v_____r_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_, lean_object* v___y_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_){
_start:
{
uint8_t v___x_417110__boxed_2411_; lean_object* v_res_2412_; 
v___x_417110__boxed_2411_ = lean_unbox(v___x_2400_);
v_res_2412_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(v___x_417110__boxed_2411_, v_____r_2401_, v___y_2402_, v___y_2403_, v___y_2404_, v___y_2405_, v___y_2406_, v___y_2407_, v___y_2408_, v___y_2409_);
lean_dec(v___y_2409_);
lean_dec_ref(v___y_2408_);
lean_dec(v___y_2407_);
lean_dec_ref(v___y_2406_);
lean_dec(v___y_2405_);
lean_dec_ref(v___y_2404_);
lean_dec(v___y_2403_);
lean_dec_ref(v___y_2402_);
return v_res_2412_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1(void){
_start:
{
lean_object* v___x_2414_; lean_object* v___x_2415_; 
v___x_2414_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__0));
v___x_2415_ = l_Lean_stringToMessageData(v___x_2414_);
return v___x_2415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1(lean_object* v_lem_2416_, lean_object* v_x_2417_, lean_object* v___y_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_declName_2427_; lean_object* v___x_2428_; uint8_t v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; 
v_declName_2427_ = lean_ctor_get(v_lem_2416_, 1);
lean_inc(v_declName_2427_);
lean_dec_ref(v_lem_2416_);
v___x_2428_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___closed__1);
v___x_2429_ = 0;
v___x_2430_ = l_Lean_MessageData_ofConstName(v_declName_2427_, v___x_2429_);
v___x_2431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2428_);
lean_ctor_set(v___x_2431_, 1, v___x_2430_);
v___x_2432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2432_, 0, v___x_2431_);
return v___x_2432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___boxed(lean_object* v_lem_2433_, lean_object* v_x_2434_, lean_object* v___y_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_){
_start:
{
lean_object* v_res_2444_; 
v_res_2444_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1(v_lem_2433_, v_x_2434_, v___y_2435_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_, v___y_2440_, v___y_2441_, v___y_2442_);
lean_dec(v___y_2442_);
lean_dec_ref(v___y_2441_);
lean_dec(v___y_2440_);
lean_dec_ref(v___y_2439_);
lean_dec(v___y_2438_);
lean_dec_ref(v___y_2437_);
lean_dec(v___y_2436_);
lean_dec_ref(v___y_2435_);
lean_dec_ref(v_x_2434_);
return v_res_2444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18(uint8_t v___x_2445_, lean_object* v_as_2446_, size_t v_sz_2447_, size_t v_i_2448_, lean_object* v_b_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_){
_start:
{
lean_object* v_a_2460_; uint8_t v___x_2464_; 
v___x_2464_ = lean_usize_dec_lt(v_i_2448_, v_sz_2447_);
if (v___x_2464_ == 0)
{
lean_object* v___x_2465_; 
v___x_2465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2465_, 0, v_b_2449_);
return v___x_2465_;
}
else
{
lean_object* v_a_2466_; lean_object* v___x_2467_; 
v_a_2466_ = lean_array_uget_borrowed(v_as_2446_, v_i_2448_);
lean_inc(v_a_2466_);
v___x_2467_ = l_Lean_MVarId_getType(v_a_2466_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2467_) == 0)
{
lean_object* v_a_2468_; lean_object* v___x_2469_; 
v_a_2468_ = lean_ctor_get(v___x_2467_, 0);
lean_inc_n(v_a_2468_, 2);
lean_dec_ref_known(v___x_2467_, 1);
v___x_2469_ = l_Lean_Meta_isClass_x3f(v_a_2468_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2469_) == 0)
{
lean_object* v_a_2470_; lean_object* v___x_2471_; 
v_a_2470_ = lean_ctor_get(v___x_2469_, 0);
lean_inc(v_a_2470_);
lean_dec_ref_known(v___x_2469_, 1);
v___x_2471_ = lean_box(0);
if (lean_obj_tag(v_a_2470_) == 0)
{
lean_dec(v_a_2468_);
goto v___jp_2472_;
}
else
{
lean_dec_ref_known(v_a_2470_, 1);
if (v___x_2445_ == 0)
{
lean_dec(v_a_2468_);
goto v___jp_2472_;
}
else
{
lean_object* v___x_2474_; lean_object* v___x_2475_; 
v___x_2474_ = lean_box(0);
v___x_2475_ = l_Lean_Meta_synthInstance_x3f(v_a_2468_, v___x_2474_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2475_) == 0)
{
lean_object* v_a_2476_; 
v_a_2476_ = lean_ctor_get(v___x_2475_, 0);
lean_inc(v_a_2476_);
lean_dec_ref_known(v___x_2475_, 1);
if (lean_obj_tag(v_a_2476_) == 1)
{
lean_object* v_val_2477_; lean_object* v___x_2478_; 
v_val_2477_ = lean_ctor_get(v_a_2476_, 0);
lean_inc(v_val_2477_);
lean_dec_ref_known(v_a_2476_, 1);
lean_inc(v_a_2466_);
v___x_2478_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_a_2466_, v_val_2477_, v___y_2455_);
if (lean_obj_tag(v___x_2478_) == 0)
{
lean_dec_ref_known(v___x_2478_, 1);
v_a_2460_ = v___x_2471_;
goto v___jp_2459_;
}
else
{
return v___x_2478_;
}
}
else
{
lean_dec(v_a_2476_);
v_a_2460_ = v___x_2471_;
goto v___jp_2459_;
}
}
else
{
lean_object* v_a_2479_; lean_object* v___x_2481_; uint8_t v_isShared_2482_; uint8_t v_isSharedCheck_2486_; 
v_a_2479_ = lean_ctor_get(v___x_2475_, 0);
v_isSharedCheck_2486_ = !lean_is_exclusive(v___x_2475_);
if (v_isSharedCheck_2486_ == 0)
{
v___x_2481_ = v___x_2475_;
v_isShared_2482_ = v_isSharedCheck_2486_;
goto v_resetjp_2480_;
}
else
{
lean_inc(v_a_2479_);
lean_dec(v___x_2475_);
v___x_2481_ = lean_box(0);
v_isShared_2482_ = v_isSharedCheck_2486_;
goto v_resetjp_2480_;
}
v_resetjp_2480_:
{
lean_object* v___x_2484_; 
if (v_isShared_2482_ == 0)
{
v___x_2484_ = v___x_2481_;
goto v_reusejp_2483_;
}
else
{
lean_object* v_reuseFailAlloc_2485_; 
v_reuseFailAlloc_2485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2485_, 0, v_a_2479_);
v___x_2484_ = v_reuseFailAlloc_2485_;
goto v_reusejp_2483_;
}
v_reusejp_2483_:
{
return v___x_2484_;
}
}
}
}
}
v___jp_2472_:
{
lean_object* v___x_2473_; 
lean_inc(v_a_2466_);
v___x_2473_ = lp_mathlib_Mathlib_Tactic_GCongr_dischargeSide(v_a_2466_, v___y_2452_, v___y_2453_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2473_) == 0)
{
lean_dec_ref_known(v___x_2473_, 1);
v_a_2460_ = v___x_2471_;
goto v___jp_2459_;
}
else
{
return v___x_2473_;
}
}
}
else
{
lean_object* v_a_2487_; lean_object* v___x_2489_; uint8_t v_isShared_2490_; uint8_t v_isSharedCheck_2494_; 
lean_dec(v_a_2468_);
v_a_2487_ = lean_ctor_get(v___x_2469_, 0);
v_isSharedCheck_2494_ = !lean_is_exclusive(v___x_2469_);
if (v_isSharedCheck_2494_ == 0)
{
v___x_2489_ = v___x_2469_;
v_isShared_2490_ = v_isSharedCheck_2494_;
goto v_resetjp_2488_;
}
else
{
lean_inc(v_a_2487_);
lean_dec(v___x_2469_);
v___x_2489_ = lean_box(0);
v_isShared_2490_ = v_isSharedCheck_2494_;
goto v_resetjp_2488_;
}
v_resetjp_2488_:
{
lean_object* v___x_2492_; 
if (v_isShared_2490_ == 0)
{
v___x_2492_ = v___x_2489_;
goto v_reusejp_2491_;
}
else
{
lean_object* v_reuseFailAlloc_2493_; 
v_reuseFailAlloc_2493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2493_, 0, v_a_2487_);
v___x_2492_ = v_reuseFailAlloc_2493_;
goto v_reusejp_2491_;
}
v_reusejp_2491_:
{
return v___x_2492_;
}
}
}
}
else
{
lean_object* v_a_2495_; lean_object* v___x_2497_; uint8_t v_isShared_2498_; uint8_t v_isSharedCheck_2502_; 
v_a_2495_ = lean_ctor_get(v___x_2467_, 0);
v_isSharedCheck_2502_ = !lean_is_exclusive(v___x_2467_);
if (v_isSharedCheck_2502_ == 0)
{
v___x_2497_ = v___x_2467_;
v_isShared_2498_ = v_isSharedCheck_2502_;
goto v_resetjp_2496_;
}
else
{
lean_inc(v_a_2495_);
lean_dec(v___x_2467_);
v___x_2497_ = lean_box(0);
v_isShared_2498_ = v_isSharedCheck_2502_;
goto v_resetjp_2496_;
}
v_resetjp_2496_:
{
lean_object* v___x_2500_; 
if (v_isShared_2498_ == 0)
{
v___x_2500_ = v___x_2497_;
goto v_reusejp_2499_;
}
else
{
lean_object* v_reuseFailAlloc_2501_; 
v_reuseFailAlloc_2501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2501_, 0, v_a_2495_);
v___x_2500_ = v_reuseFailAlloc_2501_;
goto v_reusejp_2499_;
}
v_reusejp_2499_:
{
return v___x_2500_;
}
}
}
}
v___jp_2459_:
{
size_t v___x_2461_; size_t v___x_2462_; 
v___x_2461_ = ((size_t)1ULL);
v___x_2462_ = lean_usize_add(v_i_2448_, v___x_2461_);
v_i_2448_ = v___x_2462_;
v_b_2449_ = v_a_2460_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18___boxed(lean_object* v___x_2503_, lean_object* v_as_2504_, lean_object* v_sz_2505_, lean_object* v_i_2506_, lean_object* v_b_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_){
_start:
{
uint8_t v___x_417198__boxed_2517_; size_t v_sz_boxed_2518_; size_t v_i_boxed_2519_; lean_object* v_res_2520_; 
v___x_417198__boxed_2517_ = lean_unbox(v___x_2503_);
v_sz_boxed_2518_ = lean_unbox_usize(v_sz_2505_);
lean_dec(v_sz_2505_);
v_i_boxed_2519_ = lean_unbox_usize(v_i_2506_);
lean_dec(v_i_2506_);
v_res_2520_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18(v___x_417198__boxed_2517_, v_as_2504_, v_sz_boxed_2518_, v_i_boxed_2519_, v_b_2507_, v___y_2508_, v___y_2509_, v___y_2510_, v___y_2511_, v___y_2512_, v___y_2513_, v___y_2514_, v___y_2515_);
lean_dec(v___y_2515_);
lean_dec_ref(v___y_2514_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
lean_dec(v___y_2511_);
lean_dec_ref(v___y_2510_);
lean_dec(v___y_2509_);
lean_dec_ref(v___y_2508_);
lean_dec_ref(v_as_2504_);
return v_res_2520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13(lean_object* v_as_2521_, size_t v_sz_2522_, size_t v_i_2523_, lean_object* v_b_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_){
_start:
{
lean_object* v_a_2535_; uint8_t v___x_2539_; 
v___x_2539_ = lean_usize_dec_lt(v_i_2523_, v_sz_2522_);
if (v___x_2539_ == 0)
{
lean_object* v___x_2540_; 
v___x_2540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2540_, 0, v_b_2524_);
return v___x_2540_;
}
else
{
lean_object* v_a_2541_; lean_object* v___x_2542_; 
v_a_2541_ = lean_array_uget_borrowed(v_as_2521_, v_i_2523_);
lean_inc(v_a_2541_);
v___x_2542_ = l_Lean_MVarId_getType(v_a_2541_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_);
if (lean_obj_tag(v___x_2542_) == 0)
{
lean_object* v_a_2543_; lean_object* v___x_2544_; 
v_a_2543_ = lean_ctor_get(v___x_2542_, 0);
lean_inc_n(v_a_2543_, 2);
lean_dec_ref_known(v___x_2542_, 1);
v___x_2544_ = l_Lean_Meta_isClass_x3f(v_a_2543_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_);
if (lean_obj_tag(v___x_2544_) == 0)
{
lean_object* v_a_2545_; lean_object* v___x_2546_; 
v_a_2545_ = lean_ctor_get(v___x_2544_, 0);
lean_inc(v_a_2545_);
lean_dec_ref_known(v___x_2544_, 1);
v___x_2546_ = lean_box(0);
if (lean_obj_tag(v_a_2545_) == 0)
{
lean_object* v___x_2547_; 
lean_dec(v_a_2543_);
lean_inc(v_a_2541_);
v___x_2547_ = lp_mathlib_Mathlib_Tactic_GCongr_dischargeSide(v_a_2541_, v___y_2527_, v___y_2528_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_);
if (lean_obj_tag(v___x_2547_) == 0)
{
lean_dec_ref_known(v___x_2547_, 1);
v_a_2535_ = v___x_2546_;
goto v___jp_2534_;
}
else
{
return v___x_2547_;
}
}
else
{
lean_object* v___x_2548_; lean_object* v___x_2549_; 
lean_dec_ref_known(v_a_2545_, 1);
v___x_2548_ = lean_box(0);
v___x_2549_ = l_Lean_Meta_synthInstance_x3f(v_a_2543_, v___x_2548_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_);
if (lean_obj_tag(v___x_2549_) == 0)
{
lean_object* v_a_2550_; 
v_a_2550_ = lean_ctor_get(v___x_2549_, 0);
lean_inc(v_a_2550_);
lean_dec_ref_known(v___x_2549_, 1);
if (lean_obj_tag(v_a_2550_) == 1)
{
lean_object* v_val_2551_; lean_object* v___x_2552_; 
v_val_2551_ = lean_ctor_get(v_a_2550_, 0);
lean_inc(v_val_2551_);
lean_dec_ref_known(v_a_2550_, 1);
lean_inc(v_a_2541_);
v___x_2552_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_a_2541_, v_val_2551_, v___y_2530_);
if (lean_obj_tag(v___x_2552_) == 0)
{
lean_dec_ref_known(v___x_2552_, 1);
v_a_2535_ = v___x_2546_;
goto v___jp_2534_;
}
else
{
return v___x_2552_;
}
}
else
{
lean_dec(v_a_2550_);
v_a_2535_ = v___x_2546_;
goto v___jp_2534_;
}
}
else
{
lean_object* v_a_2553_; lean_object* v___x_2555_; uint8_t v_isShared_2556_; uint8_t v_isSharedCheck_2560_; 
v_a_2553_ = lean_ctor_get(v___x_2549_, 0);
v_isSharedCheck_2560_ = !lean_is_exclusive(v___x_2549_);
if (v_isSharedCheck_2560_ == 0)
{
v___x_2555_ = v___x_2549_;
v_isShared_2556_ = v_isSharedCheck_2560_;
goto v_resetjp_2554_;
}
else
{
lean_inc(v_a_2553_);
lean_dec(v___x_2549_);
v___x_2555_ = lean_box(0);
v_isShared_2556_ = v_isSharedCheck_2560_;
goto v_resetjp_2554_;
}
v_resetjp_2554_:
{
lean_object* v___x_2558_; 
if (v_isShared_2556_ == 0)
{
v___x_2558_ = v___x_2555_;
goto v_reusejp_2557_;
}
else
{
lean_object* v_reuseFailAlloc_2559_; 
v_reuseFailAlloc_2559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2559_, 0, v_a_2553_);
v___x_2558_ = v_reuseFailAlloc_2559_;
goto v_reusejp_2557_;
}
v_reusejp_2557_:
{
return v___x_2558_;
}
}
}
}
}
else
{
lean_object* v_a_2561_; lean_object* v___x_2563_; uint8_t v_isShared_2564_; uint8_t v_isSharedCheck_2568_; 
lean_dec(v_a_2543_);
v_a_2561_ = lean_ctor_get(v___x_2544_, 0);
v_isSharedCheck_2568_ = !lean_is_exclusive(v___x_2544_);
if (v_isSharedCheck_2568_ == 0)
{
v___x_2563_ = v___x_2544_;
v_isShared_2564_ = v_isSharedCheck_2568_;
goto v_resetjp_2562_;
}
else
{
lean_inc(v_a_2561_);
lean_dec(v___x_2544_);
v___x_2563_ = lean_box(0);
v_isShared_2564_ = v_isSharedCheck_2568_;
goto v_resetjp_2562_;
}
v_resetjp_2562_:
{
lean_object* v___x_2566_; 
if (v_isShared_2564_ == 0)
{
v___x_2566_ = v___x_2563_;
goto v_reusejp_2565_;
}
else
{
lean_object* v_reuseFailAlloc_2567_; 
v_reuseFailAlloc_2567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2567_, 0, v_a_2561_);
v___x_2566_ = v_reuseFailAlloc_2567_;
goto v_reusejp_2565_;
}
v_reusejp_2565_:
{
return v___x_2566_;
}
}
}
}
else
{
lean_object* v_a_2569_; lean_object* v___x_2571_; uint8_t v_isShared_2572_; uint8_t v_isSharedCheck_2576_; 
v_a_2569_ = lean_ctor_get(v___x_2542_, 0);
v_isSharedCheck_2576_ = !lean_is_exclusive(v___x_2542_);
if (v_isSharedCheck_2576_ == 0)
{
v___x_2571_ = v___x_2542_;
v_isShared_2572_ = v_isSharedCheck_2576_;
goto v_resetjp_2570_;
}
else
{
lean_inc(v_a_2569_);
lean_dec(v___x_2542_);
v___x_2571_ = lean_box(0);
v_isShared_2572_ = v_isSharedCheck_2576_;
goto v_resetjp_2570_;
}
v_resetjp_2570_:
{
lean_object* v___x_2574_; 
if (v_isShared_2572_ == 0)
{
v___x_2574_ = v___x_2571_;
goto v_reusejp_2573_;
}
else
{
lean_object* v_reuseFailAlloc_2575_; 
v_reuseFailAlloc_2575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2575_, 0, v_a_2569_);
v___x_2574_ = v_reuseFailAlloc_2575_;
goto v_reusejp_2573_;
}
v_reusejp_2573_:
{
return v___x_2574_;
}
}
}
}
v___jp_2534_:
{
size_t v___x_2536_; size_t v___x_2537_; 
v___x_2536_ = ((size_t)1ULL);
v___x_2537_ = lean_usize_add(v_i_2523_, v___x_2536_);
v_i_2523_ = v___x_2537_;
v_b_2524_ = v_a_2535_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13___boxed(lean_object* v_as_2577_, lean_object* v_sz_2578_, lean_object* v_i_2579_, lean_object* v_b_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_){
_start:
{
size_t v_sz_boxed_2590_; size_t v_i_boxed_2591_; lean_object* v_res_2592_; 
v_sz_boxed_2590_ = lean_unbox_usize(v_sz_2578_);
lean_dec(v_sz_2578_);
v_i_boxed_2591_ = lean_unbox_usize(v_i_2579_);
lean_dec(v_i_2579_);
v_res_2592_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13(v_as_2577_, v_sz_boxed_2590_, v_i_boxed_2591_, v_b_2580_, v___y_2581_, v___y_2582_, v___y_2583_, v___y_2584_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_);
lean_dec(v___y_2588_);
lean_dec_ref(v___y_2587_);
lean_dec(v___y_2586_);
lean_dec_ref(v___y_2585_);
lean_dec(v___y_2584_);
lean_dec_ref(v___y_2583_);
lean_dec(v___y_2582_);
lean_dec_ref(v___y_2581_);
lean_dec_ref(v_as_2577_);
return v_res_2592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(lean_object* v___y_2593_){
_start:
{
lean_object* v___x_2595_; lean_object* v_traceState_2596_; lean_object* v_traces_2597_; lean_object* v___x_2598_; lean_object* v_traceState_2599_; lean_object* v_env_2600_; lean_object* v_nextMacroScope_2601_; lean_object* v_ngen_2602_; lean_object* v_auxDeclNGen_2603_; lean_object* v_cache_2604_; lean_object* v_messages_2605_; lean_object* v_infoState_2606_; lean_object* v_snapshotTasks_2607_; lean_object* v___x_2609_; uint8_t v_isShared_2610_; uint8_t v_isSharedCheck_2628_; 
v___x_2595_ = lean_st_ref_get(v___y_2593_);
v_traceState_2596_ = lean_ctor_get(v___x_2595_, 4);
lean_inc_ref(v_traceState_2596_);
lean_dec(v___x_2595_);
v_traces_2597_ = lean_ctor_get(v_traceState_2596_, 0);
lean_inc_ref(v_traces_2597_);
lean_dec_ref(v_traceState_2596_);
v___x_2598_ = lean_st_ref_take(v___y_2593_);
v_traceState_2599_ = lean_ctor_get(v___x_2598_, 4);
v_env_2600_ = lean_ctor_get(v___x_2598_, 0);
v_nextMacroScope_2601_ = lean_ctor_get(v___x_2598_, 1);
v_ngen_2602_ = lean_ctor_get(v___x_2598_, 2);
v_auxDeclNGen_2603_ = lean_ctor_get(v___x_2598_, 3);
v_cache_2604_ = lean_ctor_get(v___x_2598_, 5);
v_messages_2605_ = lean_ctor_get(v___x_2598_, 6);
v_infoState_2606_ = lean_ctor_get(v___x_2598_, 7);
v_snapshotTasks_2607_ = lean_ctor_get(v___x_2598_, 8);
v_isSharedCheck_2628_ = !lean_is_exclusive(v___x_2598_);
if (v_isSharedCheck_2628_ == 0)
{
v___x_2609_ = v___x_2598_;
v_isShared_2610_ = v_isSharedCheck_2628_;
goto v_resetjp_2608_;
}
else
{
lean_inc(v_snapshotTasks_2607_);
lean_inc(v_infoState_2606_);
lean_inc(v_messages_2605_);
lean_inc(v_cache_2604_);
lean_inc(v_traceState_2599_);
lean_inc(v_auxDeclNGen_2603_);
lean_inc(v_ngen_2602_);
lean_inc(v_nextMacroScope_2601_);
lean_inc(v_env_2600_);
lean_dec(v___x_2598_);
v___x_2609_ = lean_box(0);
v_isShared_2610_ = v_isSharedCheck_2628_;
goto v_resetjp_2608_;
}
v_resetjp_2608_:
{
uint64_t v_tid_2611_; lean_object* v___x_2613_; uint8_t v_isShared_2614_; uint8_t v_isSharedCheck_2626_; 
v_tid_2611_ = lean_ctor_get_uint64(v_traceState_2599_, sizeof(void*)*1);
v_isSharedCheck_2626_ = !lean_is_exclusive(v_traceState_2599_);
if (v_isSharedCheck_2626_ == 0)
{
lean_object* v_unused_2627_; 
v_unused_2627_ = lean_ctor_get(v_traceState_2599_, 0);
lean_dec(v_unused_2627_);
v___x_2613_ = v_traceState_2599_;
v_isShared_2614_ = v_isSharedCheck_2626_;
goto v_resetjp_2612_;
}
else
{
lean_dec(v_traceState_2599_);
v___x_2613_ = lean_box(0);
v_isShared_2614_ = v_isSharedCheck_2626_;
goto v_resetjp_2612_;
}
v_resetjp_2612_:
{
lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2619_; 
v___x_2615_ = lean_unsigned_to_nat(32u);
v___x_2616_ = lean_mk_empty_array_with_capacity(v___x_2615_);
lean_dec_ref(v___x_2616_);
v___x_2617_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__2___redArg___closed__1);
if (v_isShared_2614_ == 0)
{
lean_ctor_set(v___x_2613_, 0, v___x_2617_);
v___x_2619_ = v___x_2613_;
goto v_reusejp_2618_;
}
else
{
lean_object* v_reuseFailAlloc_2625_; 
v_reuseFailAlloc_2625_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2625_, 0, v___x_2617_);
lean_ctor_set_uint64(v_reuseFailAlloc_2625_, sizeof(void*)*1, v_tid_2611_);
v___x_2619_ = v_reuseFailAlloc_2625_;
goto v_reusejp_2618_;
}
v_reusejp_2618_:
{
lean_object* v___x_2621_; 
if (v_isShared_2610_ == 0)
{
lean_ctor_set(v___x_2609_, 4, v___x_2619_);
v___x_2621_ = v___x_2609_;
goto v_reusejp_2620_;
}
else
{
lean_object* v_reuseFailAlloc_2624_; 
v_reuseFailAlloc_2624_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2624_, 0, v_env_2600_);
lean_ctor_set(v_reuseFailAlloc_2624_, 1, v_nextMacroScope_2601_);
lean_ctor_set(v_reuseFailAlloc_2624_, 2, v_ngen_2602_);
lean_ctor_set(v_reuseFailAlloc_2624_, 3, v_auxDeclNGen_2603_);
lean_ctor_set(v_reuseFailAlloc_2624_, 4, v___x_2619_);
lean_ctor_set(v_reuseFailAlloc_2624_, 5, v_cache_2604_);
lean_ctor_set(v_reuseFailAlloc_2624_, 6, v_messages_2605_);
lean_ctor_set(v_reuseFailAlloc_2624_, 7, v_infoState_2606_);
lean_ctor_set(v_reuseFailAlloc_2624_, 8, v_snapshotTasks_2607_);
v___x_2621_ = v_reuseFailAlloc_2624_;
goto v_reusejp_2620_;
}
v_reusejp_2620_:
{
lean_object* v___x_2622_; lean_object* v___x_2623_; 
v___x_2622_ = lean_st_ref_set(v___y_2593_, v___x_2621_);
v___x_2623_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2623_, 0, v_traces_2597_);
return v___x_2623_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg___boxed(lean_object* v___y_2629_, lean_object* v___y_2630_){
_start:
{
lean_object* v_res_2631_; 
v_res_2631_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(v___y_2629_);
lean_dec(v___y_2629_);
return v_res_2631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0(uint8_t v_anyProgress_2632_, lean_object* v_____r_2633_, lean_object* v___y_2634_, lean_object* v___y_2635_, lean_object* v___y_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_){
_start:
{
lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; 
v___x_2643_ = lean_box(v_anyProgress_2632_);
v___x_2644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2644_, 0, v___x_2643_);
v___x_2645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2645_, 0, v___x_2644_);
return v___x_2645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0___boxed(lean_object* v_anyProgress_2646_, lean_object* v_____r_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_, lean_object* v___y_2656_){
_start:
{
uint8_t v_anyProgress_boxed_2657_; lean_object* v_res_2658_; 
v_anyProgress_boxed_2657_ = lean_unbox(v_anyProgress_2646_);
v_res_2658_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0(v_anyProgress_boxed_2657_, v_____r_2647_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_);
lean_dec(v___y_2655_);
lean_dec_ref(v___y_2654_);
lean_dec(v___y_2653_);
lean_dec_ref(v___y_2652_);
lean_dec(v___y_2651_);
lean_dec_ref(v___y_2650_);
lean_dec(v___y_2649_);
lean_dec_ref(v___y_2648_);
return v_res_2658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(lean_object* v_oldTraces_2659_, lean_object* v_data_2660_, lean_object* v_ref_2661_, lean_object* v_msg_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_){
_start:
{
lean_object* v_fileName_2668_; lean_object* v_fileMap_2669_; lean_object* v_options_2670_; lean_object* v_currRecDepth_2671_; lean_object* v_maxRecDepth_2672_; lean_object* v_ref_2673_; lean_object* v_currNamespace_2674_; lean_object* v_openDecls_2675_; lean_object* v_initHeartbeats_2676_; lean_object* v_maxHeartbeats_2677_; lean_object* v_quotContext_2678_; lean_object* v_currMacroScope_2679_; uint8_t v_diag_2680_; lean_object* v_cancelTk_x3f_2681_; uint8_t v_suppressElabErrors_2682_; lean_object* v_inheritedTraceOptions_2683_; lean_object* v___x_2684_; lean_object* v_traceState_2685_; lean_object* v_traces_2686_; lean_object* v_ref_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; size_t v_sz_2690_; size_t v___x_2691_; lean_object* v___x_2692_; lean_object* v_msg_2693_; lean_object* v___x_2694_; lean_object* v_a_2695_; lean_object* v___x_2697_; uint8_t v_isShared_2698_; uint8_t v_isSharedCheck_2732_; 
v_fileName_2668_ = lean_ctor_get(v___y_2665_, 0);
v_fileMap_2669_ = lean_ctor_get(v___y_2665_, 1);
v_options_2670_ = lean_ctor_get(v___y_2665_, 2);
v_currRecDepth_2671_ = lean_ctor_get(v___y_2665_, 3);
v_maxRecDepth_2672_ = lean_ctor_get(v___y_2665_, 4);
v_ref_2673_ = lean_ctor_get(v___y_2665_, 5);
v_currNamespace_2674_ = lean_ctor_get(v___y_2665_, 6);
v_openDecls_2675_ = lean_ctor_get(v___y_2665_, 7);
v_initHeartbeats_2676_ = lean_ctor_get(v___y_2665_, 8);
v_maxHeartbeats_2677_ = lean_ctor_get(v___y_2665_, 9);
v_quotContext_2678_ = lean_ctor_get(v___y_2665_, 10);
v_currMacroScope_2679_ = lean_ctor_get(v___y_2665_, 11);
v_diag_2680_ = lean_ctor_get_uint8(v___y_2665_, sizeof(void*)*14);
v_cancelTk_x3f_2681_ = lean_ctor_get(v___y_2665_, 12);
v_suppressElabErrors_2682_ = lean_ctor_get_uint8(v___y_2665_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2683_ = lean_ctor_get(v___y_2665_, 13);
v___x_2684_ = lean_st_ref_get(v___y_2666_);
v_traceState_2685_ = lean_ctor_get(v___x_2684_, 4);
lean_inc_ref(v_traceState_2685_);
lean_dec(v___x_2684_);
v_traces_2686_ = lean_ctor_get(v_traceState_2685_, 0);
lean_inc_ref(v_traces_2686_);
lean_dec_ref(v_traceState_2685_);
v_ref_2687_ = l_Lean_replaceRef(v_ref_2661_, v_ref_2673_);
lean_inc_ref(v_inheritedTraceOptions_2683_);
lean_inc(v_cancelTk_x3f_2681_);
lean_inc(v_currMacroScope_2679_);
lean_inc(v_quotContext_2678_);
lean_inc(v_maxHeartbeats_2677_);
lean_inc(v_initHeartbeats_2676_);
lean_inc(v_openDecls_2675_);
lean_inc(v_currNamespace_2674_);
lean_inc(v_maxRecDepth_2672_);
lean_inc(v_currRecDepth_2671_);
lean_inc_ref(v_options_2670_);
lean_inc_ref(v_fileMap_2669_);
lean_inc_ref(v_fileName_2668_);
v___x_2688_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2688_, 0, v_fileName_2668_);
lean_ctor_set(v___x_2688_, 1, v_fileMap_2669_);
lean_ctor_set(v___x_2688_, 2, v_options_2670_);
lean_ctor_set(v___x_2688_, 3, v_currRecDepth_2671_);
lean_ctor_set(v___x_2688_, 4, v_maxRecDepth_2672_);
lean_ctor_set(v___x_2688_, 5, v_ref_2687_);
lean_ctor_set(v___x_2688_, 6, v_currNamespace_2674_);
lean_ctor_set(v___x_2688_, 7, v_openDecls_2675_);
lean_ctor_set(v___x_2688_, 8, v_initHeartbeats_2676_);
lean_ctor_set(v___x_2688_, 9, v_maxHeartbeats_2677_);
lean_ctor_set(v___x_2688_, 10, v_quotContext_2678_);
lean_ctor_set(v___x_2688_, 11, v_currMacroScope_2679_);
lean_ctor_set(v___x_2688_, 12, v_cancelTk_x3f_2681_);
lean_ctor_set(v___x_2688_, 13, v_inheritedTraceOptions_2683_);
lean_ctor_set_uint8(v___x_2688_, sizeof(void*)*14, v_diag_2680_);
lean_ctor_set_uint8(v___x_2688_, sizeof(void*)*14 + 1, v_suppressElabErrors_2682_);
v___x_2689_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2686_);
lean_dec_ref(v_traces_2686_);
v_sz_2690_ = lean_array_size(v___x_2689_);
v___x_2691_ = ((size_t)0ULL);
v___x_2692_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__7(v_sz_2690_, v___x_2691_, v___x_2689_);
v_msg_2693_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2693_, 0, v_data_2660_);
lean_ctor_set(v_msg_2693_, 1, v_msg_2662_);
lean_ctor_set(v_msg_2693_, 2, v___x_2692_);
v___x_2694_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msg_2693_, v___y_2663_, v___y_2664_, v___x_2688_, v___y_2666_);
lean_dec_ref_known(v___x_2688_, 14);
v_a_2695_ = lean_ctor_get(v___x_2694_, 0);
v_isSharedCheck_2732_ = !lean_is_exclusive(v___x_2694_);
if (v_isSharedCheck_2732_ == 0)
{
v___x_2697_ = v___x_2694_;
v_isShared_2698_ = v_isSharedCheck_2732_;
goto v_resetjp_2696_;
}
else
{
lean_inc(v_a_2695_);
lean_dec(v___x_2694_);
v___x_2697_ = lean_box(0);
v_isShared_2698_ = v_isSharedCheck_2732_;
goto v_resetjp_2696_;
}
v_resetjp_2696_:
{
lean_object* v___x_2699_; lean_object* v_traceState_2700_; lean_object* v_env_2701_; lean_object* v_nextMacroScope_2702_; lean_object* v_ngen_2703_; lean_object* v_auxDeclNGen_2704_; lean_object* v_cache_2705_; lean_object* v_messages_2706_; lean_object* v_infoState_2707_; lean_object* v_snapshotTasks_2708_; lean_object* v___x_2710_; uint8_t v_isShared_2711_; uint8_t v_isSharedCheck_2731_; 
v___x_2699_ = lean_st_ref_take(v___y_2666_);
v_traceState_2700_ = lean_ctor_get(v___x_2699_, 4);
v_env_2701_ = lean_ctor_get(v___x_2699_, 0);
v_nextMacroScope_2702_ = lean_ctor_get(v___x_2699_, 1);
v_ngen_2703_ = lean_ctor_get(v___x_2699_, 2);
v_auxDeclNGen_2704_ = lean_ctor_get(v___x_2699_, 3);
v_cache_2705_ = lean_ctor_get(v___x_2699_, 5);
v_messages_2706_ = lean_ctor_get(v___x_2699_, 6);
v_infoState_2707_ = lean_ctor_get(v___x_2699_, 7);
v_snapshotTasks_2708_ = lean_ctor_get(v___x_2699_, 8);
v_isSharedCheck_2731_ = !lean_is_exclusive(v___x_2699_);
if (v_isSharedCheck_2731_ == 0)
{
v___x_2710_ = v___x_2699_;
v_isShared_2711_ = v_isSharedCheck_2731_;
goto v_resetjp_2709_;
}
else
{
lean_inc(v_snapshotTasks_2708_);
lean_inc(v_infoState_2707_);
lean_inc(v_messages_2706_);
lean_inc(v_cache_2705_);
lean_inc(v_traceState_2700_);
lean_inc(v_auxDeclNGen_2704_);
lean_inc(v_ngen_2703_);
lean_inc(v_nextMacroScope_2702_);
lean_inc(v_env_2701_);
lean_dec(v___x_2699_);
v___x_2710_ = lean_box(0);
v_isShared_2711_ = v_isSharedCheck_2731_;
goto v_resetjp_2709_;
}
v_resetjp_2709_:
{
uint64_t v_tid_2712_; lean_object* v___x_2714_; uint8_t v_isShared_2715_; uint8_t v_isSharedCheck_2729_; 
v_tid_2712_ = lean_ctor_get_uint64(v_traceState_2700_, sizeof(void*)*1);
v_isSharedCheck_2729_ = !lean_is_exclusive(v_traceState_2700_);
if (v_isSharedCheck_2729_ == 0)
{
lean_object* v_unused_2730_; 
v_unused_2730_ = lean_ctor_get(v_traceState_2700_, 0);
lean_dec(v_unused_2730_);
v___x_2714_ = v_traceState_2700_;
v_isShared_2715_ = v_isSharedCheck_2729_;
goto v_resetjp_2713_;
}
else
{
lean_dec(v_traceState_2700_);
v___x_2714_ = lean_box(0);
v_isShared_2715_ = v_isSharedCheck_2729_;
goto v_resetjp_2713_;
}
v_resetjp_2713_:
{
lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2719_; 
v___x_2716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2716_, 0, v_ref_2661_);
lean_ctor_set(v___x_2716_, 1, v_a_2695_);
v___x_2717_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2659_, v___x_2716_);
if (v_isShared_2715_ == 0)
{
lean_ctor_set(v___x_2714_, 0, v___x_2717_);
v___x_2719_ = v___x_2714_;
goto v_reusejp_2718_;
}
else
{
lean_object* v_reuseFailAlloc_2728_; 
v_reuseFailAlloc_2728_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2728_, 0, v___x_2717_);
lean_ctor_set_uint64(v_reuseFailAlloc_2728_, sizeof(void*)*1, v_tid_2712_);
v___x_2719_ = v_reuseFailAlloc_2728_;
goto v_reusejp_2718_;
}
v_reusejp_2718_:
{
lean_object* v___x_2721_; 
if (v_isShared_2711_ == 0)
{
lean_ctor_set(v___x_2710_, 4, v___x_2719_);
v___x_2721_ = v___x_2710_;
goto v_reusejp_2720_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v_env_2701_);
lean_ctor_set(v_reuseFailAlloc_2727_, 1, v_nextMacroScope_2702_);
lean_ctor_set(v_reuseFailAlloc_2727_, 2, v_ngen_2703_);
lean_ctor_set(v_reuseFailAlloc_2727_, 3, v_auxDeclNGen_2704_);
lean_ctor_set(v_reuseFailAlloc_2727_, 4, v___x_2719_);
lean_ctor_set(v_reuseFailAlloc_2727_, 5, v_cache_2705_);
lean_ctor_set(v_reuseFailAlloc_2727_, 6, v_messages_2706_);
lean_ctor_set(v_reuseFailAlloc_2727_, 7, v_infoState_2707_);
lean_ctor_set(v_reuseFailAlloc_2727_, 8, v_snapshotTasks_2708_);
v___x_2721_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2720_;
}
v_reusejp_2720_:
{
lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2725_; 
v___x_2722_ = lean_st_ref_set(v___y_2666_, v___x_2721_);
v___x_2723_ = lean_box(0);
if (v_isShared_2698_ == 0)
{
lean_ctor_set(v___x_2697_, 0, v___x_2723_);
v___x_2725_ = v___x_2697_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2726_; 
v_reuseFailAlloc_2726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2726_, 0, v___x_2723_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg___boxed(lean_object* v_oldTraces_2733_, lean_object* v_data_2734_, lean_object* v_ref_2735_, lean_object* v_msg_2736_, lean_object* v___y_2737_, lean_object* v___y_2738_, lean_object* v___y_2739_, lean_object* v___y_2740_, lean_object* v___y_2741_){
_start:
{
lean_object* v_res_2742_; 
v_res_2742_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(v_oldTraces_2733_, v_data_2734_, v_ref_2735_, v_msg_2736_, v___y_2737_, v___y_2738_, v___y_2739_, v___y_2740_);
lean_dec(v___y_2740_);
lean_dec_ref(v___y_2739_);
lean_dec(v___y_2738_);
lean_dec_ref(v___y_2737_);
return v_res_2742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(lean_object* v_x_2743_){
_start:
{
if (lean_obj_tag(v_x_2743_) == 0)
{
lean_object* v_a_2745_; lean_object* v___x_2747_; uint8_t v_isShared_2748_; uint8_t v_isSharedCheck_2752_; 
v_a_2745_ = lean_ctor_get(v_x_2743_, 0);
v_isSharedCheck_2752_ = !lean_is_exclusive(v_x_2743_);
if (v_isSharedCheck_2752_ == 0)
{
v___x_2747_ = v_x_2743_;
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
else
{
lean_inc(v_a_2745_);
lean_dec(v_x_2743_);
v___x_2747_ = lean_box(0);
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
v_resetjp_2746_:
{
lean_object* v___x_2750_; 
if (v_isShared_2748_ == 0)
{
lean_ctor_set_tag(v___x_2747_, 1);
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
else
{
lean_object* v_a_2753_; lean_object* v___x_2755_; uint8_t v_isShared_2756_; uint8_t v_isSharedCheck_2760_; 
v_a_2753_ = lean_ctor_get(v_x_2743_, 0);
v_isSharedCheck_2760_ = !lean_is_exclusive(v_x_2743_);
if (v_isSharedCheck_2760_ == 0)
{
v___x_2755_ = v_x_2743_;
v_isShared_2756_ = v_isSharedCheck_2760_;
goto v_resetjp_2754_;
}
else
{
lean_inc(v_a_2753_);
lean_dec(v_x_2743_);
v___x_2755_ = lean_box(0);
v_isShared_2756_ = v_isSharedCheck_2760_;
goto v_resetjp_2754_;
}
v_resetjp_2754_:
{
lean_object* v___x_2758_; 
if (v_isShared_2756_ == 0)
{
lean_ctor_set_tag(v___x_2755_, 0);
v___x_2758_ = v___x_2755_;
goto v_reusejp_2757_;
}
else
{
lean_object* v_reuseFailAlloc_2759_; 
v_reuseFailAlloc_2759_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2759_, 0, v_a_2753_);
v___x_2758_ = v_reuseFailAlloc_2759_;
goto v_reusejp_2757_;
}
v_reusejp_2757_:
{
return v___x_2758_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg___boxed(lean_object* v_x_2761_, lean_object* v___y_2762_){
_start:
{
lean_object* v_res_2763_; 
v_res_2763_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_x_2761_);
return v_res_2763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14(lean_object* v_cls_2764_, uint8_t v_collapsed_2765_, lean_object* v_tag_2766_, lean_object* v_opts_2767_, uint8_t v_clsEnabled_2768_, lean_object* v_oldTraces_2769_, lean_object* v_msg_2770_, lean_object* v_resStartStop_2771_, lean_object* v___y_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_){
_start:
{
lean_object* v_fst_2781_; lean_object* v_snd_2782_; lean_object* v___y_2784_; lean_object* v___y_2785_; lean_object* v_data_2786_; lean_object* v_fst_2797_; lean_object* v_snd_2798_; lean_object* v___x_2799_; uint8_t v___x_2800_; lean_object* v___y_2802_; lean_object* v_a_2803_; uint8_t v___y_2818_; double v___y_2849_; 
v_fst_2781_ = lean_ctor_get(v_resStartStop_2771_, 0);
lean_inc(v_fst_2781_);
v_snd_2782_ = lean_ctor_get(v_resStartStop_2771_, 1);
lean_inc(v_snd_2782_);
lean_dec_ref(v_resStartStop_2771_);
v_fst_2797_ = lean_ctor_get(v_snd_2782_, 0);
lean_inc(v_fst_2797_);
v_snd_2798_ = lean_ctor_get(v_snd_2782_, 1);
lean_inc(v_snd_2798_);
lean_dec(v_snd_2782_);
v___x_2799_ = l_Lean_trace_profiler;
v___x_2800_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_2767_, v___x_2799_);
if (v___x_2800_ == 0)
{
v___y_2818_ = v___x_2800_;
goto v___jp_2817_;
}
else
{
lean_object* v___x_2854_; uint8_t v___x_2855_; 
v___x_2854_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2855_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_2767_, v___x_2854_);
if (v___x_2855_ == 0)
{
lean_object* v___x_2856_; lean_object* v___x_2857_; double v___x_2858_; double v___x_2859_; double v___x_2860_; 
v___x_2856_ = l_Lean_trace_profiler_threshold;
v___x_2857_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_2767_, v___x_2856_);
v___x_2858_ = lean_float_of_nat(v___x_2857_);
v___x_2859_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3);
v___x_2860_ = lean_float_div(v___x_2858_, v___x_2859_);
v___y_2849_ = v___x_2860_;
goto v___jp_2848_;
}
else
{
lean_object* v___x_2861_; lean_object* v___x_2862_; double v___x_2863_; 
v___x_2861_ = l_Lean_trace_profiler_threshold;
v___x_2862_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_2767_, v___x_2861_);
v___x_2863_ = lean_float_of_nat(v___x_2862_);
v___y_2849_ = v___x_2863_;
goto v___jp_2848_;
}
}
v___jp_2783_:
{
lean_object* v___x_2787_; 
lean_inc(v___y_2785_);
v___x_2787_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(v_oldTraces_2769_, v_data_2786_, v___y_2785_, v___y_2784_, v___y_2776_, v___y_2777_, v___y_2778_, v___y_2779_);
if (lean_obj_tag(v___x_2787_) == 0)
{
lean_object* v___x_2788_; 
lean_dec_ref_known(v___x_2787_, 1);
v___x_2788_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_fst_2781_);
return v___x_2788_;
}
else
{
lean_object* v_a_2789_; lean_object* v___x_2791_; uint8_t v_isShared_2792_; uint8_t v_isSharedCheck_2796_; 
lean_dec(v_fst_2781_);
v_a_2789_ = lean_ctor_get(v___x_2787_, 0);
v_isSharedCheck_2796_ = !lean_is_exclusive(v___x_2787_);
if (v_isSharedCheck_2796_ == 0)
{
v___x_2791_ = v___x_2787_;
v_isShared_2792_ = v_isSharedCheck_2796_;
goto v_resetjp_2790_;
}
else
{
lean_inc(v_a_2789_);
lean_dec(v___x_2787_);
v___x_2791_ = lean_box(0);
v_isShared_2792_ = v_isSharedCheck_2796_;
goto v_resetjp_2790_;
}
v_resetjp_2790_:
{
lean_object* v___x_2794_; 
if (v_isShared_2792_ == 0)
{
v___x_2794_ = v___x_2791_;
goto v_reusejp_2793_;
}
else
{
lean_object* v_reuseFailAlloc_2795_; 
v_reuseFailAlloc_2795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2795_, 0, v_a_2789_);
v___x_2794_ = v_reuseFailAlloc_2795_;
goto v_reusejp_2793_;
}
v_reusejp_2793_:
{
return v___x_2794_;
}
}
}
}
v___jp_2801_:
{
uint8_t v_result_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; double v___x_2807_; lean_object* v_data_2808_; 
v_result_2804_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__7(v_fst_2781_);
v___x_2805_ = lean_box(v_result_2804_);
v___x_2806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2806_, 0, v___x_2805_);
v___x_2807_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0);
lean_inc_ref(v_tag_2766_);
lean_inc_ref(v___x_2806_);
lean_inc(v_cls_2764_);
v_data_2808_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2808_, 0, v_cls_2764_);
lean_ctor_set(v_data_2808_, 1, v___x_2806_);
lean_ctor_set(v_data_2808_, 2, v_tag_2766_);
lean_ctor_set_float(v_data_2808_, sizeof(void*)*3, v___x_2807_);
lean_ctor_set_float(v_data_2808_, sizeof(void*)*3 + 8, v___x_2807_);
lean_ctor_set_uint8(v_data_2808_, sizeof(void*)*3 + 16, v_collapsed_2765_);
if (v___x_2800_ == 0)
{
lean_dec_ref_known(v___x_2806_, 1);
lean_dec(v_snd_2798_);
lean_dec(v_fst_2797_);
lean_dec_ref(v_tag_2766_);
lean_dec(v_cls_2764_);
v___y_2784_ = v_a_2803_;
v___y_2785_ = v___y_2802_;
v_data_2786_ = v_data_2808_;
goto v___jp_2783_;
}
else
{
lean_object* v_data_2809_; double v___x_2810_; double v___x_2811_; 
lean_dec_ref_known(v_data_2808_, 3);
v_data_2809_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2809_, 0, v_cls_2764_);
lean_ctor_set(v_data_2809_, 1, v___x_2806_);
lean_ctor_set(v_data_2809_, 2, v_tag_2766_);
v___x_2810_ = lean_unbox_float(v_fst_2797_);
lean_dec(v_fst_2797_);
lean_ctor_set_float(v_data_2809_, sizeof(void*)*3, v___x_2810_);
v___x_2811_ = lean_unbox_float(v_snd_2798_);
lean_dec(v_snd_2798_);
lean_ctor_set_float(v_data_2809_, sizeof(void*)*3 + 8, v___x_2811_);
lean_ctor_set_uint8(v_data_2809_, sizeof(void*)*3 + 16, v_collapsed_2765_);
v___y_2784_ = v_a_2803_;
v___y_2785_ = v___y_2802_;
v_data_2786_ = v_data_2809_;
goto v___jp_2783_;
}
}
v___jp_2812_:
{
lean_object* v_ref_2813_; lean_object* v___x_2814_; 
v_ref_2813_ = lean_ctor_get(v___y_2778_, 5);
lean_inc(v___y_2779_);
lean_inc_ref(v___y_2778_);
lean_inc(v___y_2777_);
lean_inc_ref(v___y_2776_);
lean_inc(v___y_2775_);
lean_inc_ref(v___y_2774_);
lean_inc(v___y_2773_);
lean_inc_ref(v___y_2772_);
lean_inc(v_fst_2781_);
v___x_2814_ = lean_apply_10(v_msg_2770_, v_fst_2781_, v___y_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_, v___y_2777_, v___y_2778_, v___y_2779_, lean_box(0));
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v_a_2815_; 
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
lean_inc(v_a_2815_);
lean_dec_ref_known(v___x_2814_, 1);
v___y_2802_ = v_ref_2813_;
v_a_2803_ = v_a_2815_;
goto v___jp_2801_;
}
else
{
lean_object* v___x_2816_; 
lean_dec_ref_known(v___x_2814_, 1);
v___x_2816_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__2);
v___y_2802_ = v_ref_2813_;
v_a_2803_ = v___x_2816_;
goto v___jp_2801_;
}
}
v___jp_2817_:
{
if (v_clsEnabled_2768_ == 0)
{
if (v___y_2818_ == 0)
{
lean_object* v___x_2819_; lean_object* v_traceState_2820_; lean_object* v_env_2821_; lean_object* v_nextMacroScope_2822_; lean_object* v_ngen_2823_; lean_object* v_auxDeclNGen_2824_; lean_object* v_cache_2825_; lean_object* v_messages_2826_; lean_object* v_infoState_2827_; lean_object* v_snapshotTasks_2828_; lean_object* v___x_2830_; uint8_t v_isShared_2831_; uint8_t v_isSharedCheck_2847_; 
lean_dec(v_snd_2798_);
lean_dec(v_fst_2797_);
lean_dec_ref(v_msg_2770_);
lean_dec_ref(v_tag_2766_);
lean_dec(v_cls_2764_);
v___x_2819_ = lean_st_ref_take(v___y_2779_);
v_traceState_2820_ = lean_ctor_get(v___x_2819_, 4);
v_env_2821_ = lean_ctor_get(v___x_2819_, 0);
v_nextMacroScope_2822_ = lean_ctor_get(v___x_2819_, 1);
v_ngen_2823_ = lean_ctor_get(v___x_2819_, 2);
v_auxDeclNGen_2824_ = lean_ctor_get(v___x_2819_, 3);
v_cache_2825_ = lean_ctor_get(v___x_2819_, 5);
v_messages_2826_ = lean_ctor_get(v___x_2819_, 6);
v_infoState_2827_ = lean_ctor_get(v___x_2819_, 7);
v_snapshotTasks_2828_ = lean_ctor_get(v___x_2819_, 8);
v_isSharedCheck_2847_ = !lean_is_exclusive(v___x_2819_);
if (v_isSharedCheck_2847_ == 0)
{
v___x_2830_ = v___x_2819_;
v_isShared_2831_ = v_isSharedCheck_2847_;
goto v_resetjp_2829_;
}
else
{
lean_inc(v_snapshotTasks_2828_);
lean_inc(v_infoState_2827_);
lean_inc(v_messages_2826_);
lean_inc(v_cache_2825_);
lean_inc(v_traceState_2820_);
lean_inc(v_auxDeclNGen_2824_);
lean_inc(v_ngen_2823_);
lean_inc(v_nextMacroScope_2822_);
lean_inc(v_env_2821_);
lean_dec(v___x_2819_);
v___x_2830_ = lean_box(0);
v_isShared_2831_ = v_isSharedCheck_2847_;
goto v_resetjp_2829_;
}
v_resetjp_2829_:
{
uint64_t v_tid_2832_; lean_object* v_traces_2833_; lean_object* v___x_2835_; uint8_t v_isShared_2836_; uint8_t v_isSharedCheck_2846_; 
v_tid_2832_ = lean_ctor_get_uint64(v_traceState_2820_, sizeof(void*)*1);
v_traces_2833_ = lean_ctor_get(v_traceState_2820_, 0);
v_isSharedCheck_2846_ = !lean_is_exclusive(v_traceState_2820_);
if (v_isSharedCheck_2846_ == 0)
{
v___x_2835_ = v_traceState_2820_;
v_isShared_2836_ = v_isSharedCheck_2846_;
goto v_resetjp_2834_;
}
else
{
lean_inc(v_traces_2833_);
lean_dec(v_traceState_2820_);
v___x_2835_ = lean_box(0);
v_isShared_2836_ = v_isSharedCheck_2846_;
goto v_resetjp_2834_;
}
v_resetjp_2834_:
{
lean_object* v___x_2837_; lean_object* v___x_2839_; 
v___x_2837_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2769_, v_traces_2833_);
lean_dec_ref(v_traces_2833_);
if (v_isShared_2836_ == 0)
{
lean_ctor_set(v___x_2835_, 0, v___x_2837_);
v___x_2839_ = v___x_2835_;
goto v_reusejp_2838_;
}
else
{
lean_object* v_reuseFailAlloc_2845_; 
v_reuseFailAlloc_2845_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2845_, 0, v___x_2837_);
lean_ctor_set_uint64(v_reuseFailAlloc_2845_, sizeof(void*)*1, v_tid_2832_);
v___x_2839_ = v_reuseFailAlloc_2845_;
goto v_reusejp_2838_;
}
v_reusejp_2838_:
{
lean_object* v___x_2841_; 
if (v_isShared_2831_ == 0)
{
lean_ctor_set(v___x_2830_, 4, v___x_2839_);
v___x_2841_ = v___x_2830_;
goto v_reusejp_2840_;
}
else
{
lean_object* v_reuseFailAlloc_2844_; 
v_reuseFailAlloc_2844_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2844_, 0, v_env_2821_);
lean_ctor_set(v_reuseFailAlloc_2844_, 1, v_nextMacroScope_2822_);
lean_ctor_set(v_reuseFailAlloc_2844_, 2, v_ngen_2823_);
lean_ctor_set(v_reuseFailAlloc_2844_, 3, v_auxDeclNGen_2824_);
lean_ctor_set(v_reuseFailAlloc_2844_, 4, v___x_2839_);
lean_ctor_set(v_reuseFailAlloc_2844_, 5, v_cache_2825_);
lean_ctor_set(v_reuseFailAlloc_2844_, 6, v_messages_2826_);
lean_ctor_set(v_reuseFailAlloc_2844_, 7, v_infoState_2827_);
lean_ctor_set(v_reuseFailAlloc_2844_, 8, v_snapshotTasks_2828_);
v___x_2841_ = v_reuseFailAlloc_2844_;
goto v_reusejp_2840_;
}
v_reusejp_2840_:
{
lean_object* v___x_2842_; lean_object* v___x_2843_; 
v___x_2842_ = lean_st_ref_set(v___y_2779_, v___x_2841_);
v___x_2843_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_fst_2781_);
return v___x_2843_;
}
}
}
}
}
else
{
goto v___jp_2812_;
}
}
else
{
goto v___jp_2812_;
}
}
v___jp_2848_:
{
double v___x_2850_; double v___x_2851_; double v___x_2852_; uint8_t v___x_2853_; 
v___x_2850_ = lean_unbox_float(v_snd_2798_);
v___x_2851_ = lean_unbox_float(v_fst_2797_);
v___x_2852_ = lean_float_sub(v___x_2850_, v___x_2851_);
v___x_2853_ = lean_float_decLt(v___y_2849_, v___x_2852_);
v___y_2818_ = v___x_2853_;
goto v___jp_2817_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14___boxed(lean_object** _args){
lean_object* v_cls_2864_ = _args[0];
lean_object* v_collapsed_2865_ = _args[1];
lean_object* v_tag_2866_ = _args[2];
lean_object* v_opts_2867_ = _args[3];
lean_object* v_clsEnabled_2868_ = _args[4];
lean_object* v_oldTraces_2869_ = _args[5];
lean_object* v_msg_2870_ = _args[6];
lean_object* v_resStartStop_2871_ = _args[7];
lean_object* v___y_2872_ = _args[8];
lean_object* v___y_2873_ = _args[9];
lean_object* v___y_2874_ = _args[10];
lean_object* v___y_2875_ = _args[11];
lean_object* v___y_2876_ = _args[12];
lean_object* v___y_2877_ = _args[13];
lean_object* v___y_2878_ = _args[14];
lean_object* v___y_2879_ = _args[15];
lean_object* v___y_2880_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_2881_; uint8_t v_clsEnabled_boxed_2882_; lean_object* v_res_2883_; 
v_collapsed_boxed_2881_ = lean_unbox(v_collapsed_2865_);
v_clsEnabled_boxed_2882_ = lean_unbox(v_clsEnabled_2868_);
v_res_2883_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14(v_cls_2864_, v_collapsed_boxed_2881_, v_tag_2866_, v_opts_2867_, v_clsEnabled_boxed_2882_, v_oldTraces_2869_, v_msg_2870_, v_resStartStop_2871_, v___y_2872_, v___y_2873_, v___y_2874_, v___y_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
lean_dec(v___y_2879_);
lean_dec_ref(v___y_2878_);
lean_dec(v___y_2877_);
lean_dec_ref(v___y_2876_);
lean_dec(v___y_2875_);
lean_dec_ref(v___y_2874_);
lean_dec(v___y_2873_);
lean_dec_ref(v___y_2872_);
lean_dec_ref(v_opts_2867_);
return v_res_2883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0(lean_object* v_goal_2884_, lean_object* v_lem_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_){
_start:
{
lean_object* v___x_2895_; 
v___x_2895_ = lp_mathlib_Mathlib_Tactic_GCongr_applyGCongrLemma(v_goal_2884_, v_lem_2885_, v___y_2888_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_, v___y_2893_);
if (lean_obj_tag(v___x_2895_) == 0)
{
lean_object* v_a_2896_; lean_object* v___x_2898_; uint8_t v_isShared_2899_; uint8_t v_isSharedCheck_2904_; 
v_a_2896_ = lean_ctor_get(v___x_2895_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2895_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2898_ = v___x_2895_;
v_isShared_2899_ = v_isSharedCheck_2904_;
goto v_resetjp_2897_;
}
else
{
lean_inc(v_a_2896_);
lean_dec(v___x_2895_);
v___x_2898_ = lean_box(0);
v_isShared_2899_ = v_isSharedCheck_2904_;
goto v_resetjp_2897_;
}
v_resetjp_2897_:
{
lean_object* v___x_2900_; lean_object* v___x_2902_; 
v___x_2900_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2900_, 0, v_a_2896_);
if (v_isShared_2899_ == 0)
{
lean_ctor_set(v___x_2898_, 0, v___x_2900_);
v___x_2902_ = v___x_2898_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v___x_2900_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
else
{
lean_object* v_a_2905_; lean_object* v___x_2907_; uint8_t v_isShared_2908_; uint8_t v_isSharedCheck_2919_; 
v_a_2905_ = lean_ctor_get(v___x_2895_, 0);
v_isSharedCheck_2919_ = !lean_is_exclusive(v___x_2895_);
if (v_isSharedCheck_2919_ == 0)
{
v___x_2907_ = v___x_2895_;
v_isShared_2908_ = v_isSharedCheck_2919_;
goto v_resetjp_2906_;
}
else
{
lean_inc(v_a_2905_);
lean_dec(v___x_2895_);
v___x_2907_ = lean_box(0);
v_isShared_2908_ = v_isSharedCheck_2919_;
goto v_resetjp_2906_;
}
v_resetjp_2906_:
{
lean_object* v___x_2910_; 
lean_inc(v_a_2905_);
if (v_isShared_2908_ == 0)
{
v___x_2910_ = v___x_2907_;
goto v_reusejp_2909_;
}
else
{
lean_object* v_reuseFailAlloc_2918_; 
v_reuseFailAlloc_2918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2918_, 0, v_a_2905_);
v___x_2910_ = v_reuseFailAlloc_2918_;
goto v_reusejp_2909_;
}
v_reusejp_2909_:
{
uint8_t v___y_2912_; uint8_t v___x_2916_; 
v___x_2916_ = l_Lean_Exception_isInterrupt(v_a_2905_);
if (v___x_2916_ == 0)
{
uint8_t v___x_2917_; 
v___x_2917_ = l_Lean_Exception_isRuntime(v_a_2905_);
v___y_2912_ = v___x_2917_;
goto v___jp_2911_;
}
else
{
lean_dec(v_a_2905_);
v___y_2912_ = v___x_2916_;
goto v___jp_2911_;
}
v___jp_2911_:
{
if (v___y_2912_ == 0)
{
lean_object* v___x_2913_; lean_object* v___x_2914_; lean_object* v___x_2915_; 
lean_dec_ref(v___x_2910_);
v___x_2913_ = lean_box(v___y_2912_);
v___x_2914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2914_, 0, v___x_2913_);
v___x_2915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2915_, 0, v___x_2914_);
return v___x_2915_;
}
else
{
return v___x_2910_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0___boxed(lean_object* v_goal_2920_, lean_object* v_lem_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_, lean_object* v___y_2928_, lean_object* v___y_2929_, lean_object* v___y_2930_){
_start:
{
lean_object* v_res_2931_; 
v_res_2931_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0(v_goal_2920_, v_lem_2921_, v___y_2922_, v___y_2923_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_, v___y_2928_, v___y_2929_);
lean_dec(v___y_2929_);
lean_dec_ref(v___y_2928_);
lean_dec(v___y_2927_);
lean_dec_ref(v___y_2926_);
lean_dec(v___y_2925_);
lean_dec_ref(v___y_2924_);
lean_dec(v___y_2923_);
lean_dec_ref(v___y_2922_);
return v_res_2931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31___redArg(lean_object* v_x_2932_, lean_object* v_x_2933_){
_start:
{
if (lean_obj_tag(v_x_2933_) == 0)
{
return v_x_2932_;
}
else
{
lean_object* v_key_2934_; lean_object* v_value_2935_; lean_object* v_tail_2936_; lean_object* v___x_2938_; uint8_t v_isShared_2939_; uint8_t v_isSharedCheck_2979_; 
v_key_2934_ = lean_ctor_get(v_x_2933_, 0);
v_value_2935_ = lean_ctor_get(v_x_2933_, 1);
v_tail_2936_ = lean_ctor_get(v_x_2933_, 2);
v_isSharedCheck_2979_ = !lean_is_exclusive(v_x_2933_);
if (v_isSharedCheck_2979_ == 0)
{
v___x_2938_ = v_x_2933_;
v_isShared_2939_ = v_isSharedCheck_2979_;
goto v_resetjp_2937_;
}
else
{
lean_inc(v_tail_2936_);
lean_inc(v_value_2935_);
lean_inc(v_key_2934_);
lean_dec(v_x_2933_);
v___x_2938_ = lean_box(0);
v_isShared_2939_ = v_isSharedCheck_2979_;
goto v_resetjp_2937_;
}
v_resetjp_2937_:
{
lean_object* v_fst_2940_; lean_object* v_snd_2941_; lean_object* v___x_2942_; uint64_t v___y_2944_; uint64_t v___y_2945_; uint64_t v___y_2946_; uint64_t v___y_2967_; 
v_fst_2940_ = lean_ctor_get(v_key_2934_, 0);
v_snd_2941_ = lean_ctor_get(v_key_2934_, 1);
v___x_2942_ = lean_array_get_size(v_x_2932_);
if (lean_obj_tag(v_fst_2940_) == 0)
{
uint64_t v___x_2974_; 
v___x_2974_ = 11ULL;
v___y_2967_ = v___x_2974_;
goto v___jp_2966_;
}
else
{
lean_object* v_val_2975_; uint64_t v___x_2976_; uint64_t v___x_2977_; uint64_t v___x_2978_; 
v_val_2975_ = lean_ctor_get(v_fst_2940_, 0);
v___x_2976_ = l_Lean_Expr_hash(v_val_2975_);
v___x_2977_ = 13ULL;
v___x_2978_ = lean_uint64_mix_hash(v___x_2976_, v___x_2977_);
v___y_2967_ = v___x_2978_;
goto v___jp_2966_;
}
v___jp_2943_:
{
uint64_t v___x_2947_; uint64_t v___x_2948_; uint64_t v___x_2949_; uint64_t v___x_2950_; uint64_t v_fold_2951_; uint64_t v___x_2952_; uint64_t v___x_2953_; uint64_t v___x_2954_; size_t v___x_2955_; size_t v___x_2956_; size_t v___x_2957_; size_t v___x_2958_; size_t v___x_2959_; lean_object* v___x_2960_; lean_object* v___x_2962_; 
v___x_2947_ = lean_uint64_mix_hash(v___y_2945_, v___y_2946_);
v___x_2948_ = lean_uint64_mix_hash(v___y_2944_, v___x_2947_);
v___x_2949_ = 32ULL;
v___x_2950_ = lean_uint64_shift_right(v___x_2948_, v___x_2949_);
v_fold_2951_ = lean_uint64_xor(v___x_2948_, v___x_2950_);
v___x_2952_ = 16ULL;
v___x_2953_ = lean_uint64_shift_right(v_fold_2951_, v___x_2952_);
v___x_2954_ = lean_uint64_xor(v_fold_2951_, v___x_2953_);
v___x_2955_ = lean_uint64_to_usize(v___x_2954_);
v___x_2956_ = lean_usize_of_nat(v___x_2942_);
v___x_2957_ = ((size_t)1ULL);
v___x_2958_ = lean_usize_sub(v___x_2956_, v___x_2957_);
v___x_2959_ = lean_usize_land(v___x_2955_, v___x_2958_);
v___x_2960_ = lean_array_uget_borrowed(v_x_2932_, v___x_2959_);
lean_inc(v___x_2960_);
if (v_isShared_2939_ == 0)
{
lean_ctor_set(v___x_2938_, 2, v___x_2960_);
v___x_2962_ = v___x_2938_;
goto v_reusejp_2961_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v_key_2934_);
lean_ctor_set(v_reuseFailAlloc_2965_, 1, v_value_2935_);
lean_ctor_set(v_reuseFailAlloc_2965_, 2, v___x_2960_);
v___x_2962_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2961_;
}
v_reusejp_2961_:
{
lean_object* v___x_2963_; 
v___x_2963_ = lean_array_uset(v_x_2932_, v___x_2959_, v___x_2962_);
v_x_2932_ = v___x_2963_;
v_x_2933_ = v_tail_2936_;
goto _start;
}
}
v___jp_2966_:
{
lean_object* v_fst_2968_; lean_object* v_snd_2969_; uint64_t v___x_2970_; uint8_t v___x_2971_; 
v_fst_2968_ = lean_ctor_get(v_snd_2941_, 0);
v_snd_2969_ = lean_ctor_get(v_snd_2941_, 1);
v___x_2970_ = l_Lean_Expr_hash(v_fst_2968_);
v___x_2971_ = lean_unbox(v_snd_2969_);
if (v___x_2971_ == 0)
{
uint64_t v___x_2972_; 
v___x_2972_ = 13ULL;
v___y_2944_ = v___y_2967_;
v___y_2945_ = v___x_2970_;
v___y_2946_ = v___x_2972_;
goto v___jp_2943_;
}
else
{
uint64_t v___x_2973_; 
v___x_2973_ = 11ULL;
v___y_2944_ = v___y_2967_;
v___y_2945_ = v___x_2970_;
v___y_2946_ = v___x_2973_;
goto v___jp_2943_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13___redArg(lean_object* v_i_2980_, lean_object* v_source_2981_, lean_object* v_target_2982_){
_start:
{
lean_object* v___x_2983_; uint8_t v___x_2984_; 
v___x_2983_ = lean_array_get_size(v_source_2981_);
v___x_2984_ = lean_nat_dec_lt(v_i_2980_, v___x_2983_);
if (v___x_2984_ == 0)
{
lean_dec_ref(v_source_2981_);
lean_dec(v_i_2980_);
return v_target_2982_;
}
else
{
lean_object* v_es_2985_; lean_object* v___x_2986_; lean_object* v_source_2987_; lean_object* v_target_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; 
v_es_2985_ = lean_array_fget(v_source_2981_, v_i_2980_);
v___x_2986_ = lean_box(0);
v_source_2987_ = lean_array_fset(v_source_2981_, v_i_2980_, v___x_2986_);
v_target_2988_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31___redArg(v_target_2982_, v_es_2985_);
v___x_2989_ = lean_unsigned_to_nat(1u);
v___x_2990_ = lean_nat_add(v_i_2980_, v___x_2989_);
lean_dec(v_i_2980_);
v_i_2980_ = v___x_2990_;
v_source_2981_ = v_source_2987_;
v_target_2982_ = v_target_2988_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5___redArg(lean_object* v_data_2992_){
_start:
{
lean_object* v___x_2993_; lean_object* v___x_2994_; lean_object* v_nbuckets_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; 
v___x_2993_ = lean_array_get_size(v_data_2992_);
v___x_2994_ = lean_unsigned_to_nat(2u);
v_nbuckets_2995_ = lean_nat_mul(v___x_2993_, v___x_2994_);
v___x_2996_ = lean_unsigned_to_nat(0u);
v___x_2997_ = lean_box(0);
v___x_2998_ = lean_mk_array(v_nbuckets_2995_, v___x_2997_);
v___x_2999_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13___redArg(v___x_2996_, v_data_2992_, v___x_2998_);
return v___x_2999_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11(lean_object* v_x_3000_, lean_object* v_x_3001_){
_start:
{
if (lean_obj_tag(v_x_3000_) == 0)
{
if (lean_obj_tag(v_x_3001_) == 0)
{
uint8_t v___x_3002_; 
v___x_3002_ = 1;
return v___x_3002_;
}
else
{
uint8_t v___x_3003_; 
v___x_3003_ = 0;
return v___x_3003_;
}
}
else
{
if (lean_obj_tag(v_x_3001_) == 0)
{
uint8_t v___x_3004_; 
v___x_3004_ = 0;
return v___x_3004_;
}
else
{
lean_object* v_val_3005_; lean_object* v_val_3006_; uint8_t v___x_3007_; 
v_val_3005_ = lean_ctor_get(v_x_3000_, 0);
v_val_3006_ = lean_ctor_get(v_x_3001_, 0);
v___x_3007_ = lean_expr_eqv(v_val_3005_, v_val_3006_);
return v___x_3007_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11___boxed(lean_object* v_x_3008_, lean_object* v_x_3009_){
_start:
{
uint8_t v_res_3010_; lean_object* v_r_3011_; 
v_res_3010_ = lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11(v_x_3008_, v_x_3009_);
lean_dec(v_x_3009_);
lean_dec(v_x_3008_);
v_r_3011_ = lean_box(v_res_3010_);
return v_r_3011_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(lean_object* v_a_3012_, lean_object* v_x_3013_){
_start:
{
if (lean_obj_tag(v_x_3013_) == 0)
{
uint8_t v___x_3014_; 
v___x_3014_ = 0;
return v___x_3014_;
}
else
{
lean_object* v_key_3015_; lean_object* v_tail_3016_; uint8_t v___y_3018_; lean_object* v_fst_3020_; lean_object* v_snd_3021_; lean_object* v_fst_3022_; lean_object* v_snd_3023_; uint8_t v___x_3024_; 
v_key_3015_ = lean_ctor_get(v_x_3013_, 0);
v_tail_3016_ = lean_ctor_get(v_x_3013_, 2);
v_fst_3020_ = lean_ctor_get(v_key_3015_, 0);
v_snd_3021_ = lean_ctor_get(v_key_3015_, 1);
v_fst_3022_ = lean_ctor_get(v_a_3012_, 0);
v_snd_3023_ = lean_ctor_get(v_a_3012_, 1);
v___x_3024_ = lp_mathlib_Option_instBEq_beq___at___00Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4_spec__11(v_fst_3020_, v_fst_3022_);
if (v___x_3024_ == 0)
{
v___y_3018_ = v___x_3024_;
goto v___jp_3017_;
}
else
{
lean_object* v_fst_3025_; lean_object* v_snd_3026_; lean_object* v_fst_3027_; lean_object* v_snd_3028_; uint8_t v___x_3029_; 
v_fst_3025_ = lean_ctor_get(v_snd_3021_, 0);
v_snd_3026_ = lean_ctor_get(v_snd_3021_, 1);
v_fst_3027_ = lean_ctor_get(v_snd_3023_, 0);
v_snd_3028_ = lean_ctor_get(v_snd_3023_, 1);
v___x_3029_ = lean_expr_eqv(v_fst_3025_, v_fst_3027_);
if (v___x_3029_ == 0)
{
v___y_3018_ = v___x_3029_;
goto v___jp_3017_;
}
else
{
uint8_t v___x_3030_; 
v___x_3030_ = lean_unbox(v_snd_3026_);
if (v___x_3030_ == 0)
{
uint8_t v___x_3031_; 
v___x_3031_ = lean_unbox(v_snd_3028_);
if (v___x_3031_ == 0)
{
v___y_3018_ = v___x_3029_;
goto v___jp_3017_;
}
else
{
v_x_3013_ = v_tail_3016_;
goto _start;
}
}
else
{
uint8_t v___x_3033_; 
v___x_3033_ = lean_unbox(v_snd_3028_);
v___y_3018_ = v___x_3033_;
goto v___jp_3017_;
}
}
}
v___jp_3017_:
{
if (v___y_3018_ == 0)
{
v_x_3013_ = v_tail_3016_;
goto _start;
}
else
{
return v___y_3018_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg___boxed(lean_object* v_a_3034_, lean_object* v_x_3035_){
_start:
{
uint8_t v_res_3036_; lean_object* v_r_3037_; 
v_res_3036_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(v_a_3034_, v_x_3035_);
lean_dec(v_x_3035_);
lean_dec_ref(v_a_3034_);
v_r_3037_ = lean_box(v_res_3036_);
return v_r_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4___redArg(lean_object* v_m_3038_, lean_object* v_a_3039_, lean_object* v_b_3040_){
_start:
{
lean_object* v_size_3041_; lean_object* v_buckets_3042_; lean_object* v_fst_3043_; lean_object* v_snd_3044_; lean_object* v___x_3045_; uint64_t v___y_3047_; uint64_t v___y_3048_; uint64_t v___y_3049_; uint64_t v___y_3089_; 
v_size_3041_ = lean_ctor_get(v_m_3038_, 0);
v_buckets_3042_ = lean_ctor_get(v_m_3038_, 1);
v_fst_3043_ = lean_ctor_get(v_a_3039_, 0);
v_snd_3044_ = lean_ctor_get(v_a_3039_, 1);
v___x_3045_ = lean_array_get_size(v_buckets_3042_);
if (lean_obj_tag(v_fst_3043_) == 0)
{
uint64_t v___x_3096_; 
v___x_3096_ = 11ULL;
v___y_3089_ = v___x_3096_;
goto v___jp_3088_;
}
else
{
lean_object* v_val_3097_; uint64_t v___x_3098_; uint64_t v___x_3099_; uint64_t v___x_3100_; 
v_val_3097_ = lean_ctor_get(v_fst_3043_, 0);
v___x_3098_ = l_Lean_Expr_hash(v_val_3097_);
v___x_3099_ = 13ULL;
v___x_3100_ = lean_uint64_mix_hash(v___x_3098_, v___x_3099_);
v___y_3089_ = v___x_3100_;
goto v___jp_3088_;
}
v___jp_3046_:
{
uint64_t v___x_3050_; uint64_t v___x_3051_; uint64_t v___x_3052_; uint64_t v___x_3053_; uint64_t v_fold_3054_; uint64_t v___x_3055_; uint64_t v___x_3056_; uint64_t v___x_3057_; size_t v___x_3058_; size_t v___x_3059_; size_t v___x_3060_; size_t v___x_3061_; size_t v___x_3062_; lean_object* v_bkt_3063_; uint8_t v___x_3064_; 
v___x_3050_ = lean_uint64_mix_hash(v___y_3048_, v___y_3049_);
v___x_3051_ = lean_uint64_mix_hash(v___y_3047_, v___x_3050_);
v___x_3052_ = 32ULL;
v___x_3053_ = lean_uint64_shift_right(v___x_3051_, v___x_3052_);
v_fold_3054_ = lean_uint64_xor(v___x_3051_, v___x_3053_);
v___x_3055_ = 16ULL;
v___x_3056_ = lean_uint64_shift_right(v_fold_3054_, v___x_3055_);
v___x_3057_ = lean_uint64_xor(v_fold_3054_, v___x_3056_);
v___x_3058_ = lean_uint64_to_usize(v___x_3057_);
v___x_3059_ = lean_usize_of_nat(v___x_3045_);
v___x_3060_ = ((size_t)1ULL);
v___x_3061_ = lean_usize_sub(v___x_3059_, v___x_3060_);
v___x_3062_ = lean_usize_land(v___x_3058_, v___x_3061_);
v_bkt_3063_ = lean_array_uget_borrowed(v_buckets_3042_, v___x_3062_);
v___x_3064_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(v_a_3039_, v_bkt_3063_);
if (v___x_3064_ == 0)
{
lean_object* v___x_3066_; uint8_t v_isShared_3067_; uint8_t v_isSharedCheck_3085_; 
lean_inc_ref(v_buckets_3042_);
lean_inc(v_size_3041_);
v_isSharedCheck_3085_ = !lean_is_exclusive(v_m_3038_);
if (v_isSharedCheck_3085_ == 0)
{
lean_object* v_unused_3086_; lean_object* v_unused_3087_; 
v_unused_3086_ = lean_ctor_get(v_m_3038_, 1);
lean_dec(v_unused_3086_);
v_unused_3087_ = lean_ctor_get(v_m_3038_, 0);
lean_dec(v_unused_3087_);
v___x_3066_ = v_m_3038_;
v_isShared_3067_ = v_isSharedCheck_3085_;
goto v_resetjp_3065_;
}
else
{
lean_dec(v_m_3038_);
v___x_3066_ = lean_box(0);
v_isShared_3067_ = v_isSharedCheck_3085_;
goto v_resetjp_3065_;
}
v_resetjp_3065_:
{
lean_object* v___x_3068_; lean_object* v_size_x27_3069_; lean_object* v___x_3070_; lean_object* v_buckets_x27_3071_; lean_object* v___x_3072_; lean_object* v___x_3073_; lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; uint8_t v___x_3077_; 
v___x_3068_ = lean_unsigned_to_nat(1u);
v_size_x27_3069_ = lean_nat_add(v_size_3041_, v___x_3068_);
lean_dec(v_size_3041_);
lean_inc(v_bkt_3063_);
v___x_3070_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3070_, 0, v_a_3039_);
lean_ctor_set(v___x_3070_, 1, v_b_3040_);
lean_ctor_set(v___x_3070_, 2, v_bkt_3063_);
v_buckets_x27_3071_ = lean_array_uset(v_buckets_3042_, v___x_3062_, v___x_3070_);
v___x_3072_ = lean_unsigned_to_nat(4u);
v___x_3073_ = lean_nat_mul(v_size_x27_3069_, v___x_3072_);
v___x_3074_ = lean_unsigned_to_nat(3u);
v___x_3075_ = lean_nat_div(v___x_3073_, v___x_3074_);
lean_dec(v___x_3073_);
v___x_3076_ = lean_array_get_size(v_buckets_x27_3071_);
v___x_3077_ = lean_nat_dec_le(v___x_3075_, v___x_3076_);
lean_dec(v___x_3075_);
if (v___x_3077_ == 0)
{
lean_object* v_val_3078_; lean_object* v___x_3080_; 
v_val_3078_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5___redArg(v_buckets_x27_3071_);
if (v_isShared_3067_ == 0)
{
lean_ctor_set(v___x_3066_, 1, v_val_3078_);
lean_ctor_set(v___x_3066_, 0, v_size_x27_3069_);
v___x_3080_ = v___x_3066_;
goto v_reusejp_3079_;
}
else
{
lean_object* v_reuseFailAlloc_3081_; 
v_reuseFailAlloc_3081_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3081_, 0, v_size_x27_3069_);
lean_ctor_set(v_reuseFailAlloc_3081_, 1, v_val_3078_);
v___x_3080_ = v_reuseFailAlloc_3081_;
goto v_reusejp_3079_;
}
v_reusejp_3079_:
{
return v___x_3080_;
}
}
else
{
lean_object* v___x_3083_; 
if (v_isShared_3067_ == 0)
{
lean_ctor_set(v___x_3066_, 1, v_buckets_x27_3071_);
lean_ctor_set(v___x_3066_, 0, v_size_x27_3069_);
v___x_3083_ = v___x_3066_;
goto v_reusejp_3082_;
}
else
{
lean_object* v_reuseFailAlloc_3084_; 
v_reuseFailAlloc_3084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3084_, 0, v_size_x27_3069_);
lean_ctor_set(v_reuseFailAlloc_3084_, 1, v_buckets_x27_3071_);
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
else
{
lean_dec(v_b_3040_);
lean_dec_ref(v_a_3039_);
return v_m_3038_;
}
}
v___jp_3088_:
{
lean_object* v_fst_3090_; lean_object* v_snd_3091_; uint64_t v___x_3092_; uint8_t v___x_3093_; 
v_fst_3090_ = lean_ctor_get(v_snd_3044_, 0);
v_snd_3091_ = lean_ctor_get(v_snd_3044_, 1);
v___x_3092_ = l_Lean_Expr_hash(v_fst_3090_);
v___x_3093_ = lean_unbox(v_snd_3091_);
if (v___x_3093_ == 0)
{
uint64_t v___x_3094_; 
v___x_3094_ = 13ULL;
v___y_3047_ = v___y_3089_;
v___y_3048_ = v___x_3092_;
v___y_3049_ = v___x_3094_;
goto v___jp_3046_;
}
else
{
uint64_t v___x_3095_; 
v___x_3095_ = 11ULL;
v___y_3047_ = v___y_3089_;
v___y_3048_ = v___x_3092_;
v___y_3049_ = v___x_3095_;
goto v___jp_3046_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(lean_object* v_m_3101_, lean_object* v_a_3102_){
_start:
{
lean_object* v_buckets_3103_; lean_object* v_fst_3104_; lean_object* v_snd_3105_; lean_object* v___x_3106_; uint64_t v___y_3108_; uint64_t v___y_3109_; uint64_t v___y_3110_; uint64_t v___y_3127_; 
v_buckets_3103_ = lean_ctor_get(v_m_3101_, 1);
v_fst_3104_ = lean_ctor_get(v_a_3102_, 0);
v_snd_3105_ = lean_ctor_get(v_a_3102_, 1);
v___x_3106_ = lean_array_get_size(v_buckets_3103_);
if (lean_obj_tag(v_fst_3104_) == 0)
{
uint64_t v___x_3134_; 
v___x_3134_ = 11ULL;
v___y_3127_ = v___x_3134_;
goto v___jp_3126_;
}
else
{
lean_object* v_val_3135_; uint64_t v___x_3136_; uint64_t v___x_3137_; uint64_t v___x_3138_; 
v_val_3135_ = lean_ctor_get(v_fst_3104_, 0);
v___x_3136_ = l_Lean_Expr_hash(v_val_3135_);
v___x_3137_ = 13ULL;
v___x_3138_ = lean_uint64_mix_hash(v___x_3136_, v___x_3137_);
v___y_3127_ = v___x_3138_;
goto v___jp_3126_;
}
v___jp_3107_:
{
uint64_t v___x_3111_; uint64_t v___x_3112_; uint64_t v___x_3113_; uint64_t v___x_3114_; uint64_t v_fold_3115_; uint64_t v___x_3116_; uint64_t v___x_3117_; uint64_t v___x_3118_; size_t v___x_3119_; size_t v___x_3120_; size_t v___x_3121_; size_t v___x_3122_; size_t v___x_3123_; lean_object* v___x_3124_; uint8_t v___x_3125_; 
v___x_3111_ = lean_uint64_mix_hash(v___y_3109_, v___y_3110_);
v___x_3112_ = lean_uint64_mix_hash(v___y_3108_, v___x_3111_);
v___x_3113_ = 32ULL;
v___x_3114_ = lean_uint64_shift_right(v___x_3112_, v___x_3113_);
v_fold_3115_ = lean_uint64_xor(v___x_3112_, v___x_3114_);
v___x_3116_ = 16ULL;
v___x_3117_ = lean_uint64_shift_right(v_fold_3115_, v___x_3116_);
v___x_3118_ = lean_uint64_xor(v_fold_3115_, v___x_3117_);
v___x_3119_ = lean_uint64_to_usize(v___x_3118_);
v___x_3120_ = lean_usize_of_nat(v___x_3106_);
v___x_3121_ = ((size_t)1ULL);
v___x_3122_ = lean_usize_sub(v___x_3120_, v___x_3121_);
v___x_3123_ = lean_usize_land(v___x_3119_, v___x_3122_);
v___x_3124_ = lean_array_uget_borrowed(v_buckets_3103_, v___x_3123_);
v___x_3125_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(v_a_3102_, v___x_3124_);
return v___x_3125_;
}
v___jp_3126_:
{
lean_object* v_fst_3128_; lean_object* v_snd_3129_; uint64_t v___x_3130_; uint8_t v___x_3131_; 
v_fst_3128_ = lean_ctor_get(v_snd_3105_, 0);
v_snd_3129_ = lean_ctor_get(v_snd_3105_, 1);
v___x_3130_ = l_Lean_Expr_hash(v_fst_3128_);
v___x_3131_ = lean_unbox(v_snd_3129_);
if (v___x_3131_ == 0)
{
uint64_t v___x_3132_; 
v___x_3132_ = 13ULL;
v___y_3108_ = v___y_3127_;
v___y_3109_ = v___x_3130_;
v___y_3110_ = v___x_3132_;
goto v___jp_3107_;
}
else
{
uint64_t v___x_3133_; 
v___x_3133_ = 11ULL;
v___y_3108_ = v___y_3127_;
v___y_3109_ = v___x_3130_;
v___y_3110_ = v___x_3133_;
goto v___jp_3107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg___boxed(lean_object* v_m_3139_, lean_object* v_a_3140_){
_start:
{
uint8_t v_res_3141_; lean_object* v_r_3142_; 
v_res_3141_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(v_m_3139_, v_a_3140_);
lean_dec_ref(v_a_3140_);
lean_dec_ref(v_m_3139_);
v_r_3142_ = lean_box(v_res_3141_);
return v_r_3142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(lean_object* v_____r_3143_, lean_object* v___y_3144_, lean_object* v___y_3145_, lean_object* v___y_3146_, lean_object* v___y_3147_, lean_object* v___y_3148_, lean_object* v___y_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_){
_start:
{
lean_object* v___x_3153_; lean_object* v___x_3154_; 
v___x_3153_ = lean_box(0);
v___x_3154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3154_, 0, v___x_3153_);
return v___x_3154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0___boxed(lean_object* v_____r_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_, lean_object* v___y_3164_){
_start:
{
lean_object* v_res_3165_; 
v_res_3165_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(v_____r_3155_, v___y_3156_, v___y_3157_, v___y_3158_, v___y_3159_, v___y_3160_, v___y_3161_, v___y_3162_, v___y_3163_);
lean_dec(v___y_3163_);
lean_dec_ref(v___y_3162_);
lean_dec(v___y_3161_);
lean_dec_ref(v___y_3160_);
lean_dec(v___y_3159_);
lean_dec_ref(v___y_3158_);
lean_dec(v___y_3157_);
lean_dec_ref(v___y_3156_);
return v_res_3165_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14(lean_object* v_e_3166_){
_start:
{
if (lean_obj_tag(v_e_3166_) == 0)
{
uint8_t v___x_3167_; 
v___x_3167_ = 2;
return v___x_3167_;
}
else
{
lean_object* v_a_3168_; 
v_a_3168_ = lean_ctor_get(v_e_3166_, 0);
if (lean_obj_tag(v_a_3168_) == 0)
{
uint8_t v___x_3169_; 
v___x_3169_ = 1;
return v___x_3169_;
}
else
{
uint8_t v___x_3170_; 
v___x_3170_ = 0;
return v___x_3170_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14___boxed(lean_object* v_e_3171_){
_start:
{
uint8_t v_res_3172_; lean_object* v_r_3173_; 
v_res_3172_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14(v_e_3171_);
lean_dec_ref(v_e_3171_);
v_r_3173_ = lean_box(v_res_3172_);
return v_r_3173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10(lean_object* v_cls_3174_, uint8_t v_collapsed_3175_, lean_object* v_tag_3176_, lean_object* v_opts_3177_, uint8_t v_clsEnabled_3178_, lean_object* v_oldTraces_3179_, lean_object* v_ref_3180_, lean_object* v_msg_3181_, lean_object* v_resStartStop_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_, lean_object* v___y_3190_){
_start:
{
lean_object* v_fst_3192_; lean_object* v_snd_3193_; lean_object* v_data_3195_; lean_object* v_fst_3206_; lean_object* v_snd_3207_; lean_object* v___x_3208_; uint8_t v___x_3209_; uint8_t v___y_3220_; double v___y_3251_; 
v_fst_3192_ = lean_ctor_get(v_resStartStop_3182_, 0);
lean_inc(v_fst_3192_);
v_snd_3193_ = lean_ctor_get(v_resStartStop_3182_, 1);
lean_inc(v_snd_3193_);
lean_dec_ref(v_resStartStop_3182_);
v_fst_3206_ = lean_ctor_get(v_snd_3193_, 0);
lean_inc(v_fst_3206_);
v_snd_3207_ = lean_ctor_get(v_snd_3193_, 1);
lean_inc(v_snd_3207_);
lean_dec(v_snd_3193_);
v___x_3208_ = l_Lean_trace_profiler;
v___x_3209_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_3177_, v___x_3208_);
if (v___x_3209_ == 0)
{
v___y_3220_ = v___x_3209_;
goto v___jp_3219_;
}
else
{
lean_object* v___x_3256_; uint8_t v___x_3257_; 
v___x_3256_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3257_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_opts_3177_, v___x_3256_);
if (v___x_3257_ == 0)
{
lean_object* v___x_3258_; lean_object* v___x_3259_; double v___x_3260_; double v___x_3261_; double v___x_3262_; 
v___x_3258_ = l_Lean_trace_profiler_threshold;
v___x_3259_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_3177_, v___x_3258_);
v___x_3260_ = lean_float_of_nat(v___x_3259_);
v___x_3261_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__3);
v___x_3262_ = lean_float_div(v___x_3260_, v___x_3261_);
v___y_3251_ = v___x_3262_;
goto v___jp_3250_;
}
else
{
lean_object* v___x_3263_; lean_object* v___x_3264_; double v___x_3265_; 
v___x_3263_ = l_Lean_trace_profiler_threshold;
v___x_3264_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__8(v_opts_3177_, v___x_3263_);
v___x_3265_ = lean_float_of_nat(v___x_3264_);
v___y_3251_ = v___x_3265_;
goto v___jp_3250_;
}
}
v___jp_3194_:
{
lean_object* v___x_3196_; 
v___x_3196_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(v_oldTraces_3179_, v_data_3195_, v_ref_3180_, v_msg_3181_, v___y_3187_, v___y_3188_, v___y_3189_, v___y_3190_);
if (lean_obj_tag(v___x_3196_) == 0)
{
lean_object* v___x_3197_; 
lean_dec_ref_known(v___x_3196_, 1);
v___x_3197_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_fst_3192_);
return v___x_3197_;
}
else
{
lean_object* v_a_3198_; lean_object* v___x_3200_; uint8_t v_isShared_3201_; uint8_t v_isSharedCheck_3205_; 
lean_dec(v_fst_3192_);
v_a_3198_ = lean_ctor_get(v___x_3196_, 0);
v_isSharedCheck_3205_ = !lean_is_exclusive(v___x_3196_);
if (v_isSharedCheck_3205_ == 0)
{
v___x_3200_ = v___x_3196_;
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
else
{
lean_inc(v_a_3198_);
lean_dec(v___x_3196_);
v___x_3200_ = lean_box(0);
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
v_resetjp_3199_:
{
lean_object* v___x_3203_; 
if (v_isShared_3201_ == 0)
{
v___x_3203_ = v___x_3200_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3204_; 
v_reuseFailAlloc_3204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3204_, 0, v_a_3198_);
v___x_3203_ = v_reuseFailAlloc_3204_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
return v___x_3203_;
}
}
}
}
v___jp_3210_:
{
uint8_t v_result_3211_; lean_object* v___x_3212_; lean_object* v___x_3213_; double v___x_3214_; lean_object* v_data_3215_; 
v_result_3211_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__14(v_fst_3192_);
v___x_3212_ = lean_box(v_result_3211_);
v___x_3213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3213_, 0, v___x_3212_);
v___x_3214_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4___closed__0);
lean_inc_ref(v_tag_3176_);
lean_inc_ref(v___x_3213_);
lean_inc(v_cls_3174_);
v_data_3215_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3215_, 0, v_cls_3174_);
lean_ctor_set(v_data_3215_, 1, v___x_3213_);
lean_ctor_set(v_data_3215_, 2, v_tag_3176_);
lean_ctor_set_float(v_data_3215_, sizeof(void*)*3, v___x_3214_);
lean_ctor_set_float(v_data_3215_, sizeof(void*)*3 + 8, v___x_3214_);
lean_ctor_set_uint8(v_data_3215_, sizeof(void*)*3 + 16, v_collapsed_3175_);
if (v___x_3209_ == 0)
{
lean_dec_ref_known(v___x_3213_, 1);
lean_dec(v_snd_3207_);
lean_dec(v_fst_3206_);
lean_dec_ref(v_tag_3176_);
lean_dec(v_cls_3174_);
v_data_3195_ = v_data_3215_;
goto v___jp_3194_;
}
else
{
lean_object* v_data_3216_; double v___x_3217_; double v___x_3218_; 
lean_dec_ref_known(v_data_3215_, 3);
v_data_3216_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3216_, 0, v_cls_3174_);
lean_ctor_set(v_data_3216_, 1, v___x_3213_);
lean_ctor_set(v_data_3216_, 2, v_tag_3176_);
v___x_3217_ = lean_unbox_float(v_fst_3206_);
lean_dec(v_fst_3206_);
lean_ctor_set_float(v_data_3216_, sizeof(void*)*3, v___x_3217_);
v___x_3218_ = lean_unbox_float(v_snd_3207_);
lean_dec(v_snd_3207_);
lean_ctor_set_float(v_data_3216_, sizeof(void*)*3 + 8, v___x_3218_);
lean_ctor_set_uint8(v_data_3216_, sizeof(void*)*3 + 16, v_collapsed_3175_);
v_data_3195_ = v_data_3216_;
goto v___jp_3194_;
}
}
v___jp_3219_:
{
if (v_clsEnabled_3178_ == 0)
{
if (v___y_3220_ == 0)
{
lean_object* v___x_3221_; lean_object* v_traceState_3222_; lean_object* v_env_3223_; lean_object* v_nextMacroScope_3224_; lean_object* v_ngen_3225_; lean_object* v_auxDeclNGen_3226_; lean_object* v_cache_3227_; lean_object* v_messages_3228_; lean_object* v_infoState_3229_; lean_object* v_snapshotTasks_3230_; lean_object* v___x_3232_; uint8_t v_isShared_3233_; uint8_t v_isSharedCheck_3249_; 
lean_dec(v_snd_3207_);
lean_dec(v_fst_3206_);
lean_dec_ref(v_msg_3181_);
lean_dec(v_ref_3180_);
lean_dec_ref(v_tag_3176_);
lean_dec(v_cls_3174_);
v___x_3221_ = lean_st_ref_take(v___y_3190_);
v_traceState_3222_ = lean_ctor_get(v___x_3221_, 4);
v_env_3223_ = lean_ctor_get(v___x_3221_, 0);
v_nextMacroScope_3224_ = lean_ctor_get(v___x_3221_, 1);
v_ngen_3225_ = lean_ctor_get(v___x_3221_, 2);
v_auxDeclNGen_3226_ = lean_ctor_get(v___x_3221_, 3);
v_cache_3227_ = lean_ctor_get(v___x_3221_, 5);
v_messages_3228_ = lean_ctor_get(v___x_3221_, 6);
v_infoState_3229_ = lean_ctor_get(v___x_3221_, 7);
v_snapshotTasks_3230_ = lean_ctor_get(v___x_3221_, 8);
v_isSharedCheck_3249_ = !lean_is_exclusive(v___x_3221_);
if (v_isSharedCheck_3249_ == 0)
{
v___x_3232_ = v___x_3221_;
v_isShared_3233_ = v_isSharedCheck_3249_;
goto v_resetjp_3231_;
}
else
{
lean_inc(v_snapshotTasks_3230_);
lean_inc(v_infoState_3229_);
lean_inc(v_messages_3228_);
lean_inc(v_cache_3227_);
lean_inc(v_traceState_3222_);
lean_inc(v_auxDeclNGen_3226_);
lean_inc(v_ngen_3225_);
lean_inc(v_nextMacroScope_3224_);
lean_inc(v_env_3223_);
lean_dec(v___x_3221_);
v___x_3232_ = lean_box(0);
v_isShared_3233_ = v_isSharedCheck_3249_;
goto v_resetjp_3231_;
}
v_resetjp_3231_:
{
uint64_t v_tid_3234_; lean_object* v_traces_3235_; lean_object* v___x_3237_; uint8_t v_isShared_3238_; uint8_t v_isSharedCheck_3248_; 
v_tid_3234_ = lean_ctor_get_uint64(v_traceState_3222_, sizeof(void*)*1);
v_traces_3235_ = lean_ctor_get(v_traceState_3222_, 0);
v_isSharedCheck_3248_ = !lean_is_exclusive(v_traceState_3222_);
if (v_isSharedCheck_3248_ == 0)
{
v___x_3237_ = v_traceState_3222_;
v_isShared_3238_ = v_isSharedCheck_3248_;
goto v_resetjp_3236_;
}
else
{
lean_inc(v_traces_3235_);
lean_dec(v_traceState_3222_);
v___x_3237_ = lean_box(0);
v_isShared_3238_ = v_isSharedCheck_3248_;
goto v_resetjp_3236_;
}
v_resetjp_3236_:
{
lean_object* v___x_3239_; lean_object* v___x_3241_; 
v___x_3239_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_3179_, v_traces_3235_);
lean_dec_ref(v_traces_3235_);
if (v_isShared_3238_ == 0)
{
lean_ctor_set(v___x_3237_, 0, v___x_3239_);
v___x_3241_ = v___x_3237_;
goto v_reusejp_3240_;
}
else
{
lean_object* v_reuseFailAlloc_3247_; 
v_reuseFailAlloc_3247_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3247_, 0, v___x_3239_);
lean_ctor_set_uint64(v_reuseFailAlloc_3247_, sizeof(void*)*1, v_tid_3234_);
v___x_3241_ = v_reuseFailAlloc_3247_;
goto v_reusejp_3240_;
}
v_reusejp_3240_:
{
lean_object* v___x_3243_; 
if (v_isShared_3233_ == 0)
{
lean_ctor_set(v___x_3232_, 4, v___x_3241_);
v___x_3243_ = v___x_3232_;
goto v_reusejp_3242_;
}
else
{
lean_object* v_reuseFailAlloc_3246_; 
v_reuseFailAlloc_3246_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3246_, 0, v_env_3223_);
lean_ctor_set(v_reuseFailAlloc_3246_, 1, v_nextMacroScope_3224_);
lean_ctor_set(v_reuseFailAlloc_3246_, 2, v_ngen_3225_);
lean_ctor_set(v_reuseFailAlloc_3246_, 3, v_auxDeclNGen_3226_);
lean_ctor_set(v_reuseFailAlloc_3246_, 4, v___x_3241_);
lean_ctor_set(v_reuseFailAlloc_3246_, 5, v_cache_3227_);
lean_ctor_set(v_reuseFailAlloc_3246_, 6, v_messages_3228_);
lean_ctor_set(v_reuseFailAlloc_3246_, 7, v_infoState_3229_);
lean_ctor_set(v_reuseFailAlloc_3246_, 8, v_snapshotTasks_3230_);
v___x_3243_ = v_reuseFailAlloc_3246_;
goto v_reusejp_3242_;
}
v_reusejp_3242_:
{
lean_object* v___x_3244_; lean_object* v___x_3245_; 
v___x_3244_ = lean_st_ref_set(v___y_3190_, v___x_3243_);
v___x_3245_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_fst_3192_);
return v___x_3245_;
}
}
}
}
}
else
{
goto v___jp_3210_;
}
}
else
{
goto v___jp_3210_;
}
}
v___jp_3250_:
{
double v___x_3252_; double v___x_3253_; double v___x_3254_; uint8_t v___x_3255_; 
v___x_3252_ = lean_unbox_float(v_snd_3207_);
v___x_3253_ = lean_unbox_float(v_fst_3206_);
v___x_3254_ = lean_float_sub(v___x_3252_, v___x_3253_);
v___x_3255_ = lean_float_decLt(v___y_3251_, v___x_3254_);
v___y_3220_ = v___x_3255_;
goto v___jp_3219_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10___boxed(lean_object** _args){
lean_object* v_cls_3266_ = _args[0];
lean_object* v_collapsed_3267_ = _args[1];
lean_object* v_tag_3268_ = _args[2];
lean_object* v_opts_3269_ = _args[3];
lean_object* v_clsEnabled_3270_ = _args[4];
lean_object* v_oldTraces_3271_ = _args[5];
lean_object* v_ref_3272_ = _args[6];
lean_object* v_msg_3273_ = _args[7];
lean_object* v_resStartStop_3274_ = _args[8];
lean_object* v___y_3275_ = _args[9];
lean_object* v___y_3276_ = _args[10];
lean_object* v___y_3277_ = _args[11];
lean_object* v___y_3278_ = _args[12];
lean_object* v___y_3279_ = _args[13];
lean_object* v___y_3280_ = _args[14];
lean_object* v___y_3281_ = _args[15];
lean_object* v___y_3282_ = _args[16];
lean_object* v___y_3283_ = _args[17];
_start:
{
uint8_t v_collapsed_boxed_3284_; uint8_t v_clsEnabled_boxed_3285_; lean_object* v_res_3286_; 
v_collapsed_boxed_3284_ = lean_unbox(v_collapsed_3267_);
v_clsEnabled_boxed_3285_ = lean_unbox(v_clsEnabled_3270_);
v_res_3286_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10(v_cls_3266_, v_collapsed_boxed_3284_, v_tag_3268_, v_opts_3269_, v_clsEnabled_boxed_3285_, v_oldTraces_3271_, v_ref_3272_, v_msg_3273_, v_resStartStop_3274_, v___y_3275_, v___y_3276_, v___y_3277_, v___y_3278_, v___y_3279_, v___y_3280_, v___y_3281_, v___y_3282_);
lean_dec(v___y_3282_);
lean_dec_ref(v___y_3281_);
lean_dec(v___y_3280_);
lean_dec_ref(v___y_3279_);
lean_dec(v___y_3278_);
lean_dec_ref(v___y_3277_);
lean_dec(v___y_3276_);
lean_dec_ref(v___y_3275_);
lean_dec_ref(v_opts_3269_);
return v_res_3286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg(lean_object* v_msg_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_, lean_object* v___y_3290_, lean_object* v___y_3291_){
_start:
{
lean_object* v_ref_3293_; lean_object* v___x_3294_; lean_object* v_a_3295_; lean_object* v___x_3297_; uint8_t v_isShared_3298_; uint8_t v_isSharedCheck_3303_; 
v_ref_3293_ = lean_ctor_get(v___y_3290_, 5);
v___x_3294_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v_msg_3287_, v___y_3288_, v___y_3289_, v___y_3290_, v___y_3291_);
v_a_3295_ = lean_ctor_get(v___x_3294_, 0);
v_isSharedCheck_3303_ = !lean_is_exclusive(v___x_3294_);
if (v_isSharedCheck_3303_ == 0)
{
v___x_3297_ = v___x_3294_;
v_isShared_3298_ = v_isSharedCheck_3303_;
goto v_resetjp_3296_;
}
else
{
lean_inc(v_a_3295_);
lean_dec(v___x_3294_);
v___x_3297_ = lean_box(0);
v_isShared_3298_ = v_isSharedCheck_3303_;
goto v_resetjp_3296_;
}
v_resetjp_3296_:
{
lean_object* v___x_3299_; lean_object* v___x_3301_; 
lean_inc(v_ref_3293_);
v___x_3299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3299_, 0, v_ref_3293_);
lean_ctor_set(v___x_3299_, 1, v_a_3295_);
if (v_isShared_3298_ == 0)
{
lean_ctor_set_tag(v___x_3297_, 1);
lean_ctor_set(v___x_3297_, 0, v___x_3299_);
v___x_3301_ = v___x_3297_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3302_; 
v_reuseFailAlloc_3302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3302_, 0, v___x_3299_);
v___x_3301_ = v_reuseFailAlloc_3302_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
return v___x_3301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg___boxed(lean_object* v_msg_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_){
_start:
{
lean_object* v_res_3310_; 
v_res_3310_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg(v_msg_3304_, v___y_3305_, v___y_3306_, v___y_3307_, v___y_3308_);
lean_dec(v___y_3308_);
lean_dec_ref(v___y_3307_);
lean_dec(v___y_3306_);
lean_dec_ref(v___y_3305_);
return v_res_3310_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1(void){
_start:
{
lean_object* v___x_3319_; lean_object* v___x_3320_; 
v___x_3319_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__0));
v___x_3320_ = l_Lean_stringToMessageData(v___x_3319_);
return v___x_3320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12(lean_object* v_config_3321_, uint8_t v_forward_3322_, lean_object* v_as_3323_, size_t v_sz_3324_, size_t v_i_3325_, lean_object* v_b_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_, lean_object* v___y_3334_){
_start:
{
lean_object* v_a_3337_; uint8_t v___x_3341_; 
v___x_3341_ = lean_usize_dec_lt(v_i_3325_, v_sz_3324_);
if (v___x_3341_ == 0)
{
lean_object* v___x_3342_; 
lean_dec_ref(v_config_3321_);
v___x_3342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3342_, 0, v_b_3326_);
return v___x_3342_;
}
else
{
lean_object* v_a_3343_; lean_object* v_fst_3344_; lean_object* v_snd_3345_; lean_object* v___x_3347_; uint8_t v_isShared_3348_; uint8_t v_isSharedCheck_3509_; 
v_a_3343_ = lean_array_uget(v_as_3323_, v_i_3325_);
v_fst_3344_ = lean_ctor_get(v_a_3343_, 0);
v_snd_3345_ = lean_ctor_get(v_a_3343_, 1);
v_isSharedCheck_3509_ = !lean_is_exclusive(v_a_3343_);
if (v_isSharedCheck_3509_ == 0)
{
v___x_3347_ = v_a_3343_;
v_isShared_3348_ = v_isSharedCheck_3509_;
goto v_resetjp_3346_;
}
else
{
lean_inc(v_snd_3345_);
lean_inc(v_fst_3344_);
lean_dec(v_a_3343_);
v___x_3347_ = lean_box(0);
v_isShared_3348_ = v_isSharedCheck_3509_;
goto v_resetjp_3346_;
}
v_resetjp_3346_:
{
lean_object* v___x_3349_; lean_object* v_snd_3350_; lean_object* v___x_3352_; uint8_t v_isShared_3353_; uint8_t v_isSharedCheck_3507_; 
v___x_3349_ = lean_st_ref_get(v___y_3328_);
v_snd_3350_ = lean_ctor_get(v_b_3326_, 1);
v_isSharedCheck_3507_ = !lean_is_exclusive(v_b_3326_);
if (v_isSharedCheck_3507_ == 0)
{
lean_object* v_unused_3508_; 
v_unused_3508_ = lean_ctor_get(v_b_3326_, 0);
lean_dec(v_unused_3508_);
v___x_3352_ = v_b_3326_;
v_isShared_3353_ = v_isSharedCheck_3507_;
goto v_resetjp_3351_;
}
else
{
lean_inc(v_snd_3350_);
lean_dec(v_b_3326_);
v___x_3352_ = lean_box(0);
v_isShared_3353_ = v_isSharedCheck_3507_;
goto v_resetjp_3351_;
}
v_resetjp_3351_:
{
lean_object* v_progress_3354_; lean_object* v___x_3356_; uint8_t v_isShared_3357_; uint8_t v_isSharedCheck_3505_; 
v_progress_3354_ = lean_ctor_get(v___x_3349_, 1);
v_isSharedCheck_3505_ = !lean_is_exclusive(v___x_3349_);
if (v_isSharedCheck_3505_ == 0)
{
lean_object* v_unused_3506_; 
v_unused_3506_ = lean_ctor_get(v___x_3349_, 0);
lean_dec(v_unused_3506_);
v___x_3356_ = v___x_3349_;
v_isShared_3357_ = v_isSharedCheck_3505_;
goto v_resetjp_3355_;
}
else
{
lean_inc(v_progress_3354_);
lean_dec(v___x_3349_);
v___x_3356_ = lean_box(0);
v_isShared_3357_ = v_isSharedCheck_3505_;
goto v_resetjp_3355_;
}
v_resetjp_3355_:
{
lean_object* v___x_3358_; uint8_t v___y_3360_; uint8_t v___y_3366_; lean_object* v___y_3367_; uint8_t v_anyProgress_3396_; lean_object* v___y_3398_; lean_object* v___y_3399_; uint8_t v___y_3400_; lean_object* v___y_3401_; lean_object* v___y_3402_; lean_object* v___y_3403_; lean_object* v___y_3404_; lean_object* v___y_3405_; lean_object* v___y_3406_; lean_object* v_cls_3409_; lean_object* v___y_3411_; lean_object* v___y_3412_; uint8_t v___y_3413_; lean_object* v___y_3414_; lean_object* v___y_3415_; lean_object* v___y_3416_; lean_object* v___y_3417_; lean_object* v___y_3418_; lean_object* v___y_3419_; lean_object* v___y_3420_; uint8_t v___y_3421_; uint8_t v_anyProgress_3457_; lean_object* v___y_3458_; lean_object* v___y_3459_; lean_object* v___y_3460_; lean_object* v___y_3461_; lean_object* v___y_3462_; lean_object* v___y_3463_; lean_object* v___y_3464_; lean_object* v___y_3465_; uint8_t v___y_3485_; uint8_t v___y_3501_; 
v___x_3358_ = lean_box(0);
v_anyProgress_3396_ = 0;
v_cls_3409_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
if (lean_obj_tag(v_progress_3354_) == 2)
{
uint8_t v___x_3502_; 
lean_dec_ref_known(v_progress_3354_, 1);
lean_dec(v_snd_3345_);
v___x_3502_ = lean_unbox(v_snd_3350_);
lean_dec(v_snd_3350_);
v_anyProgress_3457_ = v___x_3502_;
v___y_3458_ = v___y_3327_;
v___y_3459_ = v___y_3328_;
v___y_3460_ = v___y_3329_;
v___y_3461_ = v___y_3330_;
v___y_3462_ = v___y_3331_;
v___y_3463_ = v___y_3332_;
v___y_3464_ = v___y_3333_;
v___y_3465_ = v___y_3334_;
goto v___jp_3456_;
}
else
{
lean_dec(v_progress_3354_);
if (v_forward_3322_ == 0)
{
uint8_t v___x_3503_; 
v___x_3503_ = lean_unbox(v_snd_3345_);
lean_dec(v_snd_3345_);
if (v___x_3503_ == 0)
{
v___y_3501_ = v___x_3341_;
goto v___jp_3500_;
}
else
{
v___y_3485_ = v___x_3341_;
goto v___jp_3484_;
}
}
else
{
uint8_t v___x_3504_; 
v___x_3504_ = lean_unbox(v_snd_3345_);
lean_dec(v_snd_3345_);
v___y_3501_ = v___x_3504_;
goto v___jp_3500_;
}
}
v___jp_3359_:
{
lean_object* v___x_3361_; lean_object* v___x_3363_; 
v___x_3361_ = lean_box(v___y_3360_);
if (v_isShared_3353_ == 0)
{
lean_ctor_set(v___x_3352_, 1, v___x_3361_);
lean_ctor_set(v___x_3352_, 0, v___x_3358_);
v___x_3363_ = v___x_3352_;
goto v_reusejp_3362_;
}
else
{
lean_object* v_reuseFailAlloc_3364_; 
v_reuseFailAlloc_3364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3364_, 0, v___x_3358_);
lean_ctor_set(v_reuseFailAlloc_3364_, 1, v___x_3361_);
v___x_3363_ = v_reuseFailAlloc_3364_;
goto v_reusejp_3362_;
}
v_reusejp_3362_:
{
v_a_3337_ = v___x_3363_;
goto v___jp_3336_;
}
}
v___jp_3365_:
{
if (lean_obj_tag(v___y_3367_) == 0)
{
lean_object* v_a_3368_; lean_object* v___x_3370_; uint8_t v_isShared_3371_; uint8_t v_isSharedCheck_3387_; 
v_a_3368_ = lean_ctor_get(v___y_3367_, 0);
v_isSharedCheck_3387_ = !lean_is_exclusive(v___y_3367_);
if (v_isSharedCheck_3387_ == 0)
{
v___x_3370_ = v___y_3367_;
v_isShared_3371_ = v_isSharedCheck_3387_;
goto v_resetjp_3369_;
}
else
{
lean_inc(v_a_3368_);
lean_dec(v___y_3367_);
v___x_3370_ = lean_box(0);
v_isShared_3371_ = v_isSharedCheck_3387_;
goto v_resetjp_3369_;
}
v_resetjp_3369_:
{
if (lean_obj_tag(v_a_3368_) == 0)
{
lean_object* v_a_3372_; lean_object* v___x_3374_; uint8_t v_isShared_3375_; uint8_t v_isSharedCheck_3386_; 
lean_del_object(v___x_3352_);
lean_dec_ref(v_config_3321_);
v_a_3372_ = lean_ctor_get(v_a_3368_, 0);
v_isSharedCheck_3386_ = !lean_is_exclusive(v_a_3368_);
if (v_isSharedCheck_3386_ == 0)
{
v___x_3374_ = v_a_3368_;
v_isShared_3375_ = v_isSharedCheck_3386_;
goto v_resetjp_3373_;
}
else
{
lean_inc(v_a_3372_);
lean_dec(v_a_3368_);
v___x_3374_ = lean_box(0);
v_isShared_3375_ = v_isSharedCheck_3386_;
goto v_resetjp_3373_;
}
v_resetjp_3373_:
{
lean_object* v___x_3377_; 
if (v_isShared_3375_ == 0)
{
lean_ctor_set_tag(v___x_3374_, 1);
v___x_3377_ = v___x_3374_;
goto v_reusejp_3376_;
}
else
{
lean_object* v_reuseFailAlloc_3385_; 
v_reuseFailAlloc_3385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3385_, 0, v_a_3372_);
v___x_3377_ = v_reuseFailAlloc_3385_;
goto v_reusejp_3376_;
}
v_reusejp_3376_:
{
lean_object* v___x_3378_; lean_object* v___x_3380_; 
v___x_3378_ = lean_box(v___y_3366_);
if (v_isShared_3348_ == 0)
{
lean_ctor_set(v___x_3347_, 1, v___x_3378_);
lean_ctor_set(v___x_3347_, 0, v___x_3377_);
v___x_3380_ = v___x_3347_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3384_; 
v_reuseFailAlloc_3384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3384_, 0, v___x_3377_);
lean_ctor_set(v_reuseFailAlloc_3384_, 1, v___x_3378_);
v___x_3380_ = v_reuseFailAlloc_3384_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
lean_object* v___x_3382_; 
if (v_isShared_3371_ == 0)
{
lean_ctor_set(v___x_3370_, 0, v___x_3380_);
v___x_3382_ = v___x_3370_;
goto v_reusejp_3381_;
}
else
{
lean_object* v_reuseFailAlloc_3383_; 
v_reuseFailAlloc_3383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3383_, 0, v___x_3380_);
v___x_3382_ = v_reuseFailAlloc_3383_;
goto v_reusejp_3381_;
}
v_reusejp_3381_:
{
return v___x_3382_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_3368_, 1);
lean_del_object(v___x_3370_);
lean_del_object(v___x_3347_);
v___y_3360_ = v___y_3366_;
goto v___jp_3359_;
}
}
}
else
{
lean_object* v_a_3388_; lean_object* v___x_3390_; uint8_t v_isShared_3391_; uint8_t v_isSharedCheck_3395_; 
lean_del_object(v___x_3352_);
lean_del_object(v___x_3347_);
lean_dec_ref(v_config_3321_);
v_a_3388_ = lean_ctor_get(v___y_3367_, 0);
v_isSharedCheck_3395_ = !lean_is_exclusive(v___y_3367_);
if (v_isSharedCheck_3395_ == 0)
{
v___x_3390_ = v___y_3367_;
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
else
{
lean_inc(v_a_3388_);
lean_dec(v___y_3367_);
v___x_3390_ = lean_box(0);
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
v_resetjp_3389_:
{
lean_object* v___x_3393_; 
if (v_isShared_3391_ == 0)
{
v___x_3393_ = v___x_3390_;
goto v_reusejp_3392_;
}
else
{
lean_object* v_reuseFailAlloc_3394_; 
v_reuseFailAlloc_3394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3394_, 0, v_a_3388_);
v___x_3393_ = v_reuseFailAlloc_3394_;
goto v_reusejp_3392_;
}
v_reusejp_3392_:
{
return v___x_3393_;
}
}
}
}
v___jp_3397_:
{
lean_object* v___x_3407_; lean_object* v___x_3408_; 
v___x_3407_ = lean_box(0);
v___x_3408_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0(v_anyProgress_3396_, v___x_3407_, v___y_3405_, v___y_3398_, v___y_3403_, v___y_3402_, v___y_3406_, v___y_3399_, v___y_3401_, v___y_3404_);
v___y_3366_ = v___y_3400_;
v___y_3367_ = v___x_3408_;
goto v___jp_3365_;
}
v___jp_3410_:
{
if (v___y_3421_ == 0)
{
lean_object* v_options_3422_; uint8_t v_hasTrace_3423_; 
v_options_3422_ = lean_ctor_get(v___y_3414_, 2);
v_hasTrace_3423_ = lean_ctor_get_uint8(v_options_3422_, sizeof(void*)*1);
if (v_hasTrace_3423_ == 0)
{
lean_dec_ref(v___y_3420_);
lean_del_object(v___x_3356_);
lean_dec(v_fst_3344_);
v___y_3398_ = v___y_3411_;
v___y_3399_ = v___y_3412_;
v___y_3400_ = v___y_3413_;
v___y_3401_ = v___y_3414_;
v___y_3402_ = v___y_3415_;
v___y_3403_ = v___y_3416_;
v___y_3404_ = v___y_3418_;
v___y_3405_ = v___y_3417_;
v___y_3406_ = v___y_3419_;
goto v___jp_3397_;
}
else
{
lean_object* v_inheritedTraceOptions_3424_; lean_object* v___x_3425_; uint8_t v___x_3426_; 
v_inheritedTraceOptions_3424_ = lean_ctor_get(v___y_3414_, 13);
v___x_3425_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_3426_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3424_, v_options_3422_, v___x_3425_);
if (v___x_3426_ == 0)
{
lean_dec_ref(v___y_3420_);
lean_del_object(v___x_3356_);
lean_dec(v_fst_3344_);
v___y_3398_ = v___y_3411_;
v___y_3399_ = v___y_3412_;
v___y_3400_ = v___y_3413_;
v___y_3401_ = v___y_3414_;
v___y_3402_ = v___y_3415_;
v___y_3403_ = v___y_3416_;
v___y_3404_ = v___y_3418_;
v___y_3405_ = v___y_3417_;
v___y_3406_ = v___y_3419_;
goto v___jp_3397_;
}
else
{
lean_object* v___x_3427_; 
v___x_3427_ = l_Lean_MVarId_getType(v_fst_3344_, v___y_3419_, v___y_3412_, v___y_3414_, v___y_3418_);
if (lean_obj_tag(v___x_3427_) == 0)
{
lean_object* v_a_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3432_; 
v_a_3428_ = lean_ctor_get(v___x_3427_, 0);
lean_inc(v_a_3428_);
lean_dec_ref_known(v___x_3427_, 1);
v___x_3429_ = l_Lean_MessageData_ofExpr(v_a_3428_);
v___x_3430_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1);
if (v_isShared_3357_ == 0)
{
lean_ctor_set_tag(v___x_3356_, 7);
lean_ctor_set(v___x_3356_, 1, v___x_3430_);
lean_ctor_set(v___x_3356_, 0, v___x_3429_);
v___x_3432_ = v___x_3356_;
goto v_reusejp_3431_;
}
else
{
lean_object* v_reuseFailAlloc_3446_; 
v_reuseFailAlloc_3446_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3446_, 0, v___x_3429_);
lean_ctor_set(v_reuseFailAlloc_3446_, 1, v___x_3430_);
v___x_3432_ = v_reuseFailAlloc_3446_;
goto v_reusejp_3431_;
}
v_reusejp_3431_:
{
lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; 
v___x_3433_ = l_Lean_Exception_toMessageData(v___y_3420_);
v___x_3434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3434_, 0, v___x_3432_);
lean_ctor_set(v___x_3434_, 1, v___x_3433_);
v___x_3435_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_3409_, v___x_3434_, v___y_3419_, v___y_3412_, v___y_3414_, v___y_3418_);
if (lean_obj_tag(v___x_3435_) == 0)
{
lean_object* v_a_3436_; lean_object* v___x_3437_; 
v_a_3436_ = lean_ctor_get(v___x_3435_, 0);
lean_inc(v_a_3436_);
lean_dec_ref_known(v___x_3435_, 1);
v___x_3437_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___lam__0(v_anyProgress_3396_, v_a_3436_, v___y_3417_, v___y_3411_, v___y_3416_, v___y_3415_, v___y_3419_, v___y_3412_, v___y_3414_, v___y_3418_);
v___y_3366_ = v___y_3413_;
v___y_3367_ = v___x_3437_;
goto v___jp_3365_;
}
else
{
lean_object* v_a_3438_; lean_object* v___x_3440_; uint8_t v_isShared_3441_; uint8_t v_isSharedCheck_3445_; 
lean_del_object(v___x_3352_);
lean_del_object(v___x_3347_);
lean_dec_ref(v_config_3321_);
v_a_3438_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3445_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3445_ == 0)
{
v___x_3440_ = v___x_3435_;
v_isShared_3441_ = v_isSharedCheck_3445_;
goto v_resetjp_3439_;
}
else
{
lean_inc(v_a_3438_);
lean_dec(v___x_3435_);
v___x_3440_ = lean_box(0);
v_isShared_3441_ = v_isSharedCheck_3445_;
goto v_resetjp_3439_;
}
v_resetjp_3439_:
{
lean_object* v___x_3443_; 
if (v_isShared_3441_ == 0)
{
v___x_3443_ = v___x_3440_;
goto v_reusejp_3442_;
}
else
{
lean_object* v_reuseFailAlloc_3444_; 
v_reuseFailAlloc_3444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3444_, 0, v_a_3438_);
v___x_3443_ = v_reuseFailAlloc_3444_;
goto v_reusejp_3442_;
}
v_reusejp_3442_:
{
return v___x_3443_;
}
}
}
}
}
else
{
lean_object* v_a_3447_; lean_object* v___x_3449_; uint8_t v_isShared_3450_; uint8_t v_isSharedCheck_3454_; 
lean_dec_ref(v___y_3420_);
lean_del_object(v___x_3356_);
lean_del_object(v___x_3352_);
lean_del_object(v___x_3347_);
lean_dec_ref(v_config_3321_);
v_a_3447_ = lean_ctor_get(v___x_3427_, 0);
v_isSharedCheck_3454_ = !lean_is_exclusive(v___x_3427_);
if (v_isSharedCheck_3454_ == 0)
{
v___x_3449_ = v___x_3427_;
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
else
{
lean_inc(v_a_3447_);
lean_dec(v___x_3427_);
v___x_3449_ = lean_box(0);
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
v_resetjp_3448_:
{
lean_object* v___x_3452_; 
if (v_isShared_3450_ == 0)
{
v___x_3452_ = v___x_3449_;
goto v_reusejp_3451_;
}
else
{
lean_object* v_reuseFailAlloc_3453_; 
v_reuseFailAlloc_3453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3453_, 0, v_a_3447_);
v___x_3452_ = v_reuseFailAlloc_3453_;
goto v_reusejp_3451_;
}
v_reusejp_3451_:
{
return v___x_3452_;
}
}
}
}
}
}
else
{
lean_object* v___x_3455_; 
lean_del_object(v___x_3356_);
lean_del_object(v___x_3352_);
lean_del_object(v___x_3347_);
lean_dec(v_fst_3344_);
lean_dec_ref(v_config_3321_);
v___x_3455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3455_, 0, v___y_3420_);
return v___x_3455_;
}
}
v___jp_3456_:
{
lean_object* v_keyedConfig_3466_; uint8_t v_trackZetaDelta_3467_; lean_object* v_zetaDeltaSet_3468_; lean_object* v_lctx_3469_; lean_object* v_localInstances_3470_; lean_object* v_defEqCtx_x3f_3471_; lean_object* v_synthPendingDepth_3472_; lean_object* v_customCanUnfoldPredicate_x3f_3473_; uint8_t v_univApprox_3474_; uint8_t v_inTypeClassResolution_3475_; uint8_t v_cacheInferType_3476_; uint8_t v___x_3477_; lean_object* v___x_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; 
v_keyedConfig_3466_ = lean_ctor_get(v___y_3462_, 0);
v_trackZetaDelta_3467_ = lean_ctor_get_uint8(v___y_3462_, sizeof(void*)*7);
v_zetaDeltaSet_3468_ = lean_ctor_get(v___y_3462_, 1);
v_lctx_3469_ = lean_ctor_get(v___y_3462_, 2);
v_localInstances_3470_ = lean_ctor_get(v___y_3462_, 3);
v_defEqCtx_x3f_3471_ = lean_ctor_get(v___y_3462_, 4);
v_synthPendingDepth_3472_ = lean_ctor_get(v___y_3462_, 5);
v_customCanUnfoldPredicate_x3f_3473_ = lean_ctor_get(v___y_3462_, 6);
v_univApprox_3474_ = lean_ctor_get_uint8(v___y_3462_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3475_ = lean_ctor_get_uint8(v___y_3462_, sizeof(void*)*7 + 2);
v_cacheInferType_3476_ = lean_ctor_get_uint8(v___y_3462_, sizeof(void*)*7 + 3);
v___x_3477_ = 3;
lean_inc_ref(v_keyedConfig_3466_);
v___x_3478_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3477_, v_keyedConfig_3466_);
lean_inc(v_customCanUnfoldPredicate_x3f_3473_);
lean_inc(v_synthPendingDepth_3472_);
lean_inc(v_defEqCtx_x3f_3471_);
lean_inc_ref(v_localInstances_3470_);
lean_inc_ref(v_lctx_3469_);
lean_inc(v_zetaDeltaSet_3468_);
v___x_3479_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3479_, 0, v___x_3478_);
lean_ctor_set(v___x_3479_, 1, v_zetaDeltaSet_3468_);
lean_ctor_set(v___x_3479_, 2, v_lctx_3469_);
lean_ctor_set(v___x_3479_, 3, v_localInstances_3470_);
lean_ctor_set(v___x_3479_, 4, v_defEqCtx_x3f_3471_);
lean_ctor_set(v___x_3479_, 5, v_synthPendingDepth_3472_);
lean_ctor_set(v___x_3479_, 6, v_customCanUnfoldPredicate_x3f_3473_);
lean_ctor_set_uint8(v___x_3479_, sizeof(void*)*7, v_trackZetaDelta_3467_);
lean_ctor_set_uint8(v___x_3479_, sizeof(void*)*7 + 1, v_univApprox_3474_);
lean_ctor_set_uint8(v___x_3479_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3475_);
lean_ctor_set_uint8(v___x_3479_, sizeof(void*)*7 + 3, v_cacheInferType_3476_);
lean_inc(v_fst_3344_);
v___x_3480_ = lp_mathlib_Lean_MVarId_applyRflOrId(v_fst_3344_, v___x_3479_, v___y_3463_, v___y_3464_, v___y_3465_);
lean_dec_ref_known(v___x_3479_, 7);
if (lean_obj_tag(v___x_3480_) == 0)
{
lean_dec_ref_known(v___x_3480_, 1);
lean_del_object(v___x_3356_);
lean_del_object(v___x_3347_);
lean_dec(v_fst_3344_);
v___y_3360_ = v_anyProgress_3457_;
goto v___jp_3359_;
}
else
{
if (lean_obj_tag(v___x_3480_) == 0)
{
lean_dec_ref_known(v___x_3480_, 1);
lean_del_object(v___x_3356_);
lean_del_object(v___x_3347_);
lean_dec(v_fst_3344_);
v___y_3360_ = v_anyProgress_3457_;
goto v___jp_3359_;
}
else
{
lean_object* v_a_3481_; uint8_t v___x_3482_; 
v_a_3481_ = lean_ctor_get(v___x_3480_, 0);
lean_inc(v_a_3481_);
lean_dec_ref_known(v___x_3480_, 1);
v___x_3482_ = l_Lean_Exception_isInterrupt(v_a_3481_);
if (v___x_3482_ == 0)
{
uint8_t v___x_3483_; 
lean_inc(v_a_3481_);
v___x_3483_ = l_Lean_Exception_isRuntime(v_a_3481_);
v___y_3411_ = v___y_3459_;
v___y_3412_ = v___y_3463_;
v___y_3413_ = v_anyProgress_3457_;
v___y_3414_ = v___y_3464_;
v___y_3415_ = v___y_3461_;
v___y_3416_ = v___y_3460_;
v___y_3417_ = v___y_3458_;
v___y_3418_ = v___y_3465_;
v___y_3419_ = v___y_3462_;
v___y_3420_ = v_a_3481_;
v___y_3421_ = v___x_3483_;
goto v___jp_3410_;
}
else
{
v___y_3411_ = v___y_3459_;
v___y_3412_ = v___y_3463_;
v___y_3413_ = v_anyProgress_3457_;
v___y_3414_ = v___y_3464_;
v___y_3415_ = v___y_3461_;
v___y_3416_ = v___y_3460_;
v___y_3417_ = v___y_3458_;
v___y_3418_ = v___y_3465_;
v___y_3419_ = v___y_3462_;
v___y_3420_ = v_a_3481_;
v___y_3421_ = v___x_3482_;
goto v___jp_3410_;
}
}
}
}
v___jp_3484_:
{
lean_object* v___x_3486_; 
lean_inc_ref(v_config_3321_);
lean_inc(v_fst_3344_);
v___x_3486_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(v_fst_3344_, v___y_3485_, v_config_3321_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_, v___y_3331_, v___y_3332_, v___y_3333_, v___y_3334_);
if (lean_obj_tag(v___x_3486_) == 0)
{
lean_object* v_a_3487_; uint8_t v___x_3488_; 
v_a_3487_ = lean_ctor_get(v___x_3486_, 0);
lean_inc(v_a_3487_);
lean_dec_ref_known(v___x_3486_, 1);
v___x_3488_ = lean_unbox(v_a_3487_);
lean_dec(v_a_3487_);
if (v___x_3488_ == 0)
{
uint8_t v___x_3489_; 
v___x_3489_ = lean_unbox(v_snd_3350_);
lean_dec(v_snd_3350_);
v_anyProgress_3457_ = v___x_3489_;
v___y_3458_ = v___y_3327_;
v___y_3459_ = v___y_3328_;
v___y_3460_ = v___y_3329_;
v___y_3461_ = v___y_3330_;
v___y_3462_ = v___y_3331_;
v___y_3463_ = v___y_3332_;
v___y_3464_ = v___y_3333_;
v___y_3465_ = v___y_3334_;
goto v___jp_3456_;
}
else
{
lean_object* v___x_3490_; lean_object* v___x_3491_; 
lean_del_object(v___x_3356_);
lean_del_object(v___x_3352_);
lean_dec(v_snd_3350_);
lean_del_object(v___x_3347_);
lean_dec(v_fst_3344_);
v___x_3490_ = lean_box(v___x_3341_);
v___x_3491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3491_, 0, v___x_3358_);
lean_ctor_set(v___x_3491_, 1, v___x_3490_);
v_a_3337_ = v___x_3491_;
goto v___jp_3336_;
}
}
else
{
lean_object* v_a_3492_; lean_object* v___x_3494_; uint8_t v_isShared_3495_; uint8_t v_isSharedCheck_3499_; 
lean_del_object(v___x_3356_);
lean_del_object(v___x_3352_);
lean_dec(v_snd_3350_);
lean_del_object(v___x_3347_);
lean_dec(v_fst_3344_);
lean_dec_ref(v_config_3321_);
v_a_3492_ = lean_ctor_get(v___x_3486_, 0);
v_isSharedCheck_3499_ = !lean_is_exclusive(v___x_3486_);
if (v_isSharedCheck_3499_ == 0)
{
v___x_3494_ = v___x_3486_;
v_isShared_3495_ = v_isSharedCheck_3499_;
goto v_resetjp_3493_;
}
else
{
lean_inc(v_a_3492_);
lean_dec(v___x_3486_);
v___x_3494_ = lean_box(0);
v_isShared_3495_ = v_isSharedCheck_3499_;
goto v_resetjp_3493_;
}
v_resetjp_3493_:
{
lean_object* v___x_3497_; 
if (v_isShared_3495_ == 0)
{
v___x_3497_ = v___x_3494_;
goto v_reusejp_3496_;
}
else
{
lean_object* v_reuseFailAlloc_3498_; 
v_reuseFailAlloc_3498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3498_, 0, v_a_3492_);
v___x_3497_ = v_reuseFailAlloc_3498_;
goto v_reusejp_3496_;
}
v_reusejp_3496_:
{
return v___x_3497_;
}
}
}
}
v___jp_3500_:
{
if (v___y_3501_ == 0)
{
v___y_3485_ = v___x_3341_;
goto v___jp_3484_;
}
else
{
v___y_3485_ = v_anyProgress_3396_;
goto v___jp_3484_;
}
}
}
}
}
}
v___jp_3336_:
{
size_t v___x_3338_; size_t v___x_3339_; 
v___x_3338_ = ((size_t)1ULL);
v___x_3339_ = lean_usize_add(v_i_3325_, v___x_3338_);
v_i_3325_ = v___x_3339_;
v_b_3326_ = v_a_3337_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15(uint8_t v___x_3510_, lean_object* v_config_3511_, uint8_t v___x_3512_, uint8_t v_forward_3513_, lean_object* v_as_3514_, size_t v_sz_3515_, size_t v_i_3516_, lean_object* v_b_3517_, lean_object* v___y_3518_, lean_object* v___y_3519_, lean_object* v___y_3520_, lean_object* v___y_3521_, lean_object* v___y_3522_, lean_object* v___y_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_){
_start:
{
lean_object* v_a_3528_; uint8_t v___x_3532_; 
v___x_3532_ = lean_usize_dec_lt(v_i_3516_, v_sz_3515_);
if (v___x_3532_ == 0)
{
lean_object* v___x_3533_; 
lean_dec_ref(v_config_3511_);
v___x_3533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3533_, 0, v_b_3517_);
return v___x_3533_;
}
else
{
lean_object* v_a_3534_; lean_object* v_fst_3535_; lean_object* v_snd_3536_; lean_object* v___x_3538_; uint8_t v_isShared_3539_; uint8_t v_isSharedCheck_3700_; 
v_a_3534_ = lean_array_uget(v_as_3514_, v_i_3516_);
v_fst_3535_ = lean_ctor_get(v_a_3534_, 0);
v_snd_3536_ = lean_ctor_get(v_a_3534_, 1);
v_isSharedCheck_3700_ = !lean_is_exclusive(v_a_3534_);
if (v_isSharedCheck_3700_ == 0)
{
v___x_3538_ = v_a_3534_;
v_isShared_3539_ = v_isSharedCheck_3700_;
goto v_resetjp_3537_;
}
else
{
lean_inc(v_snd_3536_);
lean_inc(v_fst_3535_);
lean_dec(v_a_3534_);
v___x_3538_ = lean_box(0);
v_isShared_3539_ = v_isSharedCheck_3700_;
goto v_resetjp_3537_;
}
v_resetjp_3537_:
{
lean_object* v___x_3540_; lean_object* v_snd_3541_; lean_object* v___x_3543_; uint8_t v_isShared_3544_; uint8_t v_isSharedCheck_3698_; 
v___x_3540_ = lean_st_ref_get(v___y_3519_);
v_snd_3541_ = lean_ctor_get(v_b_3517_, 1);
v_isSharedCheck_3698_ = !lean_is_exclusive(v_b_3517_);
if (v_isSharedCheck_3698_ == 0)
{
lean_object* v_unused_3699_; 
v_unused_3699_ = lean_ctor_get(v_b_3517_, 0);
lean_dec(v_unused_3699_);
v___x_3543_ = v_b_3517_;
v_isShared_3544_ = v_isSharedCheck_3698_;
goto v_resetjp_3542_;
}
else
{
lean_inc(v_snd_3541_);
lean_dec(v_b_3517_);
v___x_3543_ = lean_box(0);
v_isShared_3544_ = v_isSharedCheck_3698_;
goto v_resetjp_3542_;
}
v_resetjp_3542_:
{
lean_object* v_progress_3545_; lean_object* v___x_3547_; uint8_t v_isShared_3548_; uint8_t v_isSharedCheck_3696_; 
v_progress_3545_ = lean_ctor_get(v___x_3540_, 1);
v_isSharedCheck_3696_ = !lean_is_exclusive(v___x_3540_);
if (v_isSharedCheck_3696_ == 0)
{
lean_object* v_unused_3697_; 
v_unused_3697_ = lean_ctor_get(v___x_3540_, 0);
lean_dec(v_unused_3697_);
v___x_3547_ = v___x_3540_;
v_isShared_3548_ = v_isSharedCheck_3696_;
goto v_resetjp_3546_;
}
else
{
lean_inc(v_progress_3545_);
lean_dec(v___x_3540_);
v___x_3547_ = lean_box(0);
v_isShared_3548_ = v_isSharedCheck_3696_;
goto v_resetjp_3546_;
}
v_resetjp_3546_:
{
lean_object* v___x_3549_; uint8_t v___y_3551_; uint8_t v___y_3557_; lean_object* v___y_3558_; lean_object* v___y_3588_; lean_object* v___y_3589_; lean_object* v___y_3590_; lean_object* v___y_3591_; lean_object* v___y_3592_; uint8_t v___y_3593_; lean_object* v___y_3594_; lean_object* v___y_3595_; lean_object* v___y_3596_; lean_object* v_cls_3599_; lean_object* v___y_3601_; lean_object* v___y_3602_; lean_object* v___y_3603_; lean_object* v___y_3604_; lean_object* v___y_3605_; lean_object* v___y_3606_; uint8_t v___y_3607_; lean_object* v___y_3608_; lean_object* v___y_3609_; lean_object* v___y_3610_; uint8_t v___y_3611_; uint8_t v_anyProgress_3647_; lean_object* v___y_3648_; lean_object* v___y_3649_; lean_object* v___y_3650_; lean_object* v___y_3651_; lean_object* v___y_3652_; lean_object* v___y_3653_; lean_object* v___y_3654_; lean_object* v___y_3655_; uint8_t v___y_3675_; uint8_t v___y_3691_; 
v___x_3549_ = lean_box(0);
v_cls_3599_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
if (lean_obj_tag(v_progress_3545_) == 2)
{
uint8_t v___x_3692_; 
lean_dec_ref_known(v_progress_3545_, 1);
lean_dec(v_snd_3536_);
v___x_3692_ = lean_unbox(v_snd_3541_);
lean_dec(v_snd_3541_);
v_anyProgress_3647_ = v___x_3692_;
v___y_3648_ = v___y_3518_;
v___y_3649_ = v___y_3519_;
v___y_3650_ = v___y_3520_;
v___y_3651_ = v___y_3521_;
v___y_3652_ = v___y_3522_;
v___y_3653_ = v___y_3523_;
v___y_3654_ = v___y_3524_;
v___y_3655_ = v___y_3525_;
goto v___jp_3646_;
}
else
{
lean_dec(v_progress_3545_);
if (v___x_3510_ == 0)
{
if (v_forward_3513_ == 0)
{
uint8_t v___x_3693_; 
v___x_3693_ = lean_unbox(v_snd_3536_);
lean_dec(v_snd_3536_);
if (v___x_3693_ == 0)
{
v___y_3691_ = v___x_3532_;
goto v___jp_3690_;
}
else
{
v___y_3675_ = v___x_3512_;
goto v___jp_3674_;
}
}
else
{
uint8_t v___x_3694_; 
v___x_3694_ = lean_unbox(v_snd_3536_);
lean_dec(v_snd_3536_);
v___y_3691_ = v___x_3694_;
goto v___jp_3690_;
}
}
else
{
uint8_t v___x_3695_; 
lean_dec(v_snd_3536_);
v___x_3695_ = lean_unbox(v_snd_3541_);
lean_dec(v_snd_3541_);
v_anyProgress_3647_ = v___x_3695_;
v___y_3648_ = v___y_3518_;
v___y_3649_ = v___y_3519_;
v___y_3650_ = v___y_3520_;
v___y_3651_ = v___y_3521_;
v___y_3652_ = v___y_3522_;
v___y_3653_ = v___y_3523_;
v___y_3654_ = v___y_3524_;
v___y_3655_ = v___y_3525_;
goto v___jp_3646_;
}
}
v___jp_3550_:
{
lean_object* v___x_3552_; lean_object* v___x_3554_; 
v___x_3552_ = lean_box(v___y_3551_);
if (v_isShared_3544_ == 0)
{
lean_ctor_set(v___x_3543_, 1, v___x_3552_);
lean_ctor_set(v___x_3543_, 0, v___x_3549_);
v___x_3554_ = v___x_3543_;
goto v_reusejp_3553_;
}
else
{
lean_object* v_reuseFailAlloc_3555_; 
v_reuseFailAlloc_3555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3555_, 0, v___x_3549_);
lean_ctor_set(v_reuseFailAlloc_3555_, 1, v___x_3552_);
v___x_3554_ = v_reuseFailAlloc_3555_;
goto v_reusejp_3553_;
}
v_reusejp_3553_:
{
v_a_3528_ = v___x_3554_;
goto v___jp_3527_;
}
}
v___jp_3556_:
{
if (lean_obj_tag(v___y_3558_) == 0)
{
lean_object* v_a_3559_; lean_object* v___x_3561_; uint8_t v_isShared_3562_; uint8_t v_isSharedCheck_3578_; 
v_a_3559_ = lean_ctor_get(v___y_3558_, 0);
v_isSharedCheck_3578_ = !lean_is_exclusive(v___y_3558_);
if (v_isSharedCheck_3578_ == 0)
{
v___x_3561_ = v___y_3558_;
v_isShared_3562_ = v_isSharedCheck_3578_;
goto v_resetjp_3560_;
}
else
{
lean_inc(v_a_3559_);
lean_dec(v___y_3558_);
v___x_3561_ = lean_box(0);
v_isShared_3562_ = v_isSharedCheck_3578_;
goto v_resetjp_3560_;
}
v_resetjp_3560_:
{
if (lean_obj_tag(v_a_3559_) == 0)
{
lean_object* v_a_3563_; lean_object* v___x_3565_; uint8_t v_isShared_3566_; uint8_t v_isSharedCheck_3577_; 
lean_del_object(v___x_3543_);
lean_dec_ref(v_config_3511_);
v_a_3563_ = lean_ctor_get(v_a_3559_, 0);
v_isSharedCheck_3577_ = !lean_is_exclusive(v_a_3559_);
if (v_isSharedCheck_3577_ == 0)
{
v___x_3565_ = v_a_3559_;
v_isShared_3566_ = v_isSharedCheck_3577_;
goto v_resetjp_3564_;
}
else
{
lean_inc(v_a_3563_);
lean_dec(v_a_3559_);
v___x_3565_ = lean_box(0);
v_isShared_3566_ = v_isSharedCheck_3577_;
goto v_resetjp_3564_;
}
v_resetjp_3564_:
{
lean_object* v___x_3568_; 
if (v_isShared_3566_ == 0)
{
lean_ctor_set_tag(v___x_3565_, 1);
v___x_3568_ = v___x_3565_;
goto v_reusejp_3567_;
}
else
{
lean_object* v_reuseFailAlloc_3576_; 
v_reuseFailAlloc_3576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3576_, 0, v_a_3563_);
v___x_3568_ = v_reuseFailAlloc_3576_;
goto v_reusejp_3567_;
}
v_reusejp_3567_:
{
lean_object* v___x_3569_; lean_object* v___x_3571_; 
v___x_3569_ = lean_box(v___y_3557_);
if (v_isShared_3539_ == 0)
{
lean_ctor_set(v___x_3538_, 1, v___x_3569_);
lean_ctor_set(v___x_3538_, 0, v___x_3568_);
v___x_3571_ = v___x_3538_;
goto v_reusejp_3570_;
}
else
{
lean_object* v_reuseFailAlloc_3575_; 
v_reuseFailAlloc_3575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3575_, 0, v___x_3568_);
lean_ctor_set(v_reuseFailAlloc_3575_, 1, v___x_3569_);
v___x_3571_ = v_reuseFailAlloc_3575_;
goto v_reusejp_3570_;
}
v_reusejp_3570_:
{
lean_object* v___x_3573_; 
if (v_isShared_3562_ == 0)
{
lean_ctor_set(v___x_3561_, 0, v___x_3571_);
v___x_3573_ = v___x_3561_;
goto v_reusejp_3572_;
}
else
{
lean_object* v_reuseFailAlloc_3574_; 
v_reuseFailAlloc_3574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3574_, 0, v___x_3571_);
v___x_3573_ = v_reuseFailAlloc_3574_;
goto v_reusejp_3572_;
}
v_reusejp_3572_:
{
return v___x_3573_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_3559_, 1);
lean_del_object(v___x_3561_);
lean_del_object(v___x_3538_);
v___y_3551_ = v___y_3557_;
goto v___jp_3550_;
}
}
}
else
{
lean_object* v_a_3579_; lean_object* v___x_3581_; uint8_t v_isShared_3582_; uint8_t v_isSharedCheck_3586_; 
lean_del_object(v___x_3543_);
lean_del_object(v___x_3538_);
lean_dec_ref(v_config_3511_);
v_a_3579_ = lean_ctor_get(v___y_3558_, 0);
v_isSharedCheck_3586_ = !lean_is_exclusive(v___y_3558_);
if (v_isSharedCheck_3586_ == 0)
{
v___x_3581_ = v___y_3558_;
v_isShared_3582_ = v_isSharedCheck_3586_;
goto v_resetjp_3580_;
}
else
{
lean_inc(v_a_3579_);
lean_dec(v___y_3558_);
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
lean_object* v___x_3597_; lean_object* v___x_3598_; 
v___x_3597_ = lean_box(0);
v___x_3598_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(v___x_3510_, v___x_3597_, v___y_3591_, v___y_3588_, v___y_3595_, v___y_3594_, v___y_3589_, v___y_3592_, v___y_3596_, v___y_3590_);
v___y_3557_ = v___y_3593_;
v___y_3558_ = v___x_3598_;
goto v___jp_3556_;
}
v___jp_3600_:
{
if (v___y_3611_ == 0)
{
lean_object* v_options_3612_; uint8_t v_hasTrace_3613_; 
v_options_3612_ = lean_ctor_get(v___y_3610_, 2);
v_hasTrace_3613_ = lean_ctor_get_uint8(v_options_3612_, sizeof(void*)*1);
if (v_hasTrace_3613_ == 0)
{
lean_dec_ref(v___y_3605_);
lean_del_object(v___x_3547_);
lean_dec(v_fst_3535_);
v___y_3588_ = v___y_3601_;
v___y_3589_ = v___y_3602_;
v___y_3590_ = v___y_3604_;
v___y_3591_ = v___y_3603_;
v___y_3592_ = v___y_3606_;
v___y_3593_ = v___y_3607_;
v___y_3594_ = v___y_3609_;
v___y_3595_ = v___y_3608_;
v___y_3596_ = v___y_3610_;
goto v___jp_3587_;
}
else
{
lean_object* v_inheritedTraceOptions_3614_; lean_object* v___x_3615_; uint8_t v___x_3616_; 
v_inheritedTraceOptions_3614_ = lean_ctor_get(v___y_3610_, 13);
v___x_3615_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_3616_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3614_, v_options_3612_, v___x_3615_);
if (v___x_3616_ == 0)
{
lean_dec_ref(v___y_3605_);
lean_del_object(v___x_3547_);
lean_dec(v_fst_3535_);
v___y_3588_ = v___y_3601_;
v___y_3589_ = v___y_3602_;
v___y_3590_ = v___y_3604_;
v___y_3591_ = v___y_3603_;
v___y_3592_ = v___y_3606_;
v___y_3593_ = v___y_3607_;
v___y_3594_ = v___y_3609_;
v___y_3595_ = v___y_3608_;
v___y_3596_ = v___y_3610_;
goto v___jp_3587_;
}
else
{
lean_object* v___x_3617_; 
v___x_3617_ = l_Lean_MVarId_getType(v_fst_3535_, v___y_3602_, v___y_3606_, v___y_3610_, v___y_3604_);
if (lean_obj_tag(v___x_3617_) == 0)
{
lean_object* v_a_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; lean_object* v___x_3622_; 
v_a_3618_ = lean_ctor_get(v___x_3617_, 0);
lean_inc(v_a_3618_);
lean_dec_ref_known(v___x_3617_, 1);
v___x_3619_ = l_Lean_MessageData_ofExpr(v_a_3618_);
v___x_3620_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1);
if (v_isShared_3548_ == 0)
{
lean_ctor_set_tag(v___x_3547_, 7);
lean_ctor_set(v___x_3547_, 1, v___x_3620_);
lean_ctor_set(v___x_3547_, 0, v___x_3619_);
v___x_3622_ = v___x_3547_;
goto v_reusejp_3621_;
}
else
{
lean_object* v_reuseFailAlloc_3636_; 
v_reuseFailAlloc_3636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3636_, 0, v___x_3619_);
lean_ctor_set(v_reuseFailAlloc_3636_, 1, v___x_3620_);
v___x_3622_ = v_reuseFailAlloc_3636_;
goto v_reusejp_3621_;
}
v_reusejp_3621_:
{
lean_object* v___x_3623_; lean_object* v___x_3624_; lean_object* v___x_3625_; 
v___x_3623_ = l_Lean_Exception_toMessageData(v___y_3605_);
v___x_3624_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3624_, 0, v___x_3622_);
lean_ctor_set(v___x_3624_, 1, v___x_3623_);
v___x_3625_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_3599_, v___x_3624_, v___y_3602_, v___y_3606_, v___y_3610_, v___y_3604_);
if (lean_obj_tag(v___x_3625_) == 0)
{
lean_object* v_a_3626_; lean_object* v___x_3627_; 
v_a_3626_ = lean_ctor_get(v___x_3625_, 0);
lean_inc(v_a_3626_);
lean_dec_ref_known(v___x_3625_, 1);
v___x_3627_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(v___x_3510_, v_a_3626_, v___y_3603_, v___y_3601_, v___y_3608_, v___y_3609_, v___y_3602_, v___y_3606_, v___y_3610_, v___y_3604_);
v___y_3557_ = v___y_3607_;
v___y_3558_ = v___x_3627_;
goto v___jp_3556_;
}
else
{
lean_object* v_a_3628_; lean_object* v___x_3630_; uint8_t v_isShared_3631_; uint8_t v_isSharedCheck_3635_; 
lean_del_object(v___x_3543_);
lean_del_object(v___x_3538_);
lean_dec_ref(v_config_3511_);
v_a_3628_ = lean_ctor_get(v___x_3625_, 0);
v_isSharedCheck_3635_ = !lean_is_exclusive(v___x_3625_);
if (v_isSharedCheck_3635_ == 0)
{
v___x_3630_ = v___x_3625_;
v_isShared_3631_ = v_isSharedCheck_3635_;
goto v_resetjp_3629_;
}
else
{
lean_inc(v_a_3628_);
lean_dec(v___x_3625_);
v___x_3630_ = lean_box(0);
v_isShared_3631_ = v_isSharedCheck_3635_;
goto v_resetjp_3629_;
}
v_resetjp_3629_:
{
lean_object* v___x_3633_; 
if (v_isShared_3631_ == 0)
{
v___x_3633_ = v___x_3630_;
goto v_reusejp_3632_;
}
else
{
lean_object* v_reuseFailAlloc_3634_; 
v_reuseFailAlloc_3634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3634_, 0, v_a_3628_);
v___x_3633_ = v_reuseFailAlloc_3634_;
goto v_reusejp_3632_;
}
v_reusejp_3632_:
{
return v___x_3633_;
}
}
}
}
}
else
{
lean_object* v_a_3637_; lean_object* v___x_3639_; uint8_t v_isShared_3640_; uint8_t v_isSharedCheck_3644_; 
lean_dec_ref(v___y_3605_);
lean_del_object(v___x_3547_);
lean_del_object(v___x_3543_);
lean_del_object(v___x_3538_);
lean_dec_ref(v_config_3511_);
v_a_3637_ = lean_ctor_get(v___x_3617_, 0);
v_isSharedCheck_3644_ = !lean_is_exclusive(v___x_3617_);
if (v_isSharedCheck_3644_ == 0)
{
v___x_3639_ = v___x_3617_;
v_isShared_3640_ = v_isSharedCheck_3644_;
goto v_resetjp_3638_;
}
else
{
lean_inc(v_a_3637_);
lean_dec(v___x_3617_);
v___x_3639_ = lean_box(0);
v_isShared_3640_ = v_isSharedCheck_3644_;
goto v_resetjp_3638_;
}
v_resetjp_3638_:
{
lean_object* v___x_3642_; 
if (v_isShared_3640_ == 0)
{
v___x_3642_ = v___x_3639_;
goto v_reusejp_3641_;
}
else
{
lean_object* v_reuseFailAlloc_3643_; 
v_reuseFailAlloc_3643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3643_, 0, v_a_3637_);
v___x_3642_ = v_reuseFailAlloc_3643_;
goto v_reusejp_3641_;
}
v_reusejp_3641_:
{
return v___x_3642_;
}
}
}
}
}
}
else
{
lean_object* v___x_3645_; 
lean_del_object(v___x_3547_);
lean_del_object(v___x_3543_);
lean_del_object(v___x_3538_);
lean_dec(v_fst_3535_);
lean_dec_ref(v_config_3511_);
v___x_3645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3645_, 0, v___y_3605_);
return v___x_3645_;
}
}
v___jp_3646_:
{
lean_object* v_keyedConfig_3656_; uint8_t v_trackZetaDelta_3657_; lean_object* v_zetaDeltaSet_3658_; lean_object* v_lctx_3659_; lean_object* v_localInstances_3660_; lean_object* v_defEqCtx_x3f_3661_; lean_object* v_synthPendingDepth_3662_; lean_object* v_customCanUnfoldPredicate_x3f_3663_; uint8_t v_univApprox_3664_; uint8_t v_inTypeClassResolution_3665_; uint8_t v_cacheInferType_3666_; uint8_t v___x_3667_; lean_object* v___x_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; 
v_keyedConfig_3656_ = lean_ctor_get(v___y_3652_, 0);
v_trackZetaDelta_3657_ = lean_ctor_get_uint8(v___y_3652_, sizeof(void*)*7);
v_zetaDeltaSet_3658_ = lean_ctor_get(v___y_3652_, 1);
v_lctx_3659_ = lean_ctor_get(v___y_3652_, 2);
v_localInstances_3660_ = lean_ctor_get(v___y_3652_, 3);
v_defEqCtx_x3f_3661_ = lean_ctor_get(v___y_3652_, 4);
v_synthPendingDepth_3662_ = lean_ctor_get(v___y_3652_, 5);
v_customCanUnfoldPredicate_x3f_3663_ = lean_ctor_get(v___y_3652_, 6);
v_univApprox_3664_ = lean_ctor_get_uint8(v___y_3652_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3665_ = lean_ctor_get_uint8(v___y_3652_, sizeof(void*)*7 + 2);
v_cacheInferType_3666_ = lean_ctor_get_uint8(v___y_3652_, sizeof(void*)*7 + 3);
v___x_3667_ = 3;
lean_inc_ref(v_keyedConfig_3656_);
v___x_3668_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3667_, v_keyedConfig_3656_);
lean_inc(v_customCanUnfoldPredicate_x3f_3663_);
lean_inc(v_synthPendingDepth_3662_);
lean_inc(v_defEqCtx_x3f_3661_);
lean_inc_ref(v_localInstances_3660_);
lean_inc_ref(v_lctx_3659_);
lean_inc(v_zetaDeltaSet_3658_);
v___x_3669_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3669_, 0, v___x_3668_);
lean_ctor_set(v___x_3669_, 1, v_zetaDeltaSet_3658_);
lean_ctor_set(v___x_3669_, 2, v_lctx_3659_);
lean_ctor_set(v___x_3669_, 3, v_localInstances_3660_);
lean_ctor_set(v___x_3669_, 4, v_defEqCtx_x3f_3661_);
lean_ctor_set(v___x_3669_, 5, v_synthPendingDepth_3662_);
lean_ctor_set(v___x_3669_, 6, v_customCanUnfoldPredicate_x3f_3663_);
lean_ctor_set_uint8(v___x_3669_, sizeof(void*)*7, v_trackZetaDelta_3657_);
lean_ctor_set_uint8(v___x_3669_, sizeof(void*)*7 + 1, v_univApprox_3664_);
lean_ctor_set_uint8(v___x_3669_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3665_);
lean_ctor_set_uint8(v___x_3669_, sizeof(void*)*7 + 3, v_cacheInferType_3666_);
lean_inc(v_fst_3535_);
v___x_3670_ = lp_mathlib_Lean_MVarId_applyRflOrId(v_fst_3535_, v___x_3669_, v___y_3653_, v___y_3654_, v___y_3655_);
lean_dec_ref_known(v___x_3669_, 7);
if (lean_obj_tag(v___x_3670_) == 0)
{
lean_dec_ref_known(v___x_3670_, 1);
lean_del_object(v___x_3547_);
lean_del_object(v___x_3538_);
lean_dec(v_fst_3535_);
v___y_3551_ = v_anyProgress_3647_;
goto v___jp_3550_;
}
else
{
if (lean_obj_tag(v___x_3670_) == 0)
{
lean_dec_ref_known(v___x_3670_, 1);
lean_del_object(v___x_3547_);
lean_del_object(v___x_3538_);
lean_dec(v_fst_3535_);
v___y_3551_ = v_anyProgress_3647_;
goto v___jp_3550_;
}
else
{
lean_object* v_a_3671_; uint8_t v___x_3672_; 
v_a_3671_ = lean_ctor_get(v___x_3670_, 0);
lean_inc(v_a_3671_);
lean_dec_ref_known(v___x_3670_, 1);
v___x_3672_ = l_Lean_Exception_isInterrupt(v_a_3671_);
if (v___x_3672_ == 0)
{
uint8_t v___x_3673_; 
lean_inc(v_a_3671_);
v___x_3673_ = l_Lean_Exception_isRuntime(v_a_3671_);
v___y_3601_ = v___y_3649_;
v___y_3602_ = v___y_3652_;
v___y_3603_ = v___y_3648_;
v___y_3604_ = v___y_3655_;
v___y_3605_ = v_a_3671_;
v___y_3606_ = v___y_3653_;
v___y_3607_ = v_anyProgress_3647_;
v___y_3608_ = v___y_3650_;
v___y_3609_ = v___y_3651_;
v___y_3610_ = v___y_3654_;
v___y_3611_ = v___x_3673_;
goto v___jp_3600_;
}
else
{
v___y_3601_ = v___y_3649_;
v___y_3602_ = v___y_3652_;
v___y_3603_ = v___y_3648_;
v___y_3604_ = v___y_3655_;
v___y_3605_ = v_a_3671_;
v___y_3606_ = v___y_3653_;
v___y_3607_ = v_anyProgress_3647_;
v___y_3608_ = v___y_3650_;
v___y_3609_ = v___y_3651_;
v___y_3610_ = v___y_3654_;
v___y_3611_ = v___x_3672_;
goto v___jp_3600_;
}
}
}
}
v___jp_3674_:
{
lean_object* v___x_3676_; 
lean_inc_ref(v_config_3511_);
lean_inc(v_fst_3535_);
v___x_3676_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(v_fst_3535_, v___y_3675_, v_config_3511_, v___y_3518_, v___y_3519_, v___y_3520_, v___y_3521_, v___y_3522_, v___y_3523_, v___y_3524_, v___y_3525_);
if (lean_obj_tag(v___x_3676_) == 0)
{
lean_object* v_a_3677_; uint8_t v___x_3678_; 
v_a_3677_ = lean_ctor_get(v___x_3676_, 0);
lean_inc(v_a_3677_);
lean_dec_ref_known(v___x_3676_, 1);
v___x_3678_ = lean_unbox(v_a_3677_);
lean_dec(v_a_3677_);
if (v___x_3678_ == 0)
{
uint8_t v___x_3679_; 
v___x_3679_ = lean_unbox(v_snd_3541_);
lean_dec(v_snd_3541_);
v_anyProgress_3647_ = v___x_3679_;
v___y_3648_ = v___y_3518_;
v___y_3649_ = v___y_3519_;
v___y_3650_ = v___y_3520_;
v___y_3651_ = v___y_3521_;
v___y_3652_ = v___y_3522_;
v___y_3653_ = v___y_3523_;
v___y_3654_ = v___y_3524_;
v___y_3655_ = v___y_3525_;
goto v___jp_3646_;
}
else
{
lean_object* v___x_3680_; lean_object* v___x_3681_; 
lean_del_object(v___x_3547_);
lean_del_object(v___x_3543_);
lean_dec(v_snd_3541_);
lean_del_object(v___x_3538_);
lean_dec(v_fst_3535_);
v___x_3680_ = lean_box(v___x_3512_);
v___x_3681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3681_, 0, v___x_3549_);
lean_ctor_set(v___x_3681_, 1, v___x_3680_);
v_a_3528_ = v___x_3681_;
goto v___jp_3527_;
}
}
else
{
lean_object* v_a_3682_; lean_object* v___x_3684_; uint8_t v_isShared_3685_; uint8_t v_isSharedCheck_3689_; 
lean_del_object(v___x_3547_);
lean_del_object(v___x_3543_);
lean_dec(v_snd_3541_);
lean_del_object(v___x_3538_);
lean_dec(v_fst_3535_);
lean_dec_ref(v_config_3511_);
v_a_3682_ = lean_ctor_get(v___x_3676_, 0);
v_isSharedCheck_3689_ = !lean_is_exclusive(v___x_3676_);
if (v_isSharedCheck_3689_ == 0)
{
v___x_3684_ = v___x_3676_;
v_isShared_3685_ = v_isSharedCheck_3689_;
goto v_resetjp_3683_;
}
else
{
lean_inc(v_a_3682_);
lean_dec(v___x_3676_);
v___x_3684_ = lean_box(0);
v_isShared_3685_ = v_isSharedCheck_3689_;
goto v_resetjp_3683_;
}
v_resetjp_3683_:
{
lean_object* v___x_3687_; 
if (v_isShared_3685_ == 0)
{
v___x_3687_ = v___x_3684_;
goto v_reusejp_3686_;
}
else
{
lean_object* v_reuseFailAlloc_3688_; 
v_reuseFailAlloc_3688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3688_, 0, v_a_3682_);
v___x_3687_ = v_reuseFailAlloc_3688_;
goto v_reusejp_3686_;
}
v_reusejp_3686_:
{
return v___x_3687_;
}
}
}
}
v___jp_3690_:
{
if (v___y_3691_ == 0)
{
v___y_3675_ = v___x_3512_;
goto v___jp_3674_;
}
else
{
v___y_3675_ = v___x_3510_;
goto v___jp_3674_;
}
}
}
}
}
}
v___jp_3527_:
{
size_t v___x_3529_; size_t v___x_3530_; 
v___x_3529_ = ((size_t)1ULL);
v___x_3530_ = lean_usize_add(v_i_3516_, v___x_3529_);
v_i_3516_ = v___x_3530_;
v_b_3517_ = v_a_3528_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17(lean_object* v_config_3701_, uint8_t v___x_3702_, uint8_t v_forward_3703_, lean_object* v_as_3704_, size_t v_sz_3705_, size_t v_i_3706_, lean_object* v_b_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_){
_start:
{
lean_object* v_a_3718_; uint8_t v___x_3722_; 
v___x_3722_ = lean_usize_dec_lt(v_i_3706_, v_sz_3705_);
if (v___x_3722_ == 0)
{
lean_object* v___x_3723_; 
lean_dec_ref(v_config_3701_);
v___x_3723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3723_, 0, v_b_3707_);
return v___x_3723_;
}
else
{
lean_object* v_a_3724_; lean_object* v_fst_3725_; lean_object* v_snd_3726_; lean_object* v___x_3728_; uint8_t v_isShared_3729_; uint8_t v_isSharedCheck_3893_; 
v_a_3724_ = lean_array_uget(v_as_3704_, v_i_3706_);
v_fst_3725_ = lean_ctor_get(v_a_3724_, 0);
v_snd_3726_ = lean_ctor_get(v_a_3724_, 1);
v_isSharedCheck_3893_ = !lean_is_exclusive(v_a_3724_);
if (v_isSharedCheck_3893_ == 0)
{
v___x_3728_ = v_a_3724_;
v_isShared_3729_ = v_isSharedCheck_3893_;
goto v_resetjp_3727_;
}
else
{
lean_inc(v_snd_3726_);
lean_inc(v_fst_3725_);
lean_dec(v_a_3724_);
v___x_3728_ = lean_box(0);
v_isShared_3729_ = v_isSharedCheck_3893_;
goto v_resetjp_3727_;
}
v_resetjp_3727_:
{
lean_object* v___x_3730_; lean_object* v_snd_3731_; lean_object* v___x_3733_; uint8_t v_isShared_3734_; uint8_t v_isSharedCheck_3891_; 
v___x_3730_ = lean_st_ref_get(v___y_3709_);
v_snd_3731_ = lean_ctor_get(v_b_3707_, 1);
v_isSharedCheck_3891_ = !lean_is_exclusive(v_b_3707_);
if (v_isSharedCheck_3891_ == 0)
{
lean_object* v_unused_3892_; 
v_unused_3892_ = lean_ctor_get(v_b_3707_, 0);
lean_dec(v_unused_3892_);
v___x_3733_ = v_b_3707_;
v_isShared_3734_ = v_isSharedCheck_3891_;
goto v_resetjp_3732_;
}
else
{
lean_inc(v_snd_3731_);
lean_dec(v_b_3707_);
v___x_3733_ = lean_box(0);
v_isShared_3734_ = v_isSharedCheck_3891_;
goto v_resetjp_3732_;
}
v_resetjp_3732_:
{
lean_object* v_progress_3735_; lean_object* v___x_3737_; uint8_t v_isShared_3738_; uint8_t v_isSharedCheck_3889_; 
v_progress_3735_ = lean_ctor_get(v___x_3730_, 1);
v_isSharedCheck_3889_ = !lean_is_exclusive(v___x_3730_);
if (v_isSharedCheck_3889_ == 0)
{
lean_object* v_unused_3890_; 
v_unused_3890_ = lean_ctor_get(v___x_3730_, 0);
lean_dec(v_unused_3890_);
v___x_3737_ = v___x_3730_;
v_isShared_3738_ = v_isSharedCheck_3889_;
goto v_resetjp_3736_;
}
else
{
lean_inc(v_progress_3735_);
lean_dec(v___x_3730_);
v___x_3737_ = lean_box(0);
v_isShared_3738_ = v_isSharedCheck_3889_;
goto v_resetjp_3736_;
}
v_resetjp_3736_:
{
lean_object* v___x_3739_; uint8_t v___y_3741_; uint8_t v___y_3747_; lean_object* v___y_3748_; uint8_t v___x_3777_; lean_object* v___y_3779_; lean_object* v___y_3780_; uint8_t v___y_3781_; lean_object* v___y_3782_; lean_object* v___y_3783_; lean_object* v___y_3784_; lean_object* v___y_3785_; lean_object* v___y_3786_; lean_object* v___y_3787_; lean_object* v_cls_3790_; lean_object* v___y_3792_; lean_object* v___y_3793_; lean_object* v___y_3794_; uint8_t v___y_3795_; lean_object* v___y_3796_; lean_object* v___y_3797_; lean_object* v___y_3798_; lean_object* v___y_3799_; lean_object* v___y_3800_; lean_object* v___y_3801_; uint8_t v___y_3802_; uint8_t v_anyProgress_3838_; lean_object* v___y_3839_; lean_object* v___y_3840_; lean_object* v___y_3841_; lean_object* v___y_3842_; lean_object* v___y_3843_; lean_object* v___y_3844_; lean_object* v___y_3845_; lean_object* v___y_3846_; uint8_t v___y_3866_; uint8_t v___y_3882_; uint8_t v___y_3883_; uint8_t v___y_3885_; 
v___x_3739_ = lean_box(0);
v___x_3777_ = 0;
v_cls_3790_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
if (lean_obj_tag(v_progress_3735_) == 2)
{
lean_dec_ref_known(v_progress_3735_, 1);
if (v___x_3702_ == 0)
{
v___y_3885_ = v___x_3702_;
goto v___jp_3884_;
}
else
{
uint8_t v___x_3888_; 
lean_dec(v_snd_3726_);
v___x_3888_ = lean_unbox(v_snd_3731_);
lean_dec(v_snd_3731_);
v_anyProgress_3838_ = v___x_3888_;
v___y_3839_ = v___y_3708_;
v___y_3840_ = v___y_3709_;
v___y_3841_ = v___y_3710_;
v___y_3842_ = v___y_3711_;
v___y_3843_ = v___y_3712_;
v___y_3844_ = v___y_3713_;
v___y_3845_ = v___y_3714_;
v___y_3846_ = v___y_3715_;
goto v___jp_3837_;
}
}
else
{
lean_dec(v_progress_3735_);
v___y_3885_ = v___x_3777_;
goto v___jp_3884_;
}
v___jp_3740_:
{
lean_object* v___x_3742_; lean_object* v___x_3744_; 
v___x_3742_ = lean_box(v___y_3741_);
if (v_isShared_3734_ == 0)
{
lean_ctor_set(v___x_3733_, 1, v___x_3742_);
lean_ctor_set(v___x_3733_, 0, v___x_3739_);
v___x_3744_ = v___x_3733_;
goto v_reusejp_3743_;
}
else
{
lean_object* v_reuseFailAlloc_3745_; 
v_reuseFailAlloc_3745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3745_, 0, v___x_3739_);
lean_ctor_set(v_reuseFailAlloc_3745_, 1, v___x_3742_);
v___x_3744_ = v_reuseFailAlloc_3745_;
goto v_reusejp_3743_;
}
v_reusejp_3743_:
{
v_a_3718_ = v___x_3744_;
goto v___jp_3717_;
}
}
v___jp_3746_:
{
if (lean_obj_tag(v___y_3748_) == 0)
{
lean_object* v_a_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3768_; 
v_a_3749_ = lean_ctor_get(v___y_3748_, 0);
v_isSharedCheck_3768_ = !lean_is_exclusive(v___y_3748_);
if (v_isSharedCheck_3768_ == 0)
{
v___x_3751_ = v___y_3748_;
v_isShared_3752_ = v_isSharedCheck_3768_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_a_3749_);
lean_dec(v___y_3748_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3768_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
if (lean_obj_tag(v_a_3749_) == 0)
{
lean_object* v_a_3753_; lean_object* v___x_3755_; uint8_t v_isShared_3756_; uint8_t v_isSharedCheck_3767_; 
lean_del_object(v___x_3733_);
lean_dec_ref(v_config_3701_);
v_a_3753_ = lean_ctor_get(v_a_3749_, 0);
v_isSharedCheck_3767_ = !lean_is_exclusive(v_a_3749_);
if (v_isSharedCheck_3767_ == 0)
{
v___x_3755_ = v_a_3749_;
v_isShared_3756_ = v_isSharedCheck_3767_;
goto v_resetjp_3754_;
}
else
{
lean_inc(v_a_3753_);
lean_dec(v_a_3749_);
v___x_3755_ = lean_box(0);
v_isShared_3756_ = v_isSharedCheck_3767_;
goto v_resetjp_3754_;
}
v_resetjp_3754_:
{
lean_object* v___x_3758_; 
if (v_isShared_3756_ == 0)
{
lean_ctor_set_tag(v___x_3755_, 1);
v___x_3758_ = v___x_3755_;
goto v_reusejp_3757_;
}
else
{
lean_object* v_reuseFailAlloc_3766_; 
v_reuseFailAlloc_3766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3766_, 0, v_a_3753_);
v___x_3758_ = v_reuseFailAlloc_3766_;
goto v_reusejp_3757_;
}
v_reusejp_3757_:
{
lean_object* v___x_3759_; lean_object* v___x_3761_; 
v___x_3759_ = lean_box(v___y_3747_);
if (v_isShared_3729_ == 0)
{
lean_ctor_set(v___x_3728_, 1, v___x_3759_);
lean_ctor_set(v___x_3728_, 0, v___x_3758_);
v___x_3761_ = v___x_3728_;
goto v_reusejp_3760_;
}
else
{
lean_object* v_reuseFailAlloc_3765_; 
v_reuseFailAlloc_3765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3765_, 0, v___x_3758_);
lean_ctor_set(v_reuseFailAlloc_3765_, 1, v___x_3759_);
v___x_3761_ = v_reuseFailAlloc_3765_;
goto v_reusejp_3760_;
}
v_reusejp_3760_:
{
lean_object* v___x_3763_; 
if (v_isShared_3752_ == 0)
{
lean_ctor_set(v___x_3751_, 0, v___x_3761_);
v___x_3763_ = v___x_3751_;
goto v_reusejp_3762_;
}
else
{
lean_object* v_reuseFailAlloc_3764_; 
v_reuseFailAlloc_3764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3764_, 0, v___x_3761_);
v___x_3763_ = v_reuseFailAlloc_3764_;
goto v_reusejp_3762_;
}
v_reusejp_3762_:
{
return v___x_3763_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_3749_, 1);
lean_del_object(v___x_3751_);
lean_del_object(v___x_3728_);
v___y_3741_ = v___y_3747_;
goto v___jp_3740_;
}
}
}
else
{
lean_object* v_a_3769_; lean_object* v___x_3771_; uint8_t v_isShared_3772_; uint8_t v_isSharedCheck_3776_; 
lean_del_object(v___x_3733_);
lean_del_object(v___x_3728_);
lean_dec_ref(v_config_3701_);
v_a_3769_ = lean_ctor_get(v___y_3748_, 0);
v_isSharedCheck_3776_ = !lean_is_exclusive(v___y_3748_);
if (v_isSharedCheck_3776_ == 0)
{
v___x_3771_ = v___y_3748_;
v_isShared_3772_ = v_isSharedCheck_3776_;
goto v_resetjp_3770_;
}
else
{
lean_inc(v_a_3769_);
lean_dec(v___y_3748_);
v___x_3771_ = lean_box(0);
v_isShared_3772_ = v_isSharedCheck_3776_;
goto v_resetjp_3770_;
}
v_resetjp_3770_:
{
lean_object* v___x_3774_; 
if (v_isShared_3772_ == 0)
{
v___x_3774_ = v___x_3771_;
goto v_reusejp_3773_;
}
else
{
lean_object* v_reuseFailAlloc_3775_; 
v_reuseFailAlloc_3775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3775_, 0, v_a_3769_);
v___x_3774_ = v_reuseFailAlloc_3775_;
goto v_reusejp_3773_;
}
v_reusejp_3773_:
{
return v___x_3774_;
}
}
}
}
v___jp_3778_:
{
lean_object* v___x_3788_; lean_object* v___x_3789_; 
v___x_3788_ = lean_box(0);
v___x_3789_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(v___x_3777_, v___x_3788_, v___y_3786_, v___y_3785_, v___y_3779_, v___y_3780_, v___y_3787_, v___y_3782_, v___y_3784_, v___y_3783_);
v___y_3747_ = v___y_3781_;
v___y_3748_ = v___x_3789_;
goto v___jp_3746_;
}
v___jp_3791_:
{
if (v___y_3802_ == 0)
{
lean_object* v_options_3803_; uint8_t v_hasTrace_3804_; 
v_options_3803_ = lean_ctor_get(v___y_3798_, 2);
v_hasTrace_3804_ = lean_ctor_get_uint8(v_options_3803_, sizeof(void*)*1);
if (v_hasTrace_3804_ == 0)
{
lean_dec_ref(v___y_3797_);
lean_del_object(v___x_3737_);
lean_dec(v_fst_3725_);
v___y_3779_ = v___y_3792_;
v___y_3780_ = v___y_3793_;
v___y_3781_ = v___y_3795_;
v___y_3782_ = v___y_3794_;
v___y_3783_ = v___y_3796_;
v___y_3784_ = v___y_3798_;
v___y_3785_ = v___y_3800_;
v___y_3786_ = v___y_3799_;
v___y_3787_ = v___y_3801_;
goto v___jp_3778_;
}
else
{
lean_object* v_inheritedTraceOptions_3805_; lean_object* v___x_3806_; uint8_t v___x_3807_; 
v_inheritedTraceOptions_3805_ = lean_ctor_get(v___y_3798_, 13);
v___x_3806_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_3807_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3805_, v_options_3803_, v___x_3806_);
if (v___x_3807_ == 0)
{
lean_dec_ref(v___y_3797_);
lean_del_object(v___x_3737_);
lean_dec(v_fst_3725_);
v___y_3779_ = v___y_3792_;
v___y_3780_ = v___y_3793_;
v___y_3781_ = v___y_3795_;
v___y_3782_ = v___y_3794_;
v___y_3783_ = v___y_3796_;
v___y_3784_ = v___y_3798_;
v___y_3785_ = v___y_3800_;
v___y_3786_ = v___y_3799_;
v___y_3787_ = v___y_3801_;
goto v___jp_3778_;
}
else
{
lean_object* v___x_3808_; 
v___x_3808_ = l_Lean_MVarId_getType(v_fst_3725_, v___y_3801_, v___y_3794_, v___y_3798_, v___y_3796_);
if (lean_obj_tag(v___x_3808_) == 0)
{
lean_object* v_a_3809_; lean_object* v___x_3810_; lean_object* v___x_3811_; lean_object* v___x_3813_; 
v_a_3809_ = lean_ctor_get(v___x_3808_, 0);
lean_inc(v_a_3809_);
lean_dec_ref_known(v___x_3808_, 1);
v___x_3810_ = l_Lean_MessageData_ofExpr(v_a_3809_);
v___x_3811_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___closed__1);
if (v_isShared_3738_ == 0)
{
lean_ctor_set_tag(v___x_3737_, 7);
lean_ctor_set(v___x_3737_, 1, v___x_3811_);
lean_ctor_set(v___x_3737_, 0, v___x_3810_);
v___x_3813_ = v___x_3737_;
goto v_reusejp_3812_;
}
else
{
lean_object* v_reuseFailAlloc_3827_; 
v_reuseFailAlloc_3827_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3827_, 0, v___x_3810_);
lean_ctor_set(v_reuseFailAlloc_3827_, 1, v___x_3811_);
v___x_3813_ = v_reuseFailAlloc_3827_;
goto v_reusejp_3812_;
}
v_reusejp_3812_:
{
lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; 
v___x_3814_ = l_Lean_Exception_toMessageData(v___y_3797_);
v___x_3815_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3815_, 0, v___x_3813_);
lean_ctor_set(v___x_3815_, 1, v___x_3814_);
v___x_3816_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_3790_, v___x_3815_, v___y_3801_, v___y_3794_, v___y_3798_, v___y_3796_);
if (lean_obj_tag(v___x_3816_) == 0)
{
lean_object* v_a_3817_; lean_object* v___x_3818_; 
v_a_3817_ = lean_ctor_get(v___x_3816_, 0);
lean_inc(v_a_3817_);
lean_dec_ref_known(v___x_3816_, 1);
v___x_3818_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___lam__0(v___x_3777_, v_a_3817_, v___y_3799_, v___y_3800_, v___y_3792_, v___y_3793_, v___y_3801_, v___y_3794_, v___y_3798_, v___y_3796_);
v___y_3747_ = v___y_3795_;
v___y_3748_ = v___x_3818_;
goto v___jp_3746_;
}
else
{
lean_object* v_a_3819_; lean_object* v___x_3821_; uint8_t v_isShared_3822_; uint8_t v_isSharedCheck_3826_; 
lean_del_object(v___x_3733_);
lean_del_object(v___x_3728_);
lean_dec_ref(v_config_3701_);
v_a_3819_ = lean_ctor_get(v___x_3816_, 0);
v_isSharedCheck_3826_ = !lean_is_exclusive(v___x_3816_);
if (v_isSharedCheck_3826_ == 0)
{
v___x_3821_ = v___x_3816_;
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
else
{
lean_inc(v_a_3819_);
lean_dec(v___x_3816_);
v___x_3821_ = lean_box(0);
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
v_resetjp_3820_:
{
lean_object* v___x_3824_; 
if (v_isShared_3822_ == 0)
{
v___x_3824_ = v___x_3821_;
goto v_reusejp_3823_;
}
else
{
lean_object* v_reuseFailAlloc_3825_; 
v_reuseFailAlloc_3825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3825_, 0, v_a_3819_);
v___x_3824_ = v_reuseFailAlloc_3825_;
goto v_reusejp_3823_;
}
v_reusejp_3823_:
{
return v___x_3824_;
}
}
}
}
}
else
{
lean_object* v_a_3828_; lean_object* v___x_3830_; uint8_t v_isShared_3831_; uint8_t v_isSharedCheck_3835_; 
lean_dec_ref(v___y_3797_);
lean_del_object(v___x_3737_);
lean_del_object(v___x_3733_);
lean_del_object(v___x_3728_);
lean_dec_ref(v_config_3701_);
v_a_3828_ = lean_ctor_get(v___x_3808_, 0);
v_isSharedCheck_3835_ = !lean_is_exclusive(v___x_3808_);
if (v_isSharedCheck_3835_ == 0)
{
v___x_3830_ = v___x_3808_;
v_isShared_3831_ = v_isSharedCheck_3835_;
goto v_resetjp_3829_;
}
else
{
lean_inc(v_a_3828_);
lean_dec(v___x_3808_);
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
}
}
else
{
lean_object* v___x_3836_; 
lean_del_object(v___x_3737_);
lean_del_object(v___x_3733_);
lean_del_object(v___x_3728_);
lean_dec(v_fst_3725_);
lean_dec_ref(v_config_3701_);
v___x_3836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3836_, 0, v___y_3797_);
return v___x_3836_;
}
}
v___jp_3837_:
{
lean_object* v_keyedConfig_3847_; uint8_t v_trackZetaDelta_3848_; lean_object* v_zetaDeltaSet_3849_; lean_object* v_lctx_3850_; lean_object* v_localInstances_3851_; lean_object* v_defEqCtx_x3f_3852_; lean_object* v_synthPendingDepth_3853_; lean_object* v_customCanUnfoldPredicate_x3f_3854_; uint8_t v_univApprox_3855_; uint8_t v_inTypeClassResolution_3856_; uint8_t v_cacheInferType_3857_; uint8_t v___x_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3861_; 
v_keyedConfig_3847_ = lean_ctor_get(v___y_3843_, 0);
v_trackZetaDelta_3848_ = lean_ctor_get_uint8(v___y_3843_, sizeof(void*)*7);
v_zetaDeltaSet_3849_ = lean_ctor_get(v___y_3843_, 1);
v_lctx_3850_ = lean_ctor_get(v___y_3843_, 2);
v_localInstances_3851_ = lean_ctor_get(v___y_3843_, 3);
v_defEqCtx_x3f_3852_ = lean_ctor_get(v___y_3843_, 4);
v_synthPendingDepth_3853_ = lean_ctor_get(v___y_3843_, 5);
v_customCanUnfoldPredicate_x3f_3854_ = lean_ctor_get(v___y_3843_, 6);
v_univApprox_3855_ = lean_ctor_get_uint8(v___y_3843_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3856_ = lean_ctor_get_uint8(v___y_3843_, sizeof(void*)*7 + 2);
v_cacheInferType_3857_ = lean_ctor_get_uint8(v___y_3843_, sizeof(void*)*7 + 3);
v___x_3858_ = 3;
lean_inc_ref(v_keyedConfig_3847_);
v___x_3859_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3858_, v_keyedConfig_3847_);
lean_inc(v_customCanUnfoldPredicate_x3f_3854_);
lean_inc(v_synthPendingDepth_3853_);
lean_inc(v_defEqCtx_x3f_3852_);
lean_inc_ref(v_localInstances_3851_);
lean_inc_ref(v_lctx_3850_);
lean_inc(v_zetaDeltaSet_3849_);
v___x_3860_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3860_, 0, v___x_3859_);
lean_ctor_set(v___x_3860_, 1, v_zetaDeltaSet_3849_);
lean_ctor_set(v___x_3860_, 2, v_lctx_3850_);
lean_ctor_set(v___x_3860_, 3, v_localInstances_3851_);
lean_ctor_set(v___x_3860_, 4, v_defEqCtx_x3f_3852_);
lean_ctor_set(v___x_3860_, 5, v_synthPendingDepth_3853_);
lean_ctor_set(v___x_3860_, 6, v_customCanUnfoldPredicate_x3f_3854_);
lean_ctor_set_uint8(v___x_3860_, sizeof(void*)*7, v_trackZetaDelta_3848_);
lean_ctor_set_uint8(v___x_3860_, sizeof(void*)*7 + 1, v_univApprox_3855_);
lean_ctor_set_uint8(v___x_3860_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3856_);
lean_ctor_set_uint8(v___x_3860_, sizeof(void*)*7 + 3, v_cacheInferType_3857_);
lean_inc(v_fst_3725_);
v___x_3861_ = lp_mathlib_Lean_MVarId_applyRflOrId(v_fst_3725_, v___x_3860_, v___y_3844_, v___y_3845_, v___y_3846_);
lean_dec_ref_known(v___x_3860_, 7);
if (lean_obj_tag(v___x_3861_) == 0)
{
lean_dec_ref_known(v___x_3861_, 1);
lean_del_object(v___x_3737_);
lean_del_object(v___x_3728_);
lean_dec(v_fst_3725_);
v___y_3741_ = v_anyProgress_3838_;
goto v___jp_3740_;
}
else
{
if (lean_obj_tag(v___x_3861_) == 0)
{
lean_dec_ref_known(v___x_3861_, 1);
lean_del_object(v___x_3737_);
lean_del_object(v___x_3728_);
lean_dec(v_fst_3725_);
v___y_3741_ = v_anyProgress_3838_;
goto v___jp_3740_;
}
else
{
lean_object* v_a_3862_; uint8_t v___x_3863_; 
v_a_3862_ = lean_ctor_get(v___x_3861_, 0);
lean_inc(v_a_3862_);
lean_dec_ref_known(v___x_3861_, 1);
v___x_3863_ = l_Lean_Exception_isInterrupt(v_a_3862_);
if (v___x_3863_ == 0)
{
uint8_t v___x_3864_; 
lean_inc(v_a_3862_);
v___x_3864_ = l_Lean_Exception_isRuntime(v_a_3862_);
v___y_3792_ = v___y_3841_;
v___y_3793_ = v___y_3842_;
v___y_3794_ = v___y_3844_;
v___y_3795_ = v_anyProgress_3838_;
v___y_3796_ = v___y_3846_;
v___y_3797_ = v_a_3862_;
v___y_3798_ = v___y_3845_;
v___y_3799_ = v___y_3839_;
v___y_3800_ = v___y_3840_;
v___y_3801_ = v___y_3843_;
v___y_3802_ = v___x_3864_;
goto v___jp_3791_;
}
else
{
v___y_3792_ = v___y_3841_;
v___y_3793_ = v___y_3842_;
v___y_3794_ = v___y_3844_;
v___y_3795_ = v_anyProgress_3838_;
v___y_3796_ = v___y_3846_;
v___y_3797_ = v_a_3862_;
v___y_3798_ = v___y_3845_;
v___y_3799_ = v___y_3839_;
v___y_3800_ = v___y_3840_;
v___y_3801_ = v___y_3843_;
v___y_3802_ = v___x_3863_;
goto v___jp_3791_;
}
}
}
}
v___jp_3865_:
{
lean_object* v___x_3867_; 
lean_inc_ref(v_config_3701_);
lean_inc(v_fst_3725_);
v___x_3867_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(v_fst_3725_, v___y_3866_, v_config_3701_, v___y_3708_, v___y_3709_, v___y_3710_, v___y_3711_, v___y_3712_, v___y_3713_, v___y_3714_, v___y_3715_);
if (lean_obj_tag(v___x_3867_) == 0)
{
lean_object* v_a_3868_; uint8_t v___x_3869_; 
v_a_3868_ = lean_ctor_get(v___x_3867_, 0);
lean_inc(v_a_3868_);
lean_dec_ref_known(v___x_3867_, 1);
v___x_3869_ = lean_unbox(v_a_3868_);
lean_dec(v_a_3868_);
if (v___x_3869_ == 0)
{
uint8_t v___x_3870_; 
v___x_3870_ = lean_unbox(v_snd_3731_);
lean_dec(v_snd_3731_);
v_anyProgress_3838_ = v___x_3870_;
v___y_3839_ = v___y_3708_;
v___y_3840_ = v___y_3709_;
v___y_3841_ = v___y_3710_;
v___y_3842_ = v___y_3711_;
v___y_3843_ = v___y_3712_;
v___y_3844_ = v___y_3713_;
v___y_3845_ = v___y_3714_;
v___y_3846_ = v___y_3715_;
goto v___jp_3837_;
}
else
{
lean_object* v___x_3871_; lean_object* v___x_3872_; 
lean_del_object(v___x_3737_);
lean_del_object(v___x_3733_);
lean_dec(v_snd_3731_);
lean_del_object(v___x_3728_);
lean_dec(v_fst_3725_);
v___x_3871_ = lean_box(v___x_3702_);
v___x_3872_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3872_, 0, v___x_3739_);
lean_ctor_set(v___x_3872_, 1, v___x_3871_);
v_a_3718_ = v___x_3872_;
goto v___jp_3717_;
}
}
else
{
lean_object* v_a_3873_; lean_object* v___x_3875_; uint8_t v_isShared_3876_; uint8_t v_isSharedCheck_3880_; 
lean_del_object(v___x_3737_);
lean_del_object(v___x_3733_);
lean_dec(v_snd_3731_);
lean_del_object(v___x_3728_);
lean_dec(v_fst_3725_);
lean_dec_ref(v_config_3701_);
v_a_3873_ = lean_ctor_get(v___x_3867_, 0);
v_isSharedCheck_3880_ = !lean_is_exclusive(v___x_3867_);
if (v_isSharedCheck_3880_ == 0)
{
v___x_3875_ = v___x_3867_;
v_isShared_3876_ = v_isSharedCheck_3880_;
goto v_resetjp_3874_;
}
else
{
lean_inc(v_a_3873_);
lean_dec(v___x_3867_);
v___x_3875_ = lean_box(0);
v_isShared_3876_ = v_isSharedCheck_3880_;
goto v_resetjp_3874_;
}
v_resetjp_3874_:
{
lean_object* v___x_3878_; 
if (v_isShared_3876_ == 0)
{
v___x_3878_ = v___x_3875_;
goto v_reusejp_3877_;
}
else
{
lean_object* v_reuseFailAlloc_3879_; 
v_reuseFailAlloc_3879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3879_, 0, v_a_3873_);
v___x_3878_ = v_reuseFailAlloc_3879_;
goto v_reusejp_3877_;
}
v_reusejp_3877_:
{
return v___x_3878_;
}
}
}
}
v___jp_3881_:
{
if (v___y_3883_ == 0)
{
v___y_3866_ = v___x_3702_;
goto v___jp_3865_;
}
else
{
v___y_3866_ = v___y_3882_;
goto v___jp_3865_;
}
}
v___jp_3884_:
{
if (v_forward_3703_ == 0)
{
uint8_t v___x_3886_; 
v___x_3886_ = lean_unbox(v_snd_3726_);
lean_dec(v_snd_3726_);
if (v___x_3886_ == 0)
{
v___y_3882_ = v___y_3885_;
v___y_3883_ = v___x_3722_;
goto v___jp_3881_;
}
else
{
v___y_3866_ = v___x_3702_;
goto v___jp_3865_;
}
}
else
{
uint8_t v___x_3887_; 
v___x_3887_ = lean_unbox(v_snd_3726_);
lean_dec(v_snd_3726_);
v___y_3882_ = v___y_3885_;
v___y_3883_ = v___x_3887_;
goto v___jp_3881_;
}
}
}
}
}
}
v___jp_3717_:
{
size_t v___x_3719_; size_t v___x_3720_; 
v___x_3719_ = ((size_t)1ULL);
v___x_3720_ = lean_usize_add(v_i_3706_, v___x_3719_);
v_i_3706_ = v___x_3720_;
v_b_3707_ = v_a_3718_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma(lean_object* v_goal_3894_, lean_object* v_lem_3895_, uint8_t v_forward_3896_, lean_object* v_config_3897_, lean_object* v_a_3898_, lean_object* v_a_3899_, lean_object* v_a_3900_, lean_object* v_a_3901_, lean_object* v_a_3902_, lean_object* v_a_3903_, lean_object* v_a_3904_, lean_object* v_a_3905_){
_start:
{
lean_object* v_e_3908_; lean_object* v___y_3909_; lean_object* v___y_3910_; lean_object* v___y_3911_; lean_object* v___y_3912_; lean_object* v___y_3913_; lean_object* v___y_3914_; lean_object* v___y_3915_; lean_object* v___y_3916_; lean_object* v_options_3976_; uint8_t v_hasTrace_3977_; 
v_options_3976_ = lean_ctor_get(v_a_3904_, 2);
v_hasTrace_3977_ = lean_ctor_get_uint8(v_options_3976_, sizeof(void*)*1);
if (v_hasTrace_3977_ == 0)
{
lean_object* v___x_3978_; 
v___x_3978_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0(v_goal_3894_, v_lem_3895_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
if (lean_obj_tag(v___x_3978_) == 0)
{
lean_object* v_a_3979_; 
v_a_3979_ = lean_ctor_get(v___x_3978_, 0);
lean_inc(v_a_3979_);
lean_dec_ref_known(v___x_3978_, 1);
v_e_3908_ = v_a_3979_;
v___y_3909_ = v_a_3898_;
v___y_3910_ = v_a_3899_;
v___y_3911_ = v_a_3900_;
v___y_3912_ = v_a_3901_;
v___y_3913_ = v_a_3902_;
v___y_3914_ = v_a_3903_;
v___y_3915_ = v_a_3904_;
v___y_3916_ = v_a_3905_;
goto v___jp_3907_;
}
else
{
lean_object* v_a_3980_; lean_object* v___x_3982_; uint8_t v_isShared_3983_; uint8_t v_isSharedCheck_3987_; 
lean_dec_ref(v_config_3897_);
v_a_3980_ = lean_ctor_get(v___x_3978_, 0);
v_isSharedCheck_3987_ = !lean_is_exclusive(v___x_3978_);
if (v_isSharedCheck_3987_ == 0)
{
v___x_3982_ = v___x_3978_;
v_isShared_3983_ = v_isSharedCheck_3987_;
goto v_resetjp_3981_;
}
else
{
lean_inc(v_a_3980_);
lean_dec(v___x_3978_);
v___x_3982_ = lean_box(0);
v_isShared_3983_ = v_isSharedCheck_3987_;
goto v_resetjp_3981_;
}
v_resetjp_3981_:
{
lean_object* v___x_3985_; 
if (v_isShared_3983_ == 0)
{
v___x_3985_ = v___x_3982_;
goto v_reusejp_3984_;
}
else
{
lean_object* v_reuseFailAlloc_3986_; 
v_reuseFailAlloc_3986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3986_, 0, v_a_3980_);
v___x_3985_ = v_reuseFailAlloc_3986_;
goto v_reusejp_3984_;
}
v_reusejp_3984_:
{
return v___x_3985_;
}
}
}
}
else
{
lean_object* v_inheritedTraceOptions_3988_; lean_object* v___f_3989_; lean_object* v_cls_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; uint8_t v___x_3993_; lean_object* v___y_3995_; lean_object* v___y_3996_; lean_object* v_a_3997_; lean_object* v___y_4010_; lean_object* v___y_4011_; uint8_t v_a_4012_; lean_object* v___y_4016_; lean_object* v___y_4017_; lean_object* v_a_4018_; lean_object* v___y_4021_; lean_object* v___y_4022_; lean_object* v___y_4023_; uint8_t v___y_4024_; lean_object* v___y_4026_; lean_object* v___y_4027_; lean_object* v_a_4028_; lean_object* v___y_4038_; lean_object* v___y_4039_; lean_object* v_a_4040_; lean_object* v___y_4043_; lean_object* v___y_4044_; uint8_t v_a_4045_; lean_object* v___y_4049_; lean_object* v___y_4050_; lean_object* v___y_4051_; uint8_t v___y_4052_; 
v_inheritedTraceOptions_3988_ = lean_ctor_get(v_a_3904_, 13);
lean_inc_ref(v_lem_3895_);
v___f_3989_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__1___boxed), 11, 1);
lean_closure_set(v___f_3989_, 0, v_lem_3895_);
v_cls_3990_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
v___x_3991_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0));
v___x_3992_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_3993_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3988_, v_options_3976_, v___x_3992_);
if (v___x_3993_ == 0)
{
lean_object* v___x_4121_; uint8_t v___x_4122_; 
v___x_4121_ = l_Lean_trace_profiler;
v___x_4122_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_3976_, v___x_4121_);
if (v___x_4122_ == 0)
{
lean_object* v___x_4123_; 
lean_dec_ref(v___f_3989_);
v___x_4123_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___lam__0(v_goal_3894_, v_lem_3895_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
if (lean_obj_tag(v___x_4123_) == 0)
{
lean_object* v_a_4124_; 
v_a_4124_ = lean_ctor_get(v___x_4123_, 0);
lean_inc(v_a_4124_);
lean_dec_ref_known(v___x_4123_, 1);
v_e_3908_ = v_a_4124_;
v___y_3909_ = v_a_3898_;
v___y_3910_ = v_a_3899_;
v___y_3911_ = v_a_3900_;
v___y_3912_ = v_a_3901_;
v___y_3913_ = v_a_3902_;
v___y_3914_ = v_a_3903_;
v___y_3915_ = v_a_3904_;
v___y_3916_ = v_a_3905_;
goto v___jp_3907_;
}
else
{
lean_object* v_a_4125_; lean_object* v___x_4127_; uint8_t v_isShared_4128_; uint8_t v_isSharedCheck_4132_; 
lean_dec_ref(v_config_3897_);
v_a_4125_ = lean_ctor_get(v___x_4123_, 0);
v_isSharedCheck_4132_ = !lean_is_exclusive(v___x_4123_);
if (v_isSharedCheck_4132_ == 0)
{
v___x_4127_ = v___x_4123_;
v_isShared_4128_ = v_isSharedCheck_4132_;
goto v_resetjp_4126_;
}
else
{
lean_inc(v_a_4125_);
lean_dec(v___x_4123_);
v___x_4127_ = lean_box(0);
v_isShared_4128_ = v_isSharedCheck_4132_;
goto v_resetjp_4126_;
}
v_resetjp_4126_:
{
lean_object* v___x_4130_; 
if (v_isShared_4128_ == 0)
{
v___x_4130_ = v___x_4127_;
goto v_reusejp_4129_;
}
else
{
lean_object* v_reuseFailAlloc_4131_; 
v_reuseFailAlloc_4131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4131_, 0, v_a_4125_);
v___x_4130_ = v_reuseFailAlloc_4131_;
goto v_reusejp_4129_;
}
v_reusejp_4129_:
{
return v___x_4130_;
}
}
}
}
else
{
goto v___jp_4053_;
}
}
else
{
goto v___jp_4053_;
}
v___jp_3994_:
{
lean_object* v___x_3998_; double v___x_3999_; double v___x_4000_; double v___x_4001_; double v___x_4002_; double v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; 
v___x_3998_ = lean_io_mono_nanos_now();
v___x_3999_ = lean_float_of_nat(v___y_3995_);
v___x_4000_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4);
v___x_4001_ = lean_float_div(v___x_3999_, v___x_4000_);
v___x_4002_ = lean_float_of_nat(v___x_3998_);
v___x_4003_ = lean_float_div(v___x_4002_, v___x_4000_);
v___x_4004_ = lean_box_float(v___x_4001_);
v___x_4005_ = lean_box_float(v___x_4003_);
v___x_4006_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4006_, 0, v___x_4004_);
lean_ctor_set(v___x_4006_, 1, v___x_4005_);
v___x_4007_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4007_, 0, v_a_3997_);
lean_ctor_set(v___x_4007_, 1, v___x_4006_);
v___x_4008_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14(v_cls_3990_, v_hasTrace_3977_, v___x_3991_, v_options_3976_, v___x_3993_, v___y_3996_, v___f_3989_, v___x_4007_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
return v___x_4008_;
}
v___jp_4009_:
{
lean_object* v___x_4013_; lean_object* v___x_4014_; 
v___x_4013_ = lean_box(v_a_4012_);
v___x_4014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4014_, 0, v___x_4013_);
v___y_3995_ = v___y_4010_;
v___y_3996_ = v___y_4011_;
v_a_3997_ = v___x_4014_;
goto v___jp_3994_;
}
v___jp_4015_:
{
lean_object* v___x_4019_; 
v___x_4019_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4019_, 0, v_a_4018_);
v___y_3995_ = v___y_4016_;
v___y_3996_ = v___y_4017_;
v_a_3997_ = v___x_4019_;
goto v___jp_3994_;
}
v___jp_4020_:
{
if (v___y_4024_ == 0)
{
lean_dec_ref(v___y_4022_);
v___y_4010_ = v___y_4021_;
v___y_4011_ = v___y_4023_;
v_a_4012_ = v___y_4024_;
goto v___jp_4009_;
}
else
{
v___y_4016_ = v___y_4021_;
v___y_4017_ = v___y_4023_;
v_a_4018_ = v___y_4022_;
goto v___jp_4015_;
}
}
v___jp_4025_:
{
lean_object* v___x_4029_; double v___x_4030_; double v___x_4031_; lean_object* v___x_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; 
v___x_4029_ = lean_io_get_num_heartbeats();
v___x_4030_ = lean_float_of_nat(v___y_4026_);
v___x_4031_ = lean_float_of_nat(v___x_4029_);
v___x_4032_ = lean_box_float(v___x_4030_);
v___x_4033_ = lean_box_float(v___x_4031_);
v___x_4034_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4034_, 0, v___x_4032_);
lean_ctor_set(v___x_4034_, 1, v___x_4033_);
v___x_4035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4035_, 0, v_a_4028_);
lean_ctor_set(v___x_4035_, 1, v___x_4034_);
v___x_4036_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__14(v_cls_3990_, v_hasTrace_3977_, v___x_3991_, v_options_3976_, v___x_3993_, v___y_4027_, v___f_3989_, v___x_4035_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
return v___x_4036_;
}
v___jp_4037_:
{
lean_object* v___x_4041_; 
v___x_4041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4041_, 0, v_a_4040_);
v___y_4026_ = v___y_4038_;
v___y_4027_ = v___y_4039_;
v_a_4028_ = v___x_4041_;
goto v___jp_4025_;
}
v___jp_4042_:
{
lean_object* v___x_4046_; lean_object* v___x_4047_; 
v___x_4046_ = lean_box(v_a_4045_);
v___x_4047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4047_, 0, v___x_4046_);
v___y_4026_ = v___y_4043_;
v___y_4027_ = v___y_4044_;
v_a_4028_ = v___x_4047_;
goto v___jp_4025_;
}
v___jp_4048_:
{
if (v___y_4052_ == 0)
{
lean_dec_ref(v___y_4049_);
v___y_4043_ = v___y_4050_;
v___y_4044_ = v___y_4051_;
v_a_4045_ = v___y_4052_;
goto v___jp_4042_;
}
else
{
v___y_4038_ = v___y_4050_;
v___y_4039_ = v___y_4051_;
v_a_4040_ = v___y_4049_;
goto v___jp_4037_;
}
}
v___jp_4053_:
{
lean_object* v___x_4054_; 
v___x_4054_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(v_a_3905_);
if (lean_obj_tag(v___x_4054_) == 0)
{
lean_object* v_a_4055_; lean_object* v___x_4056_; uint8_t v___x_4057_; 
v_a_4055_ = lean_ctor_get(v___x_4054_, 0);
lean_inc(v_a_4055_);
lean_dec_ref_known(v___x_4054_, 1);
v___x_4056_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4057_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_3976_, v___x_4056_);
if (v___x_4057_ == 0)
{
lean_object* v___x_4058_; lean_object* v___x_4059_; 
v___x_4058_ = lean_io_mono_nanos_now();
v___x_4059_ = lp_mathlib_Mathlib_Tactic_GCongr_applyGCongrLemma(v_goal_3894_, v_lem_3895_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
if (lean_obj_tag(v___x_4059_) == 0)
{
lean_object* v_a_4060_; lean_object* v_fst_4061_; lean_object* v_snd_4062_; lean_object* v___x_4064_; uint8_t v_isShared_4065_; uint8_t v_isSharedCheck_4085_; 
v_a_4060_ = lean_ctor_get(v___x_4059_, 0);
lean_inc(v_a_4060_);
lean_dec_ref_known(v___x_4059_, 1);
v_fst_4061_ = lean_ctor_get(v_a_4060_, 0);
v_snd_4062_ = lean_ctor_get(v_a_4060_, 1);
v_isSharedCheck_4085_ = !lean_is_exclusive(v_a_4060_);
if (v_isSharedCheck_4085_ == 0)
{
v___x_4064_ = v_a_4060_;
v_isShared_4065_ = v_isSharedCheck_4085_;
goto v_resetjp_4063_;
}
else
{
lean_inc(v_snd_4062_);
lean_inc(v_fst_4061_);
lean_dec(v_a_4060_);
v___x_4064_ = lean_box(0);
v_isShared_4065_ = v_isSharedCheck_4085_;
goto v_resetjp_4063_;
}
v_resetjp_4063_:
{
lean_object* v___x_4066_; lean_object* v___x_4067_; lean_object* v___x_4069_; 
v___x_4066_ = lean_box(0);
v___x_4067_ = lean_box(v___x_4057_);
if (v_isShared_4065_ == 0)
{
lean_ctor_set(v___x_4064_, 1, v___x_4067_);
lean_ctor_set(v___x_4064_, 0, v___x_4066_);
v___x_4069_ = v___x_4064_;
goto v_reusejp_4068_;
}
else
{
lean_object* v_reuseFailAlloc_4084_; 
v_reuseFailAlloc_4084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4084_, 0, v___x_4066_);
lean_ctor_set(v_reuseFailAlloc_4084_, 1, v___x_4067_);
v___x_4069_ = v_reuseFailAlloc_4084_;
goto v_reusejp_4068_;
}
v_reusejp_4068_:
{
size_t v_sz_4070_; size_t v___x_4071_; lean_object* v___x_4072_; 
v_sz_4070_ = lean_array_size(v_fst_4061_);
v___x_4071_ = ((size_t)0ULL);
v___x_4072_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15(v___x_4057_, v_config_3897_, v_hasTrace_3977_, v_forward_3896_, v_fst_4061_, v_sz_4070_, v___x_4071_, v___x_4069_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
lean_dec(v_fst_4061_);
if (lean_obj_tag(v___x_4072_) == 0)
{
lean_object* v_a_4073_; lean_object* v_fst_4074_; 
v_a_4073_ = lean_ctor_get(v___x_4072_, 0);
lean_inc(v_a_4073_);
lean_dec_ref_known(v___x_4072_, 1);
v_fst_4074_ = lean_ctor_get(v_a_4073_, 0);
if (lean_obj_tag(v_fst_4074_) == 0)
{
lean_object* v_snd_4075_; uint8_t v___x_4076_; 
v_snd_4075_ = lean_ctor_get(v_a_4073_, 1);
lean_inc(v_snd_4075_);
lean_dec(v_a_4073_);
v___x_4076_ = lean_unbox(v_snd_4075_);
lean_dec(v_snd_4075_);
if (v___x_4076_ == 0)
{
lean_dec(v_snd_4062_);
v___y_4010_ = v___x_4058_;
v___y_4011_ = v_a_4055_;
v_a_4012_ = v___x_4057_;
goto v___jp_4009_;
}
else
{
lean_object* v___x_4077_; size_t v_sz_4078_; lean_object* v___x_4079_; 
v___x_4077_ = lean_box(0);
v_sz_4078_ = lean_array_size(v_snd_4062_);
v___x_4079_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__16(v___x_4057_, v_snd_4062_, v_sz_4078_, v___x_4071_, v___x_4077_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
lean_dec(v_snd_4062_);
if (lean_obj_tag(v___x_4079_) == 0)
{
lean_dec_ref_known(v___x_4079_, 1);
v___y_4010_ = v___x_4058_;
v___y_4011_ = v_a_4055_;
v_a_4012_ = v_hasTrace_3977_;
goto v___jp_4009_;
}
else
{
lean_object* v_a_4080_; 
v_a_4080_ = lean_ctor_get(v___x_4079_, 0);
lean_inc(v_a_4080_);
lean_dec_ref_known(v___x_4079_, 1);
v___y_4016_ = v___x_4058_;
v___y_4017_ = v_a_4055_;
v_a_4018_ = v_a_4080_;
goto v___jp_4015_;
}
}
}
else
{
lean_object* v_val_4081_; uint8_t v___x_4082_; 
lean_inc_ref(v_fst_4074_);
lean_dec(v_a_4073_);
lean_dec(v_snd_4062_);
v_val_4081_ = lean_ctor_get(v_fst_4074_, 0);
lean_inc(v_val_4081_);
lean_dec_ref_known(v_fst_4074_, 1);
v___x_4082_ = lean_unbox(v_val_4081_);
lean_dec(v_val_4081_);
v___y_4010_ = v___x_4058_;
v___y_4011_ = v_a_4055_;
v_a_4012_ = v___x_4082_;
goto v___jp_4009_;
}
}
else
{
lean_object* v_a_4083_; 
lean_dec(v_snd_4062_);
v_a_4083_ = lean_ctor_get(v___x_4072_, 0);
lean_inc(v_a_4083_);
lean_dec_ref_known(v___x_4072_, 1);
v___y_4016_ = v___x_4058_;
v___y_4017_ = v_a_4055_;
v_a_4018_ = v_a_4083_;
goto v___jp_4015_;
}
}
}
}
else
{
lean_object* v_a_4086_; uint8_t v___x_4087_; 
lean_dec_ref(v_config_3897_);
v_a_4086_ = lean_ctor_get(v___x_4059_, 0);
lean_inc(v_a_4086_);
lean_dec_ref_known(v___x_4059_, 1);
v___x_4087_ = l_Lean_Exception_isInterrupt(v_a_4086_);
if (v___x_4087_ == 0)
{
uint8_t v___x_4088_; 
lean_inc(v_a_4086_);
v___x_4088_ = l_Lean_Exception_isRuntime(v_a_4086_);
v___y_4021_ = v___x_4058_;
v___y_4022_ = v_a_4086_;
v___y_4023_ = v_a_4055_;
v___y_4024_ = v___x_4088_;
goto v___jp_4020_;
}
else
{
v___y_4021_ = v___x_4058_;
v___y_4022_ = v_a_4086_;
v___y_4023_ = v_a_4055_;
v___y_4024_ = v___x_4087_;
goto v___jp_4020_;
}
}
}
else
{
lean_object* v___x_4089_; lean_object* v___x_4090_; 
v___x_4089_ = lean_io_get_num_heartbeats();
v___x_4090_ = lp_mathlib_Mathlib_Tactic_GCongr_applyGCongrLemma(v_goal_3894_, v_lem_3895_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
if (lean_obj_tag(v___x_4090_) == 0)
{
lean_object* v_a_4091_; lean_object* v_fst_4092_; lean_object* v_snd_4093_; uint8_t v___x_4094_; lean_object* v___x_4095_; size_t v_sz_4096_; size_t v___x_4097_; lean_object* v___x_4098_; 
v_a_4091_ = lean_ctor_get(v___x_4090_, 0);
lean_inc(v_a_4091_);
lean_dec_ref_known(v___x_4090_, 1);
v_fst_4092_ = lean_ctor_get(v_a_4091_, 0);
lean_inc(v_fst_4092_);
v_snd_4093_ = lean_ctor_get(v_a_4091_, 1);
lean_inc(v_snd_4093_);
lean_dec(v_a_4091_);
v___x_4094_ = 0;
v___x_4095_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___closed__0));
v_sz_4096_ = lean_array_size(v_fst_4092_);
v___x_4097_ = ((size_t)0ULL);
v___x_4098_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17(v_config_3897_, v___x_4057_, v_forward_3896_, v_fst_4092_, v_sz_4096_, v___x_4097_, v___x_4095_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
lean_dec(v_fst_4092_);
if (lean_obj_tag(v___x_4098_) == 0)
{
lean_object* v_a_4099_; lean_object* v_fst_4100_; 
v_a_4099_ = lean_ctor_get(v___x_4098_, 0);
lean_inc(v_a_4099_);
lean_dec_ref_known(v___x_4098_, 1);
v_fst_4100_ = lean_ctor_get(v_a_4099_, 0);
if (lean_obj_tag(v_fst_4100_) == 0)
{
lean_object* v_snd_4101_; uint8_t v___x_4102_; 
v_snd_4101_ = lean_ctor_get(v_a_4099_, 1);
lean_inc(v_snd_4101_);
lean_dec(v_a_4099_);
v___x_4102_ = lean_unbox(v_snd_4101_);
lean_dec(v_snd_4101_);
if (v___x_4102_ == 0)
{
lean_dec(v_snd_4093_);
v___y_4043_ = v___x_4089_;
v___y_4044_ = v_a_4055_;
v_a_4045_ = v___x_4094_;
goto v___jp_4042_;
}
else
{
lean_object* v___x_4103_; size_t v_sz_4104_; lean_object* v___x_4105_; 
v___x_4103_ = lean_box(0);
v_sz_4104_ = lean_array_size(v_snd_4093_);
v___x_4105_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__18(v___x_4057_, v_snd_4093_, v_sz_4104_, v___x_4097_, v___x_4103_, v_a_3898_, v_a_3899_, v_a_3900_, v_a_3901_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_);
lean_dec(v_snd_4093_);
if (lean_obj_tag(v___x_4105_) == 0)
{
lean_dec_ref_known(v___x_4105_, 1);
v___y_4043_ = v___x_4089_;
v___y_4044_ = v_a_4055_;
v_a_4045_ = v___x_4057_;
goto v___jp_4042_;
}
else
{
lean_object* v_a_4106_; 
v_a_4106_ = lean_ctor_get(v___x_4105_, 0);
lean_inc(v_a_4106_);
lean_dec_ref_known(v___x_4105_, 1);
v___y_4038_ = v___x_4089_;
v___y_4039_ = v_a_4055_;
v_a_4040_ = v_a_4106_;
goto v___jp_4037_;
}
}
}
else
{
lean_object* v_val_4107_; uint8_t v___x_4108_; 
lean_inc_ref(v_fst_4100_);
lean_dec(v_a_4099_);
lean_dec(v_snd_4093_);
v_val_4107_ = lean_ctor_get(v_fst_4100_, 0);
lean_inc(v_val_4107_);
lean_dec_ref_known(v_fst_4100_, 1);
v___x_4108_ = lean_unbox(v_val_4107_);
lean_dec(v_val_4107_);
v___y_4043_ = v___x_4089_;
v___y_4044_ = v_a_4055_;
v_a_4045_ = v___x_4108_;
goto v___jp_4042_;
}
}
else
{
lean_object* v_a_4109_; 
lean_dec(v_snd_4093_);
v_a_4109_ = lean_ctor_get(v___x_4098_, 0);
lean_inc(v_a_4109_);
lean_dec_ref_known(v___x_4098_, 1);
v___y_4038_ = v___x_4089_;
v___y_4039_ = v_a_4055_;
v_a_4040_ = v_a_4109_;
goto v___jp_4037_;
}
}
else
{
lean_object* v_a_4110_; uint8_t v___x_4111_; 
lean_dec_ref(v_config_3897_);
v_a_4110_ = lean_ctor_get(v___x_4090_, 0);
lean_inc(v_a_4110_);
lean_dec_ref_known(v___x_4090_, 1);
v___x_4111_ = l_Lean_Exception_isInterrupt(v_a_4110_);
if (v___x_4111_ == 0)
{
uint8_t v___x_4112_; 
lean_inc(v_a_4110_);
v___x_4112_ = l_Lean_Exception_isRuntime(v_a_4110_);
v___y_4049_ = v_a_4110_;
v___y_4050_ = v___x_4089_;
v___y_4051_ = v_a_4055_;
v___y_4052_ = v___x_4112_;
goto v___jp_4048_;
}
else
{
v___y_4049_ = v_a_4110_;
v___y_4050_ = v___x_4089_;
v___y_4051_ = v_a_4055_;
v___y_4052_ = v___x_4111_;
goto v___jp_4048_;
}
}
}
}
else
{
lean_object* v_a_4113_; lean_object* v___x_4115_; uint8_t v_isShared_4116_; uint8_t v_isSharedCheck_4120_; 
lean_dec_ref(v___f_3989_);
lean_dec_ref(v_config_3897_);
lean_dec_ref(v_lem_3895_);
lean_dec(v_goal_3894_);
v_a_4113_ = lean_ctor_get(v___x_4054_, 0);
v_isSharedCheck_4120_ = !lean_is_exclusive(v___x_4054_);
if (v_isSharedCheck_4120_ == 0)
{
v___x_4115_ = v___x_4054_;
v_isShared_4116_ = v_isSharedCheck_4120_;
goto v_resetjp_4114_;
}
else
{
lean_inc(v_a_4113_);
lean_dec(v___x_4054_);
v___x_4115_ = lean_box(0);
v_isShared_4116_ = v_isSharedCheck_4120_;
goto v_resetjp_4114_;
}
v_resetjp_4114_:
{
lean_object* v___x_4118_; 
if (v_isShared_4116_ == 0)
{
v___x_4118_ = v___x_4115_;
goto v_reusejp_4117_;
}
else
{
lean_object* v_reuseFailAlloc_4119_; 
v_reuseFailAlloc_4119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4119_, 0, v_a_4113_);
v___x_4118_ = v_reuseFailAlloc_4119_;
goto v_reusejp_4117_;
}
v_reusejp_4117_:
{
return v___x_4118_;
}
}
}
}
}
v___jp_3907_:
{
if (lean_obj_tag(v_e_3908_) == 0)
{
lean_object* v_a_3917_; lean_object* v___x_3919_; uint8_t v_isShared_3920_; uint8_t v_isSharedCheck_3924_; 
lean_dec_ref(v_config_3897_);
v_a_3917_ = lean_ctor_get(v_e_3908_, 0);
v_isSharedCheck_3924_ = !lean_is_exclusive(v_e_3908_);
if (v_isSharedCheck_3924_ == 0)
{
v___x_3919_ = v_e_3908_;
v_isShared_3920_ = v_isSharedCheck_3924_;
goto v_resetjp_3918_;
}
else
{
lean_inc(v_a_3917_);
lean_dec(v_e_3908_);
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
v_reuseFailAlloc_3923_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_3925_; lean_object* v_fst_3926_; lean_object* v_snd_3927_; uint8_t v_anyProgress_3928_; lean_object* v___x_3929_; size_t v_sz_3930_; size_t v___x_3931_; lean_object* v___x_3932_; 
v_a_3925_ = lean_ctor_get(v_e_3908_, 0);
lean_inc(v_a_3925_);
lean_dec_ref_known(v_e_3908_, 1);
v_fst_3926_ = lean_ctor_get(v_a_3925_, 0);
lean_inc(v_fst_3926_);
v_snd_3927_ = lean_ctor_get(v_a_3925_, 1);
lean_inc(v_snd_3927_);
lean_dec(v_a_3925_);
v_anyProgress_3928_ = 0;
v___x_3929_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___closed__0));
v_sz_3930_ = lean_array_size(v_fst_3926_);
v___x_3931_ = ((size_t)0ULL);
v___x_3932_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12(v_config_3897_, v_forward_3896_, v_fst_3926_, v_sz_3930_, v___x_3931_, v___x_3929_, v___y_3909_, v___y_3910_, v___y_3911_, v___y_3912_, v___y_3913_, v___y_3914_, v___y_3915_, v___y_3916_);
lean_dec(v_fst_3926_);
if (lean_obj_tag(v___x_3932_) == 0)
{
lean_object* v_a_3933_; lean_object* v___x_3935_; uint8_t v_isShared_3936_; uint8_t v_isSharedCheck_3967_; 
v_a_3933_ = lean_ctor_get(v___x_3932_, 0);
v_isSharedCheck_3967_ = !lean_is_exclusive(v___x_3932_);
if (v_isSharedCheck_3967_ == 0)
{
v___x_3935_ = v___x_3932_;
v_isShared_3936_ = v_isSharedCheck_3967_;
goto v_resetjp_3934_;
}
else
{
lean_inc(v_a_3933_);
lean_dec(v___x_3932_);
v___x_3935_ = lean_box(0);
v_isShared_3936_ = v_isSharedCheck_3967_;
goto v_resetjp_3934_;
}
v_resetjp_3934_:
{
lean_object* v_fst_3937_; 
v_fst_3937_ = lean_ctor_get(v_a_3933_, 0);
if (lean_obj_tag(v_fst_3937_) == 0)
{
lean_object* v_snd_3938_; uint8_t v___x_3939_; 
v_snd_3938_ = lean_ctor_get(v_a_3933_, 1);
lean_inc(v_snd_3938_);
lean_dec(v_a_3933_);
v___x_3939_ = lean_unbox(v_snd_3938_);
if (v___x_3939_ == 0)
{
lean_object* v___x_3940_; lean_object* v___x_3942_; 
lean_dec(v_snd_3938_);
lean_dec(v_snd_3927_);
v___x_3940_ = lean_box(v_anyProgress_3928_);
if (v_isShared_3936_ == 0)
{
lean_ctor_set(v___x_3935_, 0, v___x_3940_);
v___x_3942_ = v___x_3935_;
goto v_reusejp_3941_;
}
else
{
lean_object* v_reuseFailAlloc_3943_; 
v_reuseFailAlloc_3943_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3943_, 0, v___x_3940_);
v___x_3942_ = v_reuseFailAlloc_3943_;
goto v_reusejp_3941_;
}
v_reusejp_3941_:
{
return v___x_3942_;
}
}
else
{
lean_object* v___x_3944_; size_t v_sz_3945_; lean_object* v___x_3946_; 
lean_del_object(v___x_3935_);
v___x_3944_ = lean_box(0);
v_sz_3945_ = lean_array_size(v_snd_3927_);
v___x_3946_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__13(v_snd_3927_, v_sz_3945_, v___x_3931_, v___x_3944_, v___y_3909_, v___y_3910_, v___y_3911_, v___y_3912_, v___y_3913_, v___y_3914_, v___y_3915_, v___y_3916_);
lean_dec(v_snd_3927_);
if (lean_obj_tag(v___x_3946_) == 0)
{
lean_object* v___x_3948_; uint8_t v_isShared_3949_; uint8_t v_isSharedCheck_3953_; 
v_isSharedCheck_3953_ = !lean_is_exclusive(v___x_3946_);
if (v_isSharedCheck_3953_ == 0)
{
lean_object* v_unused_3954_; 
v_unused_3954_ = lean_ctor_get(v___x_3946_, 0);
lean_dec(v_unused_3954_);
v___x_3948_ = v___x_3946_;
v_isShared_3949_ = v_isSharedCheck_3953_;
goto v_resetjp_3947_;
}
else
{
lean_dec(v___x_3946_);
v___x_3948_ = lean_box(0);
v_isShared_3949_ = v_isSharedCheck_3953_;
goto v_resetjp_3947_;
}
v_resetjp_3947_:
{
lean_object* v___x_3951_; 
if (v_isShared_3949_ == 0)
{
lean_ctor_set(v___x_3948_, 0, v_snd_3938_);
v___x_3951_ = v___x_3948_;
goto v_reusejp_3950_;
}
else
{
lean_object* v_reuseFailAlloc_3952_; 
v_reuseFailAlloc_3952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3952_, 0, v_snd_3938_);
v___x_3951_ = v_reuseFailAlloc_3952_;
goto v_reusejp_3950_;
}
v_reusejp_3950_:
{
return v___x_3951_;
}
}
}
else
{
lean_object* v_a_3955_; lean_object* v___x_3957_; uint8_t v_isShared_3958_; uint8_t v_isSharedCheck_3962_; 
lean_dec(v_snd_3938_);
v_a_3955_ = lean_ctor_get(v___x_3946_, 0);
v_isSharedCheck_3962_ = !lean_is_exclusive(v___x_3946_);
if (v_isSharedCheck_3962_ == 0)
{
v___x_3957_ = v___x_3946_;
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
else
{
lean_inc(v_a_3955_);
lean_dec(v___x_3946_);
v___x_3957_ = lean_box(0);
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
v_resetjp_3956_:
{
lean_object* v___x_3960_; 
if (v_isShared_3958_ == 0)
{
v___x_3960_ = v___x_3957_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3961_; 
v_reuseFailAlloc_3961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3961_, 0, v_a_3955_);
v___x_3960_ = v_reuseFailAlloc_3961_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
return v___x_3960_;
}
}
}
}
}
else
{
lean_object* v_val_3963_; lean_object* v___x_3965_; 
lean_inc_ref(v_fst_3937_);
lean_dec(v_a_3933_);
lean_dec(v_snd_3927_);
v_val_3963_ = lean_ctor_get(v_fst_3937_, 0);
lean_inc(v_val_3963_);
lean_dec_ref_known(v_fst_3937_, 1);
if (v_isShared_3936_ == 0)
{
lean_ctor_set(v___x_3935_, 0, v_val_3963_);
v___x_3965_ = v___x_3935_;
goto v_reusejp_3964_;
}
else
{
lean_object* v_reuseFailAlloc_3966_; 
v_reuseFailAlloc_3966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3966_, 0, v_val_3963_);
v___x_3965_ = v_reuseFailAlloc_3966_;
goto v_reusejp_3964_;
}
v_reusejp_3964_:
{
return v___x_3965_;
}
}
}
}
else
{
lean_object* v_a_3968_; lean_object* v___x_3970_; uint8_t v_isShared_3971_; uint8_t v_isSharedCheck_3975_; 
lean_dec(v_snd_3927_);
v_a_3968_ = lean_ctor_get(v___x_3932_, 0);
v_isSharedCheck_3975_ = !lean_is_exclusive(v___x_3932_);
if (v_isSharedCheck_3975_ == 0)
{
v___x_3970_ = v___x_3932_;
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
else
{
lean_inc(v_a_3968_);
lean_dec(v___x_3932_);
v___x_3970_ = lean_box(0);
v_isShared_3971_ = v_isSharedCheck_3975_;
goto v_resetjp_3969_;
}
v_resetjp_3969_:
{
lean_object* v___x_3973_; 
if (v_isShared_3971_ == 0)
{
v___x_3973_ = v___x_3970_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_3974_; 
v_reuseFailAlloc_3974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3974_, 0, v_a_3968_);
v___x_3973_ = v_reuseFailAlloc_3974_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
return v___x_3973_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(lean_object* v_snd_4133_, uint8_t v_forward_4134_, lean_object* v_config_4135_, lean_object* v___x_4136_, lean_object* v_fst_4137_, lean_object* v_e_4138_, lean_object* v_as_x27_4139_, lean_object* v_b_4140_, lean_object* v___y_4141_, lean_object* v___y_4142_, lean_object* v___y_4143_, lean_object* v___y_4144_, lean_object* v___y_4145_, lean_object* v___y_4146_, lean_object* v___y_4147_, lean_object* v___y_4148_){
_start:
{
if (lean_obj_tag(v_as_x27_4139_) == 0)
{
lean_object* v___x_4150_; 
lean_dec_ref(v_e_4138_);
lean_dec_ref(v_fst_4137_);
lean_dec_ref(v___x_4136_);
lean_dec_ref(v_config_4135_);
lean_dec_ref(v_snd_4133_);
v___x_4150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4150_, 0, v_b_4140_);
return v___x_4150_;
}
else
{
lean_object* v_head_4151_; lean_object* v_tail_4152_; uint8_t v_forGrw_4153_; lean_object* v___x_4154_; lean_object* v___x_4155_; 
lean_dec_ref(v_b_4140_);
v_head_4151_ = lean_ctor_get(v_as_x27_4139_, 0);
v_tail_4152_ = lean_ctor_get(v_as_x27_4139_, 1);
v_forGrw_4153_ = lean_ctor_get_uint8(v_head_4151_, sizeof(void*)*6);
v___x_4154_ = lean_box(0);
v___x_4155_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0));
if (v_forGrw_4153_ == 0)
{
v_as_x27_4139_ = v_tail_4152_;
v_b_4140_ = v___x_4155_;
goto _start;
}
else
{
lean_object* v___x_4157_; lean_object* v___x_4158_; 
v___x_4157_ = l_Lean_Expr_mvarId_x21(v_snd_4133_);
lean_inc_ref(v_config_4135_);
lean_inc(v_head_4151_);
v___x_4158_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma(v___x_4157_, v_head_4151_, v_forward_4134_, v_config_4135_, v___y_4141_, v___y_4142_, v___y_4143_, v___y_4144_, v___y_4145_, v___y_4146_, v___y_4147_, v___y_4148_);
if (lean_obj_tag(v___x_4158_) == 0)
{
lean_object* v_a_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4205_; 
v_a_4159_ = lean_ctor_get(v___x_4158_, 0);
v_isSharedCheck_4205_ = !lean_is_exclusive(v___x_4158_);
if (v_isSharedCheck_4205_ == 0)
{
v___x_4161_ = v___x_4158_;
v_isShared_4162_ = v_isSharedCheck_4205_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_a_4159_);
lean_dec(v___x_4158_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4205_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
uint8_t v___x_4163_; 
v___x_4163_ = lean_unbox(v_a_4159_);
lean_dec(v_a_4159_);
if (v___x_4163_ == 0)
{
lean_object* v___x_4164_; lean_object* v_cache_4165_; lean_object* v_zetaDeltaFVarIds_4166_; lean_object* v_postponed_4167_; lean_object* v_diag_4168_; lean_object* v___x_4170_; uint8_t v_isShared_4171_; uint8_t v_isSharedCheck_4177_; 
lean_del_object(v___x_4161_);
v___x_4164_ = lean_st_ref_take(v___y_4146_);
v_cache_4165_ = lean_ctor_get(v___x_4164_, 1);
v_zetaDeltaFVarIds_4166_ = lean_ctor_get(v___x_4164_, 2);
v_postponed_4167_ = lean_ctor_get(v___x_4164_, 3);
v_diag_4168_ = lean_ctor_get(v___x_4164_, 4);
v_isSharedCheck_4177_ = !lean_is_exclusive(v___x_4164_);
if (v_isSharedCheck_4177_ == 0)
{
lean_object* v_unused_4178_; 
v_unused_4178_ = lean_ctor_get(v___x_4164_, 0);
lean_dec(v_unused_4178_);
v___x_4170_ = v___x_4164_;
v_isShared_4171_ = v_isSharedCheck_4177_;
goto v_resetjp_4169_;
}
else
{
lean_inc(v_diag_4168_);
lean_inc(v_postponed_4167_);
lean_inc(v_zetaDeltaFVarIds_4166_);
lean_inc(v_cache_4165_);
lean_dec(v___x_4164_);
v___x_4170_ = lean_box(0);
v_isShared_4171_ = v_isSharedCheck_4177_;
goto v_resetjp_4169_;
}
v_resetjp_4169_:
{
lean_object* v___x_4173_; 
lean_inc_ref(v___x_4136_);
if (v_isShared_4171_ == 0)
{
lean_ctor_set(v___x_4170_, 0, v___x_4136_);
v___x_4173_ = v___x_4170_;
goto v_reusejp_4172_;
}
else
{
lean_object* v_reuseFailAlloc_4176_; 
v_reuseFailAlloc_4176_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4176_, 0, v___x_4136_);
lean_ctor_set(v_reuseFailAlloc_4176_, 1, v_cache_4165_);
lean_ctor_set(v_reuseFailAlloc_4176_, 2, v_zetaDeltaFVarIds_4166_);
lean_ctor_set(v_reuseFailAlloc_4176_, 3, v_postponed_4167_);
lean_ctor_set(v_reuseFailAlloc_4176_, 4, v_diag_4168_);
v___x_4173_ = v_reuseFailAlloc_4176_;
goto v_reusejp_4172_;
}
v_reusejp_4172_:
{
lean_object* v___x_4174_; 
v___x_4174_ = lean_st_ref_set(v___y_4146_, v___x_4173_);
v_as_x27_4139_ = v_tail_4152_;
v_b_4140_ = v___x_4155_;
goto _start;
}
}
}
else
{
lean_object* v___x_4179_; lean_object* v_a_4180_; lean_object* v___x_4182_; uint8_t v_isShared_4183_; uint8_t v_isSharedCheck_4204_; 
lean_dec_ref(v___x_4136_);
lean_dec_ref(v_config_4135_);
v___x_4179_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_fst_4137_, v___y_4146_);
v_a_4180_ = lean_ctor_get(v___x_4179_, 0);
v_isSharedCheck_4204_ = !lean_is_exclusive(v___x_4179_);
if (v_isSharedCheck_4204_ == 0)
{
v___x_4182_ = v___x_4179_;
v_isShared_4183_ = v_isSharedCheck_4204_;
goto v_resetjp_4181_;
}
else
{
lean_inc(v_a_4180_);
lean_dec(v___x_4179_);
v___x_4182_ = lean_box(0);
v_isShared_4183_ = v_isSharedCheck_4204_;
goto v_resetjp_4181_;
}
v_resetjp_4181_:
{
if (lean_obj_tag(v_e_4138_) == 7)
{
if (lean_obj_tag(v_a_4180_) == 7)
{
lean_object* v_binderName_4192_; uint8_t v_binderInfo_4193_; lean_object* v_binderType_4194_; lean_object* v_body_4195_; lean_object* v___x_4196_; lean_object* v___x_4197_; lean_object* v___x_4198_; lean_object* v___x_4199_; lean_object* v___x_4200_; lean_object* v___x_4202_; 
lean_del_object(v___x_4182_);
v_binderName_4192_ = lean_ctor_get(v_e_4138_, 0);
lean_inc(v_binderName_4192_);
v_binderInfo_4193_ = lean_ctor_get_uint8(v_e_4138_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_4138_, 3);
v_binderType_4194_ = lean_ctor_get(v_a_4180_, 1);
lean_inc_ref(v_binderType_4194_);
v_body_4195_ = lean_ctor_get(v_a_4180_, 2);
lean_inc_ref(v_body_4195_);
lean_dec_ref_known(v_a_4180_, 3);
v___x_4196_ = l_Lean_Expr_forallE___override(v_binderName_4192_, v_binderType_4194_, v_body_4195_, v_binderInfo_4193_);
v___x_4197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4197_, 0, v___x_4196_);
lean_ctor_set(v___x_4197_, 1, v_snd_4133_);
v___x_4198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4198_, 0, v___x_4197_);
v___x_4199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4199_, 0, v___x_4198_);
v___x_4200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4200_, 0, v___x_4199_);
lean_ctor_set(v___x_4200_, 1, v___x_4154_);
if (v_isShared_4162_ == 0)
{
lean_ctor_set(v___x_4161_, 0, v___x_4200_);
v___x_4202_ = v___x_4161_;
goto v_reusejp_4201_;
}
else
{
lean_object* v_reuseFailAlloc_4203_; 
v_reuseFailAlloc_4203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4203_, 0, v___x_4200_);
v___x_4202_ = v_reuseFailAlloc_4203_;
goto v_reusejp_4201_;
}
v_reusejp_4201_:
{
return v___x_4202_;
}
}
else
{
lean_dec_ref_known(v_e_4138_, 3);
lean_del_object(v___x_4161_);
goto v___jp_4184_;
}
}
else
{
lean_del_object(v___x_4161_);
lean_dec_ref(v_e_4138_);
goto v___jp_4184_;
}
v___jp_4184_:
{
lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4190_; 
v___x_4185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4185_, 0, v_a_4180_);
lean_ctor_set(v___x_4185_, 1, v_snd_4133_);
v___x_4186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4186_, 0, v___x_4185_);
v___x_4187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4187_, 0, v___x_4186_);
v___x_4188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4188_, 0, v___x_4187_);
lean_ctor_set(v___x_4188_, 1, v___x_4154_);
if (v_isShared_4183_ == 0)
{
lean_ctor_set(v___x_4182_, 0, v___x_4188_);
v___x_4190_ = v___x_4182_;
goto v_reusejp_4189_;
}
else
{
lean_object* v_reuseFailAlloc_4191_; 
v_reuseFailAlloc_4191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4191_, 0, v___x_4188_);
v___x_4190_ = v_reuseFailAlloc_4191_;
goto v_reusejp_4189_;
}
v_reusejp_4189_:
{
return v___x_4190_;
}
}
}
}
}
}
else
{
lean_object* v_a_4206_; lean_object* v___x_4208_; uint8_t v_isShared_4209_; uint8_t v_isSharedCheck_4213_; 
lean_dec_ref(v_e_4138_);
lean_dec_ref(v_fst_4137_);
lean_dec_ref(v___x_4136_);
lean_dec_ref(v_config_4135_);
lean_dec_ref(v_snd_4133_);
v_a_4206_ = lean_ctor_get(v___x_4158_, 0);
v_isSharedCheck_4213_ = !lean_is_exclusive(v___x_4158_);
if (v_isSharedCheck_4213_ == 0)
{
v___x_4208_ = v___x_4158_;
v_isShared_4209_ = v_isSharedCheck_4213_;
goto v_resetjp_4207_;
}
else
{
lean_inc(v_a_4206_);
lean_dec(v___x_4158_);
v___x_4208_ = lean_box(0);
v_isShared_4209_ = v_isSharedCheck_4213_;
goto v_resetjp_4207_;
}
v_resetjp_4207_:
{
lean_object* v___x_4211_; 
if (v_isShared_4209_ == 0)
{
v___x_4211_ = v___x_4208_;
goto v_reusejp_4210_;
}
else
{
lean_object* v_reuseFailAlloc_4212_; 
v_reuseFailAlloc_4212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4212_, 0, v_a_4206_);
v___x_4211_ = v_reuseFailAlloc_4212_;
goto v_reusejp_4210_;
}
v_reusejp_4210_:
{
return v___x_4211_;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1(void){
_start:
{
lean_object* v___x_4215_; lean_object* v___x_4216_; 
v___x_4215_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__0));
v___x_4216_ = l_Lean_stringToMessageData(v___x_4215_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2(lean_object* v___x_4217_, lean_object* v_a_4218_, lean_object* v_relName_4219_, uint8_t v_forward_4220_, lean_object* v_snd_4221_, lean_object* v_config_4222_, lean_object* v_fst_4223_, lean_object* v_____r_4224_, lean_object* v___y_4225_, lean_object* v___y_4226_, lean_object* v___y_4227_, lean_object* v___y_4228_, lean_object* v___y_4229_, lean_object* v___y_4230_, lean_object* v___y_4231_, lean_object* v___y_4232_){
_start:
{
lean_object* v___y_4235_; lean_object* v___x_4251_; 
lean_inc_ref(v_a_4218_);
v___x_4251_ = lp_mathlib_Mathlib_Tactic_GCongr_getCongrAppFnArgs(v_a_4218_);
if (lean_obj_tag(v___x_4251_) == 1)
{
lean_object* v_val_4252_; lean_object* v_fst_4253_; lean_object* v_snd_4254_; lean_object* v___x_4255_; lean_object* v___x_4256_; 
v_val_4252_ = lean_ctor_get(v___x_4251_, 0);
lean_inc(v_val_4252_);
lean_dec_ref_known(v___x_4251_, 1);
v_fst_4253_ = lean_ctor_get(v_val_4252_, 0);
lean_inc(v_fst_4253_);
v_snd_4254_ = lean_ctor_get(v_val_4252_, 1);
lean_inc(v_snd_4254_);
lean_dec(v_val_4252_);
v___x_4255_ = lean_array_get_size(v_snd_4254_);
lean_dec(v_snd_4254_);
lean_inc(v_relName_4219_);
v___x_4256_ = lp_mathlib_Mathlib_Tactic_GCongr_findGCongrLemmas_x3f_x27___redArg(v_relName_4219_, v_fst_4253_, v_forward_4220_, v___x_4255_, v___y_4232_);
if (lean_obj_tag(v___x_4256_) == 0)
{
lean_object* v_a_4257_; lean_object* v_lemmas_4259_; lean_object* v___y_4260_; lean_object* v___y_4261_; lean_object* v___y_4262_; lean_object* v___y_4263_; lean_object* v___y_4264_; lean_object* v___y_4265_; lean_object* v___y_4266_; lean_object* v___y_4267_; lean_object* v___x_4290_; uint8_t v___x_4291_; 
v_a_4257_ = lean_ctor_get(v___x_4256_, 0);
lean_inc(v_a_4257_);
lean_dec_ref_known(v___x_4256_, 1);
v___x_4290_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1));
v___x_4291_ = lean_name_eq(v_relName_4219_, v___x_4290_);
lean_dec(v_relName_4219_);
if (v___x_4291_ == 0)
{
v_lemmas_4259_ = v_a_4257_;
v___y_4260_ = v___y_4225_;
v___y_4261_ = v___y_4226_;
v___y_4262_ = v___y_4227_;
v___y_4263_ = v___y_4228_;
v___y_4264_ = v___y_4229_;
v___y_4265_ = v___y_4230_;
v___y_4266_ = v___y_4231_;
v___y_4267_ = v___y_4232_;
goto v___jp_4258_;
}
else
{
lean_object* v___x_4292_; lean_object* v___x_4293_; 
v___x_4292_ = lp_mathlib_Mathlib_Tactic_GCongr_relImpRelLemma(v___x_4255_);
v___x_4293_ = l_List_appendTR___redArg(v_a_4257_, v___x_4292_);
v_lemmas_4259_ = v___x_4293_;
v___y_4260_ = v___y_4225_;
v___y_4261_ = v___y_4226_;
v___y_4262_ = v___y_4227_;
v___y_4263_ = v___y_4228_;
v___y_4264_ = v___y_4229_;
v___y_4265_ = v___y_4230_;
v___y_4266_ = v___y_4231_;
v___y_4267_ = v___y_4232_;
goto v___jp_4258_;
}
v___jp_4258_:
{
lean_object* v___x_4268_; lean_object* v_mctx_4269_; lean_object* v___x_4270_; lean_object* v___x_4271_; 
v___x_4268_ = lean_st_ref_get(v___y_4265_);
v_mctx_4269_ = lean_ctor_get(v___x_4268_, 0);
lean_inc_ref(v_mctx_4269_);
lean_dec(v___x_4268_);
v___x_4270_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0));
v___x_4271_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(v_snd_4221_, v_forward_4220_, v_config_4222_, v_mctx_4269_, v_fst_4223_, v_a_4218_, v_lemmas_4259_, v___x_4270_, v___y_4260_, v___y_4261_, v___y_4262_, v___y_4263_, v___y_4264_, v___y_4265_, v___y_4266_, v___y_4267_);
lean_dec(v_lemmas_4259_);
if (lean_obj_tag(v___x_4271_) == 0)
{
lean_object* v_a_4272_; lean_object* v___x_4274_; uint8_t v_isShared_4275_; uint8_t v_isSharedCheck_4281_; 
v_a_4272_ = lean_ctor_get(v___x_4271_, 0);
v_isSharedCheck_4281_ = !lean_is_exclusive(v___x_4271_);
if (v_isSharedCheck_4281_ == 0)
{
v___x_4274_ = v___x_4271_;
v_isShared_4275_ = v_isSharedCheck_4281_;
goto v_resetjp_4273_;
}
else
{
lean_inc(v_a_4272_);
lean_dec(v___x_4271_);
v___x_4274_ = lean_box(0);
v_isShared_4275_ = v_isSharedCheck_4281_;
goto v_resetjp_4273_;
}
v_resetjp_4273_:
{
lean_object* v_fst_4276_; 
v_fst_4276_ = lean_ctor_get(v_a_4272_, 0);
lean_inc(v_fst_4276_);
lean_dec(v_a_4272_);
if (lean_obj_tag(v_fst_4276_) == 0)
{
lean_del_object(v___x_4274_);
v___y_4235_ = v___y_4261_;
goto v___jp_4234_;
}
else
{
lean_object* v_val_4277_; lean_object* v___x_4279_; 
lean_dec_ref(v___x_4217_);
v_val_4277_ = lean_ctor_get(v_fst_4276_, 0);
lean_inc(v_val_4277_);
lean_dec_ref_known(v_fst_4276_, 1);
if (v_isShared_4275_ == 0)
{
lean_ctor_set(v___x_4274_, 0, v_val_4277_);
v___x_4279_ = v___x_4274_;
goto v_reusejp_4278_;
}
else
{
lean_object* v_reuseFailAlloc_4280_; 
v_reuseFailAlloc_4280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4280_, 0, v_val_4277_);
v___x_4279_ = v_reuseFailAlloc_4280_;
goto v_reusejp_4278_;
}
v_reusejp_4278_:
{
return v___x_4279_;
}
}
}
}
else
{
lean_object* v_a_4282_; lean_object* v___x_4284_; uint8_t v_isShared_4285_; uint8_t v_isSharedCheck_4289_; 
lean_dec_ref(v___x_4217_);
v_a_4282_ = lean_ctor_get(v___x_4271_, 0);
v_isSharedCheck_4289_ = !lean_is_exclusive(v___x_4271_);
if (v_isSharedCheck_4289_ == 0)
{
v___x_4284_ = v___x_4271_;
v_isShared_4285_ = v_isSharedCheck_4289_;
goto v_resetjp_4283_;
}
else
{
lean_inc(v_a_4282_);
lean_dec(v___x_4271_);
v___x_4284_ = lean_box(0);
v_isShared_4285_ = v_isSharedCheck_4289_;
goto v_resetjp_4283_;
}
v_resetjp_4283_:
{
lean_object* v___x_4287_; 
if (v_isShared_4285_ == 0)
{
v___x_4287_ = v___x_4284_;
goto v_reusejp_4286_;
}
else
{
lean_object* v_reuseFailAlloc_4288_; 
v_reuseFailAlloc_4288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4288_, 0, v_a_4282_);
v___x_4287_ = v_reuseFailAlloc_4288_;
goto v_reusejp_4286_;
}
v_reusejp_4286_:
{
return v___x_4287_;
}
}
}
}
}
else
{
lean_object* v_a_4294_; lean_object* v___x_4296_; uint8_t v_isShared_4297_; uint8_t v_isSharedCheck_4301_; 
lean_dec_ref(v_fst_4223_);
lean_dec_ref(v_config_4222_);
lean_dec_ref(v_snd_4221_);
lean_dec(v_relName_4219_);
lean_dec_ref(v_a_4218_);
lean_dec_ref(v___x_4217_);
v_a_4294_ = lean_ctor_get(v___x_4256_, 0);
v_isSharedCheck_4301_ = !lean_is_exclusive(v___x_4256_);
if (v_isSharedCheck_4301_ == 0)
{
v___x_4296_ = v___x_4256_;
v_isShared_4297_ = v_isSharedCheck_4301_;
goto v_resetjp_4295_;
}
else
{
lean_inc(v_a_4294_);
lean_dec(v___x_4256_);
v___x_4296_ = lean_box(0);
v_isShared_4297_ = v_isSharedCheck_4301_;
goto v_resetjp_4295_;
}
v_resetjp_4295_:
{
lean_object* v___x_4299_; 
if (v_isShared_4297_ == 0)
{
v___x_4299_ = v___x_4296_;
goto v_reusejp_4298_;
}
else
{
lean_object* v_reuseFailAlloc_4300_; 
v_reuseFailAlloc_4300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4300_, 0, v_a_4294_);
v___x_4299_ = v_reuseFailAlloc_4300_;
goto v_reusejp_4298_;
}
v_reusejp_4298_:
{
return v___x_4299_;
}
}
}
}
else
{
lean_dec(v___x_4251_);
lean_dec_ref(v_fst_4223_);
lean_dec_ref(v_config_4222_);
lean_dec_ref(v_snd_4221_);
lean_dec(v_relName_4219_);
lean_dec_ref(v_a_4218_);
v___y_4235_ = v___y_4226_;
goto v___jp_4234_;
}
v___jp_4234_:
{
lean_object* v___x_4236_; lean_object* v_cache_4237_; lean_object* v_progress_4238_; lean_object* v___x_4240_; uint8_t v_isShared_4241_; uint8_t v_isSharedCheck_4250_; 
v___x_4236_ = lean_st_ref_take(v___y_4235_);
v_cache_4237_ = lean_ctor_get(v___x_4236_, 0);
v_progress_4238_ = lean_ctor_get(v___x_4236_, 1);
v_isSharedCheck_4250_ = !lean_is_exclusive(v___x_4236_);
if (v_isSharedCheck_4250_ == 0)
{
v___x_4240_ = v___x_4236_;
v_isShared_4241_ = v_isSharedCheck_4250_;
goto v_resetjp_4239_;
}
else
{
lean_inc(v_progress_4238_);
lean_inc(v_cache_4237_);
lean_dec(v___x_4236_);
v___x_4240_ = lean_box(0);
v_isShared_4241_ = v_isSharedCheck_4250_;
goto v_resetjp_4239_;
}
v_resetjp_4239_:
{
lean_object* v___x_4242_; lean_object* v___x_4243_; lean_object* v___x_4245_; 
v___x_4242_ = lean_box(0);
v___x_4243_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4___redArg(v_cache_4237_, v___x_4217_, v___x_4242_);
if (v_isShared_4241_ == 0)
{
lean_ctor_set(v___x_4240_, 0, v___x_4243_);
v___x_4245_ = v___x_4240_;
goto v_reusejp_4244_;
}
else
{
lean_object* v_reuseFailAlloc_4249_; 
v_reuseFailAlloc_4249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4249_, 0, v___x_4243_);
lean_ctor_set(v_reuseFailAlloc_4249_, 1, v_progress_4238_);
v___x_4245_ = v_reuseFailAlloc_4249_;
goto v_reusejp_4244_;
}
v_reusejp_4244_:
{
lean_object* v___x_4246_; lean_object* v___x_4247_; lean_object* v___x_4248_; 
v___x_4246_ = lean_st_ref_set(v___y_4235_, v___x_4245_);
v___x_4247_ = lean_box(0);
v___x_4248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4248_, 0, v___x_4247_);
return v___x_4248_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2___boxed(lean_object** _args){
lean_object* v___x_4302_ = _args[0];
lean_object* v_a_4303_ = _args[1];
lean_object* v_relName_4304_ = _args[2];
lean_object* v_forward_4305_ = _args[3];
lean_object* v_snd_4306_ = _args[4];
lean_object* v_config_4307_ = _args[5];
lean_object* v_fst_4308_ = _args[6];
lean_object* v_____r_4309_ = _args[7];
lean_object* v___y_4310_ = _args[8];
lean_object* v___y_4311_ = _args[9];
lean_object* v___y_4312_ = _args[10];
lean_object* v___y_4313_ = _args[11];
lean_object* v___y_4314_ = _args[12];
lean_object* v___y_4315_ = _args[13];
lean_object* v___y_4316_ = _args[14];
lean_object* v___y_4317_ = _args[15];
lean_object* v___y_4318_ = _args[16];
_start:
{
uint8_t v_forward_boxed_4319_; lean_object* v_res_4320_; 
v_forward_boxed_4319_ = lean_unbox(v_forward_4305_);
v_res_4320_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2(v___x_4302_, v_a_4303_, v_relName_4304_, v_forward_boxed_4319_, v_snd_4306_, v_config_4307_, v_fst_4308_, v_____r_4309_, v___y_4310_, v___y_4311_, v___y_4312_, v___y_4313_, v___y_4314_, v___y_4315_, v___y_4316_, v___y_4317_);
lean_dec(v___y_4317_);
lean_dec_ref(v___y_4316_);
lean_dec(v___y_4315_);
lean_dec_ref(v___y_4314_);
lean_dec(v___y_4313_);
lean_dec_ref(v___y_4312_);
lean_dec(v___y_4311_);
lean_dec_ref(v___y_4310_);
return v_res_4320_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3(void){
_start:
{
lean_object* v___x_4322_; lean_object* v___x_4323_; 
v___x_4322_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__2));
v___x_4323_ = l_Lean_stringToMessageData(v___x_4322_);
return v___x_4323_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5(void){
_start:
{
lean_object* v___x_4325_; lean_object* v___x_4326_; 
v___x_4325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__4));
v___x_4326_ = l_Lean_stringToMessageData(v___x_4325_);
return v___x_4326_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7(void){
_start:
{
lean_object* v___x_4328_; lean_object* v___x_4329_; 
v___x_4328_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__6));
v___x_4329_ = l_Lean_stringToMessageData(v___x_4328_);
return v___x_4329_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9(void){
_start:
{
lean_object* v___x_4331_; lean_object* v___x_4332_; 
v___x_4331_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__8));
v___x_4332_ = l_Lean_stringToMessageData(v___x_4331_);
return v___x_4332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore(lean_object* v_relName_4335_, lean_object* v_rel_x3f_4336_, lean_object* v_e_4337_, uint8_t v_forward_4338_, lean_object* v_config_4339_, lean_object* v_a_4340_, lean_object* v_a_4341_, lean_object* v_a_4342_, lean_object* v_a_4343_, lean_object* v_a_4344_, lean_object* v_a_4345_, lean_object* v_a_4346_, lean_object* v_a_4347_){
_start:
{
lean_object* v___y_4350_; lean_object* v___y_4351_; lean_object* v___y_4368_; lean_object* v___y_4369_; lean_object* v___y_4370_; lean_object* v___y_4371_; lean_object* v_lemmas_4372_; lean_object* v___y_4373_; lean_object* v___y_4374_; lean_object* v___y_4375_; lean_object* v___y_4376_; lean_object* v___y_4377_; lean_object* v___y_4378_; lean_object* v___y_4379_; lean_object* v___y_4380_; lean_object* v___y_4404_; lean_object* v___y_4405_; lean_object* v___y_4406_; lean_object* v___y_4407_; lean_object* v___y_4408_; lean_object* v___y_4409_; lean_object* v___y_4410_; lean_object* v___y_4411_; lean_object* v___y_4412_; lean_object* v___y_4413_; lean_object* v___y_4414_; lean_object* v___y_4415_; lean_object* v___y_4436_; lean_object* v___y_4437_; lean_object* v___y_4438_; lean_object* v___y_4439_; lean_object* v___y_4440_; lean_object* v___y_4441_; lean_object* v___y_4442_; lean_object* v___y_4443_; lean_object* v___y_4444_; lean_object* v___y_4445_; lean_object* v___y_4446_; lean_object* v___y_4447_; lean_object* v___y_4448_; lean_object* v___y_4449_; uint8_t v___y_4450_; lean_object* v___y_4483_; lean_object* v___y_4484_; lean_object* v___y_4485_; lean_object* v___y_4486_; lean_object* v___y_4487_; lean_object* v___y_4488_; lean_object* v___y_4489_; lean_object* v___y_4490_; lean_object* v___y_4491_; lean_object* v___y_4492_; lean_object* v___y_4493_; lean_object* v___y_4494_; lean_object* v___y_4495_; uint8_t v___y_4496_; lean_object* v_options_4502_; lean_object* v_fileName_4503_; lean_object* v_fileMap_4504_; lean_object* v_currRecDepth_4505_; lean_object* v_maxRecDepth_4506_; lean_object* v_ref_4507_; lean_object* v_currNamespace_4508_; lean_object* v_openDecls_4509_; lean_object* v_initHeartbeats_4510_; lean_object* v_maxHeartbeats_4511_; lean_object* v_quotContext_4512_; lean_object* v_currMacroScope_4513_; uint8_t v_diag_4514_; lean_object* v_cancelTk_x3f_4515_; uint8_t v_suppressElabErrors_4516_; lean_object* v_inheritedTraceOptions_4517_; uint8_t v_hasTrace_4518_; lean_object* v_cls_4519_; lean_object* v___y_4521_; lean_object* v___y_4522_; lean_object* v___y_4523_; lean_object* v___y_4524_; lean_object* v___y_4525_; lean_object* v___y_4526_; lean_object* v___y_4527_; lean_object* v___y_4528_; lean_object* v___y_4529_; lean_object* v_a_4530_; lean_object* v_e_4579_; lean_object* v___y_4580_; lean_object* v___y_4581_; lean_object* v___y_4582_; lean_object* v___y_4583_; lean_object* v___y_4584_; lean_object* v___y_4585_; lean_object* v___y_4586_; lean_object* v___y_4587_; 
v_options_4502_ = lean_ctor_get(v_a_4346_, 2);
v_fileName_4503_ = lean_ctor_get(v_a_4346_, 0);
v_fileMap_4504_ = lean_ctor_get(v_a_4346_, 1);
v_currRecDepth_4505_ = lean_ctor_get(v_a_4346_, 3);
v_maxRecDepth_4506_ = lean_ctor_get(v_a_4346_, 4);
v_ref_4507_ = lean_ctor_get(v_a_4346_, 5);
v_currNamespace_4508_ = lean_ctor_get(v_a_4346_, 6);
v_openDecls_4509_ = lean_ctor_get(v_a_4346_, 7);
v_initHeartbeats_4510_ = lean_ctor_get(v_a_4346_, 8);
v_maxHeartbeats_4511_ = lean_ctor_get(v_a_4346_, 9);
v_quotContext_4512_ = lean_ctor_get(v_a_4346_, 10);
v_currMacroScope_4513_ = lean_ctor_get(v_a_4346_, 11);
v_diag_4514_ = lean_ctor_get_uint8(v_a_4346_, sizeof(void*)*14);
v_cancelTk_x3f_4515_ = lean_ctor_get(v_a_4346_, 12);
v_suppressElabErrors_4516_ = lean_ctor_get_uint8(v_a_4346_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4517_ = lean_ctor_get(v_a_4346_, 13);
v_hasTrace_4518_ = lean_ctor_get_uint8(v_options_4502_, sizeof(void*)*1);
v_cls_4519_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn___closed__1_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_));
if (v_hasTrace_4518_ == 0)
{
lean_object* v___x_4606_; 
v___x_4606_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_4337_, v_a_4345_);
if (lean_obj_tag(v___x_4606_) == 0)
{
lean_object* v_a_4607_; 
v_a_4607_ = lean_ctor_get(v___x_4606_, 0);
lean_inc(v_a_4607_);
lean_dec_ref_known(v___x_4606_, 1);
v_e_4579_ = v_a_4607_;
v___y_4580_ = v_a_4340_;
v___y_4581_ = v_a_4341_;
v___y_4582_ = v_a_4342_;
v___y_4583_ = v_a_4343_;
v___y_4584_ = v_a_4344_;
v___y_4585_ = v_a_4345_;
v___y_4586_ = v_a_4346_;
v___y_4587_ = v_a_4347_;
goto v___jp_4578_;
}
else
{
lean_object* v_a_4608_; lean_object* v___x_4610_; uint8_t v_isShared_4611_; uint8_t v_isSharedCheck_4615_; 
lean_dec_ref(v_config_4339_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4608_ = lean_ctor_get(v___x_4606_, 0);
v_isSharedCheck_4615_ = !lean_is_exclusive(v___x_4606_);
if (v_isSharedCheck_4615_ == 0)
{
v___x_4610_ = v___x_4606_;
v_isShared_4611_ = v_isSharedCheck_4615_;
goto v_resetjp_4609_;
}
else
{
lean_inc(v_a_4608_);
lean_dec(v___x_4606_);
v___x_4610_ = lean_box(0);
v_isShared_4611_ = v_isSharedCheck_4615_;
goto v_resetjp_4609_;
}
v_resetjp_4609_:
{
lean_object* v___x_4613_; 
if (v_isShared_4611_ == 0)
{
v___x_4613_ = v___x_4610_;
goto v_reusejp_4612_;
}
else
{
lean_object* v_reuseFailAlloc_4614_; 
v_reuseFailAlloc_4614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4614_, 0, v_a_4608_);
v___x_4613_ = v_reuseFailAlloc_4614_;
goto v_reusejp_4612_;
}
v_reusejp_4612_:
{
return v___x_4613_;
}
}
}
}
else
{
lean_object* v___x_4616_; lean_object* v___x_4617_; uint8_t v___x_4618_; lean_object* v___y_4620_; lean_object* v___y_4621_; lean_object* v___y_4622_; lean_object* v_a_4623_; lean_object* v___y_4633_; lean_object* v___y_4634_; lean_object* v___y_4635_; lean_object* v_a_4636_; lean_object* v___y_4639_; lean_object* v___y_4640_; lean_object* v___y_4641_; lean_object* v_a_4642_; lean_object* v___y_4645_; lean_object* v___y_4646_; lean_object* v___y_4647_; lean_object* v___y_4648_; lean_object* v___y_4652_; lean_object* v___y_4653_; lean_object* v___y_4654_; lean_object* v___y_4655_; lean_object* v___y_4656_; lean_object* v___y_4657_; uint8_t v___y_4658_; lean_object* v___y_4685_; lean_object* v___y_4686_; lean_object* v___y_4687_; lean_object* v___y_4688_; lean_object* v___y_4689_; lean_object* v___y_4690_; uint8_t v___y_4691_; lean_object* v___y_4697_; lean_object* v___y_4698_; lean_object* v___y_4699_; lean_object* v___y_4700_; lean_object* v_a_4701_; lean_object* v___y_4737_; lean_object* v___y_4738_; lean_object* v___y_4739_; lean_object* v_a_4740_; lean_object* v___y_4753_; lean_object* v___y_4754_; lean_object* v___y_4755_; lean_object* v_a_4756_; lean_object* v___y_4759_; lean_object* v___y_4760_; lean_object* v___y_4761_; lean_object* v_a_4762_; lean_object* v___y_4765_; lean_object* v___y_4766_; lean_object* v___y_4767_; lean_object* v___y_4768_; lean_object* v___y_4772_; lean_object* v___y_4773_; lean_object* v___y_4774_; lean_object* v___y_4775_; lean_object* v___y_4776_; lean_object* v___y_4777_; uint8_t v___y_4778_; lean_object* v___y_4805_; lean_object* v___y_4806_; lean_object* v___y_4807_; lean_object* v___y_4808_; lean_object* v___y_4809_; lean_object* v___y_4810_; uint8_t v___y_4811_; lean_object* v___y_4817_; lean_object* v___y_4818_; lean_object* v___y_4819_; lean_object* v___y_4820_; lean_object* v_a_4821_; lean_object* v___y_4857_; lean_object* v___y_4858_; lean_object* v___y_4859_; lean_object* v___y_4860_; lean_object* v___y_4909_; lean_object* v___y_4910_; lean_object* v___y_4911_; lean_object* v___y_4912_; 
v___x_4616_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__0));
v___x_4617_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_4618_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4517_, v_options_4502_, v___x_4617_);
if (v___x_4618_ == 0)
{
lean_object* v___x_4940_; uint8_t v___x_4941_; 
v___x_4940_ = l_Lean_trace_profiler;
v___x_4941_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_4502_, v___x_4940_);
if (v___x_4941_ == 0)
{
lean_object* v___x_4942_; 
v___x_4942_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_4337_, v_a_4345_);
if (lean_obj_tag(v___x_4942_) == 0)
{
lean_object* v_a_4943_; 
v_a_4943_ = lean_ctor_get(v___x_4942_, 0);
lean_inc(v_a_4943_);
lean_dec_ref_known(v___x_4942_, 1);
v_e_4579_ = v_a_4943_;
v___y_4580_ = v_a_4340_;
v___y_4581_ = v_a_4341_;
v___y_4582_ = v_a_4342_;
v___y_4583_ = v_a_4343_;
v___y_4584_ = v_a_4344_;
v___y_4585_ = v_a_4345_;
v___y_4586_ = v_a_4346_;
v___y_4587_ = v_a_4347_;
goto v___jp_4578_;
}
else
{
lean_object* v_a_4944_; lean_object* v___x_4946_; uint8_t v_isShared_4947_; uint8_t v_isSharedCheck_4951_; 
lean_dec_ref(v_config_4339_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4944_ = lean_ctor_get(v___x_4942_, 0);
v_isSharedCheck_4951_ = !lean_is_exclusive(v___x_4942_);
if (v_isSharedCheck_4951_ == 0)
{
v___x_4946_ = v___x_4942_;
v_isShared_4947_ = v_isSharedCheck_4951_;
goto v_resetjp_4945_;
}
else
{
lean_inc(v_a_4944_);
lean_dec(v___x_4942_);
v___x_4946_ = lean_box(0);
v_isShared_4947_ = v_isSharedCheck_4951_;
goto v_resetjp_4945_;
}
v_resetjp_4945_:
{
lean_object* v___x_4949_; 
if (v_isShared_4947_ == 0)
{
v___x_4949_ = v___x_4946_;
goto v_reusejp_4948_;
}
else
{
lean_object* v_reuseFailAlloc_4950_; 
v_reuseFailAlloc_4950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4950_, 0, v_a_4944_);
v___x_4949_ = v_reuseFailAlloc_4950_;
goto v_reusejp_4948_;
}
v_reusejp_4948_:
{
return v___x_4949_;
}
}
}
}
else
{
goto v___jp_4920_;
}
}
else
{
goto v___jp_4920_;
}
v___jp_4619_:
{
lean_object* v___x_4624_; double v___x_4625_; double v___x_4626_; lean_object* v___x_4627_; lean_object* v___x_4628_; lean_object* v___x_4629_; lean_object* v___x_4630_; lean_object* v___x_4631_; 
v___x_4624_ = lean_io_get_num_heartbeats();
v___x_4625_ = lean_float_of_nat(v___y_4622_);
v___x_4626_ = lean_float_of_nat(v___x_4624_);
v___x_4627_ = lean_box_float(v___x_4625_);
v___x_4628_ = lean_box_float(v___x_4626_);
v___x_4629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4629_, 0, v___x_4627_);
lean_ctor_set(v___x_4629_, 1, v___x_4628_);
v___x_4630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4630_, 0, v_a_4623_);
lean_ctor_set(v___x_4630_, 1, v___x_4629_);
lean_inc(v_ref_4507_);
v___x_4631_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10(v_cls_4519_, v_hasTrace_4518_, v___x_4616_, v_options_4502_, v___x_4618_, v___y_4620_, v_ref_4507_, v___y_4621_, v___x_4630_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
return v___x_4631_;
}
v___jp_4632_:
{
lean_object* v___x_4637_; 
v___x_4637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4637_, 0, v_a_4636_);
v___y_4620_ = v___y_4633_;
v___y_4621_ = v___y_4634_;
v___y_4622_ = v___y_4635_;
v_a_4623_ = v___x_4637_;
goto v___jp_4619_;
}
v___jp_4638_:
{
lean_object* v___x_4643_; 
v___x_4643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4643_, 0, v_a_4642_);
v___y_4620_ = v___y_4639_;
v___y_4621_ = v___y_4640_;
v___y_4622_ = v___y_4641_;
v_a_4623_ = v___x_4643_;
goto v___jp_4619_;
}
v___jp_4644_:
{
if (lean_obj_tag(v___y_4648_) == 0)
{
lean_object* v_a_4649_; 
v_a_4649_ = lean_ctor_get(v___y_4648_, 0);
lean_inc(v_a_4649_);
lean_dec_ref_known(v___y_4648_, 1);
v___y_4633_ = v___y_4645_;
v___y_4634_ = v___y_4646_;
v___y_4635_ = v___y_4647_;
v_a_4636_ = v_a_4649_;
goto v___jp_4632_;
}
else
{
lean_object* v_a_4650_; 
v_a_4650_ = lean_ctor_get(v___y_4648_, 0);
lean_inc(v_a_4650_);
lean_dec_ref_known(v___y_4648_, 1);
v___y_4639_ = v___y_4645_;
v___y_4640_ = v___y_4646_;
v___y_4641_ = v___y_4647_;
v_a_4642_ = v_a_4650_;
goto v___jp_4638_;
}
}
v___jp_4651_:
{
lean_object* v___x_4659_; 
lean_inc_ref(v_a_4340_);
v___x_4659_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(v_a_4340_, v___y_4652_, v___y_4658_, v_config_4339_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4659_) == 0)
{
lean_object* v_a_4660_; lean_object* v___x_4662_; uint8_t v_isShared_4663_; uint8_t v_isSharedCheck_4682_; 
v_a_4660_ = lean_ctor_get(v___x_4659_, 0);
v_isSharedCheck_4682_ = !lean_is_exclusive(v___x_4659_);
if (v_isSharedCheck_4682_ == 0)
{
v___x_4662_ = v___x_4659_;
v_isShared_4663_ = v_isSharedCheck_4682_;
goto v_resetjp_4661_;
}
else
{
lean_inc(v_a_4660_);
lean_dec(v___x_4659_);
v___x_4662_ = lean_box(0);
v_isShared_4663_ = v_isSharedCheck_4682_;
goto v_resetjp_4661_;
}
v_resetjp_4661_:
{
uint8_t v___x_4664_; 
v___x_4664_ = lean_unbox(v_a_4660_);
lean_dec(v_a_4660_);
if (v___x_4664_ == 0)
{
lean_object* v___x_4665_; lean_object* v___x_4666_; 
lean_del_object(v___x_4662_);
lean_dec_ref(v___y_4656_);
v___x_4665_ = lean_box(0);
lean_inc(v_a_4347_);
lean_inc_ref(v_a_4346_);
lean_inc(v_a_4345_);
lean_inc_ref(v_a_4344_);
lean_inc(v_a_4343_);
lean_inc_ref(v_a_4342_);
lean_inc(v_a_4341_);
lean_inc_ref(v_a_4340_);
v___x_4666_ = lean_apply_10(v___y_4653_, v___x_4665_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_, lean_box(0));
v___y_4645_ = v___y_4654_;
v___y_4646_ = v___y_4655_;
v___y_4647_ = v___y_4657_;
v___y_4648_ = v___x_4666_;
goto v___jp_4644_;
}
else
{
lean_object* v___x_4667_; lean_object* v_cache_4668_; lean_object* v___x_4670_; uint8_t v_isShared_4671_; uint8_t v_isSharedCheck_4680_; 
lean_dec_ref(v___y_4653_);
v___x_4667_ = lean_st_ref_take(v_a_4341_);
v_cache_4668_ = lean_ctor_get(v___x_4667_, 0);
v_isSharedCheck_4680_ = !lean_is_exclusive(v___x_4667_);
if (v_isSharedCheck_4680_ == 0)
{
lean_object* v_unused_4681_; 
v_unused_4681_ = lean_ctor_get(v___x_4667_, 1);
lean_dec(v_unused_4681_);
v___x_4670_ = v___x_4667_;
v_isShared_4671_ = v_isSharedCheck_4680_;
goto v_resetjp_4669_;
}
else
{
lean_inc(v_cache_4668_);
lean_dec(v___x_4667_);
v___x_4670_ = lean_box(0);
v_isShared_4671_ = v_isSharedCheck_4680_;
goto v_resetjp_4669_;
}
v_resetjp_4669_:
{
lean_object* v___x_4672_; lean_object* v___x_4674_; 
v___x_4672_ = lean_box(1);
if (v_isShared_4671_ == 0)
{
lean_ctor_set(v___x_4670_, 1, v___x_4672_);
v___x_4674_ = v___x_4670_;
goto v_reusejp_4673_;
}
else
{
lean_object* v_reuseFailAlloc_4679_; 
v_reuseFailAlloc_4679_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4679_, 0, v_cache_4668_);
lean_ctor_set(v_reuseFailAlloc_4679_, 1, v___x_4672_);
v___x_4674_ = v_reuseFailAlloc_4679_;
goto v_reusejp_4673_;
}
v_reusejp_4673_:
{
lean_object* v___x_4675_; lean_object* v___x_4677_; 
v___x_4675_ = lean_st_ref_set(v_a_4341_, v___x_4674_);
if (v_isShared_4663_ == 0)
{
lean_ctor_set_tag(v___x_4662_, 1);
lean_ctor_set(v___x_4662_, 0, v___y_4656_);
v___x_4677_ = v___x_4662_;
goto v_reusejp_4676_;
}
else
{
lean_object* v_reuseFailAlloc_4678_; 
v_reuseFailAlloc_4678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4678_, 0, v___y_4656_);
v___x_4677_ = v_reuseFailAlloc_4678_;
goto v_reusejp_4676_;
}
v_reusejp_4676_:
{
v___y_4633_ = v___y_4654_;
v___y_4634_ = v___y_4655_;
v___y_4635_ = v___y_4657_;
v_a_4636_ = v___x_4677_;
goto v___jp_4632_;
}
}
}
}
}
}
else
{
lean_object* v_a_4683_; 
lean_dec_ref(v___y_4656_);
lean_dec_ref(v___y_4653_);
v_a_4683_ = lean_ctor_get(v___x_4659_, 0);
lean_inc(v_a_4683_);
lean_dec_ref_known(v___x_4659_, 1);
v___y_4639_ = v___y_4654_;
v___y_4640_ = v___y_4655_;
v___y_4641_ = v___y_4657_;
v_a_4642_ = v_a_4683_;
goto v___jp_4638_;
}
}
v___jp_4684_:
{
if (v___y_4691_ == 0)
{
lean_object* v___x_4692_; lean_object* v___x_4693_; 
lean_dec_ref(v___y_4689_);
lean_dec_ref(v___y_4688_);
lean_dec_ref(v_config_4339_);
v___x_4692_ = lean_box(0);
lean_inc(v_a_4347_);
lean_inc_ref(v_a_4346_);
lean_inc(v_a_4345_);
lean_inc_ref(v_a_4344_);
lean_inc(v_a_4343_);
lean_inc_ref(v_a_4342_);
lean_inc(v_a_4341_);
lean_inc_ref(v_a_4340_);
v___x_4693_ = lean_apply_10(v___y_4685_, v___x_4692_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_, lean_box(0));
v___y_4645_ = v___y_4686_;
v___y_4646_ = v___y_4687_;
v___y_4647_ = v___y_4690_;
v___y_4648_ = v___x_4693_;
goto v___jp_4644_;
}
else
{
uint8_t v_symm_4694_; lean_object* v___x_4695_; 
v_symm_4694_ = lean_ctor_get_uint8(v_a_4340_, sizeof(void*)*4);
v___x_4695_ = l_Lean_Expr_mvarId_x21(v___y_4688_);
lean_dec_ref(v___y_4688_);
if (v_forward_4338_ == 0)
{
if (v_symm_4694_ == 0)
{
v___y_4652_ = v___x_4695_;
v___y_4653_ = v___y_4685_;
v___y_4654_ = v___y_4686_;
v___y_4655_ = v___y_4687_;
v___y_4656_ = v___y_4689_;
v___y_4657_ = v___y_4690_;
v___y_4658_ = v___y_4691_;
goto v___jp_4651_;
}
else
{
v___y_4652_ = v___x_4695_;
v___y_4653_ = v___y_4685_;
v___y_4654_ = v___y_4686_;
v___y_4655_ = v___y_4687_;
v___y_4656_ = v___y_4689_;
v___y_4657_ = v___y_4690_;
v___y_4658_ = v_forward_4338_;
goto v___jp_4651_;
}
}
else
{
v___y_4652_ = v___x_4695_;
v___y_4653_ = v___y_4685_;
v___y_4654_ = v___y_4686_;
v___y_4655_ = v___y_4687_;
v___y_4656_ = v___y_4689_;
v___y_4657_ = v___y_4690_;
v___y_4658_ = v_symm_4694_;
goto v___jp_4651_;
}
}
}
v___jp_4696_:
{
lean_object* v___x_4702_; lean_object* v_cache_4703_; lean_object* v___x_4705_; uint8_t v_isShared_4706_; uint8_t v_isSharedCheck_4734_; 
v___x_4702_ = lean_st_ref_get(v_a_4341_);
v_cache_4703_ = lean_ctor_get(v___x_4702_, 0);
v_isSharedCheck_4734_ = !lean_is_exclusive(v___x_4702_);
if (v_isSharedCheck_4734_ == 0)
{
lean_object* v_unused_4735_; 
v_unused_4735_ = lean_ctor_get(v___x_4702_, 1);
lean_dec(v_unused_4735_);
v___x_4705_ = v___x_4702_;
v_isShared_4706_ = v_isSharedCheck_4734_;
goto v_resetjp_4704_;
}
else
{
lean_inc(v_cache_4703_);
lean_dec(v___x_4702_);
v___x_4705_ = lean_box(0);
v_isShared_4706_ = v_isSharedCheck_4734_;
goto v_resetjp_4704_;
}
v_resetjp_4704_:
{
lean_object* v___x_4707_; lean_object* v___x_4709_; 
v___x_4707_ = lean_box(v_forward_4338_);
lean_inc_ref(v___y_4697_);
if (v_isShared_4706_ == 0)
{
lean_ctor_set(v___x_4705_, 1, v___x_4707_);
lean_ctor_set(v___x_4705_, 0, v___y_4697_);
v___x_4709_ = v___x_4705_;
goto v_reusejp_4708_;
}
else
{
lean_object* v_reuseFailAlloc_4733_; 
v_reuseFailAlloc_4733_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4733_, 0, v___y_4697_);
lean_ctor_set(v_reuseFailAlloc_4733_, 1, v___x_4707_);
v___x_4709_ = v_reuseFailAlloc_4733_;
goto v_reusejp_4708_;
}
v_reusejp_4708_:
{
lean_object* v___x_4710_; uint8_t v___x_4711_; 
lean_inc(v_a_4701_);
v___x_4710_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4710_, 0, v_a_4701_);
lean_ctor_set(v___x_4710_, 1, v___x_4709_);
v___x_4711_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(v_cache_4703_, v___x_4710_);
lean_dec_ref(v_cache_4703_);
if (v___x_4711_ == 0)
{
lean_object* v___x_4712_; 
lean_inc_ref(v___y_4697_);
v___x_4712_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(v_a_4701_, v___y_4697_, v_forward_4338_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4712_) == 0)
{
lean_object* v_a_4713_; lean_object* v_index_4714_; lean_object* v_fst_4715_; lean_object* v_snd_4716_; lean_object* v_fst_4717_; lean_object* v_snd_4718_; lean_object* v___x_4719_; lean_object* v___f_4720_; lean_object* v___x_4721_; uint8_t v___x_4722_; 
v_a_4713_ = lean_ctor_get(v___x_4712_, 0);
lean_inc(v_a_4713_);
lean_dec_ref_known(v___x_4712_, 1);
v_index_4714_ = lean_ctor_get(v_a_4340_, 2);
v_fst_4715_ = lean_ctor_get(v_a_4713_, 0);
v_snd_4716_ = lean_ctor_get(v_a_4713_, 1);
lean_inc_n(v_snd_4716_, 2);
v_fst_4717_ = lean_ctor_get(v_index_4714_, 0);
v_snd_4718_ = lean_ctor_get(v_index_4714_, 1);
v___x_4719_ = lean_box(v_forward_4338_);
lean_inc(v_fst_4715_);
lean_inc_ref(v_config_4339_);
lean_inc_ref_n(v___y_4697_, 2);
v___f_4720_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2___boxed), 17, 7);
lean_closure_set(v___f_4720_, 0, v___x_4710_);
lean_closure_set(v___f_4720_, 1, v___y_4697_);
lean_closure_set(v___f_4720_, 2, v_relName_4335_);
lean_closure_set(v___f_4720_, 3, v___x_4719_);
lean_closure_set(v___f_4720_, 4, v_snd_4716_);
lean_closure_set(v___f_4720_, 5, v_config_4339_);
lean_closure_set(v___f_4720_, 6, v_fst_4715_);
v___x_4721_ = l_Lean_Expr_toHeadIndex(v___y_4697_);
v___x_4722_ = l_Lean_instBEqHeadIndex_beq(v___x_4721_, v_fst_4717_);
lean_dec(v___x_4721_);
if (v___x_4722_ == 0)
{
lean_dec_ref(v___y_4697_);
v___y_4685_ = v___f_4720_;
v___y_4686_ = v___y_4698_;
v___y_4687_ = v___y_4699_;
v___y_4688_ = v_snd_4716_;
v___y_4689_ = v_a_4713_;
v___y_4690_ = v___y_4700_;
v___y_4691_ = v___x_4722_;
goto v___jp_4684_;
}
else
{
lean_object* v___x_4723_; uint8_t v___x_4724_; 
v___x_4723_ = l_Lean_Expr_headNumArgs(v___y_4697_);
lean_dec_ref(v___y_4697_);
v___x_4724_ = lean_nat_dec_eq(v___x_4723_, v_snd_4718_);
lean_dec(v___x_4723_);
v___y_4685_ = v___f_4720_;
v___y_4686_ = v___y_4698_;
v___y_4687_ = v___y_4699_;
v___y_4688_ = v_snd_4716_;
v___y_4689_ = v_a_4713_;
v___y_4690_ = v___y_4700_;
v___y_4691_ = v___x_4724_;
goto v___jp_4684_;
}
}
else
{
lean_object* v_a_4725_; 
lean_dec_ref_known(v___x_4710_, 2);
lean_dec_ref(v___y_4697_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4725_ = lean_ctor_get(v___x_4712_, 0);
lean_inc(v_a_4725_);
lean_dec_ref_known(v___x_4712_, 1);
v___y_4639_ = v___y_4698_;
v___y_4640_ = v___y_4699_;
v___y_4641_ = v___y_4700_;
v_a_4642_ = v_a_4725_;
goto v___jp_4638_;
}
}
else
{
lean_dec_ref_known(v___x_4710_, 2);
lean_dec(v_a_4701_);
lean_dec_ref(v___y_4697_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
if (v___x_4618_ == 0)
{
lean_object* v___x_4726_; lean_object* v___x_4727_; 
v___x_4726_ = lean_box(0);
v___x_4727_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(v___x_4726_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
v___y_4645_ = v___y_4698_;
v___y_4646_ = v___y_4699_;
v___y_4647_ = v___y_4700_;
v___y_4648_ = v___x_4727_;
goto v___jp_4644_;
}
else
{
lean_object* v___x_4728_; lean_object* v___x_4729_; 
v___x_4728_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1);
v___x_4729_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_4519_, v___x_4728_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4729_) == 0)
{
lean_object* v_a_4730_; lean_object* v___x_4731_; 
v_a_4730_ = lean_ctor_get(v___x_4729_, 0);
lean_inc(v_a_4730_);
lean_dec_ref_known(v___x_4729_, 1);
v___x_4731_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(v_a_4730_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
v___y_4645_ = v___y_4698_;
v___y_4646_ = v___y_4699_;
v___y_4647_ = v___y_4700_;
v___y_4648_ = v___x_4731_;
goto v___jp_4644_;
}
else
{
lean_object* v_a_4732_; 
v_a_4732_ = lean_ctor_get(v___x_4729_, 0);
lean_inc(v_a_4732_);
lean_dec_ref_known(v___x_4729_, 1);
v___y_4639_ = v___y_4698_;
v___y_4640_ = v___y_4699_;
v___y_4641_ = v___y_4700_;
v_a_4642_ = v_a_4732_;
goto v___jp_4638_;
}
}
}
}
}
}
v___jp_4736_:
{
lean_object* v___x_4741_; double v___x_4742_; double v___x_4743_; double v___x_4744_; double v___x_4745_; double v___x_4746_; lean_object* v___x_4747_; lean_object* v___x_4748_; lean_object* v___x_4749_; lean_object* v___x_4750_; lean_object* v___x_4751_; 
v___x_4741_ = lean_io_mono_nanos_now();
v___x_4742_ = lean_float_of_nat(v___y_4737_);
v___x_4743_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__4);
v___x_4744_ = lean_float_div(v___x_4742_, v___x_4743_);
v___x_4745_ = lean_float_of_nat(v___x_4741_);
v___x_4746_ = lean_float_div(v___x_4745_, v___x_4743_);
v___x_4747_ = lean_box_float(v___x_4744_);
v___x_4748_ = lean_box_float(v___x_4746_);
v___x_4749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4749_, 0, v___x_4747_);
lean_ctor_set(v___x_4749_, 1, v___x_4748_);
v___x_4750_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4750_, 0, v_a_4740_);
lean_ctor_set(v___x_4750_, 1, v___x_4749_);
lean_inc(v_ref_4507_);
v___x_4751_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10(v_cls_4519_, v_hasTrace_4518_, v___x_4616_, v_options_4502_, v___x_4618_, v___y_4738_, v_ref_4507_, v___y_4739_, v___x_4750_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
return v___x_4751_;
}
v___jp_4752_:
{
lean_object* v___x_4757_; 
v___x_4757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4757_, 0, v_a_4756_);
v___y_4737_ = v___y_4753_;
v___y_4738_ = v___y_4754_;
v___y_4739_ = v___y_4755_;
v_a_4740_ = v___x_4757_;
goto v___jp_4736_;
}
v___jp_4758_:
{
lean_object* v___x_4763_; 
v___x_4763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4763_, 0, v_a_4762_);
v___y_4737_ = v___y_4759_;
v___y_4738_ = v___y_4760_;
v___y_4739_ = v___y_4761_;
v_a_4740_ = v___x_4763_;
goto v___jp_4736_;
}
v___jp_4764_:
{
if (lean_obj_tag(v___y_4768_) == 0)
{
lean_object* v_a_4769_; 
v_a_4769_ = lean_ctor_get(v___y_4768_, 0);
lean_inc(v_a_4769_);
lean_dec_ref_known(v___y_4768_, 1);
v___y_4753_ = v___y_4765_;
v___y_4754_ = v___y_4766_;
v___y_4755_ = v___y_4767_;
v_a_4756_ = v_a_4769_;
goto v___jp_4752_;
}
else
{
lean_object* v_a_4770_; 
v_a_4770_ = lean_ctor_get(v___y_4768_, 0);
lean_inc(v_a_4770_);
lean_dec_ref_known(v___y_4768_, 1);
v___y_4759_ = v___y_4765_;
v___y_4760_ = v___y_4766_;
v___y_4761_ = v___y_4767_;
v_a_4762_ = v_a_4770_;
goto v___jp_4758_;
}
}
v___jp_4771_:
{
lean_object* v___x_4779_; 
lean_inc_ref(v_a_4340_);
v___x_4779_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(v_a_4340_, v___y_4776_, v___y_4778_, v_config_4339_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4779_) == 0)
{
lean_object* v_a_4780_; lean_object* v___x_4782_; uint8_t v_isShared_4783_; uint8_t v_isSharedCheck_4802_; 
v_a_4780_ = lean_ctor_get(v___x_4779_, 0);
v_isSharedCheck_4802_ = !lean_is_exclusive(v___x_4779_);
if (v_isSharedCheck_4802_ == 0)
{
v___x_4782_ = v___x_4779_;
v_isShared_4783_ = v_isSharedCheck_4802_;
goto v_resetjp_4781_;
}
else
{
lean_inc(v_a_4780_);
lean_dec(v___x_4779_);
v___x_4782_ = lean_box(0);
v_isShared_4783_ = v_isSharedCheck_4802_;
goto v_resetjp_4781_;
}
v_resetjp_4781_:
{
uint8_t v___x_4784_; 
v___x_4784_ = lean_unbox(v_a_4780_);
lean_dec(v_a_4780_);
if (v___x_4784_ == 0)
{
lean_object* v___x_4785_; lean_object* v___x_4786_; 
lean_del_object(v___x_4782_);
lean_dec_ref(v___y_4777_);
v___x_4785_ = lean_box(0);
lean_inc(v_a_4347_);
lean_inc_ref(v_a_4346_);
lean_inc(v_a_4345_);
lean_inc_ref(v_a_4344_);
lean_inc(v_a_4343_);
lean_inc_ref(v_a_4342_);
lean_inc(v_a_4341_);
lean_inc_ref(v_a_4340_);
v___x_4786_ = lean_apply_10(v___y_4774_, v___x_4785_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_, lean_box(0));
v___y_4765_ = v___y_4772_;
v___y_4766_ = v___y_4773_;
v___y_4767_ = v___y_4775_;
v___y_4768_ = v___x_4786_;
goto v___jp_4764_;
}
else
{
lean_object* v___x_4787_; lean_object* v_cache_4788_; lean_object* v___x_4790_; uint8_t v_isShared_4791_; uint8_t v_isSharedCheck_4800_; 
lean_dec_ref(v___y_4774_);
v___x_4787_ = lean_st_ref_take(v_a_4341_);
v_cache_4788_ = lean_ctor_get(v___x_4787_, 0);
v_isSharedCheck_4800_ = !lean_is_exclusive(v___x_4787_);
if (v_isSharedCheck_4800_ == 0)
{
lean_object* v_unused_4801_; 
v_unused_4801_ = lean_ctor_get(v___x_4787_, 1);
lean_dec(v_unused_4801_);
v___x_4790_ = v___x_4787_;
v_isShared_4791_ = v_isSharedCheck_4800_;
goto v_resetjp_4789_;
}
else
{
lean_inc(v_cache_4788_);
lean_dec(v___x_4787_);
v___x_4790_ = lean_box(0);
v_isShared_4791_ = v_isSharedCheck_4800_;
goto v_resetjp_4789_;
}
v_resetjp_4789_:
{
lean_object* v___x_4792_; lean_object* v___x_4794_; 
v___x_4792_ = lean_box(1);
if (v_isShared_4791_ == 0)
{
lean_ctor_set(v___x_4790_, 1, v___x_4792_);
v___x_4794_ = v___x_4790_;
goto v_reusejp_4793_;
}
else
{
lean_object* v_reuseFailAlloc_4799_; 
v_reuseFailAlloc_4799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4799_, 0, v_cache_4788_);
lean_ctor_set(v_reuseFailAlloc_4799_, 1, v___x_4792_);
v___x_4794_ = v_reuseFailAlloc_4799_;
goto v_reusejp_4793_;
}
v_reusejp_4793_:
{
lean_object* v___x_4795_; lean_object* v___x_4797_; 
v___x_4795_ = lean_st_ref_set(v_a_4341_, v___x_4794_);
if (v_isShared_4783_ == 0)
{
lean_ctor_set_tag(v___x_4782_, 1);
lean_ctor_set(v___x_4782_, 0, v___y_4777_);
v___x_4797_ = v___x_4782_;
goto v_reusejp_4796_;
}
else
{
lean_object* v_reuseFailAlloc_4798_; 
v_reuseFailAlloc_4798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4798_, 0, v___y_4777_);
v___x_4797_ = v_reuseFailAlloc_4798_;
goto v_reusejp_4796_;
}
v_reusejp_4796_:
{
v___y_4753_ = v___y_4772_;
v___y_4754_ = v___y_4773_;
v___y_4755_ = v___y_4775_;
v_a_4756_ = v___x_4797_;
goto v___jp_4752_;
}
}
}
}
}
}
else
{
lean_object* v_a_4803_; 
lean_dec_ref(v___y_4777_);
lean_dec_ref(v___y_4774_);
v_a_4803_ = lean_ctor_get(v___x_4779_, 0);
lean_inc(v_a_4803_);
lean_dec_ref_known(v___x_4779_, 1);
v___y_4759_ = v___y_4772_;
v___y_4760_ = v___y_4773_;
v___y_4761_ = v___y_4775_;
v_a_4762_ = v_a_4803_;
goto v___jp_4758_;
}
}
v___jp_4804_:
{
if (v___y_4811_ == 0)
{
lean_object* v___x_4812_; lean_object* v___x_4813_; 
lean_dec_ref(v___y_4810_);
lean_dec_ref(v___y_4809_);
lean_dec_ref(v_config_4339_);
v___x_4812_ = lean_box(0);
lean_inc(v_a_4347_);
lean_inc_ref(v_a_4346_);
lean_inc(v_a_4345_);
lean_inc_ref(v_a_4344_);
lean_inc(v_a_4343_);
lean_inc_ref(v_a_4342_);
lean_inc(v_a_4341_);
lean_inc_ref(v_a_4340_);
v___x_4813_ = lean_apply_10(v___y_4807_, v___x_4812_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_, lean_box(0));
v___y_4765_ = v___y_4805_;
v___y_4766_ = v___y_4806_;
v___y_4767_ = v___y_4808_;
v___y_4768_ = v___x_4813_;
goto v___jp_4764_;
}
else
{
uint8_t v_symm_4814_; lean_object* v___x_4815_; 
v_symm_4814_ = lean_ctor_get_uint8(v_a_4340_, sizeof(void*)*4);
v___x_4815_ = l_Lean_Expr_mvarId_x21(v___y_4809_);
lean_dec_ref(v___y_4809_);
if (v_forward_4338_ == 0)
{
if (v_symm_4814_ == 0)
{
v___y_4772_ = v___y_4805_;
v___y_4773_ = v___y_4806_;
v___y_4774_ = v___y_4807_;
v___y_4775_ = v___y_4808_;
v___y_4776_ = v___x_4815_;
v___y_4777_ = v___y_4810_;
v___y_4778_ = v___y_4811_;
goto v___jp_4771_;
}
else
{
v___y_4772_ = v___y_4805_;
v___y_4773_ = v___y_4806_;
v___y_4774_ = v___y_4807_;
v___y_4775_ = v___y_4808_;
v___y_4776_ = v___x_4815_;
v___y_4777_ = v___y_4810_;
v___y_4778_ = v_forward_4338_;
goto v___jp_4771_;
}
}
else
{
v___y_4772_ = v___y_4805_;
v___y_4773_ = v___y_4806_;
v___y_4774_ = v___y_4807_;
v___y_4775_ = v___y_4808_;
v___y_4776_ = v___x_4815_;
v___y_4777_ = v___y_4810_;
v___y_4778_ = v_symm_4814_;
goto v___jp_4771_;
}
}
}
v___jp_4816_:
{
lean_object* v___x_4822_; lean_object* v_cache_4823_; lean_object* v___x_4825_; uint8_t v_isShared_4826_; uint8_t v_isSharedCheck_4854_; 
v___x_4822_ = lean_st_ref_get(v_a_4341_);
v_cache_4823_ = lean_ctor_get(v___x_4822_, 0);
v_isSharedCheck_4854_ = !lean_is_exclusive(v___x_4822_);
if (v_isSharedCheck_4854_ == 0)
{
lean_object* v_unused_4855_; 
v_unused_4855_ = lean_ctor_get(v___x_4822_, 1);
lean_dec(v_unused_4855_);
v___x_4825_ = v___x_4822_;
v_isShared_4826_ = v_isSharedCheck_4854_;
goto v_resetjp_4824_;
}
else
{
lean_inc(v_cache_4823_);
lean_dec(v___x_4822_);
v___x_4825_ = lean_box(0);
v_isShared_4826_ = v_isSharedCheck_4854_;
goto v_resetjp_4824_;
}
v_resetjp_4824_:
{
lean_object* v___x_4827_; lean_object* v___x_4829_; 
v___x_4827_ = lean_box(v_forward_4338_);
lean_inc_ref(v___y_4817_);
if (v_isShared_4826_ == 0)
{
lean_ctor_set(v___x_4825_, 1, v___x_4827_);
lean_ctor_set(v___x_4825_, 0, v___y_4817_);
v___x_4829_ = v___x_4825_;
goto v_reusejp_4828_;
}
else
{
lean_object* v_reuseFailAlloc_4853_; 
v_reuseFailAlloc_4853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4853_, 0, v___y_4817_);
lean_ctor_set(v_reuseFailAlloc_4853_, 1, v___x_4827_);
v___x_4829_ = v_reuseFailAlloc_4853_;
goto v_reusejp_4828_;
}
v_reusejp_4828_:
{
lean_object* v___x_4830_; uint8_t v___x_4831_; 
lean_inc(v_a_4821_);
v___x_4830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4830_, 0, v_a_4821_);
lean_ctor_set(v___x_4830_, 1, v___x_4829_);
v___x_4831_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(v_cache_4823_, v___x_4830_);
lean_dec_ref(v_cache_4823_);
if (v___x_4831_ == 0)
{
lean_object* v___x_4832_; 
lean_inc_ref(v___y_4817_);
v___x_4832_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(v_a_4821_, v___y_4817_, v_forward_4338_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4832_) == 0)
{
lean_object* v_a_4833_; lean_object* v_index_4834_; lean_object* v_fst_4835_; lean_object* v_snd_4836_; lean_object* v_fst_4837_; lean_object* v_snd_4838_; lean_object* v___x_4839_; lean_object* v___f_4840_; lean_object* v___x_4841_; uint8_t v___x_4842_; 
v_a_4833_ = lean_ctor_get(v___x_4832_, 0);
lean_inc(v_a_4833_);
lean_dec_ref_known(v___x_4832_, 1);
v_index_4834_ = lean_ctor_get(v_a_4340_, 2);
v_fst_4835_ = lean_ctor_get(v_a_4833_, 0);
v_snd_4836_ = lean_ctor_get(v_a_4833_, 1);
lean_inc_n(v_snd_4836_, 2);
v_fst_4837_ = lean_ctor_get(v_index_4834_, 0);
v_snd_4838_ = lean_ctor_get(v_index_4834_, 1);
v___x_4839_ = lean_box(v_forward_4338_);
lean_inc(v_fst_4835_);
lean_inc_ref(v_config_4339_);
lean_inc_ref_n(v___y_4817_, 2);
v___f_4840_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__2___boxed), 17, 7);
lean_closure_set(v___f_4840_, 0, v___x_4830_);
lean_closure_set(v___f_4840_, 1, v___y_4817_);
lean_closure_set(v___f_4840_, 2, v_relName_4335_);
lean_closure_set(v___f_4840_, 3, v___x_4839_);
lean_closure_set(v___f_4840_, 4, v_snd_4836_);
lean_closure_set(v___f_4840_, 5, v_config_4339_);
lean_closure_set(v___f_4840_, 6, v_fst_4835_);
v___x_4841_ = l_Lean_Expr_toHeadIndex(v___y_4817_);
v___x_4842_ = l_Lean_instBEqHeadIndex_beq(v___x_4841_, v_fst_4837_);
lean_dec(v___x_4841_);
if (v___x_4842_ == 0)
{
lean_dec_ref(v___y_4817_);
v___y_4805_ = v___y_4818_;
v___y_4806_ = v___y_4819_;
v___y_4807_ = v___f_4840_;
v___y_4808_ = v___y_4820_;
v___y_4809_ = v_snd_4836_;
v___y_4810_ = v_a_4833_;
v___y_4811_ = v___x_4842_;
goto v___jp_4804_;
}
else
{
lean_object* v___x_4843_; uint8_t v___x_4844_; 
v___x_4843_ = l_Lean_Expr_headNumArgs(v___y_4817_);
lean_dec_ref(v___y_4817_);
v___x_4844_ = lean_nat_dec_eq(v___x_4843_, v_snd_4838_);
lean_dec(v___x_4843_);
v___y_4805_ = v___y_4818_;
v___y_4806_ = v___y_4819_;
v___y_4807_ = v___f_4840_;
v___y_4808_ = v___y_4820_;
v___y_4809_ = v_snd_4836_;
v___y_4810_ = v_a_4833_;
v___y_4811_ = v___x_4844_;
goto v___jp_4804_;
}
}
else
{
lean_object* v_a_4845_; 
lean_dec_ref_known(v___x_4830_, 2);
lean_dec_ref(v___y_4817_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4845_ = lean_ctor_get(v___x_4832_, 0);
lean_inc(v_a_4845_);
lean_dec_ref_known(v___x_4832_, 1);
v___y_4759_ = v___y_4818_;
v___y_4760_ = v___y_4819_;
v___y_4761_ = v___y_4820_;
v_a_4762_ = v_a_4845_;
goto v___jp_4758_;
}
}
else
{
lean_dec_ref_known(v___x_4830_, 2);
lean_dec(v_a_4821_);
lean_dec_ref(v___y_4817_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
if (v___x_4618_ == 0)
{
lean_object* v___x_4846_; lean_object* v___x_4847_; 
v___x_4846_ = lean_box(0);
v___x_4847_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(v___x_4846_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
v___y_4765_ = v___y_4818_;
v___y_4766_ = v___y_4819_;
v___y_4767_ = v___y_4820_;
v___y_4768_ = v___x_4847_;
goto v___jp_4764_;
}
else
{
lean_object* v___x_4848_; lean_object* v___x_4849_; 
v___x_4848_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1);
v___x_4849_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_4519_, v___x_4848_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
if (lean_obj_tag(v___x_4849_) == 0)
{
lean_object* v_a_4850_; lean_object* v___x_4851_; 
v_a_4850_ = lean_ctor_get(v___x_4849_, 0);
lean_inc(v_a_4850_);
lean_dec_ref_known(v___x_4849_, 1);
v___x_4851_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___lam__0(v_a_4850_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_, v_a_4347_);
v___y_4765_ = v___y_4818_;
v___y_4766_ = v___y_4819_;
v___y_4767_ = v___y_4820_;
v___y_4768_ = v___x_4851_;
goto v___jp_4764_;
}
else
{
lean_object* v_a_4852_; 
v_a_4852_ = lean_ctor_get(v___x_4849_, 0);
lean_inc(v_a_4852_);
lean_dec_ref_known(v___x_4849_, 1);
v___y_4759_ = v___y_4818_;
v___y_4760_ = v___y_4819_;
v___y_4761_ = v___y_4820_;
v_a_4762_ = v_a_4852_;
goto v___jp_4758_;
}
}
}
}
}
}
v___jp_4856_:
{
lean_object* v___x_4861_; lean_object* v___x_4862_; lean_object* v___x_4863_; lean_object* v___x_4864_; 
v___x_4861_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4861_, 0, v___y_4859_);
lean_ctor_set(v___x_4861_, 1, v___y_4860_);
v___x_4862_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___lam__2___closed__3);
v___x_4863_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4863_, 0, v___x_4861_);
lean_ctor_set(v___x_4863_, 1, v___x_4862_);
v___x_4864_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__4_spec__5_spec__8(v___x_4863_, v_a_4344_, v_a_4345_, v___y_4858_, v_a_4347_);
lean_dec_ref(v___y_4858_);
if (lean_obj_tag(v___x_4864_) == 0)
{
lean_object* v_a_4865_; lean_object* v___x_4866_; uint8_t v___x_4867_; 
v_a_4865_ = lean_ctor_get(v___x_4864_, 0);
lean_inc(v_a_4865_);
lean_dec_ref_known(v___x_4864_, 1);
v___x_4866_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4867_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_4502_, v___x_4866_);
if (v___x_4867_ == 0)
{
lean_object* v___x_4868_; lean_object* v___x_4869_; 
v___x_4868_ = lean_io_mono_nanos_now();
v___x_4869_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_4337_, v_a_4345_);
if (lean_obj_tag(v___x_4869_) == 0)
{
if (lean_obj_tag(v_rel_x3f_4336_) == 0)
{
lean_object* v_a_4870_; 
v_a_4870_ = lean_ctor_get(v___x_4869_, 0);
lean_inc(v_a_4870_);
lean_dec_ref_known(v___x_4869_, 1);
v___y_4817_ = v_a_4870_;
v___y_4818_ = v___x_4868_;
v___y_4819_ = v___y_4857_;
v___y_4820_ = v_a_4865_;
v_a_4821_ = v_rel_x3f_4336_;
goto v___jp_4816_;
}
else
{
lean_object* v_a_4871_; lean_object* v_val_4872_; lean_object* v___x_4874_; uint8_t v_isShared_4875_; uint8_t v_isSharedCheck_4882_; 
v_a_4871_ = lean_ctor_get(v___x_4869_, 0);
lean_inc(v_a_4871_);
lean_dec_ref_known(v___x_4869_, 1);
v_val_4872_ = lean_ctor_get(v_rel_x3f_4336_, 0);
v_isSharedCheck_4882_ = !lean_is_exclusive(v_rel_x3f_4336_);
if (v_isSharedCheck_4882_ == 0)
{
v___x_4874_ = v_rel_x3f_4336_;
v_isShared_4875_ = v_isSharedCheck_4882_;
goto v_resetjp_4873_;
}
else
{
lean_inc(v_val_4872_);
lean_dec(v_rel_x3f_4336_);
v___x_4874_ = lean_box(0);
v_isShared_4875_ = v_isSharedCheck_4882_;
goto v_resetjp_4873_;
}
v_resetjp_4873_:
{
lean_object* v___x_4876_; 
v___x_4876_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_val_4872_, v_a_4345_);
if (lean_obj_tag(v___x_4876_) == 0)
{
lean_object* v_a_4877_; lean_object* v___x_4879_; 
v_a_4877_ = lean_ctor_get(v___x_4876_, 0);
lean_inc(v_a_4877_);
lean_dec_ref_known(v___x_4876_, 1);
if (v_isShared_4875_ == 0)
{
lean_ctor_set(v___x_4874_, 0, v_a_4877_);
v___x_4879_ = v___x_4874_;
goto v_reusejp_4878_;
}
else
{
lean_object* v_reuseFailAlloc_4880_; 
v_reuseFailAlloc_4880_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4880_, 0, v_a_4877_);
v___x_4879_ = v_reuseFailAlloc_4880_;
goto v_reusejp_4878_;
}
v_reusejp_4878_:
{
v___y_4817_ = v_a_4871_;
v___y_4818_ = v___x_4868_;
v___y_4819_ = v___y_4857_;
v___y_4820_ = v_a_4865_;
v_a_4821_ = v___x_4879_;
goto v___jp_4816_;
}
}
else
{
lean_object* v_a_4881_; 
lean_del_object(v___x_4874_);
lean_dec(v_a_4871_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4881_ = lean_ctor_get(v___x_4876_, 0);
lean_inc(v_a_4881_);
lean_dec_ref_known(v___x_4876_, 1);
v___y_4759_ = v___x_4868_;
v___y_4760_ = v___y_4857_;
v___y_4761_ = v_a_4865_;
v_a_4762_ = v_a_4881_;
goto v___jp_4758_;
}
}
}
}
else
{
lean_object* v_a_4883_; 
lean_dec_ref(v_config_4339_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4883_ = lean_ctor_get(v___x_4869_, 0);
lean_inc(v_a_4883_);
lean_dec_ref_known(v___x_4869_, 1);
v___y_4759_ = v___x_4868_;
v___y_4760_ = v___y_4857_;
v___y_4761_ = v_a_4865_;
v_a_4762_ = v_a_4883_;
goto v___jp_4758_;
}
}
else
{
lean_object* v___x_4884_; lean_object* v___x_4885_; 
v___x_4884_ = lean_io_get_num_heartbeats();
v___x_4885_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_4337_, v_a_4345_);
if (lean_obj_tag(v___x_4885_) == 0)
{
if (lean_obj_tag(v_rel_x3f_4336_) == 0)
{
lean_object* v_a_4886_; 
v_a_4886_ = lean_ctor_get(v___x_4885_, 0);
lean_inc(v_a_4886_);
lean_dec_ref_known(v___x_4885_, 1);
v___y_4697_ = v_a_4886_;
v___y_4698_ = v___y_4857_;
v___y_4699_ = v_a_4865_;
v___y_4700_ = v___x_4884_;
v_a_4701_ = v_rel_x3f_4336_;
goto v___jp_4696_;
}
else
{
lean_object* v_a_4887_; lean_object* v_val_4888_; lean_object* v___x_4890_; uint8_t v_isShared_4891_; uint8_t v_isSharedCheck_4898_; 
v_a_4887_ = lean_ctor_get(v___x_4885_, 0);
lean_inc(v_a_4887_);
lean_dec_ref_known(v___x_4885_, 1);
v_val_4888_ = lean_ctor_get(v_rel_x3f_4336_, 0);
v_isSharedCheck_4898_ = !lean_is_exclusive(v_rel_x3f_4336_);
if (v_isSharedCheck_4898_ == 0)
{
v___x_4890_ = v_rel_x3f_4336_;
v_isShared_4891_ = v_isSharedCheck_4898_;
goto v_resetjp_4889_;
}
else
{
lean_inc(v_val_4888_);
lean_dec(v_rel_x3f_4336_);
v___x_4890_ = lean_box(0);
v_isShared_4891_ = v_isSharedCheck_4898_;
goto v_resetjp_4889_;
}
v_resetjp_4889_:
{
lean_object* v___x_4892_; 
v___x_4892_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_val_4888_, v_a_4345_);
if (lean_obj_tag(v___x_4892_) == 0)
{
lean_object* v_a_4893_; lean_object* v___x_4895_; 
v_a_4893_ = lean_ctor_get(v___x_4892_, 0);
lean_inc(v_a_4893_);
lean_dec_ref_known(v___x_4892_, 1);
if (v_isShared_4891_ == 0)
{
lean_ctor_set(v___x_4890_, 0, v_a_4893_);
v___x_4895_ = v___x_4890_;
goto v_reusejp_4894_;
}
else
{
lean_object* v_reuseFailAlloc_4896_; 
v_reuseFailAlloc_4896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4896_, 0, v_a_4893_);
v___x_4895_ = v_reuseFailAlloc_4896_;
goto v_reusejp_4894_;
}
v_reusejp_4894_:
{
v___y_4697_ = v_a_4887_;
v___y_4698_ = v___y_4857_;
v___y_4699_ = v_a_4865_;
v___y_4700_ = v___x_4884_;
v_a_4701_ = v___x_4895_;
goto v___jp_4696_;
}
}
else
{
lean_object* v_a_4897_; 
lean_del_object(v___x_4890_);
lean_dec(v_a_4887_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4897_ = lean_ctor_get(v___x_4892_, 0);
lean_inc(v_a_4897_);
lean_dec_ref_known(v___x_4892_, 1);
v___y_4639_ = v___y_4857_;
v___y_4640_ = v_a_4865_;
v___y_4641_ = v___x_4884_;
v_a_4642_ = v_a_4897_;
goto v___jp_4638_;
}
}
}
}
else
{
lean_object* v_a_4899_; 
lean_dec_ref(v_config_4339_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4899_ = lean_ctor_get(v___x_4885_, 0);
lean_inc(v_a_4899_);
lean_dec_ref_known(v___x_4885_, 1);
v___y_4639_ = v___y_4857_;
v___y_4640_ = v_a_4865_;
v___y_4641_ = v___x_4884_;
v_a_4642_ = v_a_4899_;
goto v___jp_4638_;
}
}
}
else
{
lean_object* v_a_4900_; lean_object* v___x_4902_; uint8_t v_isShared_4903_; uint8_t v_isSharedCheck_4907_; 
lean_dec_ref(v___y_4857_);
lean_dec_ref(v_config_4339_);
lean_dec_ref(v_e_4337_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4900_ = lean_ctor_get(v___x_4864_, 0);
v_isSharedCheck_4907_ = !lean_is_exclusive(v___x_4864_);
if (v_isSharedCheck_4907_ == 0)
{
v___x_4902_ = v___x_4864_;
v_isShared_4903_ = v_isSharedCheck_4907_;
goto v_resetjp_4901_;
}
else
{
lean_inc(v_a_4900_);
lean_dec(v___x_4864_);
v___x_4902_ = lean_box(0);
v_isShared_4903_ = v_isSharedCheck_4907_;
goto v_resetjp_4901_;
}
v_resetjp_4901_:
{
lean_object* v___x_4905_; 
if (v_isShared_4903_ == 0)
{
v___x_4905_ = v___x_4902_;
goto v_reusejp_4904_;
}
else
{
lean_object* v_reuseFailAlloc_4906_; 
v_reuseFailAlloc_4906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4906_, 0, v_a_4900_);
v___x_4905_ = v_reuseFailAlloc_4906_;
goto v_reusejp_4904_;
}
v_reusejp_4904_:
{
return v___x_4905_;
}
}
}
}
v___jp_4908_:
{
lean_object* v___x_4913_; lean_object* v___x_4914_; lean_object* v___x_4915_; lean_object* v___x_4916_; 
lean_inc_ref(v___y_4912_);
v___x_4913_ = l_Lean_stringToMessageData(v___y_4912_);
v___x_4914_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4914_, 0, v___y_4910_);
lean_ctor_set(v___x_4914_, 1, v___x_4913_);
v___x_4915_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__3);
v___x_4916_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4916_, 0, v___x_4914_);
lean_ctor_set(v___x_4916_, 1, v___x_4915_);
if (lean_obj_tag(v_rel_x3f_4336_) == 0)
{
lean_object* v___x_4917_; 
v___x_4917_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__5);
v___y_4857_ = v___y_4909_;
v___y_4858_ = v___y_4911_;
v___y_4859_ = v___x_4916_;
v___y_4860_ = v___x_4917_;
goto v___jp_4856_;
}
else
{
lean_object* v_val_4918_; lean_object* v___x_4919_; 
v_val_4918_ = lean_ctor_get(v_rel_x3f_4336_, 0);
lean_inc(v_val_4918_);
v___x_4919_ = l_Lean_MessageData_ofExpr(v_val_4918_);
v___y_4857_ = v___y_4909_;
v___y_4858_ = v___y_4911_;
v___y_4859_ = v___x_4916_;
v___y_4860_ = v___x_4919_;
goto v___jp_4856_;
}
}
v___jp_4920_:
{
lean_object* v___x_4921_; 
v___x_4921_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(v_a_4347_);
if (lean_obj_tag(v___x_4921_) == 0)
{
lean_object* v_a_4922_; lean_object* v_ref_4923_; lean_object* v___x_4924_; lean_object* v___x_4925_; lean_object* v___x_4926_; lean_object* v___x_4927_; lean_object* v___x_4928_; lean_object* v___x_4929_; 
v_a_4922_ = lean_ctor_get(v___x_4921_, 0);
lean_inc(v_a_4922_);
lean_dec_ref_known(v___x_4921_, 1);
v_ref_4923_ = l_Lean_replaceRef(v_ref_4507_, v_ref_4507_);
lean_inc_ref(v_inheritedTraceOptions_4517_);
lean_inc(v_cancelTk_x3f_4515_);
lean_inc(v_currMacroScope_4513_);
lean_inc(v_quotContext_4512_);
lean_inc(v_maxHeartbeats_4511_);
lean_inc(v_initHeartbeats_4510_);
lean_inc(v_openDecls_4509_);
lean_inc(v_currNamespace_4508_);
lean_inc(v_maxRecDepth_4506_);
lean_inc(v_currRecDepth_4505_);
lean_inc_ref(v_options_4502_);
lean_inc_ref(v_fileMap_4504_);
lean_inc_ref(v_fileName_4503_);
v___x_4924_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4924_, 0, v_fileName_4503_);
lean_ctor_set(v___x_4924_, 1, v_fileMap_4504_);
lean_ctor_set(v___x_4924_, 2, v_options_4502_);
lean_ctor_set(v___x_4924_, 3, v_currRecDepth_4505_);
lean_ctor_set(v___x_4924_, 4, v_maxRecDepth_4506_);
lean_ctor_set(v___x_4924_, 5, v_ref_4923_);
lean_ctor_set(v___x_4924_, 6, v_currNamespace_4508_);
lean_ctor_set(v___x_4924_, 7, v_openDecls_4509_);
lean_ctor_set(v___x_4924_, 8, v_initHeartbeats_4510_);
lean_ctor_set(v___x_4924_, 9, v_maxHeartbeats_4511_);
lean_ctor_set(v___x_4924_, 10, v_quotContext_4512_);
lean_ctor_set(v___x_4924_, 11, v_currMacroScope_4513_);
lean_ctor_set(v___x_4924_, 12, v_cancelTk_x3f_4515_);
lean_ctor_set(v___x_4924_, 13, v_inheritedTraceOptions_4517_);
lean_ctor_set_uint8(v___x_4924_, sizeof(void*)*14, v_diag_4514_);
lean_ctor_set_uint8(v___x_4924_, sizeof(void*)*14 + 1, v_suppressElabErrors_4516_);
v___x_4925_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__7);
lean_inc_ref(v_e_4337_);
v___x_4926_ = l_Lean_MessageData_ofExpr(v_e_4337_);
v___x_4927_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4927_, 0, v___x_4925_);
lean_ctor_set(v___x_4927_, 1, v___x_4926_);
v___x_4928_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__9);
v___x_4929_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4929_, 0, v___x_4927_);
lean_ctor_set(v___x_4929_, 1, v___x_4928_);
if (v_forward_4338_ == 0)
{
lean_object* v___x_4930_; 
v___x_4930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__10));
v___y_4909_ = v_a_4922_;
v___y_4910_ = v___x_4929_;
v___y_4911_ = v___x_4924_;
v___y_4912_ = v___x_4930_;
goto v___jp_4908_;
}
else
{
lean_object* v___x_4931_; 
v___x_4931_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__11));
v___y_4909_ = v_a_4922_;
v___y_4910_ = v___x_4929_;
v___y_4911_ = v___x_4924_;
v___y_4912_ = v___x_4931_;
goto v___jp_4908_;
}
}
else
{
lean_object* v_a_4932_; lean_object* v___x_4934_; uint8_t v_isShared_4935_; uint8_t v_isSharedCheck_4939_; 
lean_dec_ref(v_config_4339_);
lean_dec_ref(v_e_4337_);
lean_dec(v_rel_x3f_4336_);
lean_dec(v_relName_4335_);
v_a_4932_ = lean_ctor_get(v___x_4921_, 0);
v_isSharedCheck_4939_ = !lean_is_exclusive(v___x_4921_);
if (v_isSharedCheck_4939_ == 0)
{
v___x_4934_ = v___x_4921_;
v_isShared_4935_ = v_isSharedCheck_4939_;
goto v_resetjp_4933_;
}
else
{
lean_inc(v_a_4932_);
lean_dec(v___x_4921_);
v___x_4934_ = lean_box(0);
v_isShared_4935_ = v_isSharedCheck_4939_;
goto v_resetjp_4933_;
}
v_resetjp_4933_:
{
lean_object* v___x_4937_; 
if (v_isShared_4935_ == 0)
{
v___x_4937_ = v___x_4934_;
goto v_reusejp_4936_;
}
else
{
lean_object* v_reuseFailAlloc_4938_; 
v_reuseFailAlloc_4938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4938_, 0, v_a_4932_);
v___x_4937_ = v_reuseFailAlloc_4938_;
goto v_reusejp_4936_;
}
v_reusejp_4936_:
{
return v___x_4937_;
}
}
}
}
}
v___jp_4349_:
{
lean_object* v___x_4352_; lean_object* v_cache_4353_; lean_object* v_progress_4354_; lean_object* v___x_4356_; uint8_t v_isShared_4357_; uint8_t v_isSharedCheck_4366_; 
v___x_4352_ = lean_st_ref_take(v___y_4351_);
v_cache_4353_ = lean_ctor_get(v___x_4352_, 0);
v_progress_4354_ = lean_ctor_get(v___x_4352_, 1);
v_isSharedCheck_4366_ = !lean_is_exclusive(v___x_4352_);
if (v_isSharedCheck_4366_ == 0)
{
v___x_4356_ = v___x_4352_;
v_isShared_4357_ = v_isSharedCheck_4366_;
goto v_resetjp_4355_;
}
else
{
lean_inc(v_progress_4354_);
lean_inc(v_cache_4353_);
lean_dec(v___x_4352_);
v___x_4356_ = lean_box(0);
v_isShared_4357_ = v_isSharedCheck_4366_;
goto v_resetjp_4355_;
}
v_resetjp_4355_:
{
lean_object* v___x_4358_; lean_object* v___x_4359_; lean_object* v___x_4361_; 
v___x_4358_ = lean_box(0);
v___x_4359_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4___redArg(v_cache_4353_, v___y_4350_, v___x_4358_);
if (v_isShared_4357_ == 0)
{
lean_ctor_set(v___x_4356_, 0, v___x_4359_);
v___x_4361_ = v___x_4356_;
goto v_reusejp_4360_;
}
else
{
lean_object* v_reuseFailAlloc_4365_; 
v_reuseFailAlloc_4365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4365_, 0, v___x_4359_);
lean_ctor_set(v_reuseFailAlloc_4365_, 1, v_progress_4354_);
v___x_4361_ = v_reuseFailAlloc_4365_;
goto v_reusejp_4360_;
}
v_reusejp_4360_:
{
lean_object* v___x_4362_; lean_object* v___x_4363_; lean_object* v___x_4364_; 
v___x_4362_ = lean_st_ref_set(v___y_4351_, v___x_4361_);
v___x_4363_ = lean_box(0);
v___x_4364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4364_, 0, v___x_4363_);
return v___x_4364_;
}
}
}
v___jp_4367_:
{
lean_object* v___x_4381_; lean_object* v_mctx_4382_; lean_object* v___x_4383_; lean_object* v___x_4384_; 
v___x_4381_ = lean_st_ref_get(v___y_4378_);
v_mctx_4382_ = lean_ctor_get(v___x_4381_, 0);
lean_inc_ref(v_mctx_4382_);
lean_dec(v___x_4381_);
v___x_4383_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___closed__0));
v___x_4384_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(v___y_4368_, v_forward_4338_, v_config_4339_, v_mctx_4382_, v___y_4371_, v___y_4369_, v_lemmas_4372_, v___x_4383_, v___y_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_, v___y_4379_, v___y_4380_);
lean_dec(v_lemmas_4372_);
if (lean_obj_tag(v___x_4384_) == 0)
{
lean_object* v_a_4385_; lean_object* v___x_4387_; uint8_t v_isShared_4388_; uint8_t v_isSharedCheck_4394_; 
v_a_4385_ = lean_ctor_get(v___x_4384_, 0);
v_isSharedCheck_4394_ = !lean_is_exclusive(v___x_4384_);
if (v_isSharedCheck_4394_ == 0)
{
v___x_4387_ = v___x_4384_;
v_isShared_4388_ = v_isSharedCheck_4394_;
goto v_resetjp_4386_;
}
else
{
lean_inc(v_a_4385_);
lean_dec(v___x_4384_);
v___x_4387_ = lean_box(0);
v_isShared_4388_ = v_isSharedCheck_4394_;
goto v_resetjp_4386_;
}
v_resetjp_4386_:
{
lean_object* v_fst_4389_; 
v_fst_4389_ = lean_ctor_get(v_a_4385_, 0);
lean_inc(v_fst_4389_);
lean_dec(v_a_4385_);
if (lean_obj_tag(v_fst_4389_) == 0)
{
lean_del_object(v___x_4387_);
v___y_4350_ = v___y_4370_;
v___y_4351_ = v___y_4374_;
goto v___jp_4349_;
}
else
{
lean_object* v_val_4390_; lean_object* v___x_4392_; 
lean_dec_ref(v___y_4370_);
v_val_4390_ = lean_ctor_get(v_fst_4389_, 0);
lean_inc(v_val_4390_);
lean_dec_ref_known(v_fst_4389_, 1);
if (v_isShared_4388_ == 0)
{
lean_ctor_set(v___x_4387_, 0, v_val_4390_);
v___x_4392_ = v___x_4387_;
goto v_reusejp_4391_;
}
else
{
lean_object* v_reuseFailAlloc_4393_; 
v_reuseFailAlloc_4393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4393_, 0, v_val_4390_);
v___x_4392_ = v_reuseFailAlloc_4393_;
goto v_reusejp_4391_;
}
v_reusejp_4391_:
{
return v___x_4392_;
}
}
}
}
else
{
lean_object* v_a_4395_; lean_object* v___x_4397_; uint8_t v_isShared_4398_; uint8_t v_isSharedCheck_4402_; 
lean_dec_ref(v___y_4370_);
v_a_4395_ = lean_ctor_get(v___x_4384_, 0);
v_isSharedCheck_4402_ = !lean_is_exclusive(v___x_4384_);
if (v_isSharedCheck_4402_ == 0)
{
v___x_4397_ = v___x_4384_;
v_isShared_4398_ = v_isSharedCheck_4402_;
goto v_resetjp_4396_;
}
else
{
lean_inc(v_a_4395_);
lean_dec(v___x_4384_);
v___x_4397_ = lean_box(0);
v_isShared_4398_ = v_isSharedCheck_4402_;
goto v_resetjp_4396_;
}
v_resetjp_4396_:
{
lean_object* v___x_4400_; 
if (v_isShared_4398_ == 0)
{
v___x_4400_ = v___x_4397_;
goto v_reusejp_4399_;
}
else
{
lean_object* v_reuseFailAlloc_4401_; 
v_reuseFailAlloc_4401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4401_, 0, v_a_4395_);
v___x_4400_ = v_reuseFailAlloc_4401_;
goto v_reusejp_4399_;
}
v_reusejp_4399_:
{
return v___x_4400_;
}
}
}
}
v___jp_4403_:
{
lean_object* v___x_4416_; 
lean_inc_ref(v___y_4405_);
v___x_4416_ = lp_mathlib_Mathlib_Tactic_GCongr_getCongrAppFnArgs(v___y_4405_);
if (lean_obj_tag(v___x_4416_) == 1)
{
lean_object* v_val_4417_; lean_object* v_fst_4418_; lean_object* v_snd_4419_; lean_object* v___x_4420_; lean_object* v___x_4421_; 
v_val_4417_ = lean_ctor_get(v___x_4416_, 0);
lean_inc(v_val_4417_);
lean_dec_ref_known(v___x_4416_, 1);
v_fst_4418_ = lean_ctor_get(v_val_4417_, 0);
lean_inc(v_fst_4418_);
v_snd_4419_ = lean_ctor_get(v_val_4417_, 1);
lean_inc(v_snd_4419_);
lean_dec(v_val_4417_);
v___x_4420_ = lean_array_get_size(v_snd_4419_);
lean_dec(v_snd_4419_);
lean_inc(v_relName_4335_);
v___x_4421_ = lp_mathlib_Mathlib_Tactic_GCongr_findGCongrLemmas_x3f_x27___redArg(v_relName_4335_, v_fst_4418_, v_forward_4338_, v___x_4420_, v___y_4415_);
if (lean_obj_tag(v___x_4421_) == 0)
{
lean_object* v_a_4422_; lean_object* v___x_4423_; uint8_t v___x_4424_; 
v_a_4422_ = lean_ctor_get(v___x_4421_, 0);
lean_inc(v_a_4422_);
lean_dec_ref_known(v___x_4421_, 1);
v___x_4423_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1));
v___x_4424_ = lean_name_eq(v_relName_4335_, v___x_4423_);
lean_dec(v_relName_4335_);
if (v___x_4424_ == 0)
{
v___y_4368_ = v___y_4404_;
v___y_4369_ = v___y_4405_;
v___y_4370_ = v___y_4406_;
v___y_4371_ = v___y_4407_;
v_lemmas_4372_ = v_a_4422_;
v___y_4373_ = v___y_4408_;
v___y_4374_ = v___y_4409_;
v___y_4375_ = v___y_4410_;
v___y_4376_ = v___y_4411_;
v___y_4377_ = v___y_4412_;
v___y_4378_ = v___y_4413_;
v___y_4379_ = v___y_4414_;
v___y_4380_ = v___y_4415_;
goto v___jp_4367_;
}
else
{
lean_object* v___x_4425_; lean_object* v___x_4426_; 
v___x_4425_ = lp_mathlib_Mathlib_Tactic_GCongr_relImpRelLemma(v___x_4420_);
v___x_4426_ = l_List_appendTR___redArg(v_a_4422_, v___x_4425_);
v___y_4368_ = v___y_4404_;
v___y_4369_ = v___y_4405_;
v___y_4370_ = v___y_4406_;
v___y_4371_ = v___y_4407_;
v_lemmas_4372_ = v___x_4426_;
v___y_4373_ = v___y_4408_;
v___y_4374_ = v___y_4409_;
v___y_4375_ = v___y_4410_;
v___y_4376_ = v___y_4411_;
v___y_4377_ = v___y_4412_;
v___y_4378_ = v___y_4413_;
v___y_4379_ = v___y_4414_;
v___y_4380_ = v___y_4415_;
goto v___jp_4367_;
}
}
else
{
lean_object* v_a_4427_; lean_object* v___x_4429_; uint8_t v_isShared_4430_; uint8_t v_isSharedCheck_4434_; 
lean_dec_ref(v___y_4407_);
lean_dec_ref(v___y_4406_);
lean_dec_ref(v___y_4405_);
lean_dec_ref(v___y_4404_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4427_ = lean_ctor_get(v___x_4421_, 0);
v_isSharedCheck_4434_ = !lean_is_exclusive(v___x_4421_);
if (v_isSharedCheck_4434_ == 0)
{
v___x_4429_ = v___x_4421_;
v_isShared_4430_ = v_isSharedCheck_4434_;
goto v_resetjp_4428_;
}
else
{
lean_inc(v_a_4427_);
lean_dec(v___x_4421_);
v___x_4429_ = lean_box(0);
v_isShared_4430_ = v_isSharedCheck_4434_;
goto v_resetjp_4428_;
}
v_resetjp_4428_:
{
lean_object* v___x_4432_; 
if (v_isShared_4430_ == 0)
{
v___x_4432_ = v___x_4429_;
goto v_reusejp_4431_;
}
else
{
lean_object* v_reuseFailAlloc_4433_; 
v_reuseFailAlloc_4433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4433_, 0, v_a_4427_);
v___x_4432_ = v_reuseFailAlloc_4433_;
goto v_reusejp_4431_;
}
v_reusejp_4431_:
{
return v___x_4432_;
}
}
}
}
else
{
lean_dec(v___x_4416_);
lean_dec_ref(v___y_4407_);
lean_dec_ref(v___y_4405_);
lean_dec_ref(v___y_4404_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v___y_4350_ = v___y_4406_;
v___y_4351_ = v___y_4409_;
goto v___jp_4349_;
}
}
v___jp_4435_:
{
lean_object* v___x_4451_; 
lean_inc_ref(v_config_4339_);
lean_inc_ref(v___y_4436_);
v___x_4451_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply(v___y_4436_, v___y_4446_, v___y_4450_, v_config_4339_, v___y_4443_, v___y_4438_, v___y_4441_, v___y_4437_);
if (lean_obj_tag(v___x_4451_) == 0)
{
lean_object* v_a_4452_; lean_object* v___x_4454_; uint8_t v_isShared_4455_; uint8_t v_isSharedCheck_4473_; 
v_a_4452_ = lean_ctor_get(v___x_4451_, 0);
v_isSharedCheck_4473_ = !lean_is_exclusive(v___x_4451_);
if (v_isSharedCheck_4473_ == 0)
{
v___x_4454_ = v___x_4451_;
v_isShared_4455_ = v_isSharedCheck_4473_;
goto v_resetjp_4453_;
}
else
{
lean_inc(v_a_4452_);
lean_dec(v___x_4451_);
v___x_4454_ = lean_box(0);
v_isShared_4455_ = v_isSharedCheck_4473_;
goto v_resetjp_4453_;
}
v_resetjp_4453_:
{
uint8_t v___x_4456_; 
v___x_4456_ = lean_unbox(v_a_4452_);
lean_dec(v_a_4452_);
if (v___x_4456_ == 0)
{
lean_del_object(v___x_4454_);
lean_dec_ref(v___y_4442_);
v___y_4404_ = v___y_4444_;
v___y_4405_ = v___y_4445_;
v___y_4406_ = v___y_4439_;
v___y_4407_ = v___y_4440_;
v___y_4408_ = v___y_4436_;
v___y_4409_ = v___y_4447_;
v___y_4410_ = v___y_4449_;
v___y_4411_ = v___y_4448_;
v___y_4412_ = v___y_4443_;
v___y_4413_ = v___y_4438_;
v___y_4414_ = v___y_4441_;
v___y_4415_ = v___y_4437_;
goto v___jp_4403_;
}
else
{
lean_object* v___x_4457_; lean_object* v_cache_4458_; lean_object* v___x_4460_; uint8_t v_isShared_4461_; uint8_t v_isSharedCheck_4471_; 
lean_dec_ref(v___y_4445_);
lean_dec_ref(v___y_4444_);
lean_dec_ref(v___y_4440_);
lean_dec_ref(v___y_4439_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v___x_4457_ = lean_st_ref_take(v___y_4447_);
v_cache_4458_ = lean_ctor_get(v___x_4457_, 0);
v_isSharedCheck_4471_ = !lean_is_exclusive(v___x_4457_);
if (v_isSharedCheck_4471_ == 0)
{
lean_object* v_unused_4472_; 
v_unused_4472_ = lean_ctor_get(v___x_4457_, 1);
lean_dec(v_unused_4472_);
v___x_4460_ = v___x_4457_;
v_isShared_4461_ = v_isSharedCheck_4471_;
goto v_resetjp_4459_;
}
else
{
lean_inc(v_cache_4458_);
lean_dec(v___x_4457_);
v___x_4460_ = lean_box(0);
v_isShared_4461_ = v_isSharedCheck_4471_;
goto v_resetjp_4459_;
}
v_resetjp_4459_:
{
lean_object* v___x_4462_; lean_object* v___x_4464_; 
v___x_4462_ = lean_box(1);
if (v_isShared_4461_ == 0)
{
lean_ctor_set(v___x_4460_, 1, v___x_4462_);
v___x_4464_ = v___x_4460_;
goto v_reusejp_4463_;
}
else
{
lean_object* v_reuseFailAlloc_4470_; 
v_reuseFailAlloc_4470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4470_, 0, v_cache_4458_);
lean_ctor_set(v_reuseFailAlloc_4470_, 1, v___x_4462_);
v___x_4464_ = v_reuseFailAlloc_4470_;
goto v_reusejp_4463_;
}
v_reusejp_4463_:
{
lean_object* v___x_4465_; lean_object* v___x_4466_; lean_object* v___x_4468_; 
v___x_4465_ = lean_st_ref_set(v___y_4447_, v___x_4464_);
v___x_4466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4466_, 0, v___y_4442_);
if (v_isShared_4455_ == 0)
{
lean_ctor_set(v___x_4454_, 0, v___x_4466_);
v___x_4468_ = v___x_4454_;
goto v_reusejp_4467_;
}
else
{
lean_object* v_reuseFailAlloc_4469_; 
v_reuseFailAlloc_4469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4469_, 0, v___x_4466_);
v___x_4468_ = v_reuseFailAlloc_4469_;
goto v_reusejp_4467_;
}
v_reusejp_4467_:
{
return v___x_4468_;
}
}
}
}
}
}
else
{
lean_object* v_a_4474_; lean_object* v___x_4476_; uint8_t v_isShared_4477_; uint8_t v_isSharedCheck_4481_; 
lean_dec_ref(v___y_4445_);
lean_dec_ref(v___y_4444_);
lean_dec_ref(v___y_4442_);
lean_dec_ref(v___y_4440_);
lean_dec_ref(v___y_4439_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4474_ = lean_ctor_get(v___x_4451_, 0);
v_isSharedCheck_4481_ = !lean_is_exclusive(v___x_4451_);
if (v_isSharedCheck_4481_ == 0)
{
v___x_4476_ = v___x_4451_;
v_isShared_4477_ = v_isSharedCheck_4481_;
goto v_resetjp_4475_;
}
else
{
lean_inc(v_a_4474_);
lean_dec(v___x_4451_);
v___x_4476_ = lean_box(0);
v_isShared_4477_ = v_isSharedCheck_4481_;
goto v_resetjp_4475_;
}
v_resetjp_4475_:
{
lean_object* v___x_4479_; 
if (v_isShared_4477_ == 0)
{
v___x_4479_ = v___x_4476_;
goto v_reusejp_4478_;
}
else
{
lean_object* v_reuseFailAlloc_4480_; 
v_reuseFailAlloc_4480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4480_, 0, v_a_4474_);
v___x_4479_ = v_reuseFailAlloc_4480_;
goto v_reusejp_4478_;
}
v_reusejp_4478_:
{
return v___x_4479_;
}
}
}
}
v___jp_4482_:
{
if (v___y_4496_ == 0)
{
lean_dec_ref(v___y_4489_);
v___y_4404_ = v___y_4491_;
v___y_4405_ = v___y_4492_;
v___y_4406_ = v___y_4486_;
v___y_4407_ = v___y_4487_;
v___y_4408_ = v___y_4483_;
v___y_4409_ = v___y_4493_;
v___y_4410_ = v___y_4495_;
v___y_4411_ = v___y_4494_;
v___y_4412_ = v___y_4490_;
v___y_4413_ = v___y_4485_;
v___y_4414_ = v___y_4488_;
v___y_4415_ = v___y_4484_;
goto v___jp_4403_;
}
else
{
uint8_t v_symm_4497_; lean_object* v___x_4498_; 
v_symm_4497_ = lean_ctor_get_uint8(v___y_4483_, sizeof(void*)*4);
v___x_4498_ = l_Lean_Expr_mvarId_x21(v___y_4491_);
if (v_forward_4338_ == 0)
{
if (v_symm_4497_ == 0)
{
v___y_4436_ = v___y_4483_;
v___y_4437_ = v___y_4484_;
v___y_4438_ = v___y_4485_;
v___y_4439_ = v___y_4486_;
v___y_4440_ = v___y_4487_;
v___y_4441_ = v___y_4488_;
v___y_4442_ = v___y_4489_;
v___y_4443_ = v___y_4490_;
v___y_4444_ = v___y_4491_;
v___y_4445_ = v___y_4492_;
v___y_4446_ = v___x_4498_;
v___y_4447_ = v___y_4493_;
v___y_4448_ = v___y_4494_;
v___y_4449_ = v___y_4495_;
v___y_4450_ = v___y_4496_;
goto v___jp_4435_;
}
else
{
v___y_4436_ = v___y_4483_;
v___y_4437_ = v___y_4484_;
v___y_4438_ = v___y_4485_;
v___y_4439_ = v___y_4486_;
v___y_4440_ = v___y_4487_;
v___y_4441_ = v___y_4488_;
v___y_4442_ = v___y_4489_;
v___y_4443_ = v___y_4490_;
v___y_4444_ = v___y_4491_;
v___y_4445_ = v___y_4492_;
v___y_4446_ = v___x_4498_;
v___y_4447_ = v___y_4493_;
v___y_4448_ = v___y_4494_;
v___y_4449_ = v___y_4495_;
v___y_4450_ = v_forward_4338_;
goto v___jp_4435_;
}
}
else
{
v___y_4436_ = v___y_4483_;
v___y_4437_ = v___y_4484_;
v___y_4438_ = v___y_4485_;
v___y_4439_ = v___y_4486_;
v___y_4440_ = v___y_4487_;
v___y_4441_ = v___y_4488_;
v___y_4442_ = v___y_4489_;
v___y_4443_ = v___y_4490_;
v___y_4444_ = v___y_4491_;
v___y_4445_ = v___y_4492_;
v___y_4446_ = v___x_4498_;
v___y_4447_ = v___y_4493_;
v___y_4448_ = v___y_4494_;
v___y_4449_ = v___y_4495_;
v___y_4450_ = v_symm_4497_;
goto v___jp_4435_;
}
}
}
v___jp_4499_:
{
lean_object* v___x_4500_; lean_object* v___x_4501_; 
v___x_4500_ = lean_box(0);
v___x_4501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4501_, 0, v___x_4500_);
return v___x_4501_;
}
v___jp_4520_:
{
lean_object* v___x_4531_; lean_object* v_cache_4532_; lean_object* v___x_4534_; uint8_t v_isShared_4535_; uint8_t v_isSharedCheck_4576_; 
v___x_4531_ = lean_st_ref_get(v___y_4525_);
v_cache_4532_ = lean_ctor_get(v___x_4531_, 0);
v_isSharedCheck_4576_ = !lean_is_exclusive(v___x_4531_);
if (v_isSharedCheck_4576_ == 0)
{
lean_object* v_unused_4577_; 
v_unused_4577_ = lean_ctor_get(v___x_4531_, 1);
lean_dec(v_unused_4577_);
v___x_4534_ = v___x_4531_;
v_isShared_4535_ = v_isSharedCheck_4576_;
goto v_resetjp_4533_;
}
else
{
lean_inc(v_cache_4532_);
lean_dec(v___x_4531_);
v___x_4534_ = lean_box(0);
v_isShared_4535_ = v_isSharedCheck_4576_;
goto v_resetjp_4533_;
}
v_resetjp_4533_:
{
lean_object* v___x_4536_; lean_object* v___x_4538_; 
v___x_4536_ = lean_box(v_forward_4338_);
lean_inc_ref(v___y_4522_);
if (v_isShared_4535_ == 0)
{
lean_ctor_set(v___x_4534_, 1, v___x_4536_);
lean_ctor_set(v___x_4534_, 0, v___y_4522_);
v___x_4538_ = v___x_4534_;
goto v_reusejp_4537_;
}
else
{
lean_object* v_reuseFailAlloc_4575_; 
v_reuseFailAlloc_4575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4575_, 0, v___y_4522_);
lean_ctor_set(v_reuseFailAlloc_4575_, 1, v___x_4536_);
v___x_4538_ = v_reuseFailAlloc_4575_;
goto v_reusejp_4537_;
}
v_reusejp_4537_:
{
lean_object* v___x_4539_; uint8_t v___x_4540_; 
lean_inc(v_a_4530_);
v___x_4539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4539_, 0, v_a_4530_);
lean_ctor_set(v___x_4539_, 1, v___x_4538_);
v___x_4540_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(v_cache_4532_, v___x_4539_);
lean_dec_ref(v_cache_4532_);
if (v___x_4540_ == 0)
{
lean_object* v___x_4541_; 
lean_inc_ref(v___y_4522_);
v___x_4541_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_makeGCongrGoal(v_a_4530_, v___y_4522_, v_forward_4338_, v___y_4528_, v___y_4524_, v___y_4527_, v___y_4523_);
if (lean_obj_tag(v___x_4541_) == 0)
{
lean_object* v_a_4542_; lean_object* v_index_4543_; lean_object* v_fst_4544_; lean_object* v_snd_4545_; lean_object* v_fst_4546_; lean_object* v_snd_4547_; lean_object* v___x_4548_; uint8_t v___x_4549_; 
v_a_4542_ = lean_ctor_get(v___x_4541_, 0);
lean_inc(v_a_4542_);
lean_dec_ref_known(v___x_4541_, 1);
v_index_4543_ = lean_ctor_get(v___y_4521_, 2);
v_fst_4544_ = lean_ctor_get(v_a_4542_, 0);
lean_inc(v_fst_4544_);
v_snd_4545_ = lean_ctor_get(v_a_4542_, 1);
lean_inc(v_snd_4545_);
v_fst_4546_ = lean_ctor_get(v_index_4543_, 0);
v_snd_4547_ = lean_ctor_get(v_index_4543_, 1);
lean_inc_ref(v___y_4522_);
v___x_4548_ = l_Lean_Expr_toHeadIndex(v___y_4522_);
v___x_4549_ = l_Lean_instBEqHeadIndex_beq(v___x_4548_, v_fst_4546_);
lean_dec(v___x_4548_);
if (v___x_4549_ == 0)
{
v___y_4483_ = v___y_4521_;
v___y_4484_ = v___y_4523_;
v___y_4485_ = v___y_4524_;
v___y_4486_ = v___x_4539_;
v___y_4487_ = v_fst_4544_;
v___y_4488_ = v___y_4527_;
v___y_4489_ = v_a_4542_;
v___y_4490_ = v___y_4528_;
v___y_4491_ = v_snd_4545_;
v___y_4492_ = v___y_4522_;
v___y_4493_ = v___y_4525_;
v___y_4494_ = v___y_4526_;
v___y_4495_ = v___y_4529_;
v___y_4496_ = v___x_4549_;
goto v___jp_4482_;
}
else
{
lean_object* v___x_4550_; uint8_t v___x_4551_; 
v___x_4550_ = l_Lean_Expr_headNumArgs(v___y_4522_);
v___x_4551_ = lean_nat_dec_eq(v___x_4550_, v_snd_4547_);
lean_dec(v___x_4550_);
v___y_4483_ = v___y_4521_;
v___y_4484_ = v___y_4523_;
v___y_4485_ = v___y_4524_;
v___y_4486_ = v___x_4539_;
v___y_4487_ = v_fst_4544_;
v___y_4488_ = v___y_4527_;
v___y_4489_ = v_a_4542_;
v___y_4490_ = v___y_4528_;
v___y_4491_ = v_snd_4545_;
v___y_4492_ = v___y_4522_;
v___y_4493_ = v___y_4525_;
v___y_4494_ = v___y_4526_;
v___y_4495_ = v___y_4529_;
v___y_4496_ = v___x_4551_;
goto v___jp_4482_;
}
}
else
{
lean_object* v_a_4552_; lean_object* v___x_4554_; uint8_t v_isShared_4555_; uint8_t v_isSharedCheck_4559_; 
lean_dec_ref_known(v___x_4539_, 2);
lean_dec_ref(v___y_4522_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4552_ = lean_ctor_get(v___x_4541_, 0);
v_isSharedCheck_4559_ = !lean_is_exclusive(v___x_4541_);
if (v_isSharedCheck_4559_ == 0)
{
v___x_4554_ = v___x_4541_;
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
else
{
lean_inc(v_a_4552_);
lean_dec(v___x_4541_);
v___x_4554_ = lean_box(0);
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
v_resetjp_4553_:
{
lean_object* v___x_4557_; 
if (v_isShared_4555_ == 0)
{
v___x_4557_ = v___x_4554_;
goto v_reusejp_4556_;
}
else
{
lean_object* v_reuseFailAlloc_4558_; 
v_reuseFailAlloc_4558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4558_, 0, v_a_4552_);
v___x_4557_ = v_reuseFailAlloc_4558_;
goto v_reusejp_4556_;
}
v_reusejp_4556_:
{
return v___x_4557_;
}
}
}
}
else
{
lean_object* v_options_4560_; uint8_t v_hasTrace_4561_; 
lean_dec_ref_known(v___x_4539_, 2);
lean_dec(v_a_4530_);
lean_dec_ref(v___y_4522_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_options_4560_ = lean_ctor_get(v___y_4527_, 2);
v_hasTrace_4561_ = lean_ctor_get_uint8(v_options_4560_, sizeof(void*)*1);
if (v_hasTrace_4561_ == 0)
{
goto v___jp_4499_;
}
else
{
lean_object* v_inheritedTraceOptions_4562_; lean_object* v___x_4563_; uint8_t v___x_4564_; 
v_inheritedTraceOptions_4562_ = lean_ctor_get(v___y_4527_, 13);
v___x_4563_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply___closed__3);
v___x_4564_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4562_, v_options_4560_, v___x_4563_);
if (v___x_4564_ == 0)
{
goto v___jp_4499_;
}
else
{
lean_object* v___x_4565_; lean_object* v___x_4566_; 
v___x_4565_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___closed__1);
v___x_4566_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_4519_, v___x_4565_, v___y_4528_, v___y_4524_, v___y_4527_, v___y_4523_);
if (lean_obj_tag(v___x_4566_) == 0)
{
lean_dec_ref_known(v___x_4566_, 1);
goto v___jp_4499_;
}
else
{
lean_object* v_a_4567_; lean_object* v___x_4569_; uint8_t v_isShared_4570_; uint8_t v_isSharedCheck_4574_; 
v_a_4567_ = lean_ctor_get(v___x_4566_, 0);
v_isSharedCheck_4574_ = !lean_is_exclusive(v___x_4566_);
if (v_isSharedCheck_4574_ == 0)
{
v___x_4569_ = v___x_4566_;
v_isShared_4570_ = v_isSharedCheck_4574_;
goto v_resetjp_4568_;
}
else
{
lean_inc(v_a_4567_);
lean_dec(v___x_4566_);
v___x_4569_ = lean_box(0);
v_isShared_4570_ = v_isSharedCheck_4574_;
goto v_resetjp_4568_;
}
v_resetjp_4568_:
{
lean_object* v___x_4572_; 
if (v_isShared_4570_ == 0)
{
v___x_4572_ = v___x_4569_;
goto v_reusejp_4571_;
}
else
{
lean_object* v_reuseFailAlloc_4573_; 
v_reuseFailAlloc_4573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4573_, 0, v_a_4567_);
v___x_4572_ = v_reuseFailAlloc_4573_;
goto v_reusejp_4571_;
}
v_reusejp_4571_:
{
return v___x_4572_;
}
}
}
}
}
}
}
}
}
v___jp_4578_:
{
if (lean_obj_tag(v_rel_x3f_4336_) == 0)
{
v___y_4521_ = v___y_4580_;
v___y_4522_ = v_e_4579_;
v___y_4523_ = v___y_4587_;
v___y_4524_ = v___y_4585_;
v___y_4525_ = v___y_4581_;
v___y_4526_ = v___y_4583_;
v___y_4527_ = v___y_4586_;
v___y_4528_ = v___y_4584_;
v___y_4529_ = v___y_4582_;
v_a_4530_ = v_rel_x3f_4336_;
goto v___jp_4520_;
}
else
{
lean_object* v_val_4588_; lean_object* v___x_4590_; uint8_t v_isShared_4591_; uint8_t v_isSharedCheck_4605_; 
v_val_4588_ = lean_ctor_get(v_rel_x3f_4336_, 0);
v_isSharedCheck_4605_ = !lean_is_exclusive(v_rel_x3f_4336_);
if (v_isSharedCheck_4605_ == 0)
{
v___x_4590_ = v_rel_x3f_4336_;
v_isShared_4591_ = v_isSharedCheck_4605_;
goto v_resetjp_4589_;
}
else
{
lean_inc(v_val_4588_);
lean_dec(v_rel_x3f_4336_);
v___x_4590_ = lean_box(0);
v_isShared_4591_ = v_isSharedCheck_4605_;
goto v_resetjp_4589_;
}
v_resetjp_4589_:
{
lean_object* v___x_4592_; 
v___x_4592_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_val_4588_, v___y_4585_);
if (lean_obj_tag(v___x_4592_) == 0)
{
lean_object* v_a_4593_; lean_object* v___x_4595_; 
v_a_4593_ = lean_ctor_get(v___x_4592_, 0);
lean_inc(v_a_4593_);
lean_dec_ref_known(v___x_4592_, 1);
if (v_isShared_4591_ == 0)
{
lean_ctor_set(v___x_4590_, 0, v_a_4593_);
v___x_4595_ = v___x_4590_;
goto v_reusejp_4594_;
}
else
{
lean_object* v_reuseFailAlloc_4596_; 
v_reuseFailAlloc_4596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4596_, 0, v_a_4593_);
v___x_4595_ = v_reuseFailAlloc_4596_;
goto v_reusejp_4594_;
}
v_reusejp_4594_:
{
v___y_4521_ = v___y_4580_;
v___y_4522_ = v_e_4579_;
v___y_4523_ = v___y_4587_;
v___y_4524_ = v___y_4585_;
v___y_4525_ = v___y_4581_;
v___y_4526_ = v___y_4583_;
v___y_4527_ = v___y_4586_;
v___y_4528_ = v___y_4584_;
v___y_4529_ = v___y_4582_;
v_a_4530_ = v___x_4595_;
goto v___jp_4520_;
}
}
else
{
lean_object* v_a_4597_; lean_object* v___x_4599_; uint8_t v_isShared_4600_; uint8_t v_isSharedCheck_4604_; 
lean_del_object(v___x_4590_);
lean_dec_ref(v_e_4579_);
lean_dec_ref(v_config_4339_);
lean_dec(v_relName_4335_);
v_a_4597_ = lean_ctor_get(v___x_4592_, 0);
v_isSharedCheck_4604_ = !lean_is_exclusive(v___x_4592_);
if (v_isSharedCheck_4604_ == 0)
{
v___x_4599_ = v___x_4592_;
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
else
{
lean_inc(v_a_4597_);
lean_dec(v___x_4592_);
v___x_4599_ = lean_box(0);
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
v_resetjp_4598_:
{
lean_object* v___x_4602_; 
if (v_isShared_4600_ == 0)
{
v___x_4602_ = v___x_4599_;
goto v_reusejp_4601_;
}
else
{
lean_object* v_reuseFailAlloc_4603_; 
v_reuseFailAlloc_4603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4603_, 0, v_a_4597_);
v___x_4602_ = v_reuseFailAlloc_4603_;
goto v_reusejp_4601_;
}
v_reusejp_4601_:
{
return v___x_4602_;
}
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0(void){
_start:
{
lean_object* v___x_4952_; lean_object* v_dummy_4953_; 
v___x_4952_ = lean_box(0);
v_dummy_4953_ = l_Lean_Expr_sort___override(v___x_4952_);
return v_dummy_4953_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2(void){
_start:
{
lean_object* v___x_4955_; lean_object* v___x_4956_; 
v___x_4955_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__1));
v___x_4956_ = l_Lean_stringToMessageData(v___x_4955_);
return v___x_4956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(lean_object* v_goal_4957_, uint8_t v_forward_4958_, lean_object* v_config_4959_, lean_object* v_a_4960_, lean_object* v_a_4961_, lean_object* v_a_4962_, lean_object* v_a_4963_, lean_object* v_a_4964_, lean_object* v_a_4965_, lean_object* v_a_4966_, lean_object* v_a_4967_){
_start:
{
lean_object* v___x_4969_; 
lean_inc(v_goal_4957_);
v___x_4969_ = l_Lean_MVarId_getType(v_goal_4957_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
if (lean_obj_tag(v___x_4969_) == 0)
{
lean_object* v_a_4970_; lean_object* v___x_4971_; 
v_a_4970_ = lean_ctor_get(v___x_4969_, 0);
lean_inc(v_a_4970_);
lean_dec_ref_known(v___x_4969_, 1);
lean_inc(v_a_4967_);
lean_inc_ref(v_a_4966_);
lean_inc(v_a_4965_);
lean_inc_ref(v_a_4964_);
v___x_4971_ = lean_whnf(v_a_4970_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
if (lean_obj_tag(v___x_4971_) == 0)
{
lean_object* v_a_4972_; lean_object* v___x_4973_; 
v_a_4972_ = lean_ctor_get(v___x_4971_, 0);
lean_inc(v_a_4972_);
lean_dec_ref_known(v___x_4971_, 1);
v___x_4973_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27(v_a_4972_);
lean_dec(v_a_4972_);
if (lean_obj_tag(v___x_4973_) == 1)
{
lean_object* v_val_4974_; lean_object* v_snd_4975_; lean_object* v_fst_4976_; lean_object* v_fst_4977_; lean_object* v_snd_4978_; lean_object* v_fst_4980_; lean_object* v_snd_4981_; 
v_val_4974_ = lean_ctor_get(v___x_4973_, 0);
lean_inc(v_val_4974_);
lean_dec_ref_known(v___x_4973_, 1);
v_snd_4975_ = lean_ctor_get(v_val_4974_, 1);
lean_inc(v_snd_4975_);
v_fst_4976_ = lean_ctor_get(v_val_4974_, 0);
lean_inc(v_fst_4976_);
lean_dec(v_val_4974_);
v_fst_4977_ = lean_ctor_get(v_snd_4975_, 0);
lean_inc(v_fst_4977_);
v_snd_4978_ = lean_ctor_get(v_snd_4975_, 1);
lean_inc(v_snd_4978_);
lean_dec(v_snd_4975_);
if (v_forward_4958_ == 0)
{
lean_object* v_fst_5010_; lean_object* v_snd_5011_; 
v_fst_5010_ = lean_ctor_get(v_snd_4978_, 0);
lean_inc(v_fst_5010_);
v_snd_5011_ = lean_ctor_get(v_snd_4978_, 1);
lean_inc(v_snd_5011_);
lean_dec(v_snd_4978_);
v_fst_4980_ = v_snd_5011_;
v_snd_4981_ = v_fst_5010_;
goto v___jp_4979_;
}
else
{
lean_object* v_fst_5012_; lean_object* v_snd_5013_; 
v_fst_5012_ = lean_ctor_get(v_snd_4978_, 0);
lean_inc(v_fst_5012_);
v_snd_5013_ = lean_ctor_get(v_snd_4978_, 1);
lean_inc(v_snd_5013_);
lean_dec(v_snd_4978_);
v_fst_4980_ = v_fst_5012_;
v_snd_4981_ = v_snd_5013_;
goto v___jp_4979_;
}
v___jp_4979_:
{
lean_object* v___x_4982_; 
v___x_4982_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore(v_fst_4976_, v_fst_4977_, v_fst_4980_, v_forward_4958_, v_config_4959_, v_a_4960_, v_a_4961_, v_a_4962_, v_a_4963_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
if (lean_obj_tag(v___x_4982_) == 0)
{
lean_object* v_a_4983_; lean_object* v___x_4985_; uint8_t v_isShared_4986_; uint8_t v_isSharedCheck_5001_; 
v_a_4983_ = lean_ctor_get(v___x_4982_, 0);
v_isSharedCheck_5001_ = !lean_is_exclusive(v___x_4982_);
if (v_isSharedCheck_5001_ == 0)
{
v___x_4985_ = v___x_4982_;
v_isShared_4986_ = v_isSharedCheck_5001_;
goto v_resetjp_4984_;
}
else
{
lean_inc(v_a_4983_);
lean_dec(v___x_4982_);
v___x_4985_ = lean_box(0);
v_isShared_4986_ = v_isSharedCheck_5001_;
goto v_resetjp_4984_;
}
v_resetjp_4984_:
{
if (lean_obj_tag(v_a_4983_) == 1)
{
lean_object* v_val_4987_; lean_object* v_fst_4988_; lean_object* v_snd_4989_; lean_object* v_dummy_4990_; lean_object* v_nargs_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; lean_object* v___x_4994_; lean_object* v___x_4995_; 
lean_del_object(v___x_4985_);
v_val_4987_ = lean_ctor_get(v_a_4983_, 0);
lean_inc(v_val_4987_);
lean_dec_ref_known(v_a_4983_, 1);
v_fst_4988_ = lean_ctor_get(v_val_4987_, 0);
lean_inc(v_fst_4988_);
v_snd_4989_ = lean_ctor_get(v_val_4987_, 1);
lean_inc(v_snd_4989_);
lean_dec(v_val_4987_);
v_dummy_4990_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__0);
v_nargs_4991_ = l_Lean_Expr_getAppNumArgs(v_snd_4981_);
lean_inc(v_nargs_4991_);
v___x_4992_ = lean_mk_array(v_nargs_4991_, v_dummy_4990_);
v___x_4993_ = lean_unsigned_to_nat(1u);
v___x_4994_ = lean_nat_sub(v_nargs_4991_, v___x_4993_);
lean_dec(v_nargs_4991_);
v___x_4995_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__1(v_fst_4988_, v_goal_4957_, v_snd_4989_, v_snd_4981_, v___x_4992_, v___x_4994_, v_a_4960_, v_a_4961_, v_a_4962_, v_a_4963_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
return v___x_4995_;
}
else
{
uint8_t v___x_4996_; lean_object* v___x_4997_; lean_object* v___x_4999_; 
lean_dec(v_a_4983_);
lean_dec_ref(v_snd_4981_);
lean_dec(v_goal_4957_);
v___x_4996_ = 0;
v___x_4997_ = lean_box(v___x_4996_);
if (v_isShared_4986_ == 0)
{
lean_ctor_set(v___x_4985_, 0, v___x_4997_);
v___x_4999_ = v___x_4985_;
goto v_reusejp_4998_;
}
else
{
lean_object* v_reuseFailAlloc_5000_; 
v_reuseFailAlloc_5000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5000_, 0, v___x_4997_);
v___x_4999_ = v_reuseFailAlloc_5000_;
goto v_reusejp_4998_;
}
v_reusejp_4998_:
{
return v___x_4999_;
}
}
}
}
else
{
lean_object* v_a_5002_; lean_object* v___x_5004_; uint8_t v_isShared_5005_; uint8_t v_isSharedCheck_5009_; 
lean_dec_ref(v_snd_4981_);
lean_dec(v_goal_4957_);
v_a_5002_ = lean_ctor_get(v___x_4982_, 0);
v_isSharedCheck_5009_ = !lean_is_exclusive(v___x_4982_);
if (v_isSharedCheck_5009_ == 0)
{
v___x_5004_ = v___x_4982_;
v_isShared_5005_ = v_isSharedCheck_5009_;
goto v_resetjp_5003_;
}
else
{
lean_inc(v_a_5002_);
lean_dec(v___x_4982_);
v___x_5004_ = lean_box(0);
v_isShared_5005_ = v_isSharedCheck_5009_;
goto v_resetjp_5003_;
}
v_resetjp_5003_:
{
lean_object* v___x_5007_; 
if (v_isShared_5005_ == 0)
{
v___x_5007_ = v___x_5004_;
goto v_reusejp_5006_;
}
else
{
lean_object* v_reuseFailAlloc_5008_; 
v_reuseFailAlloc_5008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5008_, 0, v_a_5002_);
v___x_5007_ = v_reuseFailAlloc_5008_;
goto v_reusejp_5006_;
}
v_reusejp_5006_:
{
return v___x_5007_;
}
}
}
}
}
else
{
lean_object* v___x_5014_; lean_object* v___x_5015_; lean_object* v___x_5016_; lean_object* v___x_5017_; 
lean_dec(v___x_4973_);
lean_dec_ref(v_config_4959_);
v___x_5014_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2, &lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___closed__2);
v___x_5015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5015_, 0, v_goal_4957_);
v___x_5016_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5016_, 0, v___x_5014_);
lean_ctor_set(v___x_5016_, 1, v___x_5015_);
v___x_5017_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg(v___x_5016_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
return v___x_5017_;
}
}
else
{
lean_object* v_a_5018_; lean_object* v___x_5020_; uint8_t v_isShared_5021_; uint8_t v_isSharedCheck_5025_; 
lean_dec_ref(v_config_4959_);
lean_dec(v_goal_4957_);
v_a_5018_ = lean_ctor_get(v___x_4971_, 0);
v_isSharedCheck_5025_ = !lean_is_exclusive(v___x_4971_);
if (v_isSharedCheck_5025_ == 0)
{
v___x_5020_ = v___x_4971_;
v_isShared_5021_ = v_isSharedCheck_5025_;
goto v_resetjp_5019_;
}
else
{
lean_inc(v_a_5018_);
lean_dec(v___x_4971_);
v___x_5020_ = lean_box(0);
v_isShared_5021_ = v_isSharedCheck_5025_;
goto v_resetjp_5019_;
}
v_resetjp_5019_:
{
lean_object* v___x_5023_; 
if (v_isShared_5021_ == 0)
{
v___x_5023_ = v___x_5020_;
goto v_reusejp_5022_;
}
else
{
lean_object* v_reuseFailAlloc_5024_; 
v_reuseFailAlloc_5024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5024_, 0, v_a_5018_);
v___x_5023_ = v_reuseFailAlloc_5024_;
goto v_reusejp_5022_;
}
v_reusejp_5022_:
{
return v___x_5023_;
}
}
}
}
else
{
lean_object* v_a_5026_; lean_object* v___x_5028_; uint8_t v_isShared_5029_; uint8_t v_isSharedCheck_5033_; 
lean_dec_ref(v_config_4959_);
lean_dec(v_goal_4957_);
v_a_5026_ = lean_ctor_get(v___x_4969_, 0);
v_isSharedCheck_5033_ = !lean_is_exclusive(v___x_4969_);
if (v_isSharedCheck_5033_ == 0)
{
v___x_5028_ = v___x_4969_;
v_isShared_5029_ = v_isSharedCheck_5033_;
goto v_resetjp_5027_;
}
else
{
lean_inc(v_a_5026_);
lean_dec(v___x_4969_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1(lean_object* v_goal_5034_, uint8_t v_forward_5035_, lean_object* v_config_5036_, uint8_t v___x_5037_, lean_object* v___x_5038_, lean_object* v___x_5039_, lean_object* v_lctx_5040_, uint8_t v___x_5041_, lean_object* v___y_5042_, lean_object* v___y_5043_, lean_object* v___y_5044_, lean_object* v___y_5045_, lean_object* v___y_5046_, lean_object* v___y_5047_, lean_object* v___y_5048_, lean_object* v___y_5049_){
_start:
{
uint8_t v___y_5052_; lean_object* v___y_5053_; uint8_t v___y_5082_; lean_object* v___y_5083_; uint8_t v___y_5101_; lean_object* v___y_5102_; lean_object* v___y_5103_; lean_object* v___y_5104_; lean_object* v___y_5105_; lean_object* v___y_5106_; lean_object* v___x_5142_; uint8_t v___y_5144_; lean_object* v_progress_5162_; 
v___x_5142_ = lean_st_ref_get(v___y_5043_);
v_progress_5162_ = lean_ctor_get(v___x_5142_, 1);
lean_inc(v_progress_5162_);
lean_dec(v___x_5142_);
if (lean_obj_tag(v_progress_5162_) == 0)
{
v___y_5144_ = v___x_5041_;
goto v___jp_5143_;
}
else
{
lean_dec(v_progress_5162_);
if (v___x_5037_ == 0)
{
lean_object* v___x_5163_; 
v___x_5163_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(v_goal_5034_, v_forward_5035_, v_config_5036_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_);
return v___x_5163_;
}
else
{
v___y_5144_ = v___x_5037_;
goto v___jp_5143_;
}
}
v___jp_5051_:
{
if (lean_obj_tag(v___y_5053_) == 0)
{
lean_object* v_a_5054_; lean_object* v___x_5056_; uint8_t v_isShared_5057_; uint8_t v_isSharedCheck_5080_; 
v_a_5054_ = lean_ctor_get(v___y_5053_, 0);
v_isSharedCheck_5080_ = !lean_is_exclusive(v___y_5053_);
if (v_isSharedCheck_5080_ == 0)
{
v___x_5056_ = v___y_5053_;
v_isShared_5057_ = v_isSharedCheck_5080_;
goto v_resetjp_5055_;
}
else
{
lean_inc(v_a_5054_);
lean_dec(v___y_5053_);
v___x_5056_ = lean_box(0);
v_isShared_5057_ = v_isSharedCheck_5080_;
goto v_resetjp_5055_;
}
v_resetjp_5055_:
{
uint8_t v___x_5058_; 
v___x_5058_ = lean_unbox(v_a_5054_);
lean_dec(v_a_5054_);
if (v___x_5058_ == 0)
{
lean_object* v___x_5059_; lean_object* v_lctx_5060_; lean_object* v_cache_5061_; lean_object* v___x_5063_; uint8_t v_isShared_5064_; uint8_t v_isSharedCheck_5074_; 
v___x_5059_ = lean_st_ref_take(v___y_5043_);
v_lctx_5060_ = lean_ctor_get(v___y_5046_, 2);
v_cache_5061_ = lean_ctor_get(v___x_5059_, 0);
v_isSharedCheck_5074_ = !lean_is_exclusive(v___x_5059_);
if (v_isSharedCheck_5074_ == 0)
{
lean_object* v_unused_5075_; 
v_unused_5075_ = lean_ctor_get(v___x_5059_, 1);
lean_dec(v_unused_5075_);
v___x_5063_ = v___x_5059_;
v_isShared_5064_ = v_isSharedCheck_5074_;
goto v_resetjp_5062_;
}
else
{
lean_inc(v_cache_5061_);
lean_dec(v___x_5059_);
v___x_5063_ = lean_box(0);
v_isShared_5064_ = v_isSharedCheck_5074_;
goto v_resetjp_5062_;
}
v_resetjp_5062_:
{
lean_object* v___x_5065_; lean_object* v___x_5067_; 
lean_inc_ref(v_lctx_5060_);
v___x_5065_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_5065_, 0, v_lctx_5060_);
if (v_isShared_5064_ == 0)
{
lean_ctor_set(v___x_5063_, 1, v___x_5065_);
v___x_5067_ = v___x_5063_;
goto v_reusejp_5066_;
}
else
{
lean_object* v_reuseFailAlloc_5073_; 
v_reuseFailAlloc_5073_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5073_, 0, v_cache_5061_);
lean_ctor_set(v_reuseFailAlloc_5073_, 1, v___x_5065_);
v___x_5067_ = v_reuseFailAlloc_5073_;
goto v_reusejp_5066_;
}
v_reusejp_5066_:
{
lean_object* v___x_5068_; lean_object* v___x_5069_; lean_object* v___x_5071_; 
v___x_5068_ = lean_st_ref_set(v___y_5043_, v___x_5067_);
v___x_5069_ = lean_box(v___y_5052_);
if (v_isShared_5057_ == 0)
{
lean_ctor_set(v___x_5056_, 0, v___x_5069_);
v___x_5071_ = v___x_5056_;
goto v_reusejp_5070_;
}
else
{
lean_object* v_reuseFailAlloc_5072_; 
v_reuseFailAlloc_5072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5072_, 0, v___x_5069_);
v___x_5071_ = v_reuseFailAlloc_5072_;
goto v_reusejp_5070_;
}
v_reusejp_5070_:
{
return v___x_5071_;
}
}
}
}
else
{
lean_object* v___x_5076_; lean_object* v___x_5078_; 
v___x_5076_ = lean_box(v___y_5052_);
if (v_isShared_5057_ == 0)
{
lean_ctor_set(v___x_5056_, 0, v___x_5076_);
v___x_5078_ = v___x_5056_;
goto v_reusejp_5077_;
}
else
{
lean_object* v_reuseFailAlloc_5079_; 
v_reuseFailAlloc_5079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5079_, 0, v___x_5076_);
v___x_5078_ = v_reuseFailAlloc_5079_;
goto v_reusejp_5077_;
}
v_reusejp_5077_:
{
return v___x_5078_;
}
}
}
}
else
{
return v___y_5053_;
}
}
v___jp_5081_:
{
lean_object* v___x_5084_; lean_object* v_cache_5085_; lean_object* v_zetaDeltaFVarIds_5086_; lean_object* v_postponed_5087_; lean_object* v_diag_5088_; lean_object* v___x_5090_; uint8_t v_isShared_5091_; uint8_t v_isSharedCheck_5098_; 
v___x_5084_ = lean_st_ref_take(v___y_5047_);
v_cache_5085_ = lean_ctor_get(v___x_5084_, 1);
v_zetaDeltaFVarIds_5086_ = lean_ctor_get(v___x_5084_, 2);
v_postponed_5087_ = lean_ctor_get(v___x_5084_, 3);
v_diag_5088_ = lean_ctor_get(v___x_5084_, 4);
v_isSharedCheck_5098_ = !lean_is_exclusive(v___x_5084_);
if (v_isSharedCheck_5098_ == 0)
{
lean_object* v_unused_5099_; 
v_unused_5099_ = lean_ctor_get(v___x_5084_, 0);
lean_dec(v_unused_5099_);
v___x_5090_ = v___x_5084_;
v_isShared_5091_ = v_isSharedCheck_5098_;
goto v_resetjp_5089_;
}
else
{
lean_inc(v_diag_5088_);
lean_inc(v_postponed_5087_);
lean_inc(v_zetaDeltaFVarIds_5086_);
lean_inc(v_cache_5085_);
lean_dec(v___x_5084_);
v___x_5090_ = lean_box(0);
v_isShared_5091_ = v_isSharedCheck_5098_;
goto v_resetjp_5089_;
}
v_resetjp_5089_:
{
lean_object* v___x_5093_; 
if (v_isShared_5091_ == 0)
{
lean_ctor_set(v___x_5090_, 0, v___y_5083_);
v___x_5093_ = v___x_5090_;
goto v_reusejp_5092_;
}
else
{
lean_object* v_reuseFailAlloc_5097_; 
v_reuseFailAlloc_5097_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5097_, 0, v___y_5083_);
lean_ctor_set(v_reuseFailAlloc_5097_, 1, v_cache_5085_);
lean_ctor_set(v_reuseFailAlloc_5097_, 2, v_zetaDeltaFVarIds_5086_);
lean_ctor_set(v_reuseFailAlloc_5097_, 3, v_postponed_5087_);
lean_ctor_set(v_reuseFailAlloc_5097_, 4, v_diag_5088_);
v___x_5093_ = v_reuseFailAlloc_5097_;
goto v_reusejp_5092_;
}
v_reusejp_5092_:
{
lean_object* v___x_5094_; lean_object* v___x_5095_; lean_object* v___x_5096_; 
v___x_5094_ = lean_st_ref_set(v___y_5047_, v___x_5093_);
v___x_5095_ = lean_box(v___y_5082_);
v___x_5096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5096_, 0, v___x_5095_);
return v___x_5096_;
}
}
}
v___jp_5100_:
{
lean_object* v___x_5107_; lean_object* v_cache_5108_; lean_object* v_zetaDeltaFVarIds_5109_; lean_object* v_postponed_5110_; lean_object* v_diag_5111_; lean_object* v___x_5113_; uint8_t v_isShared_5114_; uint8_t v_isSharedCheck_5140_; 
v___x_5107_ = lean_st_ref_take(v___y_5047_);
v_cache_5108_ = lean_ctor_get(v___x_5107_, 1);
v_zetaDeltaFVarIds_5109_ = lean_ctor_get(v___x_5107_, 2);
v_postponed_5110_ = lean_ctor_get(v___x_5107_, 3);
v_diag_5111_ = lean_ctor_get(v___x_5107_, 4);
v_isSharedCheck_5140_ = !lean_is_exclusive(v___x_5107_);
if (v_isSharedCheck_5140_ == 0)
{
lean_object* v_unused_5141_; 
v_unused_5141_ = lean_ctor_get(v___x_5107_, 0);
lean_dec(v_unused_5141_);
v___x_5113_ = v___x_5107_;
v_isShared_5114_ = v_isSharedCheck_5140_;
goto v_resetjp_5112_;
}
else
{
lean_inc(v_diag_5111_);
lean_inc(v_postponed_5110_);
lean_inc(v_zetaDeltaFVarIds_5109_);
lean_inc(v_cache_5108_);
lean_dec(v___x_5107_);
v___x_5113_ = lean_box(0);
v_isShared_5114_ = v_isSharedCheck_5140_;
goto v_resetjp_5112_;
}
v_resetjp_5112_:
{
lean_object* v___x_5116_; 
if (v_isShared_5114_ == 0)
{
lean_ctor_set(v___x_5113_, 0, v___y_5106_);
v___x_5116_ = v___x_5113_;
goto v_reusejp_5115_;
}
else
{
lean_object* v_reuseFailAlloc_5139_; 
v_reuseFailAlloc_5139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5139_, 0, v___y_5106_);
lean_ctor_set(v_reuseFailAlloc_5139_, 1, v_cache_5108_);
lean_ctor_set(v_reuseFailAlloc_5139_, 2, v_zetaDeltaFVarIds_5109_);
lean_ctor_set(v_reuseFailAlloc_5139_, 3, v_postponed_5110_);
lean_ctor_set(v_reuseFailAlloc_5139_, 4, v_diag_5111_);
v___x_5116_ = v_reuseFailAlloc_5139_;
goto v_reusejp_5115_;
}
v_reusejp_5115_:
{
lean_object* v___x_5117_; lean_object* v___x_5118_; 
v___x_5117_ = lean_st_ref_set(v___y_5047_, v___x_5116_);
v___x_5118_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(v_goal_5034_, v_forward_5035_, v_config_5036_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_);
if (lean_obj_tag(v___x_5118_) == 0)
{
lean_object* v_a_5119_; lean_object* v___x_5120_; lean_object* v_progress_5121_; 
v_a_5119_ = lean_ctor_get(v___x_5118_, 0);
lean_inc(v_a_5119_);
lean_dec_ref_known(v___x_5118_, 1);
v___x_5120_ = lean_st_ref_get(v___y_5043_);
v_progress_5121_ = lean_ctor_get(v___x_5120_, 1);
lean_inc(v_progress_5121_);
lean_dec(v___x_5120_);
if (lean_obj_tag(v_progress_5121_) == 0)
{
uint8_t v___x_5122_; 
lean_dec_ref(v___y_5102_);
v___x_5122_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5082_ = v___x_5122_;
v___y_5083_ = v___y_5104_;
goto v___jp_5081_;
}
else
{
lean_dec(v_progress_5121_);
if (v___x_5037_ == 0)
{
lean_object* v___x_5123_; uint8_t v___x_5124_; 
lean_dec_ref(v___y_5104_);
v___x_5123_ = lean_array_get_size(v___y_5105_);
v___x_5124_ = lean_nat_dec_lt(v___y_5103_, v___x_5123_);
if (v___x_5124_ == 0)
{
lean_object* v___x_5125_; lean_object* v___x_5126_; uint8_t v___x_5127_; 
v___x_5125_ = lean_box(v___x_5037_);
lean_inc(v___y_5049_);
lean_inc_ref(v___y_5048_);
lean_inc(v___y_5047_);
lean_inc_ref(v___y_5046_);
lean_inc(v___y_5045_);
lean_inc_ref(v___y_5044_);
lean_inc(v___y_5043_);
lean_inc_ref(v___y_5042_);
v___x_5126_ = lean_apply_10(v___y_5102_, v___x_5125_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_, lean_box(0));
v___x_5127_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5052_ = v___x_5127_;
v___y_5053_ = v___x_5126_;
goto v___jp_5051_;
}
else
{
if (v___x_5124_ == 0)
{
lean_object* v___x_5128_; lean_object* v___x_5129_; uint8_t v___x_5130_; 
v___x_5128_ = lean_box(v___x_5037_);
lean_inc(v___y_5049_);
lean_inc_ref(v___y_5048_);
lean_inc(v___y_5047_);
lean_inc_ref(v___y_5046_);
lean_inc(v___y_5045_);
lean_inc_ref(v___y_5044_);
lean_inc(v___y_5043_);
lean_inc_ref(v___y_5042_);
v___x_5129_ = lean_apply_10(v___y_5102_, v___x_5128_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_, lean_box(0));
v___x_5130_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5052_ = v___x_5130_;
v___y_5053_ = v___x_5129_;
goto v___jp_5051_;
}
else
{
size_t v___x_5131_; size_t v___x_5132_; lean_object* v___x_5133_; 
v___x_5131_ = ((size_t)0ULL);
v___x_5132_ = lean_usize_of_nat(v___x_5123_);
v___x_5133_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22(v___x_5038_, v___x_5039_, v_lctx_5040_, v___y_5101_, v___y_5105_, v___x_5131_, v___x_5132_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_);
if (lean_obj_tag(v___x_5133_) == 0)
{
lean_object* v_a_5134_; lean_object* v___x_5135_; uint8_t v___x_5136_; 
v_a_5134_ = lean_ctor_get(v___x_5133_, 0);
lean_inc(v_a_5134_);
lean_dec_ref_known(v___x_5133_, 1);
lean_inc(v___y_5049_);
lean_inc_ref(v___y_5048_);
lean_inc(v___y_5047_);
lean_inc_ref(v___y_5046_);
lean_inc(v___y_5045_);
lean_inc_ref(v___y_5044_);
lean_inc(v___y_5043_);
lean_inc_ref(v___y_5042_);
v___x_5135_ = lean_apply_10(v___y_5102_, v_a_5134_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_, v___y_5049_, lean_box(0));
v___x_5136_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5052_ = v___x_5136_;
v___y_5053_ = v___x_5135_;
goto v___jp_5051_;
}
else
{
uint8_t v___x_5137_; 
lean_dec_ref(v___y_5102_);
v___x_5137_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5052_ = v___x_5137_;
v___y_5053_ = v___x_5133_;
goto v___jp_5051_;
}
}
}
}
else
{
uint8_t v___x_5138_; 
lean_dec_ref(v___y_5102_);
v___x_5138_ = lean_unbox(v_a_5119_);
lean_dec(v_a_5119_);
v___y_5082_ = v___x_5138_;
v___y_5083_ = v___y_5104_;
goto v___jp_5081_;
}
}
}
else
{
lean_dec_ref(v___y_5104_);
lean_dec_ref(v___y_5102_);
return v___x_5118_;
}
}
}
}
v___jp_5143_:
{
lean_object* v___x_5145_; lean_object* v_mctx_5146_; lean_object* v_mvarIds_5147_; lean_object* v___x_5148_; lean_object* v___x_5149_; lean_object* v___f_5150_; lean_object* v___x_5151_; lean_object* v___x_5152_; uint8_t v___x_5153_; 
v___x_5145_ = lean_st_ref_get(v___y_5047_);
v_mctx_5146_ = lean_ctor_get(v___x_5145_, 0);
lean_inc_ref(v_mctx_5146_);
lean_dec(v___x_5145_);
v_mvarIds_5147_ = lean_ctor_get(v___y_5042_, 3);
v___x_5148_ = lean_box(v___y_5144_);
v___x_5149_ = lean_box(v___x_5037_);
v___f_5150_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__0___boxed), 12, 2);
lean_closure_set(v___f_5150_, 0, v___x_5148_);
lean_closure_set(v___f_5150_, 1, v___x_5149_);
v___x_5151_ = lean_unsigned_to_nat(0u);
v___x_5152_ = lean_array_get_size(v_mvarIds_5147_);
v___x_5153_ = lean_nat_dec_lt(v___x_5151_, v___x_5152_);
if (v___x_5153_ == 0)
{
lean_inc_ref(v_mctx_5146_);
v___y_5101_ = v___y_5144_;
v___y_5102_ = v___f_5150_;
v___y_5103_ = v___x_5151_;
v___y_5104_ = v_mctx_5146_;
v___y_5105_ = v_mvarIds_5147_;
v___y_5106_ = v_mctx_5146_;
goto v___jp_5100_;
}
else
{
lean_object* v_lctx_5154_; uint8_t v___x_5155_; 
v_lctx_5154_ = lean_ctor_get(v___y_5046_, 2);
v___x_5155_ = lean_nat_dec_le(v___x_5152_, v___x_5152_);
if (v___x_5155_ == 0)
{
if (v___x_5153_ == 0)
{
lean_inc_ref(v_mctx_5146_);
v___y_5101_ = v___y_5144_;
v___y_5102_ = v___f_5150_;
v___y_5103_ = v___x_5151_;
v___y_5104_ = v_mctx_5146_;
v___y_5105_ = v_mvarIds_5147_;
v___y_5106_ = v_mctx_5146_;
goto v___jp_5100_;
}
else
{
size_t v___x_5156_; size_t v___x_5157_; lean_object* v___x_5158_; 
v___x_5156_ = ((size_t)0ULL);
v___x_5157_ = lean_usize_of_nat(v___x_5152_);
lean_inc_ref(v_mctx_5146_);
lean_inc_ref(v_lctx_5154_);
v___x_5158_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24(v_lctx_5154_, v_mvarIds_5147_, v___x_5156_, v___x_5157_, v_mctx_5146_);
v___y_5101_ = v___y_5144_;
v___y_5102_ = v___f_5150_;
v___y_5103_ = v___x_5151_;
v___y_5104_ = v_mctx_5146_;
v___y_5105_ = v_mvarIds_5147_;
v___y_5106_ = v___x_5158_;
goto v___jp_5100_;
}
}
else
{
size_t v___x_5159_; size_t v___x_5160_; lean_object* v___x_5161_; 
v___x_5159_ = ((size_t)0ULL);
v___x_5160_ = lean_usize_of_nat(v___x_5152_);
lean_inc_ref(v_mctx_5146_);
lean_inc_ref(v_lctx_5154_);
v___x_5161_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__24(v_lctx_5154_, v_mvarIds_5147_, v___x_5159_, v___x_5160_, v_mctx_5146_);
v___y_5101_ = v___y_5144_;
v___y_5102_ = v___f_5150_;
v___y_5103_ = v___x_5151_;
v___y_5104_ = v_mctx_5146_;
v___y_5105_ = v_mvarIds_5147_;
v___y_5106_ = v___x_5161_;
goto v___jp_5100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1___boxed(lean_object** _args){
lean_object* v_goal_5164_ = _args[0];
lean_object* v_forward_5165_ = _args[1];
lean_object* v_config_5166_ = _args[2];
lean_object* v___x_5167_ = _args[3];
lean_object* v___x_5168_ = _args[4];
lean_object* v___x_5169_ = _args[5];
lean_object* v_lctx_5170_ = _args[6];
lean_object* v___x_5171_ = _args[7];
lean_object* v___y_5172_ = _args[8];
lean_object* v___y_5173_ = _args[9];
lean_object* v___y_5174_ = _args[10];
lean_object* v___y_5175_ = _args[11];
lean_object* v___y_5176_ = _args[12];
lean_object* v___y_5177_ = _args[13];
lean_object* v___y_5178_ = _args[14];
lean_object* v___y_5179_ = _args[15];
lean_object* v___y_5180_ = _args[16];
_start:
{
uint8_t v_forward_boxed_5181_; uint8_t v___x_418790__boxed_5182_; uint8_t v___x_418793__boxed_5183_; lean_object* v_res_5184_; 
v_forward_boxed_5181_ = lean_unbox(v_forward_5165_);
v___x_418790__boxed_5182_ = lean_unbox(v___x_5167_);
v___x_418793__boxed_5183_ = lean_unbox(v___x_5171_);
v_res_5184_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1(v_goal_5164_, v_forward_boxed_5181_, v_config_5166_, v___x_418790__boxed_5182_, v___x_5168_, v___x_5169_, v_lctx_5170_, v___x_418793__boxed_5183_, v___y_5172_, v___y_5173_, v___y_5174_, v___y_5175_, v___y_5176_, v___y_5177_, v___y_5178_, v___y_5179_);
lean_dec(v___y_5179_);
lean_dec_ref(v___y_5178_);
lean_dec(v___y_5177_);
lean_dec_ref(v___y_5176_);
lean_dec(v___y_5175_);
lean_dec_ref(v___y_5174_);
lean_dec(v___y_5173_);
lean_dec_ref(v___y_5172_);
lean_dec_ref(v_lctx_5170_);
lean_dec(v___x_5169_);
lean_dec(v___x_5168_);
return v_res_5184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(lean_object* v_goal_5185_, uint8_t v_forward_5186_, lean_object* v_config_5187_, lean_object* v_a_5188_, lean_object* v_a_5189_, lean_object* v_a_5190_, lean_object* v_a_5191_, lean_object* v_a_5192_, lean_object* v_a_5193_, lean_object* v_a_5194_, lean_object* v_a_5195_){
_start:
{
lean_object* v___x_5197_; 
lean_inc(v_goal_5185_);
v___x_5197_ = l_Lean_MVarId_getDecl(v_goal_5185_, v_a_5192_, v_a_5193_, v_a_5194_, v_a_5195_);
if (lean_obj_tag(v___x_5197_) == 0)
{
lean_object* v_a_5198_; lean_object* v_lctx_5199_; lean_object* v_lctx_5200_; lean_object* v___x_5201_; lean_object* v___x_5202_; uint8_t v___x_5203_; 
v_a_5198_ = lean_ctor_get(v___x_5197_, 0);
lean_inc(v_a_5198_);
lean_dec_ref_known(v___x_5197_, 1);
v_lctx_5199_ = lean_ctor_get(v_a_5192_, 2);
v_lctx_5200_ = lean_ctor_get(v_a_5198_, 1);
lean_inc_ref(v_lctx_5200_);
lean_dec(v_a_5198_);
v___x_5201_ = lean_local_ctx_num_indices(v_lctx_5200_);
lean_inc_ref(v_lctx_5199_);
v___x_5202_ = lean_local_ctx_num_indices(v_lctx_5199_);
v___x_5203_ = lean_nat_dec_eq(v___x_5201_, v___x_5202_);
if (v___x_5203_ == 0)
{
uint8_t v___x_5204_; lean_object* v___x_5205_; lean_object* v___x_5206_; lean_object* v___x_5207_; lean_object* v___f_5208_; lean_object* v___x_5209_; 
v___x_5204_ = 1;
v___x_5205_ = lean_box(v_forward_5186_);
v___x_5206_ = lean_box(v___x_5203_);
v___x_5207_ = lean_box(v___x_5204_);
lean_inc_ref(v_lctx_5199_);
lean_inc(v_goal_5185_);
v___f_5208_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___lam__1___boxed), 17, 8);
lean_closure_set(v___f_5208_, 0, v_goal_5185_);
lean_closure_set(v___f_5208_, 1, v___x_5205_);
lean_closure_set(v___f_5208_, 2, v_config_5187_);
lean_closure_set(v___f_5208_, 3, v___x_5206_);
lean_closure_set(v___f_5208_, 4, v___x_5201_);
lean_closure_set(v___f_5208_, 5, v___x_5202_);
lean_closure_set(v___f_5208_, 6, v_lctx_5199_);
lean_closure_set(v___f_5208_, 7, v___x_5207_);
v___x_5209_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg(v_goal_5185_, v___f_5208_, v_a_5188_, v_a_5189_, v_a_5190_, v_a_5191_, v_a_5192_, v_a_5193_, v_a_5194_, v_a_5195_);
return v___x_5209_;
}
else
{
lean_object* v___x_5210_; 
lean_dec(v___x_5202_);
lean_dec(v___x_5201_);
v___x_5210_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(v_goal_5185_, v_forward_5186_, v_config_5187_, v_a_5188_, v_a_5189_, v_a_5190_, v_a_5191_, v_a_5192_, v_a_5193_, v_a_5194_, v_a_5195_);
return v___x_5210_;
}
}
else
{
lean_object* v_a_5211_; lean_object* v___x_5213_; uint8_t v_isShared_5214_; uint8_t v_isSharedCheck_5218_; 
lean_dec_ref(v_config_5187_);
lean_dec(v_goal_5185_);
v_a_5211_ = lean_ctor_get(v___x_5197_, 0);
v_isSharedCheck_5218_ = !lean_is_exclusive(v___x_5197_);
if (v_isSharedCheck_5218_ == 0)
{
v___x_5213_ = v___x_5197_;
v_isShared_5214_ = v_isSharedCheck_5218_;
goto v_resetjp_5212_;
}
else
{
lean_inc(v_a_5211_);
lean_dec(v___x_5197_);
v___x_5213_ = lean_box(0);
v_isShared_5214_ = v_isSharedCheck_5218_;
goto v_resetjp_5212_;
}
v_resetjp_5212_:
{
lean_object* v___x_5216_; 
if (v_isShared_5214_ == 0)
{
v___x_5216_ = v___x_5213_;
goto v_reusejp_5215_;
}
else
{
lean_object* v_reuseFailAlloc_5217_; 
v_reuseFailAlloc_5217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5217_, 0, v_a_5211_);
v___x_5216_ = v_reuseFailAlloc_5217_;
goto v_reusejp_5215_;
}
v_reusejp_5215_:
{
return v___x_5216_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis___boxed(lean_object* v_goal_5219_, lean_object* v_forward_5220_, lean_object* v_config_5221_, lean_object* v_a_5222_, lean_object* v_a_5223_, lean_object* v_a_5224_, lean_object* v_a_5225_, lean_object* v_a_5226_, lean_object* v_a_5227_, lean_object* v_a_5228_, lean_object* v_a_5229_, lean_object* v_a_5230_){
_start:
{
uint8_t v_forward_boxed_5231_; lean_object* v_res_5232_; 
v_forward_boxed_5231_ = lean_unbox(v_forward_5220_);
v_res_5232_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis(v_goal_5219_, v_forward_boxed_5231_, v_config_5221_, v_a_5222_, v_a_5223_, v_a_5224_, v_a_5225_, v_a_5226_, v_a_5227_, v_a_5228_, v_a_5229_);
lean_dec(v_a_5229_);
lean_dec_ref(v_a_5228_);
lean_dec(v_a_5227_);
lean_dec_ref(v_a_5226_);
lean_dec(v_a_5225_);
lean_dec_ref(v_a_5224_);
lean_dec(v_a_5223_);
lean_dec_ref(v_a_5222_);
return v_res_5232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux___boxed(lean_object* v_goal_5233_, lean_object* v_forward_5234_, lean_object* v_config_5235_, lean_object* v_a_5236_, lean_object* v_a_5237_, lean_object* v_a_5238_, lean_object* v_a_5239_, lean_object* v_a_5240_, lean_object* v_a_5241_, lean_object* v_a_5242_, lean_object* v_a_5243_, lean_object* v_a_5244_){
_start:
{
uint8_t v_forward_boxed_5245_; lean_object* v_res_5246_; 
v_forward_boxed_5245_ = lean_unbox(v_forward_5234_);
v_res_5246_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux(v_goal_5233_, v_forward_boxed_5245_, v_config_5235_, v_a_5236_, v_a_5237_, v_a_5238_, v_a_5239_, v_a_5240_, v_a_5241_, v_a_5242_, v_a_5243_);
lean_dec(v_a_5243_);
lean_dec_ref(v_a_5242_);
lean_dec(v_a_5241_);
lean_dec_ref(v_a_5240_);
lean_dec(v_a_5239_);
lean_dec_ref(v_a_5238_);
lean_dec(v_a_5237_);
lean_dec_ref(v_a_5236_);
return v_res_5246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg___boxed(lean_object** _args){
lean_object* v_snd_5247_ = _args[0];
lean_object* v_forward_5248_ = _args[1];
lean_object* v_config_5249_ = _args[2];
lean_object* v___x_5250_ = _args[3];
lean_object* v_fst_5251_ = _args[4];
lean_object* v_e_5252_ = _args[5];
lean_object* v_as_x27_5253_ = _args[6];
lean_object* v_b_5254_ = _args[7];
lean_object* v___y_5255_ = _args[8];
lean_object* v___y_5256_ = _args[9];
lean_object* v___y_5257_ = _args[10];
lean_object* v___y_5258_ = _args[11];
lean_object* v___y_5259_ = _args[12];
lean_object* v___y_5260_ = _args[13];
lean_object* v___y_5261_ = _args[14];
lean_object* v___y_5262_ = _args[15];
lean_object* v___y_5263_ = _args[16];
_start:
{
uint8_t v_forward_boxed_5264_; lean_object* v_res_5265_; 
v_forward_boxed_5264_ = lean_unbox(v_forward_5248_);
v_res_5265_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(v_snd_5247_, v_forward_boxed_5264_, v_config_5249_, v___x_5250_, v_fst_5251_, v_e_5252_, v_as_x27_5253_, v_b_5254_, v___y_5255_, v___y_5256_, v___y_5257_, v___y_5258_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_);
lean_dec(v___y_5262_);
lean_dec_ref(v___y_5261_);
lean_dec(v___y_5260_);
lean_dec_ref(v___y_5259_);
lean_dec(v___y_5258_);
lean_dec_ref(v___y_5257_);
lean_dec(v___y_5256_);
lean_dec_ref(v___y_5255_);
lean_dec(v_as_x27_5253_);
return v_res_5265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12___boxed(lean_object* v_config_5266_, lean_object* v_forward_5267_, lean_object* v_as_5268_, lean_object* v_sz_5269_, lean_object* v_i_5270_, lean_object* v_b_5271_, lean_object* v___y_5272_, lean_object* v___y_5273_, lean_object* v___y_5274_, lean_object* v___y_5275_, lean_object* v___y_5276_, lean_object* v___y_5277_, lean_object* v___y_5278_, lean_object* v___y_5279_, lean_object* v___y_5280_){
_start:
{
uint8_t v_forward_boxed_5281_; size_t v_sz_boxed_5282_; size_t v_i_boxed_5283_; lean_object* v_res_5284_; 
v_forward_boxed_5281_ = lean_unbox(v_forward_5267_);
v_sz_boxed_5282_ = lean_unbox_usize(v_sz_5269_);
lean_dec(v_sz_5269_);
v_i_boxed_5283_ = lean_unbox_usize(v_i_5270_);
lean_dec(v_i_5270_);
v_res_5284_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__12(v_config_5266_, v_forward_boxed_5281_, v_as_5268_, v_sz_boxed_5282_, v_i_boxed_5283_, v_b_5271_, v___y_5272_, v___y_5273_, v___y_5274_, v___y_5275_, v___y_5276_, v___y_5277_, v___y_5278_, v___y_5279_);
lean_dec(v___y_5279_);
lean_dec_ref(v___y_5278_);
lean_dec(v___y_5277_);
lean_dec_ref(v___y_5276_);
lean_dec(v___y_5275_);
lean_dec_ref(v___y_5274_);
lean_dec(v___y_5273_);
lean_dec_ref(v___y_5272_);
lean_dec_ref(v_as_5268_);
return v_res_5284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15___boxed(lean_object** _args){
lean_object* v___x_5285_ = _args[0];
lean_object* v_config_5286_ = _args[1];
lean_object* v___x_5287_ = _args[2];
lean_object* v_forward_5288_ = _args[3];
lean_object* v_as_5289_ = _args[4];
lean_object* v_sz_5290_ = _args[5];
lean_object* v_i_5291_ = _args[6];
lean_object* v_b_5292_ = _args[7];
lean_object* v___y_5293_ = _args[8];
lean_object* v___y_5294_ = _args[9];
lean_object* v___y_5295_ = _args[10];
lean_object* v___y_5296_ = _args[11];
lean_object* v___y_5297_ = _args[12];
lean_object* v___y_5298_ = _args[13];
lean_object* v___y_5299_ = _args[14];
lean_object* v___y_5300_ = _args[15];
lean_object* v___y_5301_ = _args[16];
_start:
{
uint8_t v___x_418968__boxed_5302_; uint8_t v___x_418969__boxed_5303_; uint8_t v_forward_boxed_5304_; size_t v_sz_boxed_5305_; size_t v_i_boxed_5306_; lean_object* v_res_5307_; 
v___x_418968__boxed_5302_ = lean_unbox(v___x_5285_);
v___x_418969__boxed_5303_ = lean_unbox(v___x_5287_);
v_forward_boxed_5304_ = lean_unbox(v_forward_5288_);
v_sz_boxed_5305_ = lean_unbox_usize(v_sz_5290_);
lean_dec(v_sz_5290_);
v_i_boxed_5306_ = lean_unbox_usize(v_i_5291_);
lean_dec(v_i_5291_);
v_res_5307_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__15(v___x_418968__boxed_5302_, v_config_5286_, v___x_418969__boxed_5303_, v_forward_boxed_5304_, v_as_5289_, v_sz_boxed_5305_, v_i_boxed_5306_, v_b_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_, v___y_5298_, v___y_5299_, v___y_5300_);
lean_dec(v___y_5300_);
lean_dec_ref(v___y_5299_);
lean_dec(v___y_5298_);
lean_dec_ref(v___y_5297_);
lean_dec(v___y_5296_);
lean_dec_ref(v___y_5295_);
lean_dec(v___y_5294_);
lean_dec_ref(v___y_5293_);
lean_dec_ref(v_as_5289_);
return v_res_5307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17___boxed(lean_object* v_config_5308_, lean_object* v___x_5309_, lean_object* v_forward_5310_, lean_object* v_as_5311_, lean_object* v_sz_5312_, lean_object* v_i_5313_, lean_object* v_b_5314_, lean_object* v___y_5315_, lean_object* v___y_5316_, lean_object* v___y_5317_, lean_object* v___y_5318_, lean_object* v___y_5319_, lean_object* v___y_5320_, lean_object* v___y_5321_, lean_object* v___y_5322_, lean_object* v___y_5323_){
_start:
{
uint8_t v___x_419079__boxed_5324_; uint8_t v_forward_boxed_5325_; size_t v_sz_boxed_5326_; size_t v_i_boxed_5327_; lean_object* v_res_5328_; 
v___x_419079__boxed_5324_ = lean_unbox(v___x_5309_);
v_forward_boxed_5325_ = lean_unbox(v_forward_5310_);
v_sz_boxed_5326_ = lean_unbox_usize(v_sz_5312_);
lean_dec(v_sz_5312_);
v_i_boxed_5327_ = lean_unbox_usize(v_i_5313_);
lean_dec(v_i_5313_);
v_res_5328_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma_spec__17(v_config_5308_, v___x_419079__boxed_5324_, v_forward_boxed_5325_, v_as_5311_, v_sz_boxed_5326_, v_i_boxed_5327_, v_b_5314_, v___y_5315_, v___y_5316_, v___y_5317_, v___y_5318_, v___y_5319_, v___y_5320_, v___y_5321_, v___y_5322_);
lean_dec(v___y_5322_);
lean_dec_ref(v___y_5321_);
lean_dec(v___y_5320_);
lean_dec_ref(v___y_5319_);
lean_dec(v___y_5318_);
lean_dec_ref(v___y_5317_);
lean_dec(v___y_5316_);
lean_dec_ref(v___y_5315_);
lean_dec_ref(v_as_5311_);
return v_res_5328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma___boxed(lean_object* v_goal_5329_, lean_object* v_lem_5330_, lean_object* v_forward_5331_, lean_object* v_config_5332_, lean_object* v_a_5333_, lean_object* v_a_5334_, lean_object* v_a_5335_, lean_object* v_a_5336_, lean_object* v_a_5337_, lean_object* v_a_5338_, lean_object* v_a_5339_, lean_object* v_a_5340_, lean_object* v_a_5341_){
_start:
{
uint8_t v_forward_boxed_5342_; lean_object* v_res_5343_; 
v_forward_boxed_5342_ = lean_unbox(v_forward_5331_);
v_res_5343_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrLemma(v_goal_5329_, v_lem_5330_, v_forward_boxed_5342_, v_config_5332_, v_a_5333_, v_a_5334_, v_a_5335_, v_a_5336_, v_a_5337_, v_a_5338_, v_a_5339_, v_a_5340_);
lean_dec(v_a_5340_);
lean_dec_ref(v_a_5339_);
lean_dec(v_a_5338_);
lean_dec_ref(v_a_5337_);
lean_dec(v_a_5336_);
lean_dec_ref(v_a_5335_);
lean_dec(v_a_5334_);
lean_dec_ref(v_a_5333_);
return v_res_5343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore___boxed(lean_object* v_relName_5344_, lean_object* v_rel_x3f_5345_, lean_object* v_e_5346_, lean_object* v_forward_5347_, lean_object* v_config_5348_, lean_object* v_a_5349_, lean_object* v_a_5350_, lean_object* v_a_5351_, lean_object* v_a_5352_, lean_object* v_a_5353_, lean_object* v_a_5354_, lean_object* v_a_5355_, lean_object* v_a_5356_, lean_object* v_a_5357_){
_start:
{
uint8_t v_forward_boxed_5358_; lean_object* v_res_5359_; 
v_forward_boxed_5358_ = lean_unbox(v_forward_5347_);
v_res_5359_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore(v_relName_5344_, v_rel_x3f_5345_, v_e_5346_, v_forward_boxed_5358_, v_config_5348_, v_a_5349_, v_a_5350_, v_a_5351_, v_a_5352_, v_a_5353_, v_a_5354_, v_a_5355_, v_a_5356_);
lean_dec(v_a_5356_);
lean_dec_ref(v_a_5355_);
lean_dec(v_a_5354_);
lean_dec_ref(v_a_5353_);
lean_dec(v_a_5352_);
lean_dec_ref(v_a_5351_);
lean_dec(v_a_5350_);
lean_dec_ref(v_a_5349_);
return v_res_5359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6(lean_object* v_e_5360_, lean_object* v___y_5361_, lean_object* v___y_5362_, lean_object* v___y_5363_, lean_object* v___y_5364_, lean_object* v___y_5365_, lean_object* v___y_5366_, lean_object* v___y_5367_, lean_object* v___y_5368_){
_start:
{
lean_object* v___x_5370_; 
v___x_5370_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___redArg(v_e_5360_, v___y_5366_);
return v___x_5370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6___boxed(lean_object* v_e_5371_, lean_object* v___y_5372_, lean_object* v___y_5373_, lean_object* v___y_5374_, lean_object* v___y_5375_, lean_object* v___y_5376_, lean_object* v___y_5377_, lean_object* v___y_5378_, lean_object* v___y_5379_, lean_object* v___y_5380_){
_start:
{
lean_object* v_res_5381_; 
v_res_5381_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__6(v_e_5371_, v___y_5372_, v___y_5373_, v___y_5374_, v___y_5375_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
lean_dec(v___y_5379_);
lean_dec_ref(v___y_5378_);
lean_dec(v___y_5377_);
lean_dec_ref(v___y_5376_);
lean_dec(v___y_5375_);
lean_dec_ref(v___y_5374_);
lean_dec(v___y_5373_);
lean_dec_ref(v___y_5372_);
return v_res_5381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9(lean_object* v___y_5382_, lean_object* v___y_5383_, lean_object* v___y_5384_, lean_object* v___y_5385_, lean_object* v___y_5386_, lean_object* v___y_5387_, lean_object* v___y_5388_, lean_object* v___y_5389_){
_start:
{
lean_object* v___x_5391_; 
v___x_5391_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___redArg(v___y_5389_);
return v___x_5391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9___boxed(lean_object* v___y_5392_, lean_object* v___y_5393_, lean_object* v___y_5394_, lean_object* v___y_5395_, lean_object* v___y_5396_, lean_object* v___y_5397_, lean_object* v___y_5398_, lean_object* v___y_5399_, lean_object* v___y_5400_){
_start:
{
lean_object* v_res_5401_; 
v_res_5401_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__9(v___y_5392_, v___y_5393_, v___y_5394_, v___y_5395_, v___y_5396_, v___y_5397_, v___y_5398_, v___y_5399_);
lean_dec(v___y_5399_);
lean_dec_ref(v___y_5398_);
lean_dec(v___y_5397_);
lean_dec_ref(v___y_5396_);
lean_dec(v___y_5395_);
lean_dec_ref(v___y_5394_);
lean_dec(v___y_5393_);
lean_dec_ref(v___y_5392_);
return v_res_5401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20(lean_object* v_mvarId_5402_, lean_object* v___y_5403_, lean_object* v___y_5404_, lean_object* v___y_5405_, lean_object* v___y_5406_, lean_object* v___y_5407_, lean_object* v___y_5408_, lean_object* v___y_5409_, lean_object* v___y_5410_){
_start:
{
lean_object* v___x_5412_; 
v___x_5412_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___redArg(v_mvarId_5402_, v___y_5408_);
return v___x_5412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20___boxed(lean_object* v_mvarId_5413_, lean_object* v___y_5414_, lean_object* v___y_5415_, lean_object* v___y_5416_, lean_object* v___y_5417_, lean_object* v___y_5418_, lean_object* v___y_5419_, lean_object* v___y_5420_, lean_object* v___y_5421_, lean_object* v___y_5422_){
_start:
{
lean_object* v_res_5423_; 
v_res_5423_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__20(v_mvarId_5413_, v___y_5414_, v___y_5415_, v___y_5416_, v___y_5417_, v___y_5418_, v___y_5419_, v___y_5420_, v___y_5421_);
lean_dec(v___y_5421_);
lean_dec_ref(v___y_5420_);
lean_dec(v___y_5419_);
lean_dec_ref(v___y_5418_);
lean_dec(v___y_5417_);
lean_dec_ref(v___y_5416_);
lean_dec(v___y_5415_);
lean_dec_ref(v___y_5414_);
lean_dec(v_mvarId_5413_);
return v_res_5423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25(lean_object* v_00_u03b1_5424_, lean_object* v_mvarId_5425_, lean_object* v_x_5426_, lean_object* v___y_5427_, lean_object* v___y_5428_, lean_object* v___y_5429_, lean_object* v___y_5430_, lean_object* v___y_5431_, lean_object* v___y_5432_, lean_object* v___y_5433_, lean_object* v___y_5434_){
_start:
{
lean_object* v___x_5436_; 
v___x_5436_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___redArg(v_mvarId_5425_, v_x_5426_, v___y_5427_, v___y_5428_, v___y_5429_, v___y_5430_, v___y_5431_, v___y_5432_, v___y_5433_, v___y_5434_);
return v___x_5436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25___boxed(lean_object* v_00_u03b1_5437_, lean_object* v_mvarId_5438_, lean_object* v_x_5439_, lean_object* v___y_5440_, lean_object* v___y_5441_, lean_object* v___y_5442_, lean_object* v___y_5443_, lean_object* v___y_5444_, lean_object* v___y_5445_, lean_object* v___y_5446_, lean_object* v___y_5447_, lean_object* v___y_5448_){
_start:
{
lean_object* v_res_5449_; 
v_res_5449_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__25(v_00_u03b1_5437_, v_mvarId_5438_, v_x_5439_, v___y_5440_, v___y_5441_, v___y_5442_, v___y_5443_, v___y_5444_, v___y_5445_, v___y_5446_, v___y_5447_);
lean_dec(v___y_5447_);
lean_dec_ref(v___y_5446_);
lean_dec(v___y_5445_);
lean_dec_ref(v___y_5444_);
lean_dec(v___y_5443_);
lean_dec_ref(v___y_5442_);
lean_dec(v___y_5441_);
lean_dec_ref(v___y_5440_);
return v_res_5449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0(lean_object* v_mvarId_5450_, lean_object* v_val_5451_, lean_object* v___y_5452_, lean_object* v___y_5453_, lean_object* v___y_5454_, lean_object* v___y_5455_, lean_object* v___y_5456_, lean_object* v___y_5457_, lean_object* v___y_5458_, lean_object* v___y_5459_){
_start:
{
lean_object* v___x_5461_; 
v___x_5461_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___redArg(v_mvarId_5450_, v_val_5451_, v___y_5457_);
return v___x_5461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0___boxed(lean_object* v_mvarId_5462_, lean_object* v_val_5463_, lean_object* v___y_5464_, lean_object* v___y_5465_, lean_object* v___y_5466_, lean_object* v___y_5467_, lean_object* v___y_5468_, lean_object* v___y_5469_, lean_object* v___y_5470_, lean_object* v___y_5471_, lean_object* v___y_5472_){
_start:
{
lean_object* v_res_5473_; 
v_res_5473_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__0(v_mvarId_5462_, v_val_5463_, v___y_5464_, v___y_5465_, v___y_5466_, v___y_5467_, v___y_5468_, v___y_5469_, v___y_5470_, v___y_5471_);
lean_dec(v___y_5471_);
lean_dec_ref(v___y_5470_);
lean_dec(v___y_5469_);
lean_dec_ref(v___y_5468_);
lean_dec(v___y_5467_);
lean_dec_ref(v___y_5466_);
lean_dec(v___y_5465_);
lean_dec_ref(v___y_5464_);
return v_res_5473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2(lean_object* v_00_u03b1_5474_, lean_object* v_msg_5475_, lean_object* v___y_5476_, lean_object* v___y_5477_, lean_object* v___y_5478_, lean_object* v___y_5479_, lean_object* v___y_5480_, lean_object* v___y_5481_, lean_object* v___y_5482_, lean_object* v___y_5483_){
_start:
{
lean_object* v___x_5485_; 
v___x_5485_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___redArg(v_msg_5475_, v___y_5480_, v___y_5481_, v___y_5482_, v___y_5483_);
return v___x_5485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2___boxed(lean_object* v_00_u03b1_5486_, lean_object* v_msg_5487_, lean_object* v___y_5488_, lean_object* v___y_5489_, lean_object* v___y_5490_, lean_object* v___y_5491_, lean_object* v___y_5492_, lean_object* v___y_5493_, lean_object* v___y_5494_, lean_object* v___y_5495_, lean_object* v___y_5496_){
_start:
{
lean_object* v_res_5497_; 
v_res_5497_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesisAux_spec__2(v_00_u03b1_5486_, v_msg_5487_, v___y_5488_, v___y_5489_, v___y_5490_, v___y_5491_, v___y_5492_, v___y_5493_, v___y_5494_, v___y_5495_);
lean_dec(v___y_5495_);
lean_dec_ref(v___y_5494_);
lean_dec(v___y_5493_);
lean_dec_ref(v___y_5492_);
lean_dec(v___y_5491_);
lean_dec_ref(v___y_5490_);
lean_dec(v___y_5489_);
lean_dec_ref(v___y_5488_);
return v_res_5497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4(lean_object* v_00_u03b2_5498_, lean_object* v_m_5499_, lean_object* v_a_5500_, lean_object* v_b_5501_){
_start:
{
lean_object* v___x_5502_; 
v___x_5502_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4___redArg(v_m_5499_, v_a_5500_, v_b_5501_);
return v___x_5502_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5(lean_object* v_00_u03b2_5503_, lean_object* v_m_5504_, lean_object* v_a_5505_){
_start:
{
uint8_t v___x_5506_; 
v___x_5506_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___redArg(v_m_5504_, v_a_5505_);
return v___x_5506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5___boxed(lean_object* v_00_u03b2_5507_, lean_object* v_m_5508_, lean_object* v_a_5509_){
_start:
{
uint8_t v_res_5510_; lean_object* v_r_5511_; 
v_res_5510_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__5(v_00_u03b2_5507_, v_m_5508_, v_a_5509_);
lean_dec_ref(v_a_5509_);
lean_dec_ref(v_m_5508_);
v_r_5511_ = lean_box(v_res_5510_);
return v_r_5511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7(lean_object* v_snd_5512_, uint8_t v_forward_5513_, lean_object* v_config_5514_, lean_object* v___x_5515_, lean_object* v_fst_5516_, lean_object* v_e_5517_, lean_object* v_as_5518_, lean_object* v_as_x27_5519_, lean_object* v_b_5520_, lean_object* v_a_5521_, lean_object* v___y_5522_, lean_object* v___y_5523_, lean_object* v___y_5524_, lean_object* v___y_5525_, lean_object* v___y_5526_, lean_object* v___y_5527_, lean_object* v___y_5528_, lean_object* v___y_5529_){
_start:
{
lean_object* v___x_5531_; 
v___x_5531_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___redArg(v_snd_5512_, v_forward_5513_, v_config_5514_, v___x_5515_, v_fst_5516_, v_e_5517_, v_as_x27_5519_, v_b_5520_, v___y_5522_, v___y_5523_, v___y_5524_, v___y_5525_, v___y_5526_, v___y_5527_, v___y_5528_, v___y_5529_);
return v___x_5531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7___boxed(lean_object** _args){
lean_object* v_snd_5532_ = _args[0];
lean_object* v_forward_5533_ = _args[1];
lean_object* v_config_5534_ = _args[2];
lean_object* v___x_5535_ = _args[3];
lean_object* v_fst_5536_ = _args[4];
lean_object* v_e_5537_ = _args[5];
lean_object* v_as_5538_ = _args[6];
lean_object* v_as_x27_5539_ = _args[7];
lean_object* v_b_5540_ = _args[8];
lean_object* v_a_5541_ = _args[9];
lean_object* v___y_5542_ = _args[10];
lean_object* v___y_5543_ = _args[11];
lean_object* v___y_5544_ = _args[12];
lean_object* v___y_5545_ = _args[13];
lean_object* v___y_5546_ = _args[14];
lean_object* v___y_5547_ = _args[15];
lean_object* v___y_5548_ = _args[16];
lean_object* v___y_5549_ = _args[17];
lean_object* v___y_5550_ = _args[18];
_start:
{
uint8_t v_forward_boxed_5551_; lean_object* v_res_5552_; 
v_forward_boxed_5551_ = lean_unbox(v_forward_5533_);
v_res_5552_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__7(v_snd_5532_, v_forward_boxed_5551_, v_config_5534_, v___x_5535_, v_fst_5536_, v_e_5537_, v_as_5538_, v_as_x27_5539_, v_b_5540_, v_a_5541_, v___y_5542_, v___y_5543_, v___y_5544_, v___y_5545_, v___y_5546_, v___y_5547_, v___y_5548_, v___y_5549_);
lean_dec(v___y_5549_);
lean_dec_ref(v___y_5548_);
lean_dec(v___y_5547_);
lean_dec_ref(v___y_5546_);
lean_dec(v___y_5545_);
lean_dec_ref(v___y_5544_);
lean_dec(v___y_5543_);
lean_dec_ref(v___y_5542_);
lean_dec(v_as_x27_5539_);
lean_dec(v_as_5538_);
return v_res_5552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8(lean_object* v_cls_5553_, lean_object* v_msg_5554_, lean_object* v___y_5555_, lean_object* v___y_5556_, lean_object* v___y_5557_, lean_object* v___y_5558_, lean_object* v___y_5559_, lean_object* v___y_5560_, lean_object* v___y_5561_, lean_object* v___y_5562_){
_start:
{
lean_object* v___x_5564_; 
v___x_5564_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___redArg(v_cls_5553_, v_msg_5554_, v___y_5559_, v___y_5560_, v___y_5561_, v___y_5562_);
return v___x_5564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8___boxed(lean_object* v_cls_5565_, lean_object* v_msg_5566_, lean_object* v___y_5567_, lean_object* v___y_5568_, lean_object* v___y_5569_, lean_object* v___y_5570_, lean_object* v___y_5571_, lean_object* v___y_5572_, lean_object* v___y_5573_, lean_object* v___y_5574_, lean_object* v___y_5575_){
_start:
{
lean_object* v_res_5576_; 
v_res_5576_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__8(v_cls_5565_, v_msg_5566_, v___y_5567_, v___y_5568_, v___y_5569_, v___y_5570_, v___y_5571_, v___y_5572_, v___y_5573_, v___y_5574_);
lean_dec(v___y_5574_);
lean_dec_ref(v___y_5573_);
lean_dec(v___y_5572_);
lean_dec_ref(v___y_5571_);
lean_dec(v___y_5570_);
lean_dec_ref(v___y_5569_);
lean_dec(v___y_5568_);
lean_dec_ref(v___y_5567_);
return v_res_5576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13(lean_object* v_00_u03b1_5577_, lean_object* v_x_5578_, lean_object* v___y_5579_, lean_object* v___y_5580_, lean_object* v___y_5581_, lean_object* v___y_5582_, lean_object* v___y_5583_, lean_object* v___y_5584_, lean_object* v___y_5585_, lean_object* v___y_5586_){
_start:
{
lean_object* v___x_5588_; 
v___x_5588_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___redArg(v_x_5578_);
return v___x_5588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13___boxed(lean_object* v_00_u03b1_5589_, lean_object* v_x_5590_, lean_object* v___y_5591_, lean_object* v___y_5592_, lean_object* v___y_5593_, lean_object* v___y_5594_, lean_object* v___y_5595_, lean_object* v___y_5596_, lean_object* v___y_5597_, lean_object* v___y_5598_, lean_object* v___y_5599_){
_start:
{
lean_object* v_res_5600_; 
v_res_5600_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__13(v_00_u03b1_5589_, v_x_5590_, v___y_5591_, v___y_5592_, v___y_5593_, v___y_5594_, v___y_5595_, v___y_5596_, v___y_5597_, v___y_5598_);
lean_dec(v___y_5598_);
lean_dec_ref(v___y_5597_);
lean_dec(v___y_5596_);
lean_dec_ref(v___y_5595_);
lean_dec(v___y_5594_);
lean_dec_ref(v___y_5593_);
lean_dec(v___y_5592_);
lean_dec_ref(v___y_5591_);
return v_res_5600_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4(lean_object* v_00_u03b2_5601_, lean_object* v_a_5602_, lean_object* v_x_5603_){
_start:
{
uint8_t v___x_5604_; 
v___x_5604_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___redArg(v_a_5602_, v_x_5603_);
return v___x_5604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4___boxed(lean_object* v_00_u03b2_5605_, lean_object* v_a_5606_, lean_object* v_x_5607_){
_start:
{
uint8_t v_res_5608_; lean_object* v_r_5609_; 
v_res_5608_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__4(v_00_u03b2_5605_, v_a_5606_, v_x_5607_);
lean_dec(v_x_5607_);
lean_dec_ref(v_a_5606_);
v_r_5609_ = lean_box(v_res_5608_);
return v_r_5609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5(lean_object* v_00_u03b2_5610_, lean_object* v_data_5611_){
_start:
{
lean_object* v___x_5612_; 
v___x_5612_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5___redArg(v_data_5611_);
return v___x_5612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12(lean_object* v_oldTraces_5613_, lean_object* v_data_5614_, lean_object* v_ref_5615_, lean_object* v_msg_5616_, lean_object* v___y_5617_, lean_object* v___y_5618_, lean_object* v___y_5619_, lean_object* v___y_5620_, lean_object* v___y_5621_, lean_object* v___y_5622_, lean_object* v___y_5623_, lean_object* v___y_5624_){
_start:
{
lean_object* v___x_5626_; 
v___x_5626_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___redArg(v_oldTraces_5613_, v_data_5614_, v_ref_5615_, v_msg_5616_, v___y_5621_, v___y_5622_, v___y_5623_, v___y_5624_);
return v___x_5626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12___boxed(lean_object* v_oldTraces_5627_, lean_object* v_data_5628_, lean_object* v_ref_5629_, lean_object* v_msg_5630_, lean_object* v___y_5631_, lean_object* v___y_5632_, lean_object* v___y_5633_, lean_object* v___y_5634_, lean_object* v___y_5635_, lean_object* v___y_5636_, lean_object* v___y_5637_, lean_object* v___y_5638_, lean_object* v___y_5639_){
_start:
{
lean_object* v_res_5640_; 
v_res_5640_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__10_spec__12(v_oldTraces_5627_, v_data_5628_, v_ref_5629_, v_msg_5630_, v___y_5631_, v___y_5632_, v___y_5633_, v___y_5634_, v___y_5635_, v___y_5636_, v___y_5637_, v___y_5638_);
lean_dec(v___y_5638_);
lean_dec_ref(v___y_5637_);
lean_dec(v___y_5636_);
lean_dec_ref(v___y_5635_);
lean_dec(v___y_5634_);
lean_dec_ref(v___y_5633_);
lean_dec(v___y_5632_);
lean_dec_ref(v___y_5631_);
return v_res_5640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13(lean_object* v_00_u03b2_5641_, lean_object* v_i_5642_, lean_object* v_source_5643_, lean_object* v_target_5644_){
_start:
{
lean_object* v___x_5645_; 
v___x_5645_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13___redArg(v_i_5642_, v_source_5643_, v_target_5644_);
return v___x_5645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31(lean_object* v_00_u03b2_5646_, lean_object* v_x_5647_, lean_object* v_x_5648_){
_start:
{
lean_object* v___x_5649_; 
v___x_5649_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore_spec__4_spec__5_spec__13_spec__31___redArg(v_x_5647_, v_x_5648_);
return v___x_5649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg(lean_object* v_mvarId_5650_, lean_object* v_x_5651_, lean_object* v___y_5652_, lean_object* v___y_5653_, lean_object* v___y_5654_, lean_object* v___y_5655_){
_start:
{
lean_object* v___x_5657_; 
v___x_5657_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_5650_, v_x_5651_, v___y_5652_, v___y_5653_, v___y_5654_, v___y_5655_);
if (lean_obj_tag(v___x_5657_) == 0)
{
lean_object* v_a_5658_; lean_object* v___x_5660_; uint8_t v_isShared_5661_; uint8_t v_isSharedCheck_5665_; 
v_a_5658_ = lean_ctor_get(v___x_5657_, 0);
v_isSharedCheck_5665_ = !lean_is_exclusive(v___x_5657_);
if (v_isSharedCheck_5665_ == 0)
{
v___x_5660_ = v___x_5657_;
v_isShared_5661_ = v_isSharedCheck_5665_;
goto v_resetjp_5659_;
}
else
{
lean_inc(v_a_5658_);
lean_dec(v___x_5657_);
v___x_5660_ = lean_box(0);
v_isShared_5661_ = v_isSharedCheck_5665_;
goto v_resetjp_5659_;
}
v_resetjp_5659_:
{
lean_object* v___x_5663_; 
if (v_isShared_5661_ == 0)
{
v___x_5663_ = v___x_5660_;
goto v_reusejp_5662_;
}
else
{
lean_object* v_reuseFailAlloc_5664_; 
v_reuseFailAlloc_5664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5664_, 0, v_a_5658_);
v___x_5663_ = v_reuseFailAlloc_5664_;
goto v_reusejp_5662_;
}
v_reusejp_5662_:
{
return v___x_5663_;
}
}
}
else
{
lean_object* v_a_5666_; lean_object* v___x_5668_; uint8_t v_isShared_5669_; uint8_t v_isSharedCheck_5673_; 
v_a_5666_ = lean_ctor_get(v___x_5657_, 0);
v_isSharedCheck_5673_ = !lean_is_exclusive(v___x_5657_);
if (v_isSharedCheck_5673_ == 0)
{
v___x_5668_ = v___x_5657_;
v_isShared_5669_ = v_isSharedCheck_5673_;
goto v_resetjp_5667_;
}
else
{
lean_inc(v_a_5666_);
lean_dec(v___x_5657_);
v___x_5668_ = lean_box(0);
v_isShared_5669_ = v_isSharedCheck_5673_;
goto v_resetjp_5667_;
}
v_resetjp_5667_:
{
lean_object* v___x_5671_; 
if (v_isShared_5669_ == 0)
{
v___x_5671_ = v___x_5668_;
goto v_reusejp_5670_;
}
else
{
lean_object* v_reuseFailAlloc_5672_; 
v_reuseFailAlloc_5672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5672_, 0, v_a_5666_);
v___x_5671_ = v_reuseFailAlloc_5672_;
goto v_reusejp_5670_;
}
v_reusejp_5670_:
{
return v___x_5671_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg___boxed(lean_object* v_mvarId_5674_, lean_object* v_x_5675_, lean_object* v___y_5676_, lean_object* v___y_5677_, lean_object* v___y_5678_, lean_object* v___y_5679_, lean_object* v___y_5680_){
_start:
{
lean_object* v_res_5681_; 
v_res_5681_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg(v_mvarId_5674_, v_x_5675_, v___y_5676_, v___y_5677_, v___y_5678_, v___y_5679_);
lean_dec(v___y_5679_);
lean_dec_ref(v___y_5678_);
lean_dec(v___y_5677_);
lean_dec_ref(v___y_5676_);
return v_res_5681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7(lean_object* v_00_u03b1_5682_, lean_object* v_mvarId_5683_, lean_object* v_x_5684_, lean_object* v___y_5685_, lean_object* v___y_5686_, lean_object* v___y_5687_, lean_object* v___y_5688_){
_start:
{
lean_object* v___x_5690_; 
v___x_5690_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg(v_mvarId_5683_, v_x_5684_, v___y_5685_, v___y_5686_, v___y_5687_, v___y_5688_);
return v___x_5690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___boxed(lean_object* v_00_u03b1_5691_, lean_object* v_mvarId_5692_, lean_object* v_x_5693_, lean_object* v___y_5694_, lean_object* v___y_5695_, lean_object* v___y_5696_, lean_object* v___y_5697_, lean_object* v___y_5698_){
_start:
{
lean_object* v_res_5699_; 
v_res_5699_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7(v_00_u03b1_5691_, v_mvarId_5692_, v_x_5693_, v___y_5694_, v___y_5695_, v___y_5696_, v___y_5697_);
lean_dec(v___y_5697_);
lean_dec_ref(v___y_5696_);
lean_dec(v___y_5695_);
lean_dec_ref(v___y_5694_);
return v_res_5699_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1(void){
_start:
{
lean_object* v___x_5701_; lean_object* v___x_5702_; 
v___x_5701_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__0));
v___x_5702_ = l_Lean_stringToMessageData(v___x_5701_);
return v___x_5702_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3(void){
_start:
{
lean_object* v___x_5704_; lean_object* v___x_5705_; 
v___x_5704_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__2));
v___x_5705_ = l_Lean_stringToMessageData(v___x_5704_);
return v___x_5705_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5(void){
_start:
{
lean_object* v___x_5707_; lean_object* v___x_5708_; 
v___x_5707_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__4));
v___x_5708_ = l_Lean_stringToMessageData(v___x_5707_);
return v___x_5708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0(lean_object* v___x_5709_, uint8_t v_symm_5710_, lean_object* v_a_5711_, lean_object* v___x_5712_, lean_object* v_goal_5713_, lean_object* v_replacement_x27_5714_, lean_object* v___y_5715_, lean_object* v___y_5716_, lean_object* v___y_5717_, lean_object* v___y_5718_){
_start:
{
lean_object* v___x_5720_; lean_object* v___x_5721_; lean_object* v___x_5722_; lean_object* v___x_5723_; lean_object* v___x_5724_; lean_object* v___x_5725_; lean_object* v___x_5726_; lean_object* v___x_5727_; lean_object* v___x_5728_; lean_object* v___x_5729_; lean_object* v___x_5730_; lean_object* v___x_5731_; 
v___x_5720_ = lp_mathlib_Mathlib_Tactic_GCongr_updateRel(v___x_5709_, v_replacement_x27_5714_, v_symm_5710_);
v___x_5721_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1, &lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__1);
v___x_5722_ = l_Lean_indentExpr(v___x_5720_);
v___x_5723_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5723_, 0, v___x_5721_);
lean_ctor_set(v___x_5723_, 1, v___x_5722_);
v___x_5724_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3, &lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__3);
v___x_5725_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5725_, 0, v___x_5723_);
lean_ctor_set(v___x_5725_, 1, v___x_5724_);
v___x_5726_ = l_Lean_indentExpr(v_a_5711_);
v___x_5727_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5727_, 0, v___x_5725_);
lean_ctor_set(v___x_5727_, 1, v___x_5726_);
v___x_5728_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5, &lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__0___closed__5);
v___x_5729_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5729_, 0, v___x_5727_);
lean_ctor_set(v___x_5729_, 1, v___x_5728_);
v___x_5730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5730_, 0, v___x_5729_);
v___x_5731_ = l_Lean_Meta_throwTacticEx___redArg(v___x_5712_, v_goal_5713_, v___x_5730_, v___y_5715_, v___y_5716_, v___y_5717_, v___y_5718_);
return v___x_5731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__0___boxed(lean_object* v___x_5732_, lean_object* v_symm_5733_, lean_object* v_a_5734_, lean_object* v___x_5735_, lean_object* v_goal_5736_, lean_object* v_replacement_x27_5737_, lean_object* v___y_5738_, lean_object* v___y_5739_, lean_object* v___y_5740_, lean_object* v___y_5741_, lean_object* v___y_5742_){
_start:
{
uint8_t v_symm_boxed_5743_; lean_object* v_res_5744_; 
v_symm_boxed_5743_ = lean_unbox(v_symm_5733_);
v_res_5744_ = lp_mathlib_Lean_MVarId_grewrite___lam__0(v___x_5732_, v_symm_boxed_5743_, v_a_5734_, v___x_5735_, v_goal_5736_, v_replacement_x27_5737_, v___y_5738_, v___y_5739_, v___y_5740_, v___y_5741_);
lean_dec(v___y_5741_);
lean_dec_ref(v___y_5740_);
lean_dec(v___y_5739_);
lean_dec_ref(v___y_5738_);
return v_res_5744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__1(lean_object* v___x_5745_, lean_object* v___x_5746_, lean_object* v___x_5747_, lean_object* v_a_5748_, uint8_t v_forwardImp_5749_, lean_object* v_config_5750_, lean_object* v___x_5751_, lean_object* v___y_5752_, lean_object* v___y_5753_, lean_object* v___y_5754_, lean_object* v___y_5755_, lean_object* v___y_5756_, lean_object* v___y_5757_){
_start:
{
lean_object* v___x_5759_; lean_object* v___x_5760_; 
v___x_5759_ = lean_st_mk_ref(v___x_5745_);
v___x_5760_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteCore(v___x_5746_, v___x_5747_, v_a_5748_, v_forwardImp_5749_, v_config_5750_, v___x_5751_, v___x_5759_, v___y_5752_, v___y_5753_, v___y_5754_, v___y_5755_, v___y_5756_, v___y_5757_);
if (lean_obj_tag(v___x_5760_) == 0)
{
lean_object* v_a_5761_; lean_object* v___x_5763_; uint8_t v_isShared_5764_; uint8_t v_isSharedCheck_5770_; 
v_a_5761_ = lean_ctor_get(v___x_5760_, 0);
v_isSharedCheck_5770_ = !lean_is_exclusive(v___x_5760_);
if (v_isSharedCheck_5770_ == 0)
{
v___x_5763_ = v___x_5760_;
v_isShared_5764_ = v_isSharedCheck_5770_;
goto v_resetjp_5762_;
}
else
{
lean_inc(v_a_5761_);
lean_dec(v___x_5760_);
v___x_5763_ = lean_box(0);
v_isShared_5764_ = v_isSharedCheck_5770_;
goto v_resetjp_5762_;
}
v_resetjp_5762_:
{
lean_object* v___x_5765_; lean_object* v___x_5766_; lean_object* v___x_5768_; 
v___x_5765_ = lean_st_ref_get(v___x_5759_);
lean_dec(v___x_5759_);
v___x_5766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5766_, 0, v_a_5761_);
lean_ctor_set(v___x_5766_, 1, v___x_5765_);
if (v_isShared_5764_ == 0)
{
lean_ctor_set(v___x_5763_, 0, v___x_5766_);
v___x_5768_ = v___x_5763_;
goto v_reusejp_5767_;
}
else
{
lean_object* v_reuseFailAlloc_5769_; 
v_reuseFailAlloc_5769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5769_, 0, v___x_5766_);
v___x_5768_ = v_reuseFailAlloc_5769_;
goto v_reusejp_5767_;
}
v_reusejp_5767_:
{
return v___x_5768_;
}
}
}
else
{
lean_object* v_a_5771_; lean_object* v___x_5773_; uint8_t v_isShared_5774_; uint8_t v_isSharedCheck_5778_; 
lean_dec(v___x_5759_);
v_a_5771_ = lean_ctor_get(v___x_5760_, 0);
v_isSharedCheck_5778_ = !lean_is_exclusive(v___x_5760_);
if (v_isSharedCheck_5778_ == 0)
{
v___x_5773_ = v___x_5760_;
v_isShared_5774_ = v_isSharedCheck_5778_;
goto v_resetjp_5772_;
}
else
{
lean_inc(v_a_5771_);
lean_dec(v___x_5760_);
v___x_5773_ = lean_box(0);
v_isShared_5774_ = v_isSharedCheck_5778_;
goto v_resetjp_5772_;
}
v_resetjp_5772_:
{
lean_object* v___x_5776_; 
if (v_isShared_5774_ == 0)
{
v___x_5776_ = v___x_5773_;
goto v_reusejp_5775_;
}
else
{
lean_object* v_reuseFailAlloc_5777_; 
v_reuseFailAlloc_5777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5777_, 0, v_a_5771_);
v___x_5776_ = v_reuseFailAlloc_5777_;
goto v_reusejp_5775_;
}
v_reusejp_5775_:
{
return v___x_5776_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__1___boxed(lean_object* v___x_5779_, lean_object* v___x_5780_, lean_object* v___x_5781_, lean_object* v_a_5782_, lean_object* v_forwardImp_5783_, lean_object* v_config_5784_, lean_object* v___x_5785_, lean_object* v___y_5786_, lean_object* v___y_5787_, lean_object* v___y_5788_, lean_object* v___y_5789_, lean_object* v___y_5790_, lean_object* v___y_5791_, lean_object* v___y_5792_){
_start:
{
uint8_t v_forwardImp_boxed_5793_; lean_object* v_res_5794_; 
v_forwardImp_boxed_5793_ = lean_unbox(v_forwardImp_5783_);
v_res_5794_ = lp_mathlib_Lean_MVarId_grewrite___lam__1(v___x_5779_, v___x_5780_, v___x_5781_, v_a_5782_, v_forwardImp_boxed_5793_, v_config_5784_, v___x_5785_, v___y_5786_, v___y_5787_, v___y_5788_, v___y_5789_, v___y_5790_, v___y_5791_);
lean_dec(v___y_5791_);
lean_dec_ref(v___y_5790_);
lean_dec(v___y_5789_);
lean_dec_ref(v___y_5788_);
lean_dec(v___y_5787_);
lean_dec_ref(v___y_5786_);
lean_dec_ref(v___x_5785_);
return v_res_5794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0(lean_object* v_k_5795_, lean_object* v_b_5796_, lean_object* v___y_5797_, lean_object* v___y_5798_, lean_object* v___y_5799_, lean_object* v___y_5800_){
_start:
{
lean_object* v___x_5802_; 
lean_inc(v___y_5800_);
lean_inc_ref(v___y_5799_);
lean_inc(v___y_5798_);
lean_inc_ref(v___y_5797_);
v___x_5802_ = lean_apply_6(v_k_5795_, v_b_5796_, v___y_5797_, v___y_5798_, v___y_5799_, v___y_5800_, lean_box(0));
return v___x_5802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0___boxed(lean_object* v_k_5803_, lean_object* v_b_5804_, lean_object* v___y_5805_, lean_object* v___y_5806_, lean_object* v___y_5807_, lean_object* v___y_5808_, lean_object* v___y_5809_){
_start:
{
lean_object* v_res_5810_; 
v_res_5810_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0(v_k_5803_, v_b_5804_, v___y_5805_, v___y_5806_, v___y_5807_, v___y_5808_);
lean_dec(v___y_5808_);
lean_dec_ref(v___y_5807_);
lean_dec(v___y_5806_);
lean_dec_ref(v___y_5805_);
return v_res_5810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg(lean_object* v_name_5811_, uint8_t v_bi_5812_, lean_object* v_type_5813_, lean_object* v_k_5814_, uint8_t v_kind_5815_, lean_object* v___y_5816_, lean_object* v___y_5817_, lean_object* v___y_5818_, lean_object* v___y_5819_){
_start:
{
lean_object* v___f_5821_; lean_object* v___x_5822_; 
v___f_5821_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_5821_, 0, v_k_5814_);
v___x_5822_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_5811_, v_bi_5812_, v_type_5813_, v___f_5821_, v_kind_5815_, v___y_5816_, v___y_5817_, v___y_5818_, v___y_5819_);
if (lean_obj_tag(v___x_5822_) == 0)
{
lean_object* v_a_5823_; lean_object* v___x_5825_; uint8_t v_isShared_5826_; uint8_t v_isSharedCheck_5830_; 
v_a_5823_ = lean_ctor_get(v___x_5822_, 0);
v_isSharedCheck_5830_ = !lean_is_exclusive(v___x_5822_);
if (v_isSharedCheck_5830_ == 0)
{
v___x_5825_ = v___x_5822_;
v_isShared_5826_ = v_isSharedCheck_5830_;
goto v_resetjp_5824_;
}
else
{
lean_inc(v_a_5823_);
lean_dec(v___x_5822_);
v___x_5825_ = lean_box(0);
v_isShared_5826_ = v_isSharedCheck_5830_;
goto v_resetjp_5824_;
}
v_resetjp_5824_:
{
lean_object* v___x_5828_; 
if (v_isShared_5826_ == 0)
{
v___x_5828_ = v___x_5825_;
goto v_reusejp_5827_;
}
else
{
lean_object* v_reuseFailAlloc_5829_; 
v_reuseFailAlloc_5829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5829_, 0, v_a_5823_);
v___x_5828_ = v_reuseFailAlloc_5829_;
goto v_reusejp_5827_;
}
v_reusejp_5827_:
{
return v___x_5828_;
}
}
}
else
{
lean_object* v_a_5831_; lean_object* v___x_5833_; uint8_t v_isShared_5834_; uint8_t v_isSharedCheck_5838_; 
v_a_5831_ = lean_ctor_get(v___x_5822_, 0);
v_isSharedCheck_5838_ = !lean_is_exclusive(v___x_5822_);
if (v_isSharedCheck_5838_ == 0)
{
v___x_5833_ = v___x_5822_;
v_isShared_5834_ = v_isSharedCheck_5838_;
goto v_resetjp_5832_;
}
else
{
lean_inc(v_a_5831_);
lean_dec(v___x_5822_);
v___x_5833_ = lean_box(0);
v_isShared_5834_ = v_isSharedCheck_5838_;
goto v_resetjp_5832_;
}
v_resetjp_5832_:
{
lean_object* v___x_5836_; 
if (v_isShared_5834_ == 0)
{
v___x_5836_ = v___x_5833_;
goto v_reusejp_5835_;
}
else
{
lean_object* v_reuseFailAlloc_5837_; 
v_reuseFailAlloc_5837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5837_, 0, v_a_5831_);
v___x_5836_ = v_reuseFailAlloc_5837_;
goto v_reusejp_5835_;
}
v_reusejp_5835_:
{
return v___x_5836_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg___boxed(lean_object* v_name_5839_, lean_object* v_bi_5840_, lean_object* v_type_5841_, lean_object* v_k_5842_, lean_object* v_kind_5843_, lean_object* v___y_5844_, lean_object* v___y_5845_, lean_object* v___y_5846_, lean_object* v___y_5847_, lean_object* v___y_5848_){
_start:
{
uint8_t v_bi_boxed_5849_; uint8_t v_kind_boxed_5850_; lean_object* v_res_5851_; 
v_bi_boxed_5849_ = lean_unbox(v_bi_5840_);
v_kind_boxed_5850_ = lean_unbox(v_kind_5843_);
v_res_5851_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg(v_name_5839_, v_bi_boxed_5849_, v_type_5841_, v_k_5842_, v_kind_boxed_5850_, v___y_5844_, v___y_5845_, v___y_5846_, v___y_5847_);
lean_dec(v___y_5847_);
lean_dec_ref(v___y_5846_);
lean_dec(v___y_5845_);
lean_dec_ref(v___y_5844_);
return v_res_5851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg(lean_object* v_name_5852_, lean_object* v_type_5853_, lean_object* v_k_5854_, lean_object* v___y_5855_, lean_object* v___y_5856_, lean_object* v___y_5857_, lean_object* v___y_5858_){
_start:
{
uint8_t v___x_5860_; uint8_t v___x_5861_; lean_object* v___x_5862_; 
v___x_5860_ = 0;
v___x_5861_ = 0;
v___x_5862_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg(v_name_5852_, v___x_5860_, v_type_5853_, v_k_5854_, v___x_5861_, v___y_5855_, v___y_5856_, v___y_5857_, v___y_5858_);
return v___x_5862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg___boxed(lean_object* v_name_5863_, lean_object* v_type_5864_, lean_object* v_k_5865_, lean_object* v___y_5866_, lean_object* v___y_5867_, lean_object* v___y_5868_, lean_object* v___y_5869_, lean_object* v___y_5870_){
_start:
{
lean_object* v_res_5871_; 
v_res_5871_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg(v_name_5863_, v_type_5864_, v_k_5865_, v___y_5866_, v___y_5867_, v___y_5868_, v___y_5869_);
lean_dec(v___y_5869_);
lean_dec_ref(v___y_5868_);
lean_dec(v___y_5867_);
lean_dec_ref(v___y_5866_);
return v_res_5871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5(size_t v_sz_5874_, size_t v_i_5875_, lean_object* v_bs_5876_){
_start:
{
uint8_t v___x_5877_; 
v___x_5877_ = lean_usize_dec_lt(v_i_5875_, v_sz_5874_);
if (v___x_5877_ == 0)
{
return v_bs_5876_;
}
else
{
lean_object* v_v_5878_; lean_object* v___x_5879_; lean_object* v_bs_x27_5880_; lean_object* v___x_5881_; lean_object* v___x_5882_; lean_object* v___x_5883_; size_t v___x_5884_; size_t v___x_5885_; lean_object* v___x_5886_; 
v_v_5878_ = lean_array_uget(v_bs_5876_, v_i_5875_);
v___x_5879_ = lean_unsigned_to_nat(0u);
v_bs_x27_5880_ = lean_array_uset(v_bs_5876_, v_i_5875_, v___x_5879_);
v___x_5881_ = l_Lean_Expr_mvarId_x21(v_v_5878_);
lean_dec(v_v_5878_);
v___x_5882_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___closed__0));
v___x_5883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5883_, 0, v___x_5881_);
lean_ctor_set(v___x_5883_, 1, v___x_5882_);
v___x_5884_ = ((size_t)1ULL);
v___x_5885_ = lean_usize_add(v_i_5875_, v___x_5884_);
v___x_5886_ = lean_array_uset(v_bs_x27_5880_, v_i_5875_, v___x_5883_);
v_i_5875_ = v___x_5885_;
v_bs_5876_ = v___x_5886_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5___boxed(lean_object* v_sz_5888_, lean_object* v_i_5889_, lean_object* v_bs_5890_){
_start:
{
size_t v_sz_boxed_5891_; size_t v_i_boxed_5892_; lean_object* v_res_5893_; 
v_sz_boxed_5891_ = lean_unbox_usize(v_sz_5888_);
lean_dec(v_sz_5888_);
v_i_boxed_5892_ = lean_unbox_usize(v_i_5889_);
lean_dec(v_i_5889_);
v_res_5893_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5(v_sz_boxed_5891_, v_i_boxed_5892_, v_bs_5890_);
return v_res_5893_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0(void){
_start:
{
lean_object* v___x_5894_; lean_object* v___x_5895_; lean_object* v___x_5896_; 
v___x_5894_ = lean_box(0);
v___x_5895_ = lean_unsigned_to_nat(16u);
v___x_5896_ = lean_mk_array(v___x_5895_, v___x_5894_);
return v___x_5896_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1(void){
_start:
{
lean_object* v___x_5897_; lean_object* v___x_5898_; lean_object* v___x_5899_; 
v___x_5897_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0, &lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__0);
v___x_5898_ = lean_unsigned_to_nat(0u);
v___x_5899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5899_, 0, v___x_5898_);
lean_ctor_set(v___x_5899_, 1, v___x_5897_);
return v___x_5899_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2(void){
_start:
{
lean_object* v___x_5900_; lean_object* v___x_5901_; lean_object* v___x_5902_; 
v___x_5900_ = lean_box(0);
v___x_5901_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1, &lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__1);
v___x_5902_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5902_, 0, v___x_5901_);
lean_ctor_set(v___x_5902_, 1, v___x_5900_);
return v___x_5902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2(lean_object* v_fst_5907_, lean_object* v_fst_5908_, lean_object* v_mvarIds_5909_, lean_object* v___x_5910_, lean_object* v___x_5911_, lean_object* v_a_5912_, uint8_t v_forwardImp_5913_, lean_object* v_config_5914_, lean_object* v_snd_5915_, lean_object* v___f_5916_, uint8_t v_symm_x27_5917_, lean_object* v___y_5918_, lean_object* v___y_5919_, lean_object* v___y_5920_, lean_object* v___y_5921_){
_start:
{
lean_object* v___x_5923_; lean_object* v___x_5924_; lean_object* v___x_5925_; size_t v_sz_5926_; size_t v___x_5927_; lean_object* v___x_5928_; lean_object* v___x_5929_; lean_object* v___x_5930_; lean_object* v___x_5931_; lean_object* v___x_5932_; lean_object* v___x_5933_; lean_object* v___x_5934_; lean_object* v___f_5935_; lean_object* v___x_5936_; lean_object* v___x_5937_; lean_object* v___x_5938_; lean_object* v___x_5939_; 
lean_inc_ref(v_fst_5907_);
v___x_5923_ = l_Lean_Expr_toHeadIndex(v_fst_5907_);
v___x_5924_ = l_Lean_Expr_headNumArgs(v_fst_5907_);
lean_dec_ref(v_fst_5907_);
v___x_5925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5925_, 0, v___x_5923_);
lean_ctor_set(v___x_5925_, 1, v___x_5924_);
v_sz_5926_ = lean_array_size(v_fst_5908_);
v___x_5927_ = ((size_t)0ULL);
v___x_5928_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__5(v_sz_5926_, v___x_5927_, v_fst_5908_);
v___x_5929_ = l_Array_append___redArg(v_mvarIds_5909_, v___x_5928_);
lean_dec_ref(v___x_5928_);
v___x_5930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_getRel_x27___closed__1));
v___x_5931_ = lean_box(0);
v___x_5932_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_5932_, 0, v___x_5910_);
lean_ctor_set(v___x_5932_, 1, v___x_5911_);
lean_ctor_set(v___x_5932_, 2, v___x_5925_);
lean_ctor_set(v___x_5932_, 3, v___x_5929_);
lean_ctor_set_uint8(v___x_5932_, sizeof(void*)*4, v_symm_x27_5917_);
v___x_5933_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2, &lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__2);
v___x_5934_ = lean_box(v_forwardImp_5913_);
v___f_5935_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_grewrite___lam__1___boxed), 14, 7);
lean_closure_set(v___f_5935_, 0, v___x_5933_);
lean_closure_set(v___f_5935_, 1, v___x_5930_);
lean_closure_set(v___f_5935_, 2, v___x_5931_);
lean_closure_set(v___f_5935_, 3, v_a_5912_);
lean_closure_set(v___f_5935_, 4, v___x_5934_);
lean_closure_set(v___f_5935_, 5, v_config_5914_);
lean_closure_set(v___f_5935_, 6, v___x_5932_);
v___x_5936_ = lean_box(0);
v___x_5937_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__3));
v___x_5938_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract___closed__0));
v___x_5939_ = lp_mathlib_Mathlib_Tactic_GCongr_GCongrM_run___redArg(v___f_5935_, v___x_5936_, v___x_5937_, v___x_5938_, v___y_5918_, v___y_5919_, v___y_5920_, v___y_5921_);
if (lean_obj_tag(v___x_5939_) == 0)
{
lean_object* v_a_5940_; lean_object* v___x_5942_; uint8_t v_isShared_5943_; uint8_t v_isSharedCheck_6001_; 
v_a_5940_ = lean_ctor_get(v___x_5939_, 0);
v_isSharedCheck_6001_ = !lean_is_exclusive(v___x_5939_);
if (v_isSharedCheck_6001_ == 0)
{
v___x_5942_ = v___x_5939_;
v_isShared_5943_ = v_isSharedCheck_6001_;
goto v_resetjp_5941_;
}
else
{
lean_inc(v_a_5940_);
lean_dec(v___x_5939_);
v___x_5942_ = lean_box(0);
v_isShared_5943_ = v_isSharedCheck_6001_;
goto v_resetjp_5941_;
}
v_resetjp_5941_:
{
lean_object* v_fst_5944_; lean_object* v_fst_5945_; 
v_fst_5944_ = lean_ctor_get(v_a_5940_, 0);
lean_inc(v_fst_5944_);
v_fst_5945_ = lean_ctor_get(v_fst_5944_, 0);
lean_inc(v_fst_5945_);
if (lean_obj_tag(v_fst_5945_) == 1)
{
lean_object* v_val_5946_; lean_object* v___x_5948_; uint8_t v_isShared_5949_; uint8_t v_isSharedCheck_5988_; 
lean_dec_ref(v___f_5916_);
lean_dec_ref(v_snd_5915_);
v_val_5946_ = lean_ctor_get(v_fst_5945_, 0);
v_isSharedCheck_5988_ = !lean_is_exclusive(v_fst_5945_);
if (v_isSharedCheck_5988_ == 0)
{
v___x_5948_ = v_fst_5945_;
v_isShared_5949_ = v_isSharedCheck_5988_;
goto v_resetjp_5947_;
}
else
{
lean_inc(v_val_5946_);
lean_dec(v_fst_5945_);
v___x_5948_ = lean_box(0);
v_isShared_5949_ = v_isSharedCheck_5988_;
goto v_resetjp_5947_;
}
v_resetjp_5947_:
{
lean_object* v_snd_5950_; lean_object* v___x_5952_; uint8_t v_isShared_5953_; uint8_t v_isSharedCheck_5986_; 
v_snd_5950_ = lean_ctor_get(v_a_5940_, 1);
v_isSharedCheck_5986_ = !lean_is_exclusive(v_a_5940_);
if (v_isSharedCheck_5986_ == 0)
{
lean_object* v_unused_5987_; 
v_unused_5987_ = lean_ctor_get(v_a_5940_, 0);
lean_dec(v_unused_5987_);
v___x_5952_ = v_a_5940_;
v_isShared_5953_ = v_isSharedCheck_5986_;
goto v_resetjp_5951_;
}
else
{
lean_inc(v_snd_5950_);
lean_dec(v_a_5940_);
v___x_5952_ = lean_box(0);
v_isShared_5953_ = v_isSharedCheck_5986_;
goto v_resetjp_5951_;
}
v_resetjp_5951_:
{
lean_object* v_snd_5954_; lean_object* v___x_5956_; uint8_t v_isShared_5957_; uint8_t v_isSharedCheck_5984_; 
v_snd_5954_ = lean_ctor_get(v_fst_5944_, 1);
v_isSharedCheck_5984_ = !lean_is_exclusive(v_fst_5944_);
if (v_isSharedCheck_5984_ == 0)
{
lean_object* v_unused_5985_; 
v_unused_5985_ = lean_ctor_get(v_fst_5944_, 0);
lean_dec(v_unused_5985_);
v___x_5956_ = v_fst_5944_;
v_isShared_5957_ = v_isSharedCheck_5984_;
goto v_resetjp_5955_;
}
else
{
lean_inc(v_snd_5954_);
lean_dec(v_fst_5944_);
v___x_5956_ = lean_box(0);
v_isShared_5957_ = v_isSharedCheck_5984_;
goto v_resetjp_5955_;
}
v_resetjp_5955_:
{
lean_object* v_fst_5958_; lean_object* v_snd_5959_; lean_object* v___x_5961_; uint8_t v_isShared_5962_; uint8_t v_isSharedCheck_5983_; 
v_fst_5958_ = lean_ctor_get(v_val_5946_, 0);
v_snd_5959_ = lean_ctor_get(v_val_5946_, 1);
v_isSharedCheck_5983_ = !lean_is_exclusive(v_val_5946_);
if (v_isSharedCheck_5983_ == 0)
{
v___x_5961_ = v_val_5946_;
v_isShared_5962_ = v_isSharedCheck_5983_;
goto v_resetjp_5960_;
}
else
{
lean_inc(v_snd_5959_);
lean_inc(v_fst_5958_);
lean_dec(v_val_5946_);
v___x_5961_ = lean_box(0);
v_isShared_5962_ = v_isSharedCheck_5983_;
goto v_resetjp_5960_;
}
v_resetjp_5960_:
{
lean_object* v___y_5964_; lean_object* v_progress_5978_; 
v_progress_5978_ = lean_ctor_get(v_snd_5954_, 1);
lean_inc(v_progress_5978_);
lean_dec(v_snd_5954_);
if (lean_obj_tag(v_progress_5978_) == 2)
{
lean_object* v_lctx_5979_; lean_object* v___x_5981_; 
v_lctx_5979_ = lean_ctor_get(v_progress_5978_, 0);
lean_inc_ref(v_lctx_5979_);
lean_dec_ref_known(v_progress_5978_, 1);
if (v_isShared_5949_ == 0)
{
lean_ctor_set(v___x_5948_, 0, v_lctx_5979_);
v___x_5981_ = v___x_5948_;
goto v_reusejp_5980_;
}
else
{
lean_object* v_reuseFailAlloc_5982_; 
v_reuseFailAlloc_5982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5982_, 0, v_lctx_5979_);
v___x_5981_ = v_reuseFailAlloc_5982_;
goto v_reusejp_5980_;
}
v_reusejp_5980_:
{
v___y_5964_ = v___x_5981_;
goto v___jp_5963_;
}
}
else
{
lean_dec(v_progress_5978_);
lean_del_object(v___x_5948_);
v___y_5964_ = v___x_5931_;
goto v___jp_5963_;
}
v___jp_5963_:
{
lean_object* v_newGoals_5965_; lean_object* v___x_5967_; 
v_newGoals_5965_ = lean_ctor_get(v_snd_5950_, 0);
lean_inc_ref(v_newGoals_5965_);
lean_dec(v_snd_5950_);
if (v_isShared_5962_ == 0)
{
lean_ctor_set(v___x_5961_, 1, v_newGoals_5965_);
lean_ctor_set(v___x_5961_, 0, v_snd_5959_);
v___x_5967_ = v___x_5961_;
goto v_reusejp_5966_;
}
else
{
lean_object* v_reuseFailAlloc_5977_; 
v_reuseFailAlloc_5977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5977_, 0, v_snd_5959_);
lean_ctor_set(v_reuseFailAlloc_5977_, 1, v_newGoals_5965_);
v___x_5967_ = v_reuseFailAlloc_5977_;
goto v_reusejp_5966_;
}
v_reusejp_5966_:
{
lean_object* v___x_5969_; 
if (v_isShared_5957_ == 0)
{
lean_ctor_set(v___x_5956_, 1, v___x_5967_);
lean_ctor_set(v___x_5956_, 0, v_fst_5958_);
v___x_5969_ = v___x_5956_;
goto v_reusejp_5968_;
}
else
{
lean_object* v_reuseFailAlloc_5976_; 
v_reuseFailAlloc_5976_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5976_, 0, v_fst_5958_);
lean_ctor_set(v_reuseFailAlloc_5976_, 1, v___x_5967_);
v___x_5969_ = v_reuseFailAlloc_5976_;
goto v_reusejp_5968_;
}
v_reusejp_5968_:
{
lean_object* v___x_5971_; 
if (v_isShared_5953_ == 0)
{
lean_ctor_set(v___x_5952_, 1, v___x_5969_);
lean_ctor_set(v___x_5952_, 0, v___y_5964_);
v___x_5971_ = v___x_5952_;
goto v_reusejp_5970_;
}
else
{
lean_object* v_reuseFailAlloc_5975_; 
v_reuseFailAlloc_5975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5975_, 0, v___y_5964_);
lean_ctor_set(v_reuseFailAlloc_5975_, 1, v___x_5969_);
v___x_5971_ = v_reuseFailAlloc_5975_;
goto v_reusejp_5970_;
}
v_reusejp_5970_:
{
lean_object* v___x_5973_; 
if (v_isShared_5943_ == 0)
{
lean_ctor_set(v___x_5942_, 0, v___x_5971_);
v___x_5973_ = v___x_5942_;
goto v_reusejp_5972_;
}
else
{
lean_object* v_reuseFailAlloc_5974_; 
v_reuseFailAlloc_5974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5974_, 0, v___x_5971_);
v___x_5973_ = v_reuseFailAlloc_5974_;
goto v_reusejp_5972_;
}
v_reusejp_5972_:
{
return v___x_5973_;
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
else
{
lean_object* v___x_5989_; 
lean_dec(v_fst_5945_);
lean_dec(v_fst_5944_);
lean_del_object(v___x_5942_);
lean_dec(v_a_5940_);
lean_inc(v___y_5921_);
lean_inc_ref(v___y_5920_);
lean_inc(v___y_5919_);
lean_inc_ref(v___y_5918_);
v___x_5989_ = lean_infer_type(v_snd_5915_, v___y_5918_, v___y_5919_, v___y_5920_, v___y_5921_);
if (lean_obj_tag(v___x_5989_) == 0)
{
lean_object* v_a_5990_; lean_object* v___x_5991_; lean_object* v___x_5992_; 
v_a_5990_ = lean_ctor_get(v___x_5989_, 0);
lean_inc(v_a_5990_);
lean_dec_ref_known(v___x_5989_, 1);
v___x_5991_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__2___closed__5));
v___x_5992_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg(v___x_5991_, v_a_5990_, v___f_5916_, v___y_5918_, v___y_5919_, v___y_5920_, v___y_5921_);
return v___x_5992_;
}
else
{
lean_object* v_a_5993_; lean_object* v___x_5995_; uint8_t v_isShared_5996_; uint8_t v_isSharedCheck_6000_; 
lean_dec_ref(v___f_5916_);
v_a_5993_ = lean_ctor_get(v___x_5989_, 0);
v_isSharedCheck_6000_ = !lean_is_exclusive(v___x_5989_);
if (v_isSharedCheck_6000_ == 0)
{
v___x_5995_ = v___x_5989_;
v_isShared_5996_ = v_isSharedCheck_6000_;
goto v_resetjp_5994_;
}
else
{
lean_inc(v_a_5993_);
lean_dec(v___x_5989_);
v___x_5995_ = lean_box(0);
v_isShared_5996_ = v_isSharedCheck_6000_;
goto v_resetjp_5994_;
}
v_resetjp_5994_:
{
lean_object* v___x_5998_; 
if (v_isShared_5996_ == 0)
{
v___x_5998_ = v___x_5995_;
goto v_reusejp_5997_;
}
else
{
lean_object* v_reuseFailAlloc_5999_; 
v_reuseFailAlloc_5999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5999_, 0, v_a_5993_);
v___x_5998_ = v_reuseFailAlloc_5999_;
goto v_reusejp_5997_;
}
v_reusejp_5997_:
{
return v___x_5998_;
}
}
}
}
}
}
else
{
lean_object* v_a_6002_; lean_object* v___x_6004_; uint8_t v_isShared_6005_; uint8_t v_isSharedCheck_6009_; 
lean_dec_ref(v___f_5916_);
lean_dec_ref(v_snd_5915_);
v_a_6002_ = lean_ctor_get(v___x_5939_, 0);
v_isSharedCheck_6009_ = !lean_is_exclusive(v___x_5939_);
if (v_isSharedCheck_6009_ == 0)
{
v___x_6004_ = v___x_5939_;
v_isShared_6005_ = v_isSharedCheck_6009_;
goto v_resetjp_6003_;
}
else
{
lean_inc(v_a_6002_);
lean_dec(v___x_5939_);
v___x_6004_ = lean_box(0);
v_isShared_6005_ = v_isSharedCheck_6009_;
goto v_resetjp_6003_;
}
v_resetjp_6003_:
{
lean_object* v___x_6007_; 
if (v_isShared_6005_ == 0)
{
v___x_6007_ = v___x_6004_;
goto v_reusejp_6006_;
}
else
{
lean_object* v_reuseFailAlloc_6008_; 
v_reuseFailAlloc_6008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6008_, 0, v_a_6002_);
v___x_6007_ = v_reuseFailAlloc_6008_;
goto v_reusejp_6006_;
}
v_reusejp_6006_:
{
return v___x_6007_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__2___boxed(lean_object* v_fst_6010_, lean_object* v_fst_6011_, lean_object* v_mvarIds_6012_, lean_object* v___x_6013_, lean_object* v___x_6014_, lean_object* v_a_6015_, lean_object* v_forwardImp_6016_, lean_object* v_config_6017_, lean_object* v_snd_6018_, lean_object* v___f_6019_, lean_object* v_symm_x27_6020_, lean_object* v___y_6021_, lean_object* v___y_6022_, lean_object* v___y_6023_, lean_object* v___y_6024_, lean_object* v___y_6025_){
_start:
{
uint8_t v_forwardImp_boxed_6026_; uint8_t v_symm_x27_boxed_6027_; lean_object* v_res_6028_; 
v_forwardImp_boxed_6026_ = lean_unbox(v_forwardImp_6016_);
v_symm_x27_boxed_6027_ = lean_unbox(v_symm_x27_6020_);
v_res_6028_ = lp_mathlib_Lean_MVarId_grewrite___lam__2(v_fst_6010_, v_fst_6011_, v_mvarIds_6012_, v___x_6013_, v___x_6014_, v_a_6015_, v_forwardImp_boxed_6026_, v_config_6017_, v_snd_6018_, v___f_6019_, v_symm_x27_boxed_6027_, v___y_6021_, v___y_6022_, v___y_6023_, v___y_6024_);
lean_dec(v___y_6024_);
lean_dec_ref(v___y_6023_);
lean_dec(v___y_6022_);
lean_dec_ref(v___y_6021_);
return v_res_6028_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg(lean_object* v_keys_6029_, lean_object* v_i_6030_, lean_object* v_k_6031_){
_start:
{
lean_object* v___x_6032_; uint8_t v___x_6033_; 
v___x_6032_ = lean_array_get_size(v_keys_6029_);
v___x_6033_ = lean_nat_dec_lt(v_i_6030_, v___x_6032_);
if (v___x_6033_ == 0)
{
lean_dec(v_i_6030_);
return v___x_6033_;
}
else
{
lean_object* v_k_x27_6034_; uint8_t v___x_6035_; 
v_k_x27_6034_ = lean_array_fget_borrowed(v_keys_6029_, v_i_6030_);
v___x_6035_ = l_Lean_instBEqMVarId_beq(v_k_6031_, v_k_x27_6034_);
if (v___x_6035_ == 0)
{
lean_object* v___x_6036_; lean_object* v___x_6037_; 
v___x_6036_ = lean_unsigned_to_nat(1u);
v___x_6037_ = lean_nat_add(v_i_6030_, v___x_6036_);
lean_dec(v_i_6030_);
v_i_6030_ = v___x_6037_;
goto _start;
}
else
{
lean_dec(v_i_6030_);
return v___x_6035_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg___boxed(lean_object* v_keys_6039_, lean_object* v_i_6040_, lean_object* v_k_6041_){
_start:
{
uint8_t v_res_6042_; lean_object* v_r_6043_; 
v_res_6042_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg(v_keys_6039_, v_i_6040_, v_k_6041_);
lean_dec(v_k_6041_);
lean_dec_ref(v_keys_6039_);
v_r_6043_ = lean_box(v_res_6042_);
return v_r_6043_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg(lean_object* v_x_6044_, size_t v_x_6045_, lean_object* v_x_6046_){
_start:
{
if (lean_obj_tag(v_x_6044_) == 0)
{
lean_object* v_es_6047_; lean_object* v___x_6048_; size_t v___x_6049_; size_t v___x_6050_; lean_object* v_j_6051_; lean_object* v___x_6052_; 
v_es_6047_ = lean_ctor_get(v_x_6044_, 0);
v___x_6048_ = lean_box(2);
v___x_6049_ = ((size_t)31ULL);
v___x_6050_ = lean_usize_land(v_x_6045_, v___x_6049_);
v_j_6051_ = lean_usize_to_nat(v___x_6050_);
v___x_6052_ = lean_array_get_borrowed(v___x_6048_, v_es_6047_, v_j_6051_);
lean_dec(v_j_6051_);
switch(lean_obj_tag(v___x_6052_))
{
case 0:
{
lean_object* v_key_6053_; uint8_t v___x_6054_; 
v_key_6053_ = lean_ctor_get(v___x_6052_, 0);
v___x_6054_ = l_Lean_instBEqMVarId_beq(v_x_6046_, v_key_6053_);
return v___x_6054_;
}
case 1:
{
lean_object* v_node_6055_; size_t v___x_6056_; size_t v___x_6057_; 
v_node_6055_ = lean_ctor_get(v___x_6052_, 0);
v___x_6056_ = ((size_t)5ULL);
v___x_6057_ = lean_usize_shift_right(v_x_6045_, v___x_6056_);
v_x_6044_ = v_node_6055_;
v_x_6045_ = v___x_6057_;
goto _start;
}
default: 
{
uint8_t v___x_6059_; 
v___x_6059_ = 0;
return v___x_6059_;
}
}
}
else
{
lean_object* v_ks_6060_; lean_object* v___x_6061_; uint8_t v___x_6062_; 
v_ks_6060_ = lean_ctor_get(v_x_6044_, 0);
v___x_6061_ = lean_unsigned_to_nat(0u);
v___x_6062_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg(v_ks_6060_, v___x_6061_, v_x_6046_);
return v___x_6062_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_x_6063_, lean_object* v_x_6064_, lean_object* v_x_6065_){
_start:
{
size_t v_x_18807__boxed_6066_; uint8_t v_res_6067_; lean_object* v_r_6068_; 
v_x_18807__boxed_6066_ = lean_unbox_usize(v_x_6064_);
lean_dec(v_x_6064_);
v_res_6067_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg(v_x_6063_, v_x_18807__boxed_6066_, v_x_6065_);
lean_dec(v_x_6065_);
lean_dec_ref(v_x_6063_);
v_r_6068_ = lean_box(v_res_6067_);
return v_r_6068_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg(lean_object* v_x_6069_, lean_object* v_x_6070_){
_start:
{
uint64_t v___x_6071_; size_t v___x_6072_; uint8_t v___x_6073_; 
v___x_6071_ = l_Lean_instHashableMVarId_hash(v_x_6070_);
v___x_6072_ = lean_uint64_to_usize(v___x_6071_);
v___x_6073_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg(v_x_6069_, v___x_6072_, v_x_6070_);
return v___x_6073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg___boxed(lean_object* v_x_6074_, lean_object* v_x_6075_){
_start:
{
uint8_t v_res_6076_; lean_object* v_r_6077_; 
v_res_6076_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg(v_x_6074_, v_x_6075_);
lean_dec(v_x_6075_);
lean_dec_ref(v_x_6074_);
v_r_6077_ = lean_box(v_res_6076_);
return v_r_6077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg(lean_object* v_mvarId_6078_, lean_object* v___y_6079_){
_start:
{
lean_object* v___x_6081_; lean_object* v_mctx_6082_; lean_object* v_eAssignment_6083_; uint8_t v___x_6084_; lean_object* v___x_6085_; lean_object* v___x_6086_; 
v___x_6081_ = lean_st_ref_get(v___y_6079_);
v_mctx_6082_ = lean_ctor_get(v___x_6081_, 0);
lean_inc_ref(v_mctx_6082_);
lean_dec(v___x_6081_);
v_eAssignment_6083_ = lean_ctor_get(v_mctx_6082_, 8);
lean_inc_ref(v_eAssignment_6083_);
lean_dec_ref(v_mctx_6082_);
v___x_6084_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg(v_eAssignment_6083_, v_mvarId_6078_);
lean_dec_ref(v_eAssignment_6083_);
v___x_6085_ = lean_box(v___x_6084_);
v___x_6086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6086_, 0, v___x_6085_);
return v___x_6086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg___boxed(lean_object* v_mvarId_6087_, lean_object* v___y_6088_, lean_object* v___y_6089_){
_start:
{
lean_object* v_res_6090_; 
v_res_6090_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg(v_mvarId_6087_, v___y_6088_);
lean_dec(v___y_6088_);
lean_dec(v_mvarId_6087_);
return v_res_6090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4(lean_object* v_as_6091_, size_t v_i_6092_, size_t v_stop_6093_, lean_object* v_b_6094_, lean_object* v___y_6095_, lean_object* v___y_6096_, lean_object* v___y_6097_, lean_object* v___y_6098_){
_start:
{
lean_object* v_a_6101_; uint8_t v___x_6105_; 
v___x_6105_ = lean_usize_dec_eq(v_i_6092_, v_stop_6093_);
if (v___x_6105_ == 0)
{
lean_object* v___x_6106_; lean_object* v___x_6109_; 
v___x_6106_ = lean_array_uget_borrowed(v_as_6091_, v_i_6092_);
v___x_6109_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg(v___x_6106_, v___y_6096_);
if (lean_obj_tag(v___x_6109_) == 0)
{
lean_object* v_a_6110_; uint8_t v___x_6111_; 
v_a_6110_ = lean_ctor_get(v___x_6109_, 0);
lean_inc(v_a_6110_);
lean_dec_ref_known(v___x_6109_, 1);
v___x_6111_ = lean_unbox(v_a_6110_);
lean_dec(v_a_6110_);
if (v___x_6111_ == 0)
{
goto v___jp_6107_;
}
else
{
v_a_6101_ = v_b_6094_;
goto v___jp_6100_;
}
}
else
{
if (lean_obj_tag(v___x_6109_) == 0)
{
lean_object* v_a_6112_; uint8_t v___x_6113_; 
v_a_6112_ = lean_ctor_get(v___x_6109_, 0);
lean_inc(v_a_6112_);
lean_dec_ref_known(v___x_6109_, 1);
v___x_6113_ = lean_unbox(v_a_6112_);
lean_dec(v_a_6112_);
if (v___x_6113_ == 0)
{
v_a_6101_ = v_b_6094_;
goto v___jp_6100_;
}
else
{
goto v___jp_6107_;
}
}
else
{
lean_object* v_a_6114_; lean_object* v___x_6116_; uint8_t v_isShared_6117_; uint8_t v_isSharedCheck_6121_; 
lean_dec_ref(v_b_6094_);
v_a_6114_ = lean_ctor_get(v___x_6109_, 0);
v_isSharedCheck_6121_ = !lean_is_exclusive(v___x_6109_);
if (v_isSharedCheck_6121_ == 0)
{
v___x_6116_ = v___x_6109_;
v_isShared_6117_ = v_isSharedCheck_6121_;
goto v_resetjp_6115_;
}
else
{
lean_inc(v_a_6114_);
lean_dec(v___x_6109_);
v___x_6116_ = lean_box(0);
v_isShared_6117_ = v_isSharedCheck_6121_;
goto v_resetjp_6115_;
}
v_resetjp_6115_:
{
lean_object* v___x_6119_; 
if (v_isShared_6117_ == 0)
{
v___x_6119_ = v___x_6116_;
goto v_reusejp_6118_;
}
else
{
lean_object* v_reuseFailAlloc_6120_; 
v_reuseFailAlloc_6120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6120_, 0, v_a_6114_);
v___x_6119_ = v_reuseFailAlloc_6120_;
goto v_reusejp_6118_;
}
v_reusejp_6118_:
{
return v___x_6119_;
}
}
}
}
v___jp_6107_:
{
lean_object* v___x_6108_; 
lean_inc(v___x_6106_);
v___x_6108_ = lean_array_push(v_b_6094_, v___x_6106_);
v_a_6101_ = v___x_6108_;
goto v___jp_6100_;
}
}
else
{
lean_object* v___x_6122_; 
v___x_6122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6122_, 0, v_b_6094_);
return v___x_6122_;
}
v___jp_6100_:
{
size_t v___x_6102_; size_t v___x_6103_; 
v___x_6102_ = ((size_t)1ULL);
v___x_6103_ = lean_usize_add(v_i_6092_, v___x_6102_);
v_i_6092_ = v___x_6103_;
v_b_6094_ = v_a_6101_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4___boxed(lean_object* v_as_6123_, lean_object* v_i_6124_, lean_object* v_stop_6125_, lean_object* v_b_6126_, lean_object* v___y_6127_, lean_object* v___y_6128_, lean_object* v___y_6129_, lean_object* v___y_6130_, lean_object* v___y_6131_){
_start:
{
size_t v_i_boxed_6132_; size_t v_stop_boxed_6133_; lean_object* v_res_6134_; 
v_i_boxed_6132_ = lean_unbox_usize(v_i_6124_);
lean_dec(v_i_6124_);
v_stop_boxed_6133_ = lean_unbox_usize(v_stop_6125_);
lean_dec(v_stop_6125_);
v_res_6134_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4(v_as_6123_, v_i_boxed_6132_, v_stop_boxed_6133_, v_b_6126_, v___y_6127_, v___y_6128_, v___y_6129_, v___y_6130_);
lean_dec(v___y_6130_);
lean_dec_ref(v___y_6129_);
lean_dec(v___y_6128_);
lean_dec_ref(v___y_6127_);
lean_dec_ref(v_as_6123_);
return v_res_6134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1(size_t v_sz_6135_, size_t v_i_6136_, lean_object* v_bs_6137_){
_start:
{
uint8_t v___x_6138_; 
v___x_6138_ = lean_usize_dec_lt(v_i_6136_, v_sz_6135_);
if (v___x_6138_ == 0)
{
return v_bs_6137_;
}
else
{
lean_object* v_v_6139_; lean_object* v___x_6140_; lean_object* v_bs_x27_6141_; lean_object* v___x_6142_; size_t v___x_6143_; size_t v___x_6144_; lean_object* v___x_6145_; 
v_v_6139_ = lean_array_uget(v_bs_6137_, v_i_6136_);
v___x_6140_ = lean_unsigned_to_nat(0u);
v_bs_x27_6141_ = lean_array_uset(v_bs_6137_, v_i_6136_, v___x_6140_);
v___x_6142_ = l_Lean_Expr_mvarId_x21(v_v_6139_);
lean_dec(v_v_6139_);
v___x_6143_ = ((size_t)1ULL);
v___x_6144_ = lean_usize_add(v_i_6136_, v___x_6143_);
v___x_6145_ = lean_array_uset(v_bs_x27_6141_, v_i_6136_, v___x_6142_);
v_i_6136_ = v___x_6144_;
v_bs_6137_ = v___x_6145_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1___boxed(lean_object* v_sz_6147_, lean_object* v_i_6148_, lean_object* v_bs_6149_){
_start:
{
size_t v_sz_boxed_6150_; size_t v_i_boxed_6151_; lean_object* v_res_6152_; 
v_sz_boxed_6150_ = lean_unbox_usize(v_sz_6147_);
lean_dec(v_sz_6147_);
v_i_boxed_6151_ = lean_unbox_usize(v_i_6148_);
lean_dec(v_i_6148_);
v_res_6152_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1(v_sz_boxed_6150_, v_i_boxed_6151_, v_bs_6149_);
return v_res_6152_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3(lean_object* v_a_6153_, lean_object* v_as_6154_, size_t v_i_6155_, size_t v_stop_6156_){
_start:
{
uint8_t v___x_6157_; 
v___x_6157_ = lean_usize_dec_eq(v_i_6155_, v_stop_6156_);
if (v___x_6157_ == 0)
{
lean_object* v___x_6158_; uint8_t v___x_6159_; 
v___x_6158_ = lean_array_uget_borrowed(v_as_6154_, v_i_6155_);
v___x_6159_ = l_Lean_instBEqMVarId_beq(v_a_6153_, v___x_6158_);
if (v___x_6159_ == 0)
{
size_t v___x_6160_; size_t v___x_6161_; 
v___x_6160_ = ((size_t)1ULL);
v___x_6161_ = lean_usize_add(v_i_6155_, v___x_6160_);
v_i_6155_ = v___x_6161_;
goto _start;
}
else
{
return v___x_6159_;
}
}
else
{
uint8_t v___x_6163_; 
v___x_6163_ = 0;
return v___x_6163_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3___boxed(lean_object* v_a_6164_, lean_object* v_as_6165_, lean_object* v_i_6166_, lean_object* v_stop_6167_){
_start:
{
size_t v_i_boxed_6168_; size_t v_stop_boxed_6169_; uint8_t v_res_6170_; lean_object* v_r_6171_; 
v_i_boxed_6168_ = lean_unbox_usize(v_i_6166_);
lean_dec(v_i_6166_);
v_stop_boxed_6169_ = lean_unbox_usize(v_stop_6167_);
lean_dec(v_stop_6167_);
v_res_6170_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3(v_a_6164_, v_as_6165_, v_i_boxed_6168_, v_stop_boxed_6169_);
lean_dec_ref(v_as_6165_);
lean_dec(v_a_6164_);
v_r_6171_ = lean_box(v_res_6170_);
return v_r_6171_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2(lean_object* v_as_6172_, lean_object* v_a_6173_){
_start:
{
lean_object* v___x_6174_; lean_object* v___x_6175_; uint8_t v___x_6176_; 
v___x_6174_ = lean_unsigned_to_nat(0u);
v___x_6175_ = lean_array_get_size(v_as_6172_);
v___x_6176_ = lean_nat_dec_lt(v___x_6174_, v___x_6175_);
if (v___x_6176_ == 0)
{
return v___x_6176_;
}
else
{
if (v___x_6176_ == 0)
{
return v___x_6176_;
}
else
{
size_t v___x_6177_; size_t v___x_6178_; uint8_t v___x_6179_; 
v___x_6177_ = ((size_t)0ULL);
v___x_6178_ = lean_usize_of_nat(v___x_6175_);
v___x_6179_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_MVarId_grewrite_spec__2_spec__3(v_a_6173_, v_as_6172_, v___x_6177_, v___x_6178_);
return v___x_6179_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2___boxed(lean_object* v_as_6180_, lean_object* v_a_6181_){
_start:
{
uint8_t v_res_6182_; lean_object* v_r_6183_; 
v_res_6182_ = lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2(v_as_6180_, v_a_6181_);
lean_dec(v_a_6181_);
lean_dec_ref(v_as_6180_);
v_r_6183_ = lean_box(v_res_6182_);
return v_r_6183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3(lean_object* v_a_6184_, lean_object* v_as_6185_, size_t v_i_6186_, size_t v_stop_6187_, lean_object* v_b_6188_){
_start:
{
lean_object* v___y_6190_; uint8_t v___x_6194_; 
v___x_6194_ = lean_usize_dec_eq(v_i_6186_, v_stop_6187_);
if (v___x_6194_ == 0)
{
lean_object* v___x_6195_; uint8_t v___x_6196_; 
v___x_6195_ = lean_array_uget_borrowed(v_as_6185_, v_i_6186_);
v___x_6196_ = lp_mathlib_Array_contains___at___00Lean_MVarId_grewrite_spec__2(v_a_6184_, v___x_6195_);
if (v___x_6196_ == 0)
{
lean_object* v___x_6197_; 
lean_inc(v___x_6195_);
v___x_6197_ = lean_array_push(v_b_6188_, v___x_6195_);
v___y_6190_ = v___x_6197_;
goto v___jp_6189_;
}
else
{
v___y_6190_ = v_b_6188_;
goto v___jp_6189_;
}
}
else
{
return v_b_6188_;
}
v___jp_6189_:
{
size_t v___x_6191_; size_t v___x_6192_; 
v___x_6191_ = ((size_t)1ULL);
v___x_6192_ = lean_usize_add(v_i_6186_, v___x_6191_);
v_i_6186_ = v___x_6192_;
v_b_6188_ = v___y_6190_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3___boxed(lean_object* v_a_6198_, lean_object* v_as_6199_, lean_object* v_i_6200_, lean_object* v_stop_6201_, lean_object* v_b_6202_){
_start:
{
size_t v_i_boxed_6203_; size_t v_stop_boxed_6204_; lean_object* v_res_6205_; 
v_i_boxed_6203_ = lean_unbox_usize(v_i_6200_);
lean_dec(v_i_6200_);
v_stop_boxed_6204_ = lean_unbox_usize(v_stop_6201_);
lean_dec(v_stop_6201_);
v_res_6205_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3(v_a_6198_, v_as_6199_, v_i_boxed_6203_, v_stop_boxed_6204_, v_b_6202_);
lean_dec_ref(v_as_6199_);
lean_dec_ref(v_a_6198_);
return v_res_6205_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1(void){
_start:
{
lean_object* v___x_6207_; lean_object* v___x_6208_; 
v___x_6207_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__0));
v___x_6208_ = l_Lean_stringToMessageData(v___x_6207_);
return v___x_6208_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3(void){
_start:
{
lean_object* v___x_6210_; lean_object* v___x_6211_; 
v___x_6210_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__2));
v___x_6211_ = l_Lean_stringToMessageData(v___x_6210_);
return v___x_6211_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5(void){
_start:
{
lean_object* v___x_6213_; lean_object* v___x_6214_; 
v___x_6213_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__4));
v___x_6214_ = l_Lean_stringToMessageData(v___x_6213_);
return v___x_6214_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7(void){
_start:
{
lean_object* v___x_6216_; lean_object* v___x_6217_; 
v___x_6216_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__6));
v___x_6217_ = l_Lean_stringToMessageData(v___x_6216_);
return v___x_6217_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19(void){
_start:
{
lean_object* v___x_6236_; lean_object* v___x_6237_; 
v___x_6236_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__18));
v___x_6237_ = l_Lean_stringToMessageData(v___x_6236_);
return v___x_6237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3(lean_object* v_goal_6238_, lean_object* v___x_6239_, lean_object* v_hrel_6240_, lean_object* v_config_6241_, lean_object* v_e_6242_, uint8_t v_symm_6243_, lean_object* v_mvarIds_6244_, uint8_t v_forwardImp_6245_, lean_object* v___y_6246_, lean_object* v___y_6247_, lean_object* v___y_6248_, lean_object* v___y_6249_){
_start:
{
lean_object* v___y_6252_; lean_object* v___y_6253_; lean_object* v___y_6254_; lean_object* v___y_6255_; lean_object* v___y_6256_; lean_object* v___y_6262_; size_t v___y_6263_; lean_object* v___y_6264_; lean_object* v___y_6265_; lean_object* v___y_6266_; lean_object* v___y_6267_; lean_object* v___y_6268_; lean_object* v___y_6269_; lean_object* v___y_6270_; lean_object* v_a_6271_; lean_object* v___y_6291_; lean_object* v___y_6292_; lean_object* v___y_6293_; size_t v___y_6294_; lean_object* v___y_6295_; lean_object* v___y_6296_; lean_object* v___y_6297_; lean_object* v___y_6298_; lean_object* v___y_6299_; lean_object* v___y_6300_; lean_object* v___y_6311_; lean_object* v___y_6312_; lean_object* v___y_6313_; lean_object* v___y_6314_; lean_object* v___y_6315_; lean_object* v___y_6316_; lean_object* v___y_6317_; lean_object* v___y_6318_; uint8_t v___y_6319_; lean_object* v___y_6320_; lean_object* v___y_6321_; uint8_t v___y_6322_; lean_object* v___y_6346_; lean_object* v___y_6347_; uint8_t v___y_6348_; lean_object* v_fst_6349_; lean_object* v_snd_6350_; lean_object* v___y_6351_; lean_object* v___y_6352_; lean_object* v___y_6353_; lean_object* v___y_6354_; lean_object* v___y_6364_; lean_object* v___y_6365_; lean_object* v___y_6366_; lean_object* v___y_6367_; uint8_t v___y_6368_; lean_object* v___y_6369_; lean_object* v___y_6370_; lean_object* v___y_6371_; uint8_t v___y_6384_; lean_object* v___y_6385_; lean_object* v___y_6386_; lean_object* v___y_6387_; lean_object* v___y_6388_; lean_object* v___y_6389_; lean_object* v___y_6390_; uint8_t v___y_6391_; lean_object* v___y_6392_; lean_object* v___y_6393_; lean_object* v___y_6394_; uint8_t v___y_6395_; lean_object* v___y_6416_; lean_object* v___y_6417_; lean_object* v___y_6418_; lean_object* v___y_6419_; lean_object* v___y_6420_; lean_object* v___y_6421_; lean_object* v___y_6422_; lean_object* v___y_6423_; lean_object* v___y_6424_; lean_object* v___y_6425_; lean_object* v___y_6426_; uint8_t v___y_6427_; lean_object* v___y_6428_; lean_object* v___y_6429_; uint8_t v___y_6430_; lean_object* v___y_6436_; lean_object* v___y_6437_; lean_object* v___y_6438_; lean_object* v___y_6439_; lean_object* v___y_6440_; lean_object* v___y_6441_; lean_object* v___y_6442_; lean_object* v___y_6443_; lean_object* v___x_6471_; 
lean_inc(v___x_6239_);
lean_inc(v_goal_6238_);
v___x_6471_ = l_Lean_MVarId_checkNotAssigned(v_goal_6238_, v___x_6239_, v___y_6246_, v___y_6247_, v___y_6248_, v___y_6249_);
if (lean_obj_tag(v___x_6471_) == 0)
{
lean_object* v___x_6472_; 
lean_dec_ref_known(v___x_6471_, 1);
lean_inc(v___y_6249_);
lean_inc_ref(v___y_6248_);
lean_inc(v___y_6247_);
lean_inc_ref(v___y_6246_);
lean_inc_ref(v_hrel_6240_);
v___x_6472_ = lean_infer_type(v_hrel_6240_, v___y_6246_, v___y_6247_, v___y_6248_, v___y_6249_);
if (lean_obj_tag(v___x_6472_) == 0)
{
lean_object* v_a_6473_; lean_object* v___x_6474_; lean_object* v_a_6475_; lean_object* v___x_6477_; uint8_t v_isShared_6478_; uint8_t v_isSharedCheck_6713_; 
v_a_6473_ = lean_ctor_get(v___x_6472_, 0);
lean_inc(v_a_6473_);
lean_dec_ref_known(v___x_6472_, 1);
v___x_6474_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(v_a_6473_, v___y_6247_);
v_a_6475_ = lean_ctor_get(v___x_6474_, 0);
v_isSharedCheck_6713_ = !lean_is_exclusive(v___x_6474_);
if (v_isSharedCheck_6713_ == 0)
{
v___x_6477_ = v___x_6474_;
v_isShared_6478_ = v_isSharedCheck_6713_;
goto v_resetjp_6476_;
}
else
{
lean_inc(v_a_6475_);
lean_dec(v___x_6474_);
v___x_6477_ = lean_box(0);
v_isShared_6478_ = v_isSharedCheck_6713_;
goto v_resetjp_6476_;
}
v_resetjp_6476_:
{
lean_object* v_toConfig_6479_; uint8_t v_useRewrite_6480_; uint8_t v_implicationHyp_6481_; uint8_t v_useKAbstract_6482_; lean_object* v___y_6484_; uint8_t v___y_6485_; lean_object* v___y_6486_; lean_object* v___y_6487_; lean_object* v___y_6488_; lean_object* v___y_6489_; lean_object* v___y_6490_; lean_object* v___y_6491_; lean_object* v___y_6492_; lean_object* v___y_6493_; lean_object* v___y_6494_; uint8_t v___y_6495_; lean_object* v___y_6496_; lean_object* v___y_6497_; lean_object* v___y_6498_; lean_object* v___y_6499_; lean_object* v___y_6560_; lean_object* v___y_6561_; lean_object* v___y_6562_; lean_object* v___y_6563_; lean_object* v___y_6564_; lean_object* v___y_6565_; lean_object* v___y_6566_; lean_object* v___y_6567_; lean_object* v___y_6568_; lean_object* v___y_6569_; lean_object* v___y_6570_; lean_object* v___y_6571_; uint8_t v___y_6572_; uint8_t v___y_6573_; lean_object* v_fst_6574_; lean_object* v_snd_6575_; lean_object* v___y_6598_; lean_object* v___y_6599_; lean_object* v___y_6600_; lean_object* v___y_6601_; lean_object* v___y_6602_; uint8_t v___y_6603_; lean_object* v___y_6604_; lean_object* v___y_6605_; lean_object* v___y_6606_; lean_object* v___y_6607_; uint8_t v___y_6608_; lean_object* v___y_6621_; lean_object* v___y_6622_; lean_object* v___y_6623_; lean_object* v___y_6624_; lean_object* v___y_6625_; uint8_t v___y_6626_; lean_object* v___y_6627_; lean_object* v___y_6628_; lean_object* v___y_6629_; uint8_t v___y_6630_; lean_object* v_maxMVars_x3f_6650_; lean_object* v___y_6651_; lean_object* v___y_6652_; lean_object* v___y_6653_; lean_object* v___y_6654_; 
v_toConfig_6479_ = lean_ctor_get(v_config_6241_, 0);
v_useRewrite_6480_ = lean_ctor_get_uint8(v_config_6241_, sizeof(void*)*1);
v_implicationHyp_6481_ = lean_ctor_get_uint8(v_config_6241_, sizeof(void*)*1 + 1);
v_useKAbstract_6482_ = lean_ctor_get_uint8(v_config_6241_, sizeof(void*)*1 + 2);
if (v_implicationHyp_6481_ == 0)
{
lean_object* v___x_6691_; 
v___x_6691_ = lean_box(0);
v_maxMVars_x3f_6650_ = v___x_6691_;
v___y_6651_ = v___y_6246_;
v___y_6652_ = v___y_6247_;
v___y_6653_ = v___y_6248_;
v___y_6654_ = v___y_6249_;
goto v___jp_6649_;
}
else
{
lean_object* v___x_6692_; lean_object* v_zero_6693_; uint8_t v_isZero_6694_; 
lean_inc(v_a_6475_);
v___x_6692_ = l_Lean_Expr_getForallArity(v_a_6475_);
v_zero_6693_ = lean_unsigned_to_nat(0u);
v_isZero_6694_ = lean_nat_dec_eq(v___x_6692_, v_zero_6693_);
if (v_isZero_6694_ == 0)
{
lean_object* v_one_6695_; lean_object* v_n_6696_; lean_object* v___x_6697_; 
v_one_6695_ = lean_unsigned_to_nat(1u);
v_n_6696_ = lean_nat_sub(v___x_6692_, v_one_6695_);
lean_dec(v___x_6692_);
v___x_6697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6697_, 0, v_n_6696_);
v_maxMVars_x3f_6650_ = v___x_6697_;
v___y_6651_ = v___y_6246_;
v___y_6652_ = v___y_6247_;
v___y_6653_ = v___y_6248_;
v___y_6654_ = v___y_6249_;
goto v___jp_6649_;
}
else
{
lean_object* v___x_6698_; lean_object* v___x_6699_; lean_object* v___x_6700_; lean_object* v___x_6701_; lean_object* v___x_6702_; lean_object* v___x_6703_; 
lean_dec(v___x_6692_);
v___x_6698_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__17));
v___x_6699_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__19);
lean_inc(v_a_6475_);
v___x_6700_ = l_Lean_MessageData_ofExpr(v_a_6475_);
v___x_6701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6701_, 0, v___x_6699_);
lean_ctor_set(v___x_6701_, 1, v___x_6700_);
v___x_6702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6702_, 0, v___x_6701_);
lean_inc(v_goal_6238_);
v___x_6703_ = l_Lean_Meta_throwTacticEx___redArg(v___x_6698_, v_goal_6238_, v___x_6702_, v___y_6246_, v___y_6247_, v___y_6248_, v___y_6249_);
if (lean_obj_tag(v___x_6703_) == 0)
{
lean_object* v_a_6704_; 
v_a_6704_ = lean_ctor_get(v___x_6703_, 0);
lean_inc(v_a_6704_);
lean_dec_ref_known(v___x_6703_, 1);
v_maxMVars_x3f_6650_ = v_a_6704_;
v___y_6651_ = v___y_6246_;
v___y_6652_ = v___y_6247_;
v___y_6653_ = v___y_6248_;
v___y_6654_ = v___y_6249_;
goto v___jp_6649_;
}
else
{
lean_object* v_a_6705_; lean_object* v___x_6707_; uint8_t v_isShared_6708_; uint8_t v_isSharedCheck_6712_; 
lean_del_object(v___x_6477_);
lean_dec(v_a_6475_);
lean_dec(v___y_6249_);
lean_dec_ref(v___y_6248_);
lean_dec(v___y_6247_);
lean_dec_ref(v___y_6246_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6705_ = lean_ctor_get(v___x_6703_, 0);
v_isSharedCheck_6712_ = !lean_is_exclusive(v___x_6703_);
if (v_isSharedCheck_6712_ == 0)
{
v___x_6707_ = v___x_6703_;
v_isShared_6708_ = v_isSharedCheck_6712_;
goto v_resetjp_6706_;
}
else
{
lean_inc(v_a_6705_);
lean_dec(v___x_6703_);
v___x_6707_ = lean_box(0);
v_isShared_6708_ = v_isSharedCheck_6712_;
goto v_resetjp_6706_;
}
v_resetjp_6706_:
{
lean_object* v___x_6710_; 
if (v_isShared_6708_ == 0)
{
v___x_6710_ = v___x_6707_;
goto v_reusejp_6709_;
}
else
{
lean_object* v_reuseFailAlloc_6711_; 
v_reuseFailAlloc_6711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6711_, 0, v_a_6705_);
v___x_6710_ = v_reuseFailAlloc_6711_;
goto v_reusejp_6709_;
}
v_reusejp_6709_:
{
return v___x_6710_;
}
}
}
}
}
v___jp_6483_:
{
lean_object* v___x_6500_; 
v___x_6500_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract_spec__0___redArg(v_e_6242_, v___y_6486_);
if (v_useKAbstract_6482_ == 0)
{
lean_object* v_a_6501_; lean_object* v___x_6503_; uint8_t v_isShared_6504_; uint8_t v_isSharedCheck_6546_; 
v_a_6501_ = lean_ctor_get(v___x_6500_, 0);
v_isSharedCheck_6546_ = !lean_is_exclusive(v___x_6500_);
if (v_isSharedCheck_6546_ == 0)
{
v___x_6503_ = v___x_6500_;
v_isShared_6504_ = v_isSharedCheck_6546_;
goto v_resetjp_6502_;
}
else
{
lean_inc(v_a_6501_);
lean_dec(v___x_6500_);
v___x_6503_ = lean_box(0);
v_isShared_6504_ = v_isSharedCheck_6546_;
goto v_resetjp_6502_;
}
v_resetjp_6502_:
{
lean_object* v_keyedConfig_6505_; uint8_t v_trackZetaDelta_6506_; lean_object* v_zetaDeltaSet_6507_; lean_object* v_lctx_6508_; lean_object* v_localInstances_6509_; lean_object* v_defEqCtx_x3f_6510_; lean_object* v_synthPendingDepth_6511_; lean_object* v_customCanUnfoldPredicate_x3f_6512_; uint8_t v_univApprox_6513_; uint8_t v_inTypeClassResolution_6514_; uint8_t v_cacheInferType_6515_; lean_object* v___x_6516_; lean_object* v___x_6517_; lean_object* v___x_6518_; 
v_keyedConfig_6505_ = lean_ctor_get(v___y_6493_, 0);
v_trackZetaDelta_6506_ = lean_ctor_get_uint8(v___y_6493_, sizeof(void*)*7);
v_zetaDeltaSet_6507_ = lean_ctor_get(v___y_6493_, 1);
v_lctx_6508_ = lean_ctor_get(v___y_6493_, 2);
v_localInstances_6509_ = lean_ctor_get(v___y_6493_, 3);
v_defEqCtx_x3f_6510_ = lean_ctor_get(v___y_6493_, 4);
v_synthPendingDepth_6511_ = lean_ctor_get(v___y_6493_, 5);
v_customCanUnfoldPredicate_x3f_6512_ = lean_ctor_get(v___y_6493_, 6);
v_univApprox_6513_ = lean_ctor_get_uint8(v___y_6493_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_6514_ = lean_ctor_get_uint8(v___y_6493_, sizeof(void*)*7 + 2);
v_cacheInferType_6515_ = lean_ctor_get_uint8(v___y_6493_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_6505_);
v___x_6516_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___y_6495_, v_keyedConfig_6505_);
lean_inc(v_customCanUnfoldPredicate_x3f_6512_);
lean_inc(v_synthPendingDepth_6511_);
lean_inc(v_defEqCtx_x3f_6510_);
lean_inc_ref(v_localInstances_6509_);
lean_inc_ref(v_lctx_6508_);
lean_inc(v_zetaDeltaSet_6507_);
v___x_6517_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_6517_, 0, v___x_6516_);
lean_ctor_set(v___x_6517_, 1, v_zetaDeltaSet_6507_);
lean_ctor_set(v___x_6517_, 2, v_lctx_6508_);
lean_ctor_set(v___x_6517_, 3, v_localInstances_6509_);
lean_ctor_set(v___x_6517_, 4, v_defEqCtx_x3f_6510_);
lean_ctor_set(v___x_6517_, 5, v_synthPendingDepth_6511_);
lean_ctor_set(v___x_6517_, 6, v_customCanUnfoldPredicate_x3f_6512_);
lean_ctor_set_uint8(v___x_6517_, sizeof(void*)*7, v_trackZetaDelta_6506_);
lean_ctor_set_uint8(v___x_6517_, sizeof(void*)*7 + 1, v_univApprox_6513_);
lean_ctor_set_uint8(v___x_6517_, sizeof(void*)*7 + 2, v_inTypeClassResolution_6514_);
lean_ctor_set_uint8(v___x_6517_, sizeof(void*)*7 + 3, v_cacheInferType_6515_);
lean_inc(v___y_6497_);
lean_inc_ref(v___y_6491_);
lean_inc(v___y_6486_);
lean_inc_ref(v___x_6517_);
lean_inc_ref(v___y_6496_);
v___x_6518_ = lean_whnf(v___y_6496_, v___x_6517_, v___y_6486_, v___y_6491_, v___y_6497_);
if (lean_obj_tag(v___x_6518_) == 0)
{
lean_object* v_a_6519_; lean_object* v___x_6520_; 
v_a_6519_ = lean_ctor_get(v___x_6518_, 0);
lean_inc(v_a_6519_);
lean_dec_ref_known(v___x_6518_, 1);
v___x_6520_ = lp_mathlib_Mathlib_Tactic_GCongr_getRel(v_a_6519_);
lean_dec(v_a_6519_);
if (lean_obj_tag(v___x_6520_) == 1)
{
lean_object* v_val_6521_; lean_object* v_snd_6522_; lean_object* v_fst_6523_; lean_object* v_snd_6524_; lean_object* v___x_6525_; lean_object* v___f_6526_; lean_object* v___x_6527_; lean_object* v___f_6528_; uint8_t v___x_6529_; 
lean_del_object(v___x_6503_);
v_val_6521_ = lean_ctor_get(v___x_6520_, 0);
lean_inc(v_val_6521_);
lean_dec_ref_known(v___x_6520_, 1);
v_snd_6522_ = lean_ctor_get(v_val_6521_, 1);
lean_inc(v_snd_6522_);
lean_dec(v_val_6521_);
v_fst_6523_ = lean_ctor_get(v_snd_6522_, 0);
lean_inc(v_fst_6523_);
v_snd_6524_ = lean_ctor_get(v_snd_6522_, 1);
lean_inc(v_snd_6524_);
lean_dec(v_snd_6522_);
v___x_6525_ = lean_box(v_symm_6243_);
lean_inc(v_goal_6238_);
lean_inc(v___x_6239_);
lean_inc(v_a_6501_);
lean_inc_ref(v___y_6484_);
v___f_6526_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_grewrite___lam__0___boxed), 11, 5);
lean_closure_set(v___f_6526_, 0, v___y_6484_);
lean_closure_set(v___f_6526_, 1, v___x_6525_);
lean_closure_set(v___f_6526_, 2, v_a_6501_);
lean_closure_set(v___f_6526_, 3, v___x_6239_);
lean_closure_set(v___f_6526_, 4, v_goal_6238_);
v___x_6527_ = lean_box(v_forwardImp_6245_);
v___f_6528_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_grewrite___lam__2___boxed), 16, 10);
lean_closure_set(v___f_6528_, 0, v___y_6489_);
lean_closure_set(v___f_6528_, 1, v___y_6487_);
lean_closure_set(v___f_6528_, 2, v_mvarIds_6244_);
lean_closure_set(v___f_6528_, 3, v___y_6488_);
lean_closure_set(v___f_6528_, 4, v___y_6484_);
lean_closure_set(v___f_6528_, 5, v_a_6501_);
lean_closure_set(v___f_6528_, 6, v___x_6527_);
lean_closure_set(v___f_6528_, 7, v_config_6241_);
lean_closure_set(v___f_6528_, 8, v___y_6499_);
lean_closure_set(v___f_6528_, 9, v___f_6526_);
v___x_6529_ = lean_expr_eqv(v_fst_6523_, v___y_6490_);
if (v___x_6529_ == 0)
{
v___y_6416_ = v_snd_6524_;
v___y_6417_ = v_fst_6523_;
v___y_6418_ = v___y_6491_;
v___y_6419_ = v___y_6492_;
v___y_6420_ = v___y_6494_;
v___y_6421_ = v___f_6528_;
v___y_6422_ = v___y_6490_;
v___y_6423_ = v___y_6497_;
v___y_6424_ = v___y_6496_;
v___y_6425_ = v___y_6498_;
v___y_6426_ = v___y_6493_;
v___y_6427_ = v___y_6485_;
v___y_6428_ = v___y_6486_;
v___y_6429_ = v___x_6517_;
v___y_6430_ = v___x_6529_;
goto v___jp_6415_;
}
else
{
uint8_t v___x_6530_; 
v___x_6530_ = lean_expr_eqv(v_snd_6524_, v___y_6498_);
v___y_6416_ = v_snd_6524_;
v___y_6417_ = v_fst_6523_;
v___y_6418_ = v___y_6491_;
v___y_6419_ = v___y_6492_;
v___y_6420_ = v___y_6494_;
v___y_6421_ = v___f_6528_;
v___y_6422_ = v___y_6490_;
v___y_6423_ = v___y_6497_;
v___y_6424_ = v___y_6496_;
v___y_6425_ = v___y_6498_;
v___y_6426_ = v___y_6493_;
v___y_6427_ = v___y_6485_;
v___y_6428_ = v___y_6486_;
v___y_6429_ = v___x_6517_;
v___y_6430_ = v___x_6530_;
goto v___jp_6415_;
}
}
else
{
lean_object* v___x_6531_; lean_object* v___x_6532_; lean_object* v___x_6533_; lean_object* v___x_6535_; 
lean_dec(v___x_6520_);
lean_dec(v_a_6501_);
lean_dec_ref(v___y_6499_);
lean_dec_ref(v___y_6498_);
lean_dec_ref(v___y_6490_);
lean_dec_ref(v___y_6489_);
lean_dec_ref(v___y_6488_);
lean_dec_ref(v___y_6487_);
lean_dec_ref(v___y_6484_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_config_6241_);
v___x_6531_ = l_Lean_MessageData_ofExpr(v___y_6496_);
v___x_6532_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1);
v___x_6533_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6533_, 0, v___x_6531_);
lean_ctor_set(v___x_6533_, 1, v___x_6532_);
if (v_isShared_6504_ == 0)
{
lean_ctor_set_tag(v___x_6503_, 1);
lean_ctor_set(v___x_6503_, 0, v___x_6533_);
v___x_6535_ = v___x_6503_;
goto v_reusejp_6534_;
}
else
{
lean_object* v_reuseFailAlloc_6537_; 
v_reuseFailAlloc_6537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6537_, 0, v___x_6533_);
v___x_6535_ = v_reuseFailAlloc_6537_;
goto v_reusejp_6534_;
}
v_reusejp_6534_:
{
lean_object* v___x_6536_; 
lean_inc(v_goal_6238_);
lean_inc(v___x_6239_);
v___x_6536_ = l_Lean_Meta_throwTacticEx___redArg(v___x_6239_, v_goal_6238_, v___x_6535_, v___x_6517_, v___y_6486_, v___y_6491_, v___y_6497_);
lean_dec_ref_known(v___x_6517_, 7);
v___y_6364_ = v___y_6491_;
v___y_6365_ = v___y_6494_;
v___y_6366_ = v___y_6492_;
v___y_6367_ = v___y_6493_;
v___y_6368_ = v___y_6485_;
v___y_6369_ = v___y_6486_;
v___y_6370_ = v___y_6497_;
v___y_6371_ = v___x_6536_;
goto v___jp_6363_;
}
}
}
else
{
lean_object* v_a_6538_; lean_object* v___x_6540_; uint8_t v_isShared_6541_; uint8_t v_isSharedCheck_6545_; 
lean_dec_ref_known(v___x_6517_, 7);
lean_del_object(v___x_6503_);
lean_dec(v_a_6501_);
lean_dec_ref(v___y_6499_);
lean_dec_ref(v___y_6498_);
lean_dec(v___y_6497_);
lean_dec_ref(v___y_6496_);
lean_dec_ref(v___y_6494_);
lean_dec_ref(v___y_6493_);
lean_dec_ref(v___y_6492_);
lean_dec_ref(v___y_6491_);
lean_dec_ref(v___y_6490_);
lean_dec_ref(v___y_6489_);
lean_dec_ref(v___y_6488_);
lean_dec_ref(v___y_6487_);
lean_dec(v___y_6486_);
lean_dec_ref(v___y_6484_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6538_ = lean_ctor_get(v___x_6518_, 0);
v_isSharedCheck_6545_ = !lean_is_exclusive(v___x_6518_);
if (v_isSharedCheck_6545_ == 0)
{
v___x_6540_ = v___x_6518_;
v_isShared_6541_ = v_isSharedCheck_6545_;
goto v_resetjp_6539_;
}
else
{
lean_inc(v_a_6538_);
lean_dec(v___x_6518_);
v___x_6540_ = lean_box(0);
v_isShared_6541_ = v_isSharedCheck_6545_;
goto v_resetjp_6539_;
}
v_resetjp_6539_:
{
lean_object* v___x_6543_; 
if (v_isShared_6541_ == 0)
{
v___x_6543_ = v___x_6540_;
goto v_reusejp_6542_;
}
else
{
lean_object* v_reuseFailAlloc_6544_; 
v_reuseFailAlloc_6544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6544_, 0, v_a_6538_);
v___x_6543_ = v_reuseFailAlloc_6544_;
goto v_reusejp_6542_;
}
v_reusejp_6542_:
{
return v___x_6543_;
}
}
}
}
}
else
{
lean_object* v_a_6547_; lean_object* v___x_6548_; 
lean_dec_ref(v___y_6498_);
lean_dec_ref(v___y_6496_);
lean_dec_ref(v___y_6490_);
lean_dec_ref(v___y_6487_);
lean_dec_ref(v___y_6484_);
lean_dec_ref(v_mvarIds_6244_);
v_a_6547_ = lean_ctor_get(v___x_6500_, 0);
lean_inc(v_a_6547_);
lean_dec_ref(v___x_6500_);
lean_inc(v_goal_6238_);
v___x_6548_ = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_grewriteUsingKAbstract(v_goal_6238_, v_a_6547_, v___y_6488_, v___y_6489_, v___y_6499_, v_forwardImp_6245_, v_config_6241_, v___y_6493_, v___y_6486_, v___y_6491_, v___y_6497_);
lean_dec_ref(v___y_6499_);
if (lean_obj_tag(v___x_6548_) == 0)
{
lean_object* v_a_6549_; lean_object* v___x_6550_; 
v_a_6549_ = lean_ctor_get(v___x_6548_, 0);
lean_inc(v_a_6549_);
lean_dec_ref_known(v___x_6548_, 1);
v___x_6550_ = lean_box(0);
v___y_6346_ = v___y_6492_;
v___y_6347_ = v___y_6494_;
v___y_6348_ = v___y_6485_;
v_fst_6349_ = v___x_6550_;
v_snd_6350_ = v_a_6549_;
v___y_6351_ = v___y_6493_;
v___y_6352_ = v___y_6486_;
v___y_6353_ = v___y_6491_;
v___y_6354_ = v___y_6497_;
goto v___jp_6345_;
}
else
{
lean_object* v_a_6551_; lean_object* v___x_6553_; uint8_t v_isShared_6554_; uint8_t v_isSharedCheck_6558_; 
lean_dec(v___y_6497_);
lean_dec_ref(v___y_6494_);
lean_dec_ref(v___y_6493_);
lean_dec_ref(v___y_6492_);
lean_dec_ref(v___y_6491_);
lean_dec(v___y_6486_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6551_ = lean_ctor_get(v___x_6548_, 0);
v_isSharedCheck_6558_ = !lean_is_exclusive(v___x_6548_);
if (v_isSharedCheck_6558_ == 0)
{
v___x_6553_ = v___x_6548_;
v_isShared_6554_ = v_isSharedCheck_6558_;
goto v_resetjp_6552_;
}
else
{
lean_inc(v_a_6551_);
lean_dec(v___x_6548_);
v___x_6553_ = lean_box(0);
v_isShared_6554_ = v_isSharedCheck_6558_;
goto v_resetjp_6552_;
}
v_resetjp_6552_:
{
lean_object* v___x_6556_; 
if (v_isShared_6554_ == 0)
{
v___x_6556_ = v___x_6553_;
goto v_reusejp_6555_;
}
else
{
lean_object* v_reuseFailAlloc_6557_; 
v_reuseFailAlloc_6557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6557_, 0, v_a_6551_);
v___x_6556_ = v_reuseFailAlloc_6557_;
goto v_reusejp_6555_;
}
v_reusejp_6555_:
{
return v___x_6556_;
}
}
}
}
}
v___jp_6559_:
{
lean_object* v___x_6576_; uint8_t v___x_6577_; 
v___x_6576_ = l_Lean_Expr_getAppFn(v_fst_6574_);
v___x_6577_ = l_Lean_Expr_isMVar(v___x_6576_);
lean_dec_ref(v___x_6576_);
if (v___x_6577_ == 0)
{
lean_del_object(v___x_6477_);
v___y_6484_ = v___y_6561_;
v___y_6485_ = v___y_6573_;
v___y_6486_ = v___y_6570_;
v___y_6487_ = v___y_6562_;
v___y_6488_ = v___y_6563_;
v___y_6489_ = v_fst_6574_;
v___y_6490_ = v___y_6560_;
v___y_6491_ = v___y_6565_;
v___y_6492_ = v___y_6567_;
v___y_6493_ = v___y_6569_;
v___y_6494_ = v___y_6568_;
v___y_6495_ = v___y_6572_;
v___y_6496_ = v___y_6571_;
v___y_6497_ = v___y_6566_;
v___y_6498_ = v___y_6564_;
v___y_6499_ = v_snd_6575_;
goto v___jp_6483_;
}
else
{
lean_object* v___x_6578_; lean_object* v___x_6579_; lean_object* v___x_6580_; lean_object* v___x_6581_; lean_object* v___x_6582_; lean_object* v___x_6583_; lean_object* v___x_6584_; lean_object* v___x_6586_; 
v___x_6578_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__3);
lean_inc_ref(v_fst_6574_);
v___x_6579_ = l_Lean_indentExpr(v_fst_6574_);
v___x_6580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6580_, 0, v___x_6578_);
lean_ctor_set(v___x_6580_, 1, v___x_6579_);
v___x_6581_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__5);
v___x_6582_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6582_, 0, v___x_6580_);
lean_ctor_set(v___x_6582_, 1, v___x_6581_);
lean_inc_ref(v___y_6571_);
v___x_6583_ = l_Lean_indentExpr(v___y_6571_);
v___x_6584_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6584_, 0, v___x_6582_);
lean_ctor_set(v___x_6584_, 1, v___x_6583_);
if (v_isShared_6478_ == 0)
{
lean_ctor_set_tag(v___x_6477_, 1);
lean_ctor_set(v___x_6477_, 0, v___x_6584_);
v___x_6586_ = v___x_6477_;
goto v_reusejp_6585_;
}
else
{
lean_object* v_reuseFailAlloc_6596_; 
v_reuseFailAlloc_6596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6596_, 0, v___x_6584_);
v___x_6586_ = v_reuseFailAlloc_6596_;
goto v_reusejp_6585_;
}
v_reusejp_6585_:
{
lean_object* v___x_6587_; 
lean_inc(v_goal_6238_);
lean_inc(v___x_6239_);
v___x_6587_ = l_Lean_Meta_throwTacticEx___redArg(v___x_6239_, v_goal_6238_, v___x_6586_, v___y_6569_, v___y_6570_, v___y_6565_, v___y_6566_);
if (lean_obj_tag(v___x_6587_) == 0)
{
lean_dec_ref_known(v___x_6587_, 1);
v___y_6484_ = v___y_6561_;
v___y_6485_ = v___y_6573_;
v___y_6486_ = v___y_6570_;
v___y_6487_ = v___y_6562_;
v___y_6488_ = v___y_6563_;
v___y_6489_ = v_fst_6574_;
v___y_6490_ = v___y_6560_;
v___y_6491_ = v___y_6565_;
v___y_6492_ = v___y_6567_;
v___y_6493_ = v___y_6569_;
v___y_6494_ = v___y_6568_;
v___y_6495_ = v___y_6572_;
v___y_6496_ = v___y_6571_;
v___y_6497_ = v___y_6566_;
v___y_6498_ = v___y_6564_;
v___y_6499_ = v_snd_6575_;
goto v___jp_6483_;
}
else
{
lean_object* v_a_6588_; lean_object* v___x_6590_; uint8_t v_isShared_6591_; uint8_t v_isSharedCheck_6595_; 
lean_dec_ref(v_snd_6575_);
lean_dec_ref(v_fst_6574_);
lean_dec_ref(v___y_6571_);
lean_dec(v___y_6570_);
lean_dec_ref(v___y_6569_);
lean_dec_ref(v___y_6568_);
lean_dec_ref(v___y_6567_);
lean_dec(v___y_6566_);
lean_dec_ref(v___y_6565_);
lean_dec_ref(v___y_6564_);
lean_dec_ref(v___y_6563_);
lean_dec_ref(v___y_6562_);
lean_dec_ref(v___y_6561_);
lean_dec_ref(v___y_6560_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6588_ = lean_ctor_get(v___x_6587_, 0);
v_isSharedCheck_6595_ = !lean_is_exclusive(v___x_6587_);
if (v_isSharedCheck_6595_ == 0)
{
v___x_6590_ = v___x_6587_;
v_isShared_6591_ = v_isSharedCheck_6595_;
goto v_resetjp_6589_;
}
else
{
lean_inc(v_a_6588_);
lean_dec(v___x_6587_);
v___x_6590_ = lean_box(0);
v_isShared_6591_ = v_isSharedCheck_6595_;
goto v_resetjp_6589_;
}
v_resetjp_6589_:
{
lean_object* v___x_6593_; 
if (v_isShared_6591_ == 0)
{
v___x_6593_ = v___x_6590_;
goto v_reusejp_6592_;
}
else
{
lean_object* v_reuseFailAlloc_6594_; 
v_reuseFailAlloc_6594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6594_, 0, v_a_6588_);
v___x_6593_ = v_reuseFailAlloc_6594_;
goto v_reusejp_6592_;
}
v_reusejp_6592_:
{
return v___x_6593_;
}
}
}
}
}
}
v___jp_6597_:
{
lean_object* v___x_6609_; 
v___x_6609_ = lp_mathlib_Mathlib_Tactic_GCongr_getRel(v___y_6602_);
if (lean_obj_tag(v___x_6609_) == 1)
{
lean_object* v_val_6610_; lean_object* v_snd_6611_; lean_object* v_fst_6612_; lean_object* v_snd_6613_; lean_object* v___x_6614_; 
v_val_6610_ = lean_ctor_get(v___x_6609_, 0);
lean_inc(v_val_6610_);
lean_dec_ref_known(v___x_6609_, 1);
v_snd_6611_ = lean_ctor_get(v_val_6610_, 1);
lean_inc(v_snd_6611_);
lean_dec(v_val_6610_);
v_fst_6612_ = lean_ctor_get(v_snd_6611_, 0);
lean_inc(v_fst_6612_);
v_snd_6613_ = lean_ctor_get(v_snd_6611_, 1);
lean_inc(v_snd_6613_);
lean_dec(v_snd_6611_);
lean_inc_ref(v_hrel_6240_);
v___x_6614_ = l_Lean_mkAppN(v_hrel_6240_, v___y_6605_);
if (v_symm_6243_ == 0)
{
lean_inc(v_snd_6613_);
lean_inc(v_fst_6612_);
v___y_6560_ = v_fst_6612_;
v___y_6561_ = v___y_6598_;
v___y_6562_ = v___y_6599_;
v___y_6563_ = v___x_6614_;
v___y_6564_ = v_snd_6613_;
v___y_6565_ = v___y_6600_;
v___y_6566_ = v___y_6601_;
v___y_6567_ = v___y_6605_;
v___y_6568_ = v___y_6604_;
v___y_6569_ = v___y_6606_;
v___y_6570_ = v___y_6607_;
v___y_6571_ = v___y_6602_;
v___y_6572_ = v___y_6603_;
v___y_6573_ = v___y_6608_;
v_fst_6574_ = v_fst_6612_;
v_snd_6575_ = v_snd_6613_;
goto v___jp_6559_;
}
else
{
lean_inc(v_snd_6613_);
lean_inc(v_fst_6612_);
v___y_6560_ = v_fst_6612_;
v___y_6561_ = v___y_6598_;
v___y_6562_ = v___y_6599_;
v___y_6563_ = v___x_6614_;
v___y_6564_ = v_snd_6613_;
v___y_6565_ = v___y_6600_;
v___y_6566_ = v___y_6601_;
v___y_6567_ = v___y_6605_;
v___y_6568_ = v___y_6604_;
v___y_6569_ = v___y_6606_;
v___y_6570_ = v___y_6607_;
v___y_6571_ = v___y_6602_;
v___y_6572_ = v___y_6603_;
v___y_6573_ = v___y_6608_;
v_fst_6574_ = v_snd_6613_;
v_snd_6575_ = v_fst_6612_;
goto v___jp_6559_;
}
}
else
{
lean_object* v___x_6615_; lean_object* v___x_6616_; lean_object* v___x_6617_; lean_object* v___x_6618_; lean_object* v___x_6619_; 
lean_dec(v___x_6609_);
lean_dec_ref(v___y_6605_);
lean_dec_ref(v___y_6604_);
lean_dec_ref(v___y_6599_);
lean_dec_ref(v___y_6598_);
lean_del_object(v___x_6477_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
v___x_6615_ = l_Lean_MessageData_ofExpr(v___y_6602_);
v___x_6616_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__7);
v___x_6617_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6617_, 0, v___x_6615_);
lean_ctor_set(v___x_6617_, 1, v___x_6616_);
v___x_6618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6618_, 0, v___x_6617_);
v___x_6619_ = l_Lean_Meta_throwTacticEx___redArg(v___x_6239_, v_goal_6238_, v___x_6618_, v___y_6606_, v___y_6607_, v___y_6600_, v___y_6601_);
lean_dec(v___y_6601_);
lean_dec_ref(v___y_6600_);
lean_dec(v___y_6607_);
lean_dec_ref(v___y_6606_);
return v___x_6619_;
}
}
v___jp_6620_:
{
if (v___y_6630_ == 0)
{
lean_inc_ref(v___y_6621_);
v___y_6598_ = v___y_6621_;
v___y_6599_ = v___y_6622_;
v___y_6600_ = v___y_6624_;
v___y_6601_ = v___y_6625_;
v___y_6602_ = v___y_6621_;
v___y_6603_ = v___y_6626_;
v___y_6604_ = v___y_6623_;
v___y_6605_ = v___y_6627_;
v___y_6606_ = v___y_6628_;
v___y_6607_ = v___y_6629_;
v___y_6608_ = v___y_6630_;
goto v___jp_6597_;
}
else
{
if (v_useRewrite_6480_ == 0)
{
lean_inc_ref(v___y_6621_);
v___y_6598_ = v___y_6621_;
v___y_6599_ = v___y_6622_;
v___y_6600_ = v___y_6624_;
v___y_6601_ = v___y_6625_;
v___y_6602_ = v___y_6621_;
v___y_6603_ = v___y_6626_;
v___y_6604_ = v___y_6623_;
v___y_6605_ = v___y_6627_;
v___y_6606_ = v___y_6628_;
v___y_6607_ = v___y_6629_;
v___y_6608_ = v_useRewrite_6480_;
goto v___jp_6597_;
}
else
{
lean_object* v___x_6631_; 
lean_inc_ref(v_toConfig_6479_);
lean_dec_ref(v___y_6627_);
lean_dec_ref(v___y_6623_);
lean_dec_ref(v___y_6622_);
lean_dec_ref(v___y_6621_);
lean_del_object(v___x_6477_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_config_6241_);
lean_dec(v___x_6239_);
lean_inc_ref(v_e_6242_);
v___x_6631_ = l_Lean_MVarId_rewrite(v_goal_6238_, v_e_6242_, v_hrel_6240_, v_symm_6243_, v_toConfig_6479_, v___y_6628_, v___y_6629_, v___y_6624_, v___y_6625_);
if (lean_obj_tag(v___x_6631_) == 0)
{
lean_object* v_a_6632_; 
v_a_6632_ = lean_ctor_get(v___x_6631_, 0);
lean_inc(v_a_6632_);
lean_dec_ref_known(v___x_6631_, 1);
if (v_forwardImp_6245_ == 0)
{
lean_object* v_eNew_6633_; lean_object* v_eqProof_6634_; lean_object* v_mvarIds_6635_; lean_object* v___x_6636_; 
v_eNew_6633_ = lean_ctor_get(v_a_6632_, 0);
lean_inc_ref(v_eNew_6633_);
v_eqProof_6634_ = lean_ctor_get(v_a_6632_, 1);
lean_inc_ref(v_eqProof_6634_);
v_mvarIds_6635_ = lean_ctor_get(v_a_6632_, 2);
lean_inc(v_mvarIds_6635_);
lean_dec(v_a_6632_);
v___x_6636_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__10));
v___y_6436_ = v___y_6624_;
v___y_6437_ = v___y_6625_;
v___y_6438_ = v___y_6628_;
v___y_6439_ = v___y_6629_;
v___y_6440_ = v_mvarIds_6635_;
v___y_6441_ = v_eqProof_6634_;
v___y_6442_ = v_eNew_6633_;
v___y_6443_ = v___x_6636_;
goto v___jp_6435_;
}
else
{
lean_object* v_eNew_6637_; lean_object* v_eqProof_6638_; lean_object* v_mvarIds_6639_; lean_object* v___x_6640_; 
v_eNew_6637_ = lean_ctor_get(v_a_6632_, 0);
lean_inc_ref(v_eNew_6637_);
v_eqProof_6638_ = lean_ctor_get(v_a_6632_, 1);
lean_inc_ref(v_eqProof_6638_);
v_mvarIds_6639_ = lean_ctor_get(v_a_6632_, 2);
lean_inc(v_mvarIds_6639_);
lean_dec(v_a_6632_);
v___x_6640_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__12));
v___y_6436_ = v___y_6624_;
v___y_6437_ = v___y_6625_;
v___y_6438_ = v___y_6628_;
v___y_6439_ = v___y_6629_;
v___y_6440_ = v_mvarIds_6639_;
v___y_6441_ = v_eqProof_6638_;
v___y_6442_ = v_eNew_6637_;
v___y_6443_ = v___x_6640_;
goto v___jp_6435_;
}
}
else
{
lean_object* v_a_6641_; lean_object* v___x_6643_; uint8_t v_isShared_6644_; uint8_t v_isSharedCheck_6648_; 
lean_dec(v___y_6629_);
lean_dec_ref(v___y_6628_);
lean_dec(v___y_6625_);
lean_dec_ref(v___y_6624_);
lean_dec_ref(v_e_6242_);
v_a_6641_ = lean_ctor_get(v___x_6631_, 0);
v_isSharedCheck_6648_ = !lean_is_exclusive(v___x_6631_);
if (v_isSharedCheck_6648_ == 0)
{
v___x_6643_ = v___x_6631_;
v_isShared_6644_ = v_isSharedCheck_6648_;
goto v_resetjp_6642_;
}
else
{
lean_inc(v_a_6641_);
lean_dec(v___x_6631_);
v___x_6643_ = lean_box(0);
v_isShared_6644_ = v_isSharedCheck_6648_;
goto v_resetjp_6642_;
}
v_resetjp_6642_:
{
lean_object* v___x_6646_; 
if (v_isShared_6644_ == 0)
{
v___x_6646_ = v___x_6643_;
goto v_reusejp_6645_;
}
else
{
lean_object* v_reuseFailAlloc_6647_; 
v_reuseFailAlloc_6647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6647_, 0, v_a_6641_);
v___x_6646_ = v_reuseFailAlloc_6647_;
goto v_reusejp_6645_;
}
v_reusejp_6645_:
{
return v___x_6646_;
}
}
}
}
}
}
v___jp_6649_:
{
lean_object* v_keyedConfig_6655_; uint8_t v_trackZetaDelta_6656_; lean_object* v_zetaDeltaSet_6657_; lean_object* v_lctx_6658_; lean_object* v_localInstances_6659_; lean_object* v_defEqCtx_x3f_6660_; lean_object* v_synthPendingDepth_6661_; lean_object* v_customCanUnfoldPredicate_x3f_6662_; uint8_t v_univApprox_6663_; uint8_t v_inTypeClassResolution_6664_; uint8_t v_cacheInferType_6665_; uint8_t v___x_6666_; uint8_t v___x_6667_; lean_object* v___x_6668_; lean_object* v___x_6669_; lean_object* v___x_6670_; 
v_keyedConfig_6655_ = lean_ctor_get(v___y_6651_, 0);
v_trackZetaDelta_6656_ = lean_ctor_get_uint8(v___y_6651_, sizeof(void*)*7);
v_zetaDeltaSet_6657_ = lean_ctor_get(v___y_6651_, 1);
v_lctx_6658_ = lean_ctor_get(v___y_6651_, 2);
v_localInstances_6659_ = lean_ctor_get(v___y_6651_, 3);
v_defEqCtx_x3f_6660_ = lean_ctor_get(v___y_6651_, 4);
v_synthPendingDepth_6661_ = lean_ctor_get(v___y_6651_, 5);
v_customCanUnfoldPredicate_x3f_6662_ = lean_ctor_get(v___y_6651_, 6);
v_univApprox_6663_ = lean_ctor_get_uint8(v___y_6651_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_6664_ = lean_ctor_get_uint8(v___y_6651_, sizeof(void*)*7 + 2);
v_cacheInferType_6665_ = lean_ctor_get_uint8(v___y_6651_, sizeof(void*)*7 + 3);
v___x_6666_ = 0;
v___x_6667_ = 2;
lean_inc_ref(v_keyedConfig_6655_);
v___x_6668_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_6667_, v_keyedConfig_6655_);
lean_inc(v_customCanUnfoldPredicate_x3f_6662_);
lean_inc(v_synthPendingDepth_6661_);
lean_inc(v_defEqCtx_x3f_6660_);
lean_inc_ref(v_localInstances_6659_);
lean_inc_ref(v_lctx_6658_);
lean_inc(v_zetaDeltaSet_6657_);
v___x_6669_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_6669_, 0, v___x_6668_);
lean_ctor_set(v___x_6669_, 1, v_zetaDeltaSet_6657_);
lean_ctor_set(v___x_6669_, 2, v_lctx_6658_);
lean_ctor_set(v___x_6669_, 3, v_localInstances_6659_);
lean_ctor_set(v___x_6669_, 4, v_defEqCtx_x3f_6660_);
lean_ctor_set(v___x_6669_, 5, v_synthPendingDepth_6661_);
lean_ctor_set(v___x_6669_, 6, v_customCanUnfoldPredicate_x3f_6662_);
lean_ctor_set_uint8(v___x_6669_, sizeof(void*)*7, v_trackZetaDelta_6656_);
lean_ctor_set_uint8(v___x_6669_, sizeof(void*)*7 + 1, v_univApprox_6663_);
lean_ctor_set_uint8(v___x_6669_, sizeof(void*)*7 + 2, v_inTypeClassResolution_6664_);
lean_ctor_set_uint8(v___x_6669_, sizeof(void*)*7 + 3, v_cacheInferType_6665_);
v___x_6670_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_6475_, v_maxMVars_x3f_6650_, v___x_6666_, v___x_6669_, v___y_6652_, v___y_6653_, v___y_6654_);
lean_dec_ref_known(v___x_6669_, 7);
if (lean_obj_tag(v___x_6670_) == 0)
{
lean_object* v_a_6671_; lean_object* v_snd_6672_; lean_object* v_fst_6673_; lean_object* v_fst_6674_; lean_object* v_snd_6675_; lean_object* v___x_6676_; lean_object* v___x_6677_; lean_object* v___x_6678_; uint8_t v___x_6679_; 
v_a_6671_ = lean_ctor_get(v___x_6670_, 0);
lean_inc(v_a_6671_);
lean_dec_ref_known(v___x_6670_, 1);
v_snd_6672_ = lean_ctor_get(v_a_6671_, 1);
lean_inc(v_snd_6672_);
v_fst_6673_ = lean_ctor_get(v_a_6671_, 0);
lean_inc(v_fst_6673_);
lean_dec(v_a_6671_);
v_fst_6674_ = lean_ctor_get(v_snd_6672_, 0);
lean_inc(v_fst_6674_);
v_snd_6675_ = lean_ctor_get(v_snd_6672_, 1);
lean_inc(v_snd_6675_);
lean_dec(v_snd_6672_);
v___x_6676_ = l_Lean_Expr_cleanupAnnotations(v_snd_6675_);
v___x_6677_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__14));
v___x_6678_ = lean_unsigned_to_nat(2u);
v___x_6679_ = l_Lean_Expr_isAppOfArity(v___x_6676_, v___x_6677_, v___x_6678_);
if (v___x_6679_ == 0)
{
lean_object* v___x_6680_; lean_object* v___x_6681_; uint8_t v___x_6682_; 
v___x_6680_ = ((lean_object*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__15));
v___x_6681_ = lean_unsigned_to_nat(3u);
v___x_6682_ = l_Lean_Expr_isAppOfArity(v___x_6676_, v___x_6680_, v___x_6681_);
lean_inc(v_fst_6673_);
v___y_6621_ = v___x_6676_;
v___y_6622_ = v_fst_6673_;
v___y_6623_ = v_fst_6674_;
v___y_6624_ = v___y_6653_;
v___y_6625_ = v___y_6654_;
v___y_6626_ = v___x_6667_;
v___y_6627_ = v_fst_6673_;
v___y_6628_ = v___y_6651_;
v___y_6629_ = v___y_6652_;
v___y_6630_ = v___x_6682_;
goto v___jp_6620_;
}
else
{
lean_inc(v_fst_6673_);
v___y_6621_ = v___x_6676_;
v___y_6622_ = v_fst_6673_;
v___y_6623_ = v_fst_6674_;
v___y_6624_ = v___y_6653_;
v___y_6625_ = v___y_6654_;
v___y_6626_ = v___x_6667_;
v___y_6627_ = v_fst_6673_;
v___y_6628_ = v___y_6651_;
v___y_6629_ = v___y_6652_;
v___y_6630_ = v___x_6679_;
goto v___jp_6620_;
}
}
else
{
lean_object* v_a_6683_; lean_object* v___x_6685_; uint8_t v_isShared_6686_; uint8_t v_isSharedCheck_6690_; 
lean_dec(v___y_6654_);
lean_dec_ref(v___y_6653_);
lean_dec(v___y_6652_);
lean_dec_ref(v___y_6651_);
lean_del_object(v___x_6477_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6683_ = lean_ctor_get(v___x_6670_, 0);
v_isSharedCheck_6690_ = !lean_is_exclusive(v___x_6670_);
if (v_isSharedCheck_6690_ == 0)
{
v___x_6685_ = v___x_6670_;
v_isShared_6686_ = v_isSharedCheck_6690_;
goto v_resetjp_6684_;
}
else
{
lean_inc(v_a_6683_);
lean_dec(v___x_6670_);
v___x_6685_ = lean_box(0);
v_isShared_6686_ = v_isSharedCheck_6690_;
goto v_resetjp_6684_;
}
v_resetjp_6684_:
{
lean_object* v___x_6688_; 
if (v_isShared_6686_ == 0)
{
v___x_6688_ = v___x_6685_;
goto v_reusejp_6687_;
}
else
{
lean_object* v_reuseFailAlloc_6689_; 
v_reuseFailAlloc_6689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6689_, 0, v_a_6683_);
v___x_6688_ = v_reuseFailAlloc_6689_;
goto v_reusejp_6687_;
}
v_reusejp_6687_:
{
return v___x_6688_;
}
}
}
}
}
}
else
{
lean_object* v_a_6714_; lean_object* v___x_6716_; uint8_t v_isShared_6717_; uint8_t v_isSharedCheck_6721_; 
lean_dec(v___y_6249_);
lean_dec_ref(v___y_6248_);
lean_dec(v___y_6247_);
lean_dec_ref(v___y_6246_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6714_ = lean_ctor_get(v___x_6472_, 0);
v_isSharedCheck_6721_ = !lean_is_exclusive(v___x_6472_);
if (v_isSharedCheck_6721_ == 0)
{
v___x_6716_ = v___x_6472_;
v_isShared_6717_ = v_isSharedCheck_6721_;
goto v_resetjp_6715_;
}
else
{
lean_inc(v_a_6714_);
lean_dec(v___x_6472_);
v___x_6716_ = lean_box(0);
v_isShared_6717_ = v_isSharedCheck_6721_;
goto v_resetjp_6715_;
}
v_resetjp_6715_:
{
lean_object* v___x_6719_; 
if (v_isShared_6717_ == 0)
{
v___x_6719_ = v___x_6716_;
goto v_reusejp_6718_;
}
else
{
lean_object* v_reuseFailAlloc_6720_; 
v_reuseFailAlloc_6720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6720_, 0, v_a_6714_);
v___x_6719_ = v_reuseFailAlloc_6720_;
goto v_reusejp_6718_;
}
v_reusejp_6718_:
{
return v___x_6719_;
}
}
}
}
else
{
lean_object* v_a_6722_; lean_object* v___x_6724_; uint8_t v_isShared_6725_; uint8_t v_isSharedCheck_6729_; 
lean_dec(v___y_6249_);
lean_dec_ref(v___y_6248_);
lean_dec(v___y_6247_);
lean_dec_ref(v___y_6246_);
lean_dec_ref(v_mvarIds_6244_);
lean_dec_ref(v_e_6242_);
lean_dec_ref(v_config_6241_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6722_ = lean_ctor_get(v___x_6471_, 0);
v_isSharedCheck_6729_ = !lean_is_exclusive(v___x_6471_);
if (v_isSharedCheck_6729_ == 0)
{
v___x_6724_ = v___x_6471_;
v_isShared_6725_ = v_isSharedCheck_6729_;
goto v_resetjp_6723_;
}
else
{
lean_inc(v_a_6722_);
lean_dec(v___x_6471_);
v___x_6724_ = lean_box(0);
v_isShared_6725_ = v_isSharedCheck_6729_;
goto v_resetjp_6723_;
}
v_resetjp_6723_:
{
lean_object* v___x_6727_; 
if (v_isShared_6725_ == 0)
{
v___x_6727_ = v___x_6724_;
goto v_reusejp_6726_;
}
else
{
lean_object* v_reuseFailAlloc_6728_; 
v_reuseFailAlloc_6728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6728_, 0, v_a_6722_);
v___x_6727_ = v_reuseFailAlloc_6728_;
goto v_reusejp_6726_;
}
v_reusejp_6726_:
{
return v___x_6727_;
}
}
}
v___jp_6251_:
{
lean_object* v___x_6257_; lean_object* v___x_6258_; lean_object* v___x_6259_; lean_object* v___x_6260_; 
v___x_6257_ = l_Array_append___redArg(v___y_6252_, v___y_6256_);
lean_dec_ref(v___y_6256_);
v___x_6258_ = lean_array_to_list(v___x_6257_);
v___x_6259_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_6259_, 0, v___y_6255_);
lean_ctor_set(v___x_6259_, 1, v___y_6254_);
lean_ctor_set(v___x_6259_, 2, v___x_6258_);
lean_ctor_set(v___x_6259_, 3, v___y_6253_);
v___x_6260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6260_, 0, v___x_6259_);
return v___x_6260_;
}
v___jp_6261_:
{
lean_object* v___x_6272_; 
v___x_6272_ = l_Lean_Meta_getMVarsNoDelayed(v_hrel_6240_, v___y_6267_, v___y_6266_, v___y_6269_, v___y_6270_);
lean_dec(v___y_6270_);
lean_dec_ref(v___y_6269_);
lean_dec(v___y_6266_);
lean_dec_ref(v___y_6267_);
if (lean_obj_tag(v___x_6272_) == 0)
{
lean_object* v_a_6273_; lean_object* v___x_6274_; lean_object* v___x_6275_; uint8_t v___x_6276_; 
v_a_6273_ = lean_ctor_get(v___x_6272_, 0);
lean_inc(v_a_6273_);
lean_dec_ref_known(v___x_6272_, 1);
v___x_6274_ = lean_array_get_size(v_a_6273_);
v___x_6275_ = lean_mk_empty_array_with_capacity(v___y_6268_);
v___x_6276_ = lean_nat_dec_lt(v___y_6268_, v___x_6274_);
if (v___x_6276_ == 0)
{
lean_dec(v_a_6273_);
v___y_6252_ = v_a_6271_;
v___y_6253_ = v___y_6262_;
v___y_6254_ = v___y_6265_;
v___y_6255_ = v___y_6264_;
v___y_6256_ = v___x_6275_;
goto v___jp_6251_;
}
else
{
uint8_t v___x_6277_; 
v___x_6277_ = lean_nat_dec_le(v___x_6274_, v___x_6274_);
if (v___x_6277_ == 0)
{
if (v___x_6276_ == 0)
{
lean_dec(v_a_6273_);
v___y_6252_ = v_a_6271_;
v___y_6253_ = v___y_6262_;
v___y_6254_ = v___y_6265_;
v___y_6255_ = v___y_6264_;
v___y_6256_ = v___x_6275_;
goto v___jp_6251_;
}
else
{
size_t v___x_6278_; lean_object* v___x_6279_; 
v___x_6278_ = lean_usize_of_nat(v___x_6274_);
v___x_6279_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3(v_a_6271_, v_a_6273_, v___y_6263_, v___x_6278_, v___x_6275_);
lean_dec(v_a_6273_);
v___y_6252_ = v_a_6271_;
v___y_6253_ = v___y_6262_;
v___y_6254_ = v___y_6265_;
v___y_6255_ = v___y_6264_;
v___y_6256_ = v___x_6279_;
goto v___jp_6251_;
}
}
else
{
size_t v___x_6280_; lean_object* v___x_6281_; 
v___x_6280_ = lean_usize_of_nat(v___x_6274_);
v___x_6281_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__3(v_a_6271_, v_a_6273_, v___y_6263_, v___x_6280_, v___x_6275_);
lean_dec(v_a_6273_);
v___y_6252_ = v_a_6271_;
v___y_6253_ = v___y_6262_;
v___y_6254_ = v___y_6265_;
v___y_6255_ = v___y_6264_;
v___y_6256_ = v___x_6281_;
goto v___jp_6251_;
}
}
}
else
{
lean_object* v_a_6282_; lean_object* v___x_6284_; uint8_t v_isShared_6285_; uint8_t v_isSharedCheck_6289_; 
lean_dec_ref(v_a_6271_);
lean_dec_ref(v___y_6265_);
lean_dec_ref(v___y_6264_);
lean_dec(v___y_6262_);
v_a_6282_ = lean_ctor_get(v___x_6272_, 0);
v_isSharedCheck_6289_ = !lean_is_exclusive(v___x_6272_);
if (v_isSharedCheck_6289_ == 0)
{
v___x_6284_ = v___x_6272_;
v_isShared_6285_ = v_isSharedCheck_6289_;
goto v_resetjp_6283_;
}
else
{
lean_inc(v_a_6282_);
lean_dec(v___x_6272_);
v___x_6284_ = lean_box(0);
v_isShared_6285_ = v_isSharedCheck_6289_;
goto v_resetjp_6283_;
}
v_resetjp_6283_:
{
lean_object* v___x_6287_; 
if (v_isShared_6285_ == 0)
{
v___x_6287_ = v___x_6284_;
goto v_reusejp_6286_;
}
else
{
lean_object* v_reuseFailAlloc_6288_; 
v_reuseFailAlloc_6288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6288_, 0, v_a_6282_);
v___x_6287_ = v_reuseFailAlloc_6288_;
goto v_reusejp_6286_;
}
v_reusejp_6286_:
{
return v___x_6287_;
}
}
}
}
v___jp_6290_:
{
if (lean_obj_tag(v___y_6300_) == 0)
{
lean_object* v_a_6301_; 
v_a_6301_ = lean_ctor_get(v___y_6300_, 0);
lean_inc(v_a_6301_);
lean_dec_ref_known(v___y_6300_, 1);
v___y_6262_ = v___y_6291_;
v___y_6263_ = v___y_6294_;
v___y_6264_ = v___y_6293_;
v___y_6265_ = v___y_6292_;
v___y_6266_ = v___y_6295_;
v___y_6267_ = v___y_6296_;
v___y_6268_ = v___y_6297_;
v___y_6269_ = v___y_6298_;
v___y_6270_ = v___y_6299_;
v_a_6271_ = v_a_6301_;
goto v___jp_6261_;
}
else
{
lean_object* v_a_6302_; lean_object* v___x_6304_; uint8_t v_isShared_6305_; uint8_t v_isSharedCheck_6309_; 
lean_dec(v___y_6299_);
lean_dec_ref(v___y_6298_);
lean_dec_ref(v___y_6296_);
lean_dec(v___y_6295_);
lean_dec_ref(v___y_6293_);
lean_dec_ref(v___y_6292_);
lean_dec(v___y_6291_);
lean_dec_ref(v_hrel_6240_);
v_a_6302_ = lean_ctor_get(v___y_6300_, 0);
v_isSharedCheck_6309_ = !lean_is_exclusive(v___y_6300_);
if (v_isSharedCheck_6309_ == 0)
{
v___x_6304_ = v___y_6300_;
v_isShared_6305_ = v_isSharedCheck_6309_;
goto v_resetjp_6303_;
}
else
{
lean_inc(v_a_6302_);
lean_dec(v___y_6300_);
v___x_6304_ = lean_box(0);
v_isShared_6305_ = v_isSharedCheck_6309_;
goto v_resetjp_6303_;
}
v_resetjp_6303_:
{
lean_object* v___x_6307_; 
if (v_isShared_6305_ == 0)
{
v___x_6307_ = v___x_6304_;
goto v_reusejp_6306_;
}
else
{
lean_object* v_reuseFailAlloc_6308_; 
v_reuseFailAlloc_6308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6308_, 0, v_a_6302_);
v___x_6307_ = v_reuseFailAlloc_6308_;
goto v_reusejp_6306_;
}
v_reusejp_6306_:
{
return v___x_6307_;
}
}
}
}
v___jp_6310_:
{
lean_object* v___x_6323_; 
v___x_6323_ = l_Lean_Meta_postprocessAppMVars(v___x_6239_, v_goal_6238_, v___y_6315_, v___y_6314_, v___y_6322_, v___y_6319_, v___y_6317_, v___y_6316_, v___y_6318_, v___y_6321_);
if (lean_obj_tag(v___x_6323_) == 0)
{
size_t v_sz_6324_; size_t v___x_6325_; lean_object* v___x_6326_; lean_object* v___x_6327_; lean_object* v___x_6328_; lean_object* v___x_6329_; lean_object* v___x_6330_; uint8_t v___x_6331_; 
lean_dec_ref_known(v___x_6323_, 1);
v_sz_6324_ = lean_array_size(v___y_6315_);
v___x_6325_ = ((size_t)0ULL);
v___x_6326_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_grewrite_spec__1(v_sz_6324_, v___x_6325_, v___y_6315_);
v___x_6327_ = l_Array_append___redArg(v___y_6320_, v___x_6326_);
lean_dec_ref(v___x_6326_);
v___x_6328_ = lean_unsigned_to_nat(0u);
v___x_6329_ = lean_array_get_size(v___x_6327_);
v___x_6330_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_processGCongrHypothesis_spec__22___closed__2));
v___x_6331_ = lean_nat_dec_lt(v___x_6328_, v___x_6329_);
if (v___x_6331_ == 0)
{
lean_dec_ref(v___x_6327_);
v___y_6262_ = v___y_6311_;
v___y_6263_ = v___x_6325_;
v___y_6264_ = v___y_6313_;
v___y_6265_ = v___y_6312_;
v___y_6266_ = v___y_6316_;
v___y_6267_ = v___y_6317_;
v___y_6268_ = v___x_6328_;
v___y_6269_ = v___y_6318_;
v___y_6270_ = v___y_6321_;
v_a_6271_ = v___x_6330_;
goto v___jp_6261_;
}
else
{
uint8_t v___x_6332_; 
v___x_6332_ = lean_nat_dec_le(v___x_6329_, v___x_6329_);
if (v___x_6332_ == 0)
{
if (v___x_6331_ == 0)
{
lean_dec_ref(v___x_6327_);
v___y_6262_ = v___y_6311_;
v___y_6263_ = v___x_6325_;
v___y_6264_ = v___y_6313_;
v___y_6265_ = v___y_6312_;
v___y_6266_ = v___y_6316_;
v___y_6267_ = v___y_6317_;
v___y_6268_ = v___x_6328_;
v___y_6269_ = v___y_6318_;
v___y_6270_ = v___y_6321_;
v_a_6271_ = v___x_6330_;
goto v___jp_6261_;
}
else
{
size_t v___x_6333_; lean_object* v___x_6334_; 
v___x_6333_ = lean_usize_of_nat(v___x_6329_);
v___x_6334_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4(v___x_6327_, v___x_6325_, v___x_6333_, v___x_6330_, v___y_6317_, v___y_6316_, v___y_6318_, v___y_6321_);
lean_dec_ref(v___x_6327_);
v___y_6291_ = v___y_6311_;
v___y_6292_ = v___y_6312_;
v___y_6293_ = v___y_6313_;
v___y_6294_ = v___x_6325_;
v___y_6295_ = v___y_6316_;
v___y_6296_ = v___y_6317_;
v___y_6297_ = v___x_6328_;
v___y_6298_ = v___y_6318_;
v___y_6299_ = v___y_6321_;
v___y_6300_ = v___x_6334_;
goto v___jp_6290_;
}
}
else
{
size_t v___x_6335_; lean_object* v___x_6336_; 
v___x_6335_ = lean_usize_of_nat(v___x_6329_);
v___x_6336_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_grewrite_spec__4(v___x_6327_, v___x_6325_, v___x_6335_, v___x_6330_, v___y_6317_, v___y_6316_, v___y_6318_, v___y_6321_);
lean_dec_ref(v___x_6327_);
v___y_6291_ = v___y_6311_;
v___y_6292_ = v___y_6312_;
v___y_6293_ = v___y_6313_;
v___y_6294_ = v___x_6325_;
v___y_6295_ = v___y_6316_;
v___y_6296_ = v___y_6317_;
v___y_6297_ = v___x_6328_;
v___y_6298_ = v___y_6318_;
v___y_6299_ = v___y_6321_;
v___y_6300_ = v___x_6336_;
goto v___jp_6290_;
}
}
}
else
{
lean_object* v_a_6337_; lean_object* v___x_6339_; uint8_t v_isShared_6340_; uint8_t v_isSharedCheck_6344_; 
lean_dec(v___y_6321_);
lean_dec_ref(v___y_6320_);
lean_dec_ref(v___y_6318_);
lean_dec_ref(v___y_6317_);
lean_dec(v___y_6316_);
lean_dec_ref(v___y_6315_);
lean_dec_ref(v___y_6313_);
lean_dec_ref(v___y_6312_);
lean_dec(v___y_6311_);
lean_dec_ref(v_hrel_6240_);
v_a_6337_ = lean_ctor_get(v___x_6323_, 0);
v_isSharedCheck_6344_ = !lean_is_exclusive(v___x_6323_);
if (v_isSharedCheck_6344_ == 0)
{
v___x_6339_ = v___x_6323_;
v_isShared_6340_ = v_isSharedCheck_6344_;
goto v_resetjp_6338_;
}
else
{
lean_inc(v_a_6337_);
lean_dec(v___x_6323_);
v___x_6339_ = lean_box(0);
v_isShared_6340_ = v_isSharedCheck_6344_;
goto v_resetjp_6338_;
}
v_resetjp_6338_:
{
lean_object* v___x_6342_; 
if (v_isShared_6340_ == 0)
{
v___x_6342_ = v___x_6339_;
goto v_reusejp_6341_;
}
else
{
lean_object* v_reuseFailAlloc_6343_; 
v_reuseFailAlloc_6343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6343_, 0, v_a_6337_);
v___x_6342_ = v_reuseFailAlloc_6343_;
goto v_reusejp_6341_;
}
v_reusejp_6341_:
{
return v___x_6342_;
}
}
}
}
v___jp_6345_:
{
lean_object* v_snd_6355_; lean_object* v_fst_6356_; lean_object* v_fst_6357_; lean_object* v_snd_6358_; lean_object* v_options_6359_; lean_object* v___x_6360_; uint8_t v___x_6361_; 
v_snd_6355_ = lean_ctor_get(v_snd_6350_, 1);
lean_inc(v_snd_6355_);
v_fst_6356_ = lean_ctor_get(v_snd_6350_, 0);
lean_inc(v_fst_6356_);
lean_dec_ref(v_snd_6350_);
v_fst_6357_ = lean_ctor_get(v_snd_6355_, 0);
lean_inc(v_fst_6357_);
v_snd_6358_ = lean_ctor_get(v_snd_6355_, 1);
lean_inc(v_snd_6358_);
lean_dec(v_snd_6355_);
v_options_6359_ = lean_ctor_get(v___y_6353_, 2);
v___x_6360_ = l_Lean_Meta_tactic_skipAssignedInstances;
v___x_6361_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_GRewriteLemma_apply_spec__3(v_options_6359_, v___x_6360_);
if (v___x_6361_ == 0)
{
uint8_t v___x_6362_; 
v___x_6362_ = 1;
v___y_6311_ = v_fst_6349_;
v___y_6312_ = v_fst_6357_;
v___y_6313_ = v_fst_6356_;
v___y_6314_ = v___y_6347_;
v___y_6315_ = v___y_6346_;
v___y_6316_ = v___y_6352_;
v___y_6317_ = v___y_6351_;
v___y_6318_ = v___y_6353_;
v___y_6319_ = v___y_6348_;
v___y_6320_ = v_snd_6358_;
v___y_6321_ = v___y_6354_;
v___y_6322_ = v___x_6362_;
goto v___jp_6310_;
}
else
{
v___y_6311_ = v_fst_6349_;
v___y_6312_ = v_fst_6357_;
v___y_6313_ = v_fst_6356_;
v___y_6314_ = v___y_6347_;
v___y_6315_ = v___y_6346_;
v___y_6316_ = v___y_6352_;
v___y_6317_ = v___y_6351_;
v___y_6318_ = v___y_6353_;
v___y_6319_ = v___y_6348_;
v___y_6320_ = v_snd_6358_;
v___y_6321_ = v___y_6354_;
v___y_6322_ = v___y_6348_;
goto v___jp_6310_;
}
}
v___jp_6363_:
{
if (lean_obj_tag(v___y_6371_) == 0)
{
lean_object* v_a_6372_; lean_object* v_fst_6373_; lean_object* v_snd_6374_; 
v_a_6372_ = lean_ctor_get(v___y_6371_, 0);
lean_inc(v_a_6372_);
lean_dec_ref_known(v___y_6371_, 1);
v_fst_6373_ = lean_ctor_get(v_a_6372_, 0);
lean_inc(v_fst_6373_);
v_snd_6374_ = lean_ctor_get(v_a_6372_, 1);
lean_inc(v_snd_6374_);
lean_dec(v_a_6372_);
v___y_6346_ = v___y_6366_;
v___y_6347_ = v___y_6365_;
v___y_6348_ = v___y_6368_;
v_fst_6349_ = v_fst_6373_;
v_snd_6350_ = v_snd_6374_;
v___y_6351_ = v___y_6367_;
v___y_6352_ = v___y_6369_;
v___y_6353_ = v___y_6364_;
v___y_6354_ = v___y_6370_;
goto v___jp_6345_;
}
else
{
lean_object* v_a_6375_; lean_object* v___x_6377_; uint8_t v_isShared_6378_; uint8_t v_isSharedCheck_6382_; 
lean_dec(v___y_6370_);
lean_dec(v___y_6369_);
lean_dec_ref(v___y_6367_);
lean_dec_ref(v___y_6366_);
lean_dec_ref(v___y_6365_);
lean_dec_ref(v___y_6364_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6375_ = lean_ctor_get(v___y_6371_, 0);
v_isSharedCheck_6382_ = !lean_is_exclusive(v___y_6371_);
if (v_isSharedCheck_6382_ == 0)
{
v___x_6377_ = v___y_6371_;
v_isShared_6378_ = v_isSharedCheck_6382_;
goto v_resetjp_6376_;
}
else
{
lean_inc(v_a_6375_);
lean_dec(v___y_6371_);
v___x_6377_ = lean_box(0);
v_isShared_6378_ = v_isSharedCheck_6382_;
goto v_resetjp_6376_;
}
v_resetjp_6376_:
{
lean_object* v___x_6380_; 
if (v_isShared_6378_ == 0)
{
v___x_6380_ = v___x_6377_;
goto v_reusejp_6379_;
}
else
{
lean_object* v_reuseFailAlloc_6381_; 
v_reuseFailAlloc_6381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6381_, 0, v_a_6375_);
v___x_6380_ = v_reuseFailAlloc_6381_;
goto v_reusejp_6379_;
}
v_reusejp_6379_:
{
return v___x_6380_;
}
}
}
}
v___jp_6383_:
{
if (v___y_6395_ == 0)
{
lean_object* v___x_6396_; lean_object* v___x_6397_; lean_object* v___x_6398_; lean_object* v___x_6399_; lean_object* v___x_6400_; 
v___x_6396_ = l_Lean_MessageData_ofExpr(v___y_6385_);
v___x_6397_ = lean_obj_once(&lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1, &lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1_once, _init_lp_mathlib_Lean_MVarId_grewrite___lam__3___closed__1);
v___x_6398_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6398_, 0, v___x_6396_);
lean_ctor_set(v___x_6398_, 1, v___x_6397_);
v___x_6399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6399_, 0, v___x_6398_);
lean_inc(v_goal_6238_);
lean_inc(v___x_6239_);
v___x_6400_ = l_Lean_Meta_throwTacticEx___redArg(v___x_6239_, v_goal_6238_, v___x_6399_, v___y_6394_, v___y_6392_, v___y_6386_, v___y_6393_);
if (lean_obj_tag(v___x_6400_) == 0)
{
lean_object* v_a_6401_; lean_object* v___x_6402_; 
v_a_6401_ = lean_ctor_get(v___x_6400_, 0);
lean_inc(v_a_6401_);
lean_dec_ref_known(v___x_6400_, 1);
lean_inc(v___y_6393_);
lean_inc_ref(v___y_6386_);
lean_inc(v___y_6392_);
v___x_6402_ = lean_apply_6(v___y_6389_, v_a_6401_, v___y_6394_, v___y_6392_, v___y_6386_, v___y_6393_, lean_box(0));
v___y_6364_ = v___y_6386_;
v___y_6365_ = v___y_6388_;
v___y_6366_ = v___y_6387_;
v___y_6367_ = v___y_6390_;
v___y_6368_ = v___y_6391_;
v___y_6369_ = v___y_6392_;
v___y_6370_ = v___y_6393_;
v___y_6371_ = v___x_6402_;
goto v___jp_6363_;
}
else
{
lean_object* v_a_6403_; lean_object* v___x_6405_; uint8_t v_isShared_6406_; uint8_t v_isSharedCheck_6410_; 
lean_dec_ref(v___y_6394_);
lean_dec(v___y_6393_);
lean_dec(v___y_6392_);
lean_dec_ref(v___y_6390_);
lean_dec_ref(v___y_6389_);
lean_dec_ref(v___y_6388_);
lean_dec_ref(v___y_6387_);
lean_dec_ref(v___y_6386_);
lean_dec_ref(v_hrel_6240_);
lean_dec(v___x_6239_);
lean_dec(v_goal_6238_);
v_a_6403_ = lean_ctor_get(v___x_6400_, 0);
v_isSharedCheck_6410_ = !lean_is_exclusive(v___x_6400_);
if (v_isSharedCheck_6410_ == 0)
{
v___x_6405_ = v___x_6400_;
v_isShared_6406_ = v_isSharedCheck_6410_;
goto v_resetjp_6404_;
}
else
{
lean_inc(v_a_6403_);
lean_dec(v___x_6400_);
v___x_6405_ = lean_box(0);
v_isShared_6406_ = v_isSharedCheck_6410_;
goto v_resetjp_6404_;
}
v_resetjp_6404_:
{
lean_object* v___x_6408_; 
if (v_isShared_6406_ == 0)
{
v___x_6408_ = v___x_6405_;
goto v_reusejp_6407_;
}
else
{
lean_object* v_reuseFailAlloc_6409_; 
v_reuseFailAlloc_6409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6409_, 0, v_a_6403_);
v___x_6408_ = v_reuseFailAlloc_6409_;
goto v_reusejp_6407_;
}
v_reusejp_6407_:
{
return v___x_6408_;
}
}
}
}
else
{
lean_dec_ref(v___y_6385_);
if (v_symm_6243_ == 0)
{
lean_object* v___x_6411_; lean_object* v___x_6412_; 
v___x_6411_ = lean_box(v___y_6395_);
lean_inc(v___y_6393_);
lean_inc_ref(v___y_6386_);
lean_inc(v___y_6392_);
v___x_6412_ = lean_apply_6(v___y_6389_, v___x_6411_, v___y_6394_, v___y_6392_, v___y_6386_, v___y_6393_, lean_box(0));
v___y_6364_ = v___y_6386_;
v___y_6365_ = v___y_6388_;
v___y_6366_ = v___y_6387_;
v___y_6367_ = v___y_6390_;
v___y_6368_ = v___y_6391_;
v___y_6369_ = v___y_6392_;
v___y_6370_ = v___y_6393_;
v___y_6371_ = v___x_6412_;
goto v___jp_6363_;
}
else
{
lean_object* v___x_6413_; lean_object* v___x_6414_; 
v___x_6413_ = lean_box(v___y_6384_);
lean_inc(v___y_6393_);
lean_inc_ref(v___y_6386_);
lean_inc(v___y_6392_);
v___x_6414_ = lean_apply_6(v___y_6389_, v___x_6413_, v___y_6394_, v___y_6392_, v___y_6386_, v___y_6393_, lean_box(0));
v___y_6364_ = v___y_6386_;
v___y_6365_ = v___y_6388_;
v___y_6366_ = v___y_6387_;
v___y_6367_ = v___y_6390_;
v___y_6368_ = v___y_6391_;
v___y_6369_ = v___y_6392_;
v___y_6370_ = v___y_6393_;
v___y_6371_ = v___x_6414_;
goto v___jp_6363_;
}
}
}
v___jp_6415_:
{
if (v___y_6430_ == 0)
{
uint8_t v___x_6431_; 
v___x_6431_ = lean_expr_eqv(v___y_6417_, v___y_6425_);
lean_dec_ref(v___y_6425_);
lean_dec_ref(v___y_6417_);
if (v___x_6431_ == 0)
{
lean_dec_ref(v___y_6422_);
lean_dec_ref(v___y_6416_);
v___y_6384_ = v___y_6430_;
v___y_6385_ = v___y_6424_;
v___y_6386_ = v___y_6418_;
v___y_6387_ = v___y_6419_;
v___y_6388_ = v___y_6420_;
v___y_6389_ = v___y_6421_;
v___y_6390_ = v___y_6426_;
v___y_6391_ = v___y_6427_;
v___y_6392_ = v___y_6428_;
v___y_6393_ = v___y_6423_;
v___y_6394_ = v___y_6429_;
v___y_6395_ = v___x_6431_;
goto v___jp_6383_;
}
else
{
uint8_t v___x_6432_; 
v___x_6432_ = lean_expr_eqv(v___y_6416_, v___y_6422_);
lean_dec_ref(v___y_6422_);
lean_dec_ref(v___y_6416_);
v___y_6384_ = v___y_6430_;
v___y_6385_ = v___y_6424_;
v___y_6386_ = v___y_6418_;
v___y_6387_ = v___y_6419_;
v___y_6388_ = v___y_6420_;
v___y_6389_ = v___y_6421_;
v___y_6390_ = v___y_6426_;
v___y_6391_ = v___y_6427_;
v___y_6392_ = v___y_6428_;
v___y_6393_ = v___y_6423_;
v___y_6394_ = v___y_6429_;
v___y_6395_ = v___x_6432_;
goto v___jp_6383_;
}
}
else
{
lean_object* v___x_6433_; lean_object* v___x_6434_; 
lean_dec_ref(v___y_6425_);
lean_dec_ref(v___y_6424_);
lean_dec_ref(v___y_6422_);
lean_dec_ref(v___y_6417_);
lean_dec_ref(v___y_6416_);
v___x_6433_ = lean_box(v_symm_6243_);
lean_inc(v___y_6423_);
lean_inc_ref(v___y_6418_);
lean_inc(v___y_6428_);
v___x_6434_ = lean_apply_6(v___y_6421_, v___x_6433_, v___y_6429_, v___y_6428_, v___y_6418_, v___y_6423_, lean_box(0));
v___y_6364_ = v___y_6418_;
v___y_6365_ = v___y_6420_;
v___y_6366_ = v___y_6419_;
v___y_6367_ = v___y_6426_;
v___y_6368_ = v___y_6427_;
v___y_6369_ = v___y_6428_;
v___y_6370_ = v___y_6423_;
v___y_6371_ = v___x_6434_;
goto v___jp_6363_;
}
}
v___jp_6435_:
{
lean_object* v___x_6444_; lean_object* v___x_6445_; lean_object* v___x_6446_; lean_object* v___x_6447_; lean_object* v___x_6448_; lean_object* v___x_6449_; lean_object* v___x_6450_; lean_object* v___x_6451_; lean_object* v___x_6452_; 
v___x_6444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6444_, 0, v_e_6242_);
lean_inc_ref(v___y_6442_);
v___x_6445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6445_, 0, v___y_6442_);
v___x_6446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6446_, 0, v___y_6441_);
v___x_6447_ = lean_unsigned_to_nat(3u);
v___x_6448_ = lean_mk_empty_array_with_capacity(v___x_6447_);
v___x_6449_ = lean_array_push(v___x_6448_, v___x_6444_);
v___x_6450_ = lean_array_push(v___x_6449_, v___x_6445_);
v___x_6451_ = lean_array_push(v___x_6450_, v___x_6446_);
lean_inc(v___y_6443_);
v___x_6452_ = l_Lean_Meta_mkAppOptM(v___y_6443_, v___x_6451_, v___y_6438_, v___y_6439_, v___y_6436_, v___y_6437_);
lean_dec(v___y_6437_);
lean_dec_ref(v___y_6436_);
lean_dec(v___y_6439_);
lean_dec_ref(v___y_6438_);
if (lean_obj_tag(v___x_6452_) == 0)
{
lean_object* v_a_6453_; lean_object* v___x_6455_; uint8_t v_isShared_6456_; uint8_t v_isSharedCheck_6462_; 
v_a_6453_ = lean_ctor_get(v___x_6452_, 0);
v_isSharedCheck_6462_ = !lean_is_exclusive(v___x_6452_);
if (v_isSharedCheck_6462_ == 0)
{
v___x_6455_ = v___x_6452_;
v_isShared_6456_ = v_isSharedCheck_6462_;
goto v_resetjp_6454_;
}
else
{
lean_inc(v_a_6453_);
lean_dec(v___x_6452_);
v___x_6455_ = lean_box(0);
v_isShared_6456_ = v_isSharedCheck_6462_;
goto v_resetjp_6454_;
}
v_resetjp_6454_:
{
lean_object* v___x_6457_; lean_object* v___x_6458_; lean_object* v___x_6460_; 
v___x_6457_ = lean_box(0);
v___x_6458_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_6458_, 0, v___y_6442_);
lean_ctor_set(v___x_6458_, 1, v_a_6453_);
lean_ctor_set(v___x_6458_, 2, v___y_6440_);
lean_ctor_set(v___x_6458_, 3, v___x_6457_);
if (v_isShared_6456_ == 0)
{
lean_ctor_set(v___x_6455_, 0, v___x_6458_);
v___x_6460_ = v___x_6455_;
goto v_reusejp_6459_;
}
else
{
lean_object* v_reuseFailAlloc_6461_; 
v_reuseFailAlloc_6461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6461_, 0, v___x_6458_);
v___x_6460_ = v_reuseFailAlloc_6461_;
goto v_reusejp_6459_;
}
v_reusejp_6459_:
{
return v___x_6460_;
}
}
}
else
{
lean_object* v_a_6463_; lean_object* v___x_6465_; uint8_t v_isShared_6466_; uint8_t v_isSharedCheck_6470_; 
lean_dec_ref(v___y_6442_);
lean_dec(v___y_6440_);
v_a_6463_ = lean_ctor_get(v___x_6452_, 0);
v_isSharedCheck_6470_ = !lean_is_exclusive(v___x_6452_);
if (v_isSharedCheck_6470_ == 0)
{
v___x_6465_ = v___x_6452_;
v_isShared_6466_ = v_isSharedCheck_6470_;
goto v_resetjp_6464_;
}
else
{
lean_inc(v_a_6463_);
lean_dec(v___x_6452_);
v___x_6465_ = lean_box(0);
v_isShared_6466_ = v_isSharedCheck_6470_;
goto v_resetjp_6464_;
}
v_resetjp_6464_:
{
lean_object* v___x_6468_; 
if (v_isShared_6466_ == 0)
{
v___x_6468_ = v___x_6465_;
goto v_reusejp_6467_;
}
else
{
lean_object* v_reuseFailAlloc_6469_; 
v_reuseFailAlloc_6469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6469_, 0, v_a_6463_);
v___x_6468_ = v_reuseFailAlloc_6469_;
goto v_reusejp_6467_;
}
v_reusejp_6467_:
{
return v___x_6468_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___lam__3___boxed(lean_object* v_goal_6730_, lean_object* v___x_6731_, lean_object* v_hrel_6732_, lean_object* v_config_6733_, lean_object* v_e_6734_, lean_object* v_symm_6735_, lean_object* v_mvarIds_6736_, lean_object* v_forwardImp_6737_, lean_object* v___y_6738_, lean_object* v___y_6739_, lean_object* v___y_6740_, lean_object* v___y_6741_, lean_object* v___y_6742_){
_start:
{
uint8_t v_symm_boxed_6743_; uint8_t v_forwardImp_boxed_6744_; lean_object* v_res_6745_; 
v_symm_boxed_6743_ = lean_unbox(v_symm_6735_);
v_forwardImp_boxed_6744_ = lean_unbox(v_forwardImp_6737_);
v_res_6745_ = lp_mathlib_Lean_MVarId_grewrite___lam__3(v_goal_6730_, v___x_6731_, v_hrel_6732_, v_config_6733_, v_e_6734_, v_symm_boxed_6743_, v_mvarIds_6736_, v_forwardImp_boxed_6744_, v___y_6738_, v___y_6739_, v___y_6740_, v___y_6741_);
return v_res_6745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite(lean_object* v_goal_6746_, lean_object* v_e_6747_, lean_object* v_hrel_6748_, lean_object* v_mvarIds_6749_, uint8_t v_forwardImp_6750_, uint8_t v_symm_6751_, lean_object* v_config_6752_, lean_object* v_a_6753_, lean_object* v_a_6754_, lean_object* v_a_6755_, lean_object* v_a_6756_){
_start:
{
lean_object* v___x_6758_; lean_object* v___x_6759_; lean_object* v___x_6760_; lean_object* v___f_6761_; lean_object* v___x_6762_; 
v___x_6758_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_dischargeMain___closed__1));
v___x_6759_ = lean_box(v_symm_6751_);
v___x_6760_ = lean_box(v_forwardImp_6750_);
lean_inc(v_goal_6746_);
v___f_6761_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_grewrite___lam__3___boxed), 13, 8);
lean_closure_set(v___f_6761_, 0, v_goal_6746_);
lean_closure_set(v___f_6761_, 1, v___x_6758_);
lean_closure_set(v___f_6761_, 2, v_hrel_6748_);
lean_closure_set(v___f_6761_, 3, v_config_6752_);
lean_closure_set(v___f_6761_, 4, v_e_6747_);
lean_closure_set(v___f_6761_, 5, v___x_6759_);
lean_closure_set(v___f_6761_, 6, v_mvarIds_6749_);
lean_closure_set(v___f_6761_, 7, v___x_6760_);
v___x_6762_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_grewrite_spec__7___redArg(v_goal_6746_, v___f_6761_, v_a_6753_, v_a_6754_, v_a_6755_, v_a_6756_);
return v___x_6762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_grewrite___boxed(lean_object* v_goal_6763_, lean_object* v_e_6764_, lean_object* v_hrel_6765_, lean_object* v_mvarIds_6766_, lean_object* v_forwardImp_6767_, lean_object* v_symm_6768_, lean_object* v_config_6769_, lean_object* v_a_6770_, lean_object* v_a_6771_, lean_object* v_a_6772_, lean_object* v_a_6773_, lean_object* v_a_6774_){
_start:
{
uint8_t v_forwardImp_boxed_6775_; uint8_t v_symm_boxed_6776_; lean_object* v_res_6777_; 
v_forwardImp_boxed_6775_ = lean_unbox(v_forwardImp_6767_);
v_symm_boxed_6776_ = lean_unbox(v_symm_6768_);
v_res_6777_ = lp_mathlib_Lean_MVarId_grewrite(v_goal_6763_, v_e_6764_, v_hrel_6765_, v_mvarIds_6766_, v_forwardImp_boxed_6775_, v_symm_boxed_6776_, v_config_6769_, v_a_6770_, v_a_6771_, v_a_6772_, v_a_6773_);
lean_dec(v_a_6773_);
lean_dec_ref(v_a_6772_);
lean_dec(v_a_6771_);
lean_dec_ref(v_a_6770_);
return v_res_6777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0(lean_object* v_mvarId_6778_, lean_object* v___y_6779_, lean_object* v___y_6780_, lean_object* v___y_6781_, lean_object* v___y_6782_){
_start:
{
lean_object* v___x_6784_; 
v___x_6784_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___redArg(v_mvarId_6778_, v___y_6780_);
return v___x_6784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0___boxed(lean_object* v_mvarId_6785_, lean_object* v___y_6786_, lean_object* v___y_6787_, lean_object* v___y_6788_, lean_object* v___y_6789_, lean_object* v___y_6790_){
_start:
{
lean_object* v_res_6791_; 
v_res_6791_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0(v_mvarId_6785_, v___y_6786_, v___y_6787_, v___y_6788_, v___y_6789_);
lean_dec(v___y_6789_);
lean_dec_ref(v___y_6788_);
lean_dec(v___y_6787_);
lean_dec_ref(v___y_6786_);
lean_dec(v_mvarId_6785_);
return v_res_6791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8(lean_object* v_00_u03b1_6792_, lean_object* v_name_6793_, uint8_t v_bi_6794_, lean_object* v_type_6795_, lean_object* v_k_6796_, uint8_t v_kind_6797_, lean_object* v___y_6798_, lean_object* v___y_6799_, lean_object* v___y_6800_, lean_object* v___y_6801_){
_start:
{
lean_object* v___x_6803_; 
v___x_6803_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___redArg(v_name_6793_, v_bi_6794_, v_type_6795_, v_k_6796_, v_kind_6797_, v___y_6798_, v___y_6799_, v___y_6800_, v___y_6801_);
return v___x_6803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8___boxed(lean_object* v_00_u03b1_6804_, lean_object* v_name_6805_, lean_object* v_bi_6806_, lean_object* v_type_6807_, lean_object* v_k_6808_, lean_object* v_kind_6809_, lean_object* v___y_6810_, lean_object* v___y_6811_, lean_object* v___y_6812_, lean_object* v___y_6813_, lean_object* v___y_6814_){
_start:
{
uint8_t v_bi_boxed_6815_; uint8_t v_kind_boxed_6816_; lean_object* v_res_6817_; 
v_bi_boxed_6815_ = lean_unbox(v_bi_6806_);
v_kind_boxed_6816_ = lean_unbox(v_kind_6809_);
v_res_6817_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6_spec__8(v_00_u03b1_6804_, v_name_6805_, v_bi_boxed_6815_, v_type_6807_, v_k_6808_, v_kind_boxed_6816_, v___y_6810_, v___y_6811_, v___y_6812_, v___y_6813_);
lean_dec(v___y_6813_);
lean_dec_ref(v___y_6812_);
lean_dec(v___y_6811_);
lean_dec_ref(v___y_6810_);
return v_res_6817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6(lean_object* v_00_u03b1_6818_, lean_object* v_name_6819_, lean_object* v_type_6820_, lean_object* v_k_6821_, lean_object* v___y_6822_, lean_object* v___y_6823_, lean_object* v___y_6824_, lean_object* v___y_6825_){
_start:
{
lean_object* v___x_6827_; 
v___x_6827_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___redArg(v_name_6819_, v_type_6820_, v_k_6821_, v___y_6822_, v___y_6823_, v___y_6824_, v___y_6825_);
return v___x_6827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6___boxed(lean_object* v_00_u03b1_6828_, lean_object* v_name_6829_, lean_object* v_type_6830_, lean_object* v_k_6831_, lean_object* v___y_6832_, lean_object* v___y_6833_, lean_object* v___y_6834_, lean_object* v___y_6835_, lean_object* v___y_6836_){
_start:
{
lean_object* v_res_6837_; 
v_res_6837_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_MVarId_grewrite_spec__6(v_00_u03b1_6828_, v_name_6829_, v_type_6830_, v_k_6831_, v___y_6832_, v___y_6833_, v___y_6834_, v___y_6835_);
lean_dec(v___y_6835_);
lean_dec_ref(v___y_6834_);
lean_dec(v___y_6833_);
lean_dec_ref(v___y_6832_);
return v_res_6837_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0(lean_object* v_00_u03b2_6838_, lean_object* v_x_6839_, lean_object* v_x_6840_){
_start:
{
uint8_t v___x_6841_; 
v___x_6841_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___redArg(v_x_6839_, v_x_6840_);
return v___x_6841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0___boxed(lean_object* v_00_u03b2_6842_, lean_object* v_x_6843_, lean_object* v_x_6844_){
_start:
{
uint8_t v_res_6845_; lean_object* v_r_6846_; 
v_res_6845_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0(v_00_u03b2_6842_, v_x_6843_, v_x_6844_);
lean_dec(v_x_6844_);
lean_dec_ref(v_x_6843_);
v_r_6846_ = lean_box(v_res_6845_);
return v_r_6846_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_6847_, lean_object* v_x_6848_, size_t v_x_6849_, lean_object* v_x_6850_){
_start:
{
uint8_t v___x_6851_; 
v___x_6851_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___redArg(v_x_6848_, v_x_6849_, v_x_6850_);
return v___x_6851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_6852_, lean_object* v_x_6853_, lean_object* v_x_6854_, lean_object* v_x_6855_){
_start:
{
size_t v_x_20118__boxed_6856_; uint8_t v_res_6857_; lean_object* v_r_6858_; 
v_x_20118__boxed_6856_ = lean_unbox_usize(v_x_6854_);
lean_dec(v_x_6854_);
v_res_6857_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2(v_00_u03b2_6852_, v_x_6853_, v_x_20118__boxed_6856_, v_x_6855_);
lean_dec(v_x_6855_);
lean_dec_ref(v_x_6853_);
v_r_6858_ = lean_box(v_res_6857_);
return v_r_6858_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10(lean_object* v_00_u03b2_6859_, lean_object* v_keys_6860_, lean_object* v_vals_6861_, lean_object* v_heq_6862_, lean_object* v_i_6863_, lean_object* v_k_6864_){
_start:
{
uint8_t v___x_6865_; 
v___x_6865_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___redArg(v_keys_6860_, v_i_6863_, v_k_6864_);
return v___x_6865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10___boxed(lean_object* v_00_u03b2_6866_, lean_object* v_keys_6867_, lean_object* v_vals_6868_, lean_object* v_heq_6869_, lean_object* v_i_6870_, lean_object* v_k_6871_){
_start:
{
uint8_t v_res_6872_; lean_object* v_r_6873_; 
v_res_6872_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Lean_MVarId_grewrite_spec__0_spec__0_spec__2_spec__10(v_00_u03b2_6866_, v_keys_6867_, v_vals_6868_, v_heq_6869_, v_i_6870_, v_k_6871_);
lean_dec(v_k_6871_);
lean_dec_ref(v_vals_6868_);
lean_dec_ref(v_keys_6867_);
v_r_6873_ = lean_box(v_res_6872_);
return v_r_6873_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GCongr_Core(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GRewrite_Core(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GCongr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GCongr_Core(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_GRewrite_Core(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GCongr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_GRewrite_Core_0__Mathlib_Tactic_GRewrite_initFn_00___x40_Mathlib_Tactic_GRewrite_Core_3959275054____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr_Core(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_GRewrite_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GRewrite_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_GRewrite_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_GRewrite_Core(builtin);
}
#ifdef __cplusplus
}
#endif
