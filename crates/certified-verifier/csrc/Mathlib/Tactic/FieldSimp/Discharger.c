// Lean compiler output
// Module: Mathlib.Tactic.FieldSimp.Discharger
// Imports: public import Init public meta import Init import all Lean.Meta.Tactic.Simp.Rewrite public import Mathlib.Tactic.Positivity.Core public import Mathlib.Util.DischargerAsTactic
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
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
double lean_float_of_nat(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocs___redArg(lean_object*);
uint32_t lean_uint32_add(uint32_t, uint32_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setSimpTheorems(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkOfEqTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_Positivity_solve(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Simp_Rewrite_0__Lean_Meta_Simp_dischargeUsingAssumption_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__5___redArg(lean_object*);
extern lean_object* l_Lean_trace_profiler;
uint8_t l_Lean_Option_get___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__3(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* l_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__7(lean_object*, lean_object*);
double lean_float_div(double, double);
lean_object* lean_io_mono_nanos_now();
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__2___redArg(lean_object*);
lean_object* lp_mathlib_wrapSimpDischarger(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "discharge "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5___boxed(lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0;
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "two_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 9, 102, 68, 219, 110, 92, 129)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "three_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(135, 61, 57, 90, 114, 212, 7, 211)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "four_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 48, 179, 160, 201, 61, 62, 73)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mul_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__6_value),LEAN_SCALAR_PTR_LITERAL(104, 59, 168, 170, 0, 6, 10, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "pow_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__8_value),LEAN_SCALAR_PTR_LITERAL(111, 117, 199, 58, 253, 19, 118, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "zpow_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__10_value),LEAN_SCALAR_PTR_LITERAL(98, 69, 60, 128, 243, 215, 186, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "cast_add_one_ne_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__12_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__13_value),LEAN_SCALAR_PTR_LITERAL(120, 10, 172, 187, 234, 245, 244, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__28_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "field_simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__31_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__32_value),LEAN_SCALAR_PTR_LITERAL(134, 173, 236, 139, 174, 242, 46, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__35_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__36_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "FieldSimp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "tacticField_simp_discharge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__31_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 95, 53, 248, 72, 241, 7, 199)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 104, 14, 73, 80, 168, 95, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "field_simp_discharge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__0));
v___x_3_ = l_Lean_stringToMessageData(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg(lean_object* v_prop_4_){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_6_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___closed__1);
v___x_7_ = l_Lean_MessageData_ofExpr(v_prop_4_);
v___x_8_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_6_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg___boxed(lean_object* v_prop_10_, lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg(v_prop_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage(lean_object* v_00_u03b5_13_, lean_object* v_prop_14_, lean_object* v_x_15_, lean_object* v_a_16_, lean_object* v_a_17_, lean_object* v_a_18_, lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v_a_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___redArg(v_prop_14_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___boxed(lean_object* v_00_u03b5_25_, lean_object* v_prop_26_, lean_object* v_x_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_, lean_object* v_a_31_, lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage(v_00_u03b5_25_, v_prop_26_, v_x_27_, v_a_28_, v_a_29_, v_a_30_, v_a_31_, v_a_32_, v_a_33_, v_a_34_);
lean_dec(v_a_34_);
lean_dec_ref(v_a_33_);
lean_dec(v_a_32_);
lean_dec_ref(v_a_31_);
lean_dec(v_a_30_);
lean_dec_ref(v_a_29_);
lean_dec(v_a_28_);
lean_dec_ref(v_x_27_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(lean_object* v_l_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_40_; lean_object* v_mctx_41_; lean_object* v___x_42_; lean_object* v_fst_43_; lean_object* v_snd_44_; lean_object* v___x_45_; lean_object* v_cache_46_; lean_object* v_zetaDeltaFVarIds_47_; lean_object* v_postponed_48_; lean_object* v_diag_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_58_; 
v___x_40_ = lean_st_ref_get(v___y_38_);
v_mctx_41_ = lean_ctor_get(v___x_40_, 0);
lean_inc_ref(v_mctx_41_);
lean_dec(v___x_40_);
v___x_42_ = lean_instantiate_level_mvars(v_mctx_41_, v_l_37_);
v_fst_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_fst_43_);
v_snd_44_ = lean_ctor_get(v___x_42_, 1);
lean_inc(v_snd_44_);
lean_dec_ref(v___x_42_);
v___x_45_ = lean_st_ref_take(v___y_38_);
v_cache_46_ = lean_ctor_get(v___x_45_, 1);
v_zetaDeltaFVarIds_47_ = lean_ctor_get(v___x_45_, 2);
v_postponed_48_ = lean_ctor_get(v___x_45_, 3);
v_diag_49_ = lean_ctor_get(v___x_45_, 4);
v_isSharedCheck_58_ = !lean_is_exclusive(v___x_45_);
if (v_isSharedCheck_58_ == 0)
{
lean_object* v_unused_59_; 
v_unused_59_ = lean_ctor_get(v___x_45_, 0);
lean_dec(v_unused_59_);
v___x_51_ = v___x_45_;
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_diag_49_);
lean_inc(v_postponed_48_);
lean_inc(v_zetaDeltaFVarIds_47_);
lean_inc(v_cache_46_);
lean_dec(v___x_45_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_54_; 
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 0, v_fst_43_);
v___x_54_ = v___x_51_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v_fst_43_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v_cache_46_);
lean_ctor_set(v_reuseFailAlloc_57_, 2, v_zetaDeltaFVarIds_47_);
lean_ctor_set(v_reuseFailAlloc_57_, 3, v_postponed_48_);
lean_ctor_set(v_reuseFailAlloc_57_, 4, v_diag_49_);
v___x_54_ = v_reuseFailAlloc_57_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_st_ref_set(v___y_38_, v___x_54_);
v___x_56_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_56_, 0, v_snd_44_);
return v___x_56_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg___boxed(lean_object* v_l_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(v_l_60_, v___y_61_);
lean_dec(v___y_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0(lean_object* v_l_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(v_l_64_, v___y_66_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___boxed(lean_object* v_l_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0(v_l_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(lean_object* v_k_78_, uint8_t v_allowLevelAssignments_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_79_, v_k_78_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
if (lean_obj_tag(v___x_85_) == 0)
{
lean_object* v_a_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_93_; 
v_a_86_ = lean_ctor_get(v___x_85_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_93_ == 0)
{
v___x_88_ = v___x_85_;
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_a_86_);
lean_dec(v___x_85_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_91_; 
if (v_isShared_89_ == 0)
{
v___x_91_ = v___x_88_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v_a_86_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
else
{
lean_object* v_a_94_; lean_object* v___x_96_; uint8_t v_isShared_97_; uint8_t v_isSharedCheck_101_; 
v_a_94_ = lean_ctor_get(v___x_85_, 0);
v_isSharedCheck_101_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_101_ == 0)
{
v___x_96_ = v___x_85_;
v_isShared_97_ = v_isSharedCheck_101_;
goto v_resetjp_95_;
}
else
{
lean_inc(v_a_94_);
lean_dec(v___x_85_);
v___x_96_ = lean_box(0);
v_isShared_97_ = v_isSharedCheck_101_;
goto v_resetjp_95_;
}
v_resetjp_95_:
{
lean_object* v___x_99_; 
if (v_isShared_97_ == 0)
{
v___x_99_ = v___x_96_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v_a_94_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg___boxed(lean_object* v_k_102_, lean_object* v_allowLevelAssignments_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_109_; lean_object* v_res_110_; 
v_allowLevelAssignments_boxed_109_ = lean_unbox(v_allowLevelAssignments_103_);
v_res_110_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(v_k_102_, v_allowLevelAssignments_boxed_109_, v___y_104_, v___y_105_, v___y_106_, v___y_107_);
lean_dec(v___y_107_);
lean_dec_ref(v___y_106_);
lean_dec(v___y_105_);
lean_dec_ref(v___y_104_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1(lean_object* v_00_u03b1_111_, lean_object* v_k_112_, uint8_t v_allowLevelAssignments_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(v_k_112_, v_allowLevelAssignments_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___boxed(lean_object* v_00_u03b1_120_, lean_object* v_k_121_, lean_object* v_allowLevelAssignments_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_128_; lean_object* v_res_129_; 
v_allowLevelAssignments_boxed_128_ = lean_unbox(v_allowLevelAssignments_122_);
v_res_129_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1(v_00_u03b1_120_, v_k_121_, v_allowLevelAssignments_boxed_128_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
return v_res_129_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0(void){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_131_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__0);
v___x_132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2(lean_object* v_00_u03b2_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2___closed__1);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0(void){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_135_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_136_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__0);
v___x_137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3(lean_object* v_00_u03b2_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3___closed__1);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0(lean_object* v_prop_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = l_Lean_Meta_mkFreshLevelMVar(v___y_144_, v___y_145_, v___y_146_, v___y_147_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_151_; lean_object* v___x_152_; uint8_t v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v_a_150_ = lean_ctor_get(v___x_149_, 0);
lean_inc_n(v_a_150_, 2);
lean_dec_ref_known(v___x_149_, 1);
v___x_151_ = l_Lean_Expr_sort___override(v_a_150_);
v___x_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
v___x_153_ = 0;
v___x_154_ = lean_box(0);
v___x_155_ = l_Lean_Meta_mkFreshExprMVar(v___x_152_, v___x_153_, v___x_154_, v___y_144_, v___y_145_, v___y_146_, v___y_147_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v_a_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v_a_156_ = lean_ctor_get(v___x_155_, 0);
lean_inc_n(v_a_156_, 2);
lean_dec_ref_known(v___x_155_, 1);
v___x_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_157_, 0, v_a_156_);
lean_inc_ref(v___x_157_);
v___x_158_ = l_Lean_Meta_mkFreshExprMVar(v___x_157_, v___x_153_, v___x_154_, v___y_144_, v___y_145_, v___y_146_, v___y_147_);
if (lean_obj_tag(v___x_158_) == 0)
{
lean_object* v_a_159_; lean_object* v___x_160_; 
v_a_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc(v_a_159_);
lean_dec_ref_known(v___x_158_, 1);
v___x_160_ = l_Lean_Meta_mkFreshExprMVar(v___x_157_, v___x_153_, v___x_154_, v___y_144_, v___y_145_, v___y_146_, v___y_147_);
if (lean_obj_tag(v___x_160_) == 0)
{
lean_object* v_a_161_; lean_object* v_keyedConfig_162_; uint8_t v_trackZetaDelta_163_; lean_object* v_zetaDeltaSet_164_; lean_object* v_lctx_165_; lean_object* v_localInstances_166_; lean_object* v_defEqCtx_x3f_167_; lean_object* v_synthPendingDepth_168_; lean_object* v_customCanUnfoldPredicate_x3f_169_; uint8_t v_univApprox_170_; uint8_t v_inTypeClassResolution_171_; uint8_t v_cacheInferType_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_261_; 
v_a_161_ = lean_ctor_get(v___x_160_, 0);
lean_inc(v_a_161_);
lean_dec_ref_known(v___x_160_, 1);
v_keyedConfig_162_ = lean_ctor_get(v___y_144_, 0);
v_trackZetaDelta_163_ = lean_ctor_get_uint8(v___y_144_, sizeof(void*)*7);
v_zetaDeltaSet_164_ = lean_ctor_get(v___y_144_, 1);
v_lctx_165_ = lean_ctor_get(v___y_144_, 2);
v_localInstances_166_ = lean_ctor_get(v___y_144_, 3);
v_defEqCtx_x3f_167_ = lean_ctor_get(v___y_144_, 4);
v_synthPendingDepth_168_ = lean_ctor_get(v___y_144_, 5);
v_customCanUnfoldPredicate_x3f_169_ = lean_ctor_get(v___y_144_, 6);
v_univApprox_170_ = lean_ctor_get_uint8(v___y_144_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_171_ = lean_ctor_get_uint8(v___y_144_, sizeof(void*)*7 + 2);
v_cacheInferType_172_ = lean_ctor_get_uint8(v___y_144_, sizeof(void*)*7 + 3);
v_isSharedCheck_261_ = !lean_is_exclusive(v___y_144_);
if (v_isSharedCheck_261_ == 0)
{
v___x_174_ = v___y_144_;
v_isShared_175_ = v_isSharedCheck_261_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_169_);
lean_inc(v_synthPendingDepth_168_);
lean_inc(v_defEqCtx_x3f_167_);
lean_inc(v_localInstances_166_);
lean_inc(v_lctx_165_);
lean_inc(v_zetaDeltaSet_164_);
lean_inc(v_keyedConfig_162_);
lean_dec(v___y_144_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_261_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; uint8_t v___x_183_; lean_object* v___x_184_; lean_object* v___x_186_; 
v___x_176_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1));
v___x_177_ = lean_box(0);
lean_inc(v_a_150_);
v___x_178_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_178_, 0, v_a_150_);
lean_ctor_set(v___x_178_, 1, v___x_177_);
v___x_179_ = l_Lean_Expr_const___override(v___x_176_, v___x_178_);
lean_inc(v_a_156_);
v___x_180_ = l_Lean_Expr_app___override(v___x_179_, v_a_156_);
lean_inc(v_a_159_);
v___x_181_ = l_Lean_Expr_app___override(v___x_180_, v_a_159_);
lean_inc(v_a_161_);
v___x_182_ = l_Lean_Expr_app___override(v___x_181_, v_a_161_);
v___x_183_ = 2;
v___x_184_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_183_, v_keyedConfig_162_);
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 0, v___x_184_);
v___x_186_ = v___x_174_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v___x_184_);
lean_ctor_set(v_reuseFailAlloc_260_, 1, v_zetaDeltaSet_164_);
lean_ctor_set(v_reuseFailAlloc_260_, 2, v_lctx_165_);
lean_ctor_set(v_reuseFailAlloc_260_, 3, v_localInstances_166_);
lean_ctor_set(v_reuseFailAlloc_260_, 4, v_defEqCtx_x3f_167_);
lean_ctor_set(v_reuseFailAlloc_260_, 5, v_synthPendingDepth_168_);
lean_ctor_set(v_reuseFailAlloc_260_, 6, v_customCanUnfoldPredicate_x3f_169_);
lean_ctor_set_uint8(v_reuseFailAlloc_260_, sizeof(void*)*7, v_trackZetaDelta_163_);
lean_ctor_set_uint8(v_reuseFailAlloc_260_, sizeof(void*)*7 + 1, v_univApprox_170_);
lean_ctor_set_uint8(v_reuseFailAlloc_260_, sizeof(void*)*7 + 2, v_inTypeClassResolution_171_);
lean_ctor_set_uint8(v_reuseFailAlloc_260_, sizeof(void*)*7 + 3, v_cacheInferType_172_);
v___x_186_ = v_reuseFailAlloc_260_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
lean_object* v___x_187_; 
v___x_187_ = l_Lean_Meta_isExprDefEq(v___x_182_, v_prop_143_, v___x_186_, v___y_145_, v___y_146_, v___y_147_);
lean_dec_ref(v___x_186_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_object* v_a_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_251_; 
v_a_188_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_251_ == 0)
{
v___x_190_ = v___x_187_;
v_isShared_191_ = v_isSharedCheck_251_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_a_188_);
lean_dec(v___x_187_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_251_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
uint8_t v___x_192_; 
v___x_192_ = lean_unbox(v_a_188_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_198_; 
v___x_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_193_, 0, v_a_161_);
lean_ctor_set(v___x_193_, 1, v_a_188_);
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v_a_159_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v_a_156_);
lean_ctor_set(v___x_195_, 1, v___x_194_);
v___x_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_196_, 0, v_a_150_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
if (v_isShared_191_ == 0)
{
lean_ctor_set(v___x_190_, 0, v___x_196_);
v___x_198_ = v___x_190_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
else
{
lean_object* v___x_200_; 
lean_del_object(v___x_190_);
v___x_200_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(v_a_150_, v___y_145_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___x_202_; 
v_a_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_a_201_);
lean_dec_ref_known(v___x_200_, 1);
v___x_202_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_156_, v___y_145_);
if (lean_obj_tag(v___x_202_) == 0)
{
lean_object* v_a_203_; lean_object* v___x_204_; 
v_a_203_ = lean_ctor_get(v___x_202_, 0);
lean_inc(v_a_203_);
lean_dec_ref_known(v___x_202_, 1);
v___x_204_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_159_, v___y_145_);
if (lean_obj_tag(v___x_204_) == 0)
{
lean_object* v_a_205_; lean_object* v___x_206_; 
v_a_205_ = lean_ctor_get(v___x_204_, 0);
lean_inc(v_a_205_);
lean_dec_ref_known(v___x_204_, 1);
v___x_206_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_161_, v___y_145_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_218_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_218_ == 0)
{
v___x_209_ = v___x_206_;
v_isShared_210_ = v_isSharedCheck_218_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_206_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_218_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_216_; 
v___x_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_211_, 0, v_a_207_);
lean_ctor_set(v___x_211_, 1, v_a_188_);
v___x_212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_212_, 0, v_a_205_);
lean_ctor_set(v___x_212_, 1, v___x_211_);
v___x_213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_213_, 0, v_a_203_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v_a_201_);
lean_ctor_set(v___x_214_, 1, v___x_213_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 0, v___x_214_);
v___x_216_ = v___x_209_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
else
{
lean_object* v_a_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_226_; 
lean_dec(v_a_205_);
lean_dec(v_a_203_);
lean_dec(v_a_201_);
lean_dec(v_a_188_);
v_a_219_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_226_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_226_ == 0)
{
v___x_221_ = v___x_206_;
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_a_219_);
lean_dec(v___x_206_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_224_; 
if (v_isShared_222_ == 0)
{
v___x_224_ = v___x_221_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_a_219_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
}
}
else
{
lean_object* v_a_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_234_; 
lean_dec(v_a_203_);
lean_dec(v_a_201_);
lean_dec(v_a_188_);
lean_dec(v_a_161_);
v_a_227_ = lean_ctor_get(v___x_204_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_204_);
if (v_isSharedCheck_234_ == 0)
{
v___x_229_ = v___x_204_;
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_a_227_);
lean_dec(v___x_204_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
if (v_isShared_230_ == 0)
{
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_a_227_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
}
else
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
lean_dec(v_a_201_);
lean_dec(v_a_188_);
lean_dec(v_a_161_);
lean_dec(v_a_159_);
v_a_235_ = lean_ctor_get(v___x_202_, 0);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_202_);
if (v_isSharedCheck_242_ == 0)
{
v___x_237_ = v___x_202_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_202_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_235_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
lean_dec(v_a_188_);
lean_dec(v_a_161_);
lean_dec(v_a_159_);
lean_dec(v_a_156_);
v_a_243_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_200_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_200_);
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
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec(v_a_161_);
lean_dec(v_a_159_);
lean_dec(v_a_156_);
lean_dec(v_a_150_);
v_a_252_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_187_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_187_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
}
}
else
{
lean_object* v_a_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_269_; 
lean_dec(v_a_159_);
lean_dec(v_a_156_);
lean_dec(v_a_150_);
lean_dec_ref(v___y_144_);
lean_dec_ref(v_prop_143_);
v_a_262_ = lean_ctor_get(v___x_160_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_269_ == 0)
{
v___x_264_ = v___x_160_;
v_isShared_265_ = v_isSharedCheck_269_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_a_262_);
lean_dec(v___x_160_);
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
else
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
lean_dec_ref_known(v___x_157_, 1);
lean_dec(v_a_156_);
lean_dec(v_a_150_);
lean_dec_ref(v___y_144_);
lean_dec_ref(v_prop_143_);
v_a_270_ = lean_ctor_get(v___x_158_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v___x_158_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_158_);
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
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
lean_dec(v_a_150_);
lean_dec_ref(v___y_144_);
lean_dec_ref(v_prop_143_);
v_a_278_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___x_155_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v___x_155_);
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
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_dec_ref(v___y_144_);
lean_dec_ref(v_prop_143_);
v_a_286_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_149_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_149_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___boxed(lean_object* v_prop_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0(v_prop_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(lean_object* v_r_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_310_, 0, v_r_301_);
v___x_311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1___boxed(lean_object* v_r_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(v_r_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
lean_dec(v___y_315_);
lean_dec_ref(v___y_314_);
lean_dec(v___y_313_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3(lean_object* v_prop_322_, uint8_t v___x_323_, uint8_t v_hasTrace_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = l_Lean_Meta_mkFreshLevelMVar(v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v_a_331_; lean_object* v___x_332_; lean_object* v___x_333_; uint8_t v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v_a_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc_n(v_a_331_, 2);
lean_dec_ref_known(v___x_330_, 1);
v___x_332_ = l_Lean_Expr_sort___override(v_a_331_);
v___x_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
v___x_334_ = 0;
v___x_335_ = lean_box(0);
v___x_336_ = l_Lean_Meta_mkFreshExprMVar(v___x_333_, v___x_334_, v___x_335_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_336_) == 0)
{
lean_object* v_a_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v_a_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc_n(v_a_337_, 2);
lean_dec_ref_known(v___x_336_, 1);
v___x_338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_338_, 0, v_a_337_);
lean_inc_ref(v___x_338_);
v___x_339_ = l_Lean_Meta_mkFreshExprMVar(v___x_338_, v___x_334_, v___x_335_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_339_) == 0)
{
lean_object* v_a_340_; lean_object* v___x_341_; 
v_a_340_ = lean_ctor_get(v___x_339_, 0);
lean_inc(v_a_340_);
lean_dec_ref_known(v___x_339_, 1);
v___x_341_ = l_Lean_Meta_mkFreshExprMVar(v___x_338_, v___x_334_, v___x_335_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_341_) == 0)
{
lean_object* v_a_342_; lean_object* v_keyedConfig_343_; uint8_t v_trackZetaDelta_344_; lean_object* v_zetaDeltaSet_345_; lean_object* v_lctx_346_; lean_object* v_localInstances_347_; lean_object* v_defEqCtx_x3f_348_; lean_object* v_synthPendingDepth_349_; lean_object* v_customCanUnfoldPredicate_x3f_350_; uint8_t v_univApprox_351_; uint8_t v_inTypeClassResolution_352_; uint8_t v_cacheInferType_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_444_; 
v_a_342_ = lean_ctor_get(v___x_341_, 0);
lean_inc(v_a_342_);
lean_dec_ref_known(v___x_341_, 1);
v_keyedConfig_343_ = lean_ctor_get(v___y_325_, 0);
v_trackZetaDelta_344_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7);
v_zetaDeltaSet_345_ = lean_ctor_get(v___y_325_, 1);
v_lctx_346_ = lean_ctor_get(v___y_325_, 2);
v_localInstances_347_ = lean_ctor_get(v___y_325_, 3);
v_defEqCtx_x3f_348_ = lean_ctor_get(v___y_325_, 4);
v_synthPendingDepth_349_ = lean_ctor_get(v___y_325_, 5);
v_customCanUnfoldPredicate_x3f_350_ = lean_ctor_get(v___y_325_, 6);
v_univApprox_351_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_352_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 2);
v_cacheInferType_353_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 3);
v_isSharedCheck_444_ = !lean_is_exclusive(v___y_325_);
if (v_isSharedCheck_444_ == 0)
{
v___x_355_ = v___y_325_;
v_isShared_356_ = v_isSharedCheck_444_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_350_);
lean_inc(v_synthPendingDepth_349_);
lean_inc(v_defEqCtx_x3f_348_);
lean_inc(v_localInstances_347_);
lean_inc(v_lctx_346_);
lean_inc(v_zetaDeltaSet_345_);
lean_inc(v_keyedConfig_343_);
lean_dec(v___y_325_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_444_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; uint8_t v___x_364_; lean_object* v___x_365_; lean_object* v___x_367_; 
v___x_357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1));
v___x_358_ = lean_box(0);
lean_inc(v_a_331_);
v___x_359_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_359_, 0, v_a_331_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v___x_360_ = l_Lean_Expr_const___override(v___x_357_, v___x_359_);
lean_inc(v_a_337_);
v___x_361_ = l_Lean_Expr_app___override(v___x_360_, v_a_337_);
lean_inc(v_a_340_);
v___x_362_ = l_Lean_Expr_app___override(v___x_361_, v_a_340_);
lean_inc(v_a_342_);
v___x_363_ = l_Lean_Expr_app___override(v___x_362_, v_a_342_);
v___x_364_ = 2;
v___x_365_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_364_, v_keyedConfig_343_);
if (v_isShared_356_ == 0)
{
lean_ctor_set(v___x_355_, 0, v___x_365_);
v___x_367_ = v___x_355_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_365_);
lean_ctor_set(v_reuseFailAlloc_443_, 1, v_zetaDeltaSet_345_);
lean_ctor_set(v_reuseFailAlloc_443_, 2, v_lctx_346_);
lean_ctor_set(v_reuseFailAlloc_443_, 3, v_localInstances_347_);
lean_ctor_set(v_reuseFailAlloc_443_, 4, v_defEqCtx_x3f_348_);
lean_ctor_set(v_reuseFailAlloc_443_, 5, v_synthPendingDepth_349_);
lean_ctor_set(v_reuseFailAlloc_443_, 6, v_customCanUnfoldPredicate_x3f_350_);
lean_ctor_set_uint8(v_reuseFailAlloc_443_, sizeof(void*)*7, v_trackZetaDelta_344_);
lean_ctor_set_uint8(v_reuseFailAlloc_443_, sizeof(void*)*7 + 1, v_univApprox_351_);
lean_ctor_set_uint8(v_reuseFailAlloc_443_, sizeof(void*)*7 + 2, v_inTypeClassResolution_352_);
lean_ctor_set_uint8(v_reuseFailAlloc_443_, sizeof(void*)*7 + 3, v_cacheInferType_353_);
v___x_367_ = v_reuseFailAlloc_443_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
lean_object* v___x_368_; 
v___x_368_ = l_Lean_Meta_isExprDefEq(v___x_363_, v_prop_322_, v___x_367_, v___y_326_, v___y_327_, v___y_328_);
lean_dec_ref(v___x_367_);
if (lean_obj_tag(v___x_368_) == 0)
{
lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_434_; 
v_a_369_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_434_ == 0)
{
v___x_371_ = v___x_368_;
v_isShared_372_ = v_isSharedCheck_434_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_368_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_434_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
uint8_t v___x_373_; 
v___x_373_ = lean_unbox(v_a_369_);
lean_dec(v_a_369_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_380_; 
v___x_374_ = lean_box(v___x_323_);
v___x_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_375_, 0, v_a_342_);
lean_ctor_set(v___x_375_, 1, v___x_374_);
v___x_376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_376_, 0, v_a_340_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v_a_337_);
lean_ctor_set(v___x_377_, 1, v___x_376_);
v___x_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_378_, 0, v_a_331_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 0, v___x_378_);
v___x_380_ = v___x_371_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v___x_378_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
else
{
lean_object* v___x_382_; 
lean_del_object(v___x_371_);
v___x_382_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(v_a_331_, v___y_326_);
if (lean_obj_tag(v___x_382_) == 0)
{
lean_object* v_a_383_; lean_object* v___x_384_; 
v_a_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_a_383_);
lean_dec_ref_known(v___x_382_, 1);
v___x_384_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_337_, v___y_326_);
if (lean_obj_tag(v___x_384_) == 0)
{
lean_object* v_a_385_; lean_object* v___x_386_; 
v_a_385_ = lean_ctor_get(v___x_384_, 0);
lean_inc(v_a_385_);
lean_dec_ref_known(v___x_384_, 1);
v___x_386_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_340_, v___y_326_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_object* v_a_387_; lean_object* v___x_388_; 
v_a_387_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_a_387_);
lean_dec_ref_known(v___x_386_, 1);
v___x_388_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_342_, v___y_326_);
if (lean_obj_tag(v___x_388_) == 0)
{
lean_object* v_a_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_401_; 
v_a_389_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_401_ == 0)
{
v___x_391_ = v___x_388_;
v_isShared_392_ = v_isSharedCheck_401_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_a_389_);
lean_dec(v___x_388_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_401_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_399_; 
v___x_393_ = lean_box(v_hasTrace_324_);
v___x_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_394_, 0, v_a_389_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
v___x_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_395_, 0, v_a_387_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_396_, 0, v_a_385_);
lean_ctor_set(v___x_396_, 1, v___x_395_);
v___x_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_397_, 0, v_a_383_);
lean_ctor_set(v___x_397_, 1, v___x_396_);
if (v_isShared_392_ == 0)
{
lean_ctor_set(v___x_391_, 0, v___x_397_);
v___x_399_ = v___x_391_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_397_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
else
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_409_; 
lean_dec(v_a_387_);
lean_dec(v_a_385_);
lean_dec(v_a_383_);
v_a_402_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_409_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_409_ == 0)
{
v___x_404_ = v___x_388_;
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_388_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v_a_402_);
v___x_407_ = v_reuseFailAlloc_408_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
return v___x_407_;
}
}
}
}
else
{
lean_object* v_a_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_417_; 
lean_dec(v_a_385_);
lean_dec(v_a_383_);
lean_dec(v_a_342_);
v_a_410_ = lean_ctor_get(v___x_386_, 0);
v_isSharedCheck_417_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_417_ == 0)
{
v___x_412_ = v___x_386_;
v_isShared_413_ = v_isSharedCheck_417_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_a_410_);
lean_dec(v___x_386_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_417_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___x_415_; 
if (v_isShared_413_ == 0)
{
v___x_415_ = v___x_412_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_a_410_);
v___x_415_ = v_reuseFailAlloc_416_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
return v___x_415_;
}
}
}
}
else
{
lean_object* v_a_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_425_; 
lean_dec(v_a_383_);
lean_dec(v_a_342_);
lean_dec(v_a_340_);
v_a_418_ = lean_ctor_get(v___x_384_, 0);
v_isSharedCheck_425_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_425_ == 0)
{
v___x_420_ = v___x_384_;
v_isShared_421_ = v_isSharedCheck_425_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_a_418_);
lean_dec(v___x_384_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_425_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_423_; 
if (v_isShared_421_ == 0)
{
v___x_423_ = v___x_420_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_424_; 
v_reuseFailAlloc_424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_424_, 0, v_a_418_);
v___x_423_ = v_reuseFailAlloc_424_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
return v___x_423_;
}
}
}
}
else
{
lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_433_; 
lean_dec(v_a_342_);
lean_dec(v_a_340_);
lean_dec(v_a_337_);
v_a_426_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_433_ == 0)
{
v___x_428_ = v___x_382_;
v_isShared_429_ = v_isSharedCheck_433_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_382_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_433_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v___x_431_; 
if (v_isShared_429_ == 0)
{
v___x_431_ = v___x_428_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_432_; 
v_reuseFailAlloc_432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_432_, 0, v_a_426_);
v___x_431_ = v_reuseFailAlloc_432_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
return v___x_431_;
}
}
}
}
}
}
else
{
lean_object* v_a_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_442_; 
lean_dec(v_a_342_);
lean_dec(v_a_340_);
lean_dec(v_a_337_);
lean_dec(v_a_331_);
v_a_435_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_442_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_442_ == 0)
{
v___x_437_ = v___x_368_;
v_isShared_438_ = v_isSharedCheck_442_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_a_435_);
lean_dec(v___x_368_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_442_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v___x_440_; 
if (v_isShared_438_ == 0)
{
v___x_440_ = v___x_437_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v_a_435_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
}
}
}
}
else
{
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_452_; 
lean_dec(v_a_340_);
lean_dec(v_a_337_);
lean_dec(v_a_331_);
lean_dec_ref(v___y_325_);
lean_dec_ref(v_prop_322_);
v_a_445_ = lean_ctor_get(v___x_341_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_341_);
if (v_isSharedCheck_452_ == 0)
{
v___x_447_ = v___x_341_;
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_341_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_450_; 
if (v_isShared_448_ == 0)
{
v___x_450_ = v___x_447_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_a_445_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
else
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
lean_dec_ref_known(v___x_338_, 1);
lean_dec(v_a_337_);
lean_dec(v_a_331_);
lean_dec_ref(v___y_325_);
lean_dec_ref(v_prop_322_);
v_a_453_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_339_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_339_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_a_453_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
else
{
lean_object* v_a_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_468_; 
lean_dec(v_a_331_);
lean_dec_ref(v___y_325_);
lean_dec_ref(v_prop_322_);
v_a_461_ = lean_ctor_get(v___x_336_, 0);
v_isSharedCheck_468_ = !lean_is_exclusive(v___x_336_);
if (v_isSharedCheck_468_ == 0)
{
v___x_463_ = v___x_336_;
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_a_461_);
lean_dec(v___x_336_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_466_; 
if (v_isShared_464_ == 0)
{
v___x_466_ = v___x_463_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v_a_461_);
v___x_466_ = v_reuseFailAlloc_467_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
return v___x_466_;
}
}
}
}
else
{
lean_object* v_a_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_476_; 
lean_dec_ref(v___y_325_);
lean_dec_ref(v_prop_322_);
v_a_469_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_476_ == 0)
{
v___x_471_ = v___x_330_;
v_isShared_472_ = v_isSharedCheck_476_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_a_469_);
lean_dec(v___x_330_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3___boxed(lean_object* v_prop_477_, lean_object* v___x_478_, lean_object* v_hasTrace_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
uint8_t v___x_117566__boxed_485_; uint8_t v_hasTrace_boxed_486_; lean_object* v_res_487_; 
v___x_117566__boxed_485_ = lean_unbox(v___x_478_);
v_hasTrace_boxed_486_ = lean_unbox(v_hasTrace_479_);
v_res_487_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3(v_prop_477_, v___x_117566__boxed_485_, v_hasTrace_boxed_486_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
lean_dec(v___y_483_);
lean_dec_ref(v___y_482_);
lean_dec(v___y_481_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2(lean_object* v_prop_488_, uint8_t v___x_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = l_Lean_Meta_mkFreshLevelMVar(v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_495_) == 0)
{
lean_object* v_a_496_; lean_object* v___x_497_; lean_object* v___x_498_; uint8_t v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v_a_496_ = lean_ctor_get(v___x_495_, 0);
lean_inc_n(v_a_496_, 2);
lean_dec_ref_known(v___x_495_, 1);
v___x_497_ = l_Lean_Expr_sort___override(v_a_496_);
v___x_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_498_, 0, v___x_497_);
v___x_499_ = 0;
v___x_500_ = lean_box(0);
v___x_501_ = l_Lean_Meta_mkFreshExprMVar(v___x_498_, v___x_499_, v___x_500_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_501_) == 0)
{
lean_object* v_a_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v_a_502_ = lean_ctor_get(v___x_501_, 0);
lean_inc_n(v_a_502_, 2);
lean_dec_ref_known(v___x_501_, 1);
v___x_503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_503_, 0, v_a_502_);
lean_inc_ref(v___x_503_);
v___x_504_ = l_Lean_Meta_mkFreshExprMVar(v___x_503_, v___x_499_, v___x_500_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_504_) == 0)
{
lean_object* v_a_505_; lean_object* v___x_506_; 
v_a_505_ = lean_ctor_get(v___x_504_, 0);
lean_inc(v_a_505_);
lean_dec_ref_known(v___x_504_, 1);
v___x_506_ = l_Lean_Meta_mkFreshExprMVar(v___x_503_, v___x_499_, v___x_500_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
if (lean_obj_tag(v___x_506_) == 0)
{
lean_object* v_a_507_; lean_object* v_keyedConfig_508_; uint8_t v_trackZetaDelta_509_; lean_object* v_zetaDeltaSet_510_; lean_object* v_lctx_511_; lean_object* v_localInstances_512_; lean_object* v_defEqCtx_x3f_513_; lean_object* v_synthPendingDepth_514_; lean_object* v_customCanUnfoldPredicate_x3f_515_; uint8_t v_univApprox_516_; uint8_t v_inTypeClassResolution_517_; uint8_t v_cacheInferType_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_608_; 
v_a_507_ = lean_ctor_get(v___x_506_, 0);
lean_inc(v_a_507_);
lean_dec_ref_known(v___x_506_, 1);
v_keyedConfig_508_ = lean_ctor_get(v___y_490_, 0);
v_trackZetaDelta_509_ = lean_ctor_get_uint8(v___y_490_, sizeof(void*)*7);
v_zetaDeltaSet_510_ = lean_ctor_get(v___y_490_, 1);
v_lctx_511_ = lean_ctor_get(v___y_490_, 2);
v_localInstances_512_ = lean_ctor_get(v___y_490_, 3);
v_defEqCtx_x3f_513_ = lean_ctor_get(v___y_490_, 4);
v_synthPendingDepth_514_ = lean_ctor_get(v___y_490_, 5);
v_customCanUnfoldPredicate_x3f_515_ = lean_ctor_get(v___y_490_, 6);
v_univApprox_516_ = lean_ctor_get_uint8(v___y_490_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_517_ = lean_ctor_get_uint8(v___y_490_, sizeof(void*)*7 + 2);
v_cacheInferType_518_ = lean_ctor_get_uint8(v___y_490_, sizeof(void*)*7 + 3);
v_isSharedCheck_608_ = !lean_is_exclusive(v___y_490_);
if (v_isSharedCheck_608_ == 0)
{
v___x_520_ = v___y_490_;
v_isShared_521_ = v_isSharedCheck_608_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_515_);
lean_inc(v_synthPendingDepth_514_);
lean_inc(v_defEqCtx_x3f_513_);
lean_inc(v_localInstances_512_);
lean_inc(v_lctx_511_);
lean_inc(v_zetaDeltaSet_510_);
lean_inc(v_keyedConfig_508_);
lean_dec(v___y_490_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_608_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; uint8_t v___x_529_; lean_object* v___x_530_; lean_object* v___x_532_; 
v___x_522_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___closed__1));
v___x_523_ = lean_box(0);
lean_inc(v_a_496_);
v___x_524_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_524_, 0, v_a_496_);
lean_ctor_set(v___x_524_, 1, v___x_523_);
v___x_525_ = l_Lean_Expr_const___override(v___x_522_, v___x_524_);
lean_inc(v_a_502_);
v___x_526_ = l_Lean_Expr_app___override(v___x_525_, v_a_502_);
lean_inc(v_a_505_);
v___x_527_ = l_Lean_Expr_app___override(v___x_526_, v_a_505_);
lean_inc(v_a_507_);
v___x_528_ = l_Lean_Expr_app___override(v___x_527_, v_a_507_);
v___x_529_ = 2;
v___x_530_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_529_, v_keyedConfig_508_);
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 0, v___x_530_);
v___x_532_ = v___x_520_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_530_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_zetaDeltaSet_510_);
lean_ctor_set(v_reuseFailAlloc_607_, 2, v_lctx_511_);
lean_ctor_set(v_reuseFailAlloc_607_, 3, v_localInstances_512_);
lean_ctor_set(v_reuseFailAlloc_607_, 4, v_defEqCtx_x3f_513_);
lean_ctor_set(v_reuseFailAlloc_607_, 5, v_synthPendingDepth_514_);
lean_ctor_set(v_reuseFailAlloc_607_, 6, v_customCanUnfoldPredicate_x3f_515_);
lean_ctor_set_uint8(v_reuseFailAlloc_607_, sizeof(void*)*7, v_trackZetaDelta_509_);
lean_ctor_set_uint8(v_reuseFailAlloc_607_, sizeof(void*)*7 + 1, v_univApprox_516_);
lean_ctor_set_uint8(v_reuseFailAlloc_607_, sizeof(void*)*7 + 2, v_inTypeClassResolution_517_);
lean_ctor_set_uint8(v_reuseFailAlloc_607_, sizeof(void*)*7 + 3, v_cacheInferType_518_);
v___x_532_ = v_reuseFailAlloc_607_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
lean_object* v___x_533_; 
v___x_533_ = l_Lean_Meta_isExprDefEq(v___x_528_, v_prop_488_, v___x_532_, v___y_491_, v___y_492_, v___y_493_);
lean_dec_ref(v___x_532_);
if (lean_obj_tag(v___x_533_) == 0)
{
lean_object* v_a_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_598_; 
v_a_534_ = lean_ctor_get(v___x_533_, 0);
v_isSharedCheck_598_ = !lean_is_exclusive(v___x_533_);
if (v_isSharedCheck_598_ == 0)
{
v___x_536_ = v___x_533_;
v_isShared_537_ = v_isSharedCheck_598_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_a_534_);
lean_dec(v___x_533_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_598_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
uint8_t v___x_538_; 
v___x_538_ = lean_unbox(v_a_534_);
if (v___x_538_ == 0)
{
lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_544_; 
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v_a_507_);
lean_ctor_set(v___x_539_, 1, v_a_534_);
v___x_540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_540_, 0, v_a_505_);
lean_ctor_set(v___x_540_, 1, v___x_539_);
v___x_541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_541_, 0, v_a_502_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v_a_496_);
lean_ctor_set(v___x_542_, 1, v___x_541_);
if (v_isShared_537_ == 0)
{
lean_ctor_set(v___x_536_, 0, v___x_542_);
v___x_544_ = v___x_536_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
else
{
lean_object* v___x_546_; 
lean_del_object(v___x_536_);
lean_dec(v_a_534_);
v___x_546_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_FieldSimp_discharge_spec__0___redArg(v_a_496_, v___y_491_);
if (lean_obj_tag(v___x_546_) == 0)
{
lean_object* v_a_547_; lean_object* v___x_548_; 
v_a_547_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_a_547_);
lean_dec_ref_known(v___x_546_, 1);
v___x_548_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_502_, v___y_491_);
if (lean_obj_tag(v___x_548_) == 0)
{
lean_object* v_a_549_; lean_object* v___x_550_; 
v_a_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_a_549_);
lean_dec_ref_known(v___x_548_, 1);
v___x_550_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_505_, v___y_491_);
if (lean_obj_tag(v___x_550_) == 0)
{
lean_object* v_a_551_; lean_object* v___x_552_; 
v_a_551_ = lean_ctor_get(v___x_550_, 0);
lean_inc(v_a_551_);
lean_dec_ref_known(v___x_550_, 1);
v___x_552_ = l_Lean_instantiateMVars___at___00Lean_Meta_Simp_dischargeEqnThmHypothesis_x3f_spec__1___redArg(v_a_507_, v___y_491_);
if (lean_obj_tag(v___x_552_) == 0)
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_565_; 
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_565_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_565_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_565_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_563_; 
v___x_557_ = lean_box(v___x_489_);
v___x_558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_558_, 0, v_a_553_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
v___x_559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_559_, 0, v_a_551_);
lean_ctor_set(v___x_559_, 1, v___x_558_);
v___x_560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_560_, 0, v_a_549_);
lean_ctor_set(v___x_560_, 1, v___x_559_);
v___x_561_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_561_, 0, v_a_547_);
lean_ctor_set(v___x_561_, 1, v___x_560_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_561_);
v___x_563_ = v___x_555_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v___x_561_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
else
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_573_; 
lean_dec(v_a_551_);
lean_dec(v_a_549_);
lean_dec(v_a_547_);
v_a_566_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_573_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_573_ == 0)
{
v___x_568_ = v___x_552_;
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_552_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_571_; 
if (v_isShared_569_ == 0)
{
v___x_571_ = v___x_568_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_572_; 
v_reuseFailAlloc_572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_572_, 0, v_a_566_);
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
else
{
lean_object* v_a_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_581_; 
lean_dec(v_a_549_);
lean_dec(v_a_547_);
lean_dec(v_a_507_);
v_a_574_ = lean_ctor_get(v___x_550_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_550_);
if (v_isSharedCheck_581_ == 0)
{
v___x_576_ = v___x_550_;
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_a_574_);
lean_dec(v___x_550_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v___x_579_; 
if (v_isShared_577_ == 0)
{
v___x_579_ = v___x_576_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v_a_574_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
}
else
{
lean_object* v_a_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_589_; 
lean_dec(v_a_547_);
lean_dec(v_a_507_);
lean_dec(v_a_505_);
v_a_582_ = lean_ctor_get(v___x_548_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_548_);
if (v_isSharedCheck_589_ == 0)
{
v___x_584_ = v___x_548_;
v_isShared_585_ = v_isSharedCheck_589_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_a_582_);
lean_dec(v___x_548_);
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
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
lean_dec(v_a_507_);
lean_dec(v_a_505_);
lean_dec(v_a_502_);
v_a_590_ = lean_ctor_get(v___x_546_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_546_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_546_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_546_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_590_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
}
}
else
{
lean_object* v_a_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_606_; 
lean_dec(v_a_507_);
lean_dec(v_a_505_);
lean_dec(v_a_502_);
lean_dec(v_a_496_);
v_a_599_ = lean_ctor_get(v___x_533_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_533_);
if (v_isSharedCheck_606_ == 0)
{
v___x_601_ = v___x_533_;
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_a_599_);
lean_dec(v___x_533_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_604_; 
if (v_isShared_602_ == 0)
{
v___x_604_ = v___x_601_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_a_599_);
v___x_604_ = v_reuseFailAlloc_605_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
return v___x_604_;
}
}
}
}
}
}
else
{
lean_object* v_a_609_; lean_object* v___x_611_; uint8_t v_isShared_612_; uint8_t v_isSharedCheck_616_; 
lean_dec(v_a_505_);
lean_dec(v_a_502_);
lean_dec(v_a_496_);
lean_dec_ref(v___y_490_);
lean_dec_ref(v_prop_488_);
v_a_609_ = lean_ctor_get(v___x_506_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_506_);
if (v_isSharedCheck_616_ == 0)
{
v___x_611_ = v___x_506_;
v_isShared_612_ = v_isSharedCheck_616_;
goto v_resetjp_610_;
}
else
{
lean_inc(v_a_609_);
lean_dec(v___x_506_);
v___x_611_ = lean_box(0);
v_isShared_612_ = v_isSharedCheck_616_;
goto v_resetjp_610_;
}
v_resetjp_610_:
{
lean_object* v___x_614_; 
if (v_isShared_612_ == 0)
{
v___x_614_ = v___x_611_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v_a_609_);
v___x_614_ = v_reuseFailAlloc_615_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
return v___x_614_;
}
}
}
}
else
{
lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_624_; 
lean_dec_ref_known(v___x_503_, 1);
lean_dec(v_a_502_);
lean_dec(v_a_496_);
lean_dec_ref(v___y_490_);
lean_dec_ref(v_prop_488_);
v_a_617_ = lean_ctor_get(v___x_504_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_504_);
if (v_isSharedCheck_624_ == 0)
{
v___x_619_ = v___x_504_;
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_504_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___x_622_; 
if (v_isShared_620_ == 0)
{
v___x_622_ = v___x_619_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_a_617_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
}
else
{
lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_632_; 
lean_dec(v_a_496_);
lean_dec_ref(v___y_490_);
lean_dec_ref(v_prop_488_);
v_a_625_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_632_ == 0)
{
v___x_627_ = v___x_501_;
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_501_);
v___x_627_ = lean_box(0);
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
v_resetjp_626_:
{
lean_object* v___x_630_; 
if (v_isShared_628_ == 0)
{
v___x_630_ = v___x_627_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_a_625_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
else
{
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
lean_dec_ref(v___y_490_);
lean_dec_ref(v_prop_488_);
v_a_633_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_495_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_495_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_a_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2___boxed(lean_object* v_prop_641_, lean_object* v___x_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_){
_start:
{
uint8_t v___x_117860__boxed_648_; lean_object* v_res_649_; 
v___x_117860__boxed_648_ = lean_unbox(v___x_642_);
v_res_649_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2(v_prop_641_, v___x_117860__boxed_648_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v___y_644_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg(uint8_t v___x_650_, lean_object* v_x_651_, lean_object* v_x_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
if (lean_obj_tag(v_x_652_) == 0)
{
lean_object* v___x_658_; 
v___x_658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_658_, 0, v_x_651_);
return v___x_658_;
}
else
{
lean_object* v_head_659_; lean_object* v_tail_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v_head_659_ = lean_ctor_get(v_x_652_, 0);
lean_inc(v_head_659_);
v_tail_660_ = lean_ctor_get(v_x_652_, 1);
lean_inc(v_tail_660_);
lean_dec_ref_known(v_x_652_, 2);
v___x_661_ = lean_unsigned_to_nat(1000u);
v___x_662_ = l_Lean_Meta_SimpTheorems_addConst(v_x_651_, v_head_659_, v___x_650_, v___x_650_, v___x_661_, v___y_653_, v___y_654_, v___y_655_, v___y_656_);
if (lean_obj_tag(v___x_662_) == 0)
{
lean_object* v_a_663_; 
v_a_663_ = lean_ctor_get(v___x_662_, 0);
lean_inc(v_a_663_);
lean_dec_ref_known(v___x_662_, 1);
v_x_651_ = v_a_663_;
v_x_652_ = v_tail_660_;
goto _start;
}
else
{
lean_dec(v_tail_660_);
return v___x_662_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg___boxed(lean_object* v___x_665_, lean_object* v_x_666_, lean_object* v_x_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
uint8_t v___x_118149__boxed_673_; lean_object* v_res_674_; 
v___x_118149__boxed_673_ = lean_unbox(v___x_665_);
v_res_674_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg(v___x_118149__boxed_673_, v_x_666_, v_x_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
lean_dec(v___y_671_);
lean_dec_ref(v___y_670_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
return v_res_674_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5(lean_object* v_e_675_){
_start:
{
if (lean_obj_tag(v_e_675_) == 0)
{
uint8_t v___x_676_; 
v___x_676_ = 2;
return v___x_676_;
}
else
{
lean_object* v_a_677_; 
v_a_677_ = lean_ctor_get(v_e_675_, 0);
if (lean_obj_tag(v_a_677_) == 0)
{
uint8_t v___x_678_; 
v___x_678_ = 1;
return v___x_678_;
}
else
{
uint8_t v___x_679_; 
v___x_679_ = 0;
return v___x_679_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5___boxed(lean_object* v_e_680_){
_start:
{
uint8_t v_res_681_; lean_object* v_r_682_; 
v_res_681_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5(v_e_680_);
lean_dec_ref(v_e_680_);
v_r_682_ = lean_box(v_res_681_);
return v_r_682_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0(void){
_start:
{
lean_object* v___x_683_; double v___x_684_; 
v___x_683_ = lean_unsigned_to_nat(0u);
v___x_684_ = lean_float_of_nat(v___x_683_);
return v___x_684_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2(void){
_start:
{
lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_686_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__1));
v___x_687_ = l_Lean_stringToMessageData(v___x_686_);
return v___x_687_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3(void){
_start:
{
lean_object* v___x_688_; double v___x_689_; 
v___x_688_ = lean_unsigned_to_nat(1000u);
v___x_689_ = lean_float_of_nat(v___x_688_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5(lean_object* v_cls_690_, uint8_t v_collapsed_691_, lean_object* v_tag_692_, lean_object* v_opts_693_, uint8_t v_clsEnabled_694_, lean_object* v_oldTraces_695_, lean_object* v_msg_696_, lean_object* v_resStartStop_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_){
_start:
{
lean_object* v_fst_706_; lean_object* v_snd_707_; lean_object* v___y_709_; lean_object* v___y_710_; lean_object* v_data_711_; lean_object* v_fst_722_; lean_object* v_snd_723_; lean_object* v___x_724_; uint8_t v___x_725_; lean_object* v___y_727_; lean_object* v_a_728_; uint8_t v___y_743_; double v___y_774_; 
v_fst_706_ = lean_ctor_get(v_resStartStop_697_, 0);
lean_inc(v_fst_706_);
v_snd_707_ = lean_ctor_get(v_resStartStop_697_, 1);
lean_inc(v_snd_707_);
lean_dec_ref(v_resStartStop_697_);
v_fst_722_ = lean_ctor_get(v_snd_707_, 0);
lean_inc(v_fst_722_);
v_snd_723_ = lean_ctor_get(v_snd_707_, 1);
lean_inc(v_snd_723_);
lean_dec(v_snd_707_);
v___x_724_ = l_Lean_trace_profiler;
v___x_725_ = l_Lean_Option_get___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__3(v_opts_693_, v___x_724_);
if (v___x_725_ == 0)
{
v___y_743_ = v___x_725_;
goto v___jp_742_;
}
else
{
lean_object* v___x_779_; uint8_t v___x_780_; 
v___x_779_ = l_Lean_trace_profiler_useHeartbeats;
v___x_780_ = l_Lean_Option_get___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__3(v_opts_693_, v___x_779_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; lean_object* v___x_782_; double v___x_783_; double v___x_784_; double v___x_785_; 
v___x_781_ = l_Lean_trace_profiler_threshold;
v___x_782_ = l_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__7(v_opts_693_, v___x_781_);
v___x_783_ = lean_float_of_nat(v___x_782_);
v___x_784_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__3);
v___x_785_ = lean_float_div(v___x_783_, v___x_784_);
v___y_774_ = v___x_785_;
goto v___jp_773_;
}
else
{
lean_object* v___x_786_; lean_object* v___x_787_; double v___x_788_; 
v___x_786_ = l_Lean_trace_profiler_threshold;
v___x_787_ = l_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__7(v_opts_693_, v___x_786_);
v___x_788_ = lean_float_of_nat(v___x_787_);
v___y_774_ = v___x_788_;
goto v___jp_773_;
}
}
v___jp_708_:
{
lean_object* v___x_712_; 
lean_inc(v___y_709_);
v___x_712_ = l___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__4___redArg(v_oldTraces_695_, v_data_711_, v___y_709_, v___y_710_, v___y_701_, v___y_702_, v___y_703_, v___y_704_);
if (lean_obj_tag(v___x_712_) == 0)
{
lean_object* v___x_713_; 
lean_dec_ref_known(v___x_712_, 1);
v___x_713_ = l_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__5___redArg(v_fst_706_);
return v___x_713_;
}
else
{
lean_object* v_a_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_721_; 
lean_dec(v_fst_706_);
v_a_714_ = lean_ctor_get(v___x_712_, 0);
v_isSharedCheck_721_ = !lean_is_exclusive(v___x_712_);
if (v_isSharedCheck_721_ == 0)
{
v___x_716_ = v___x_712_;
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_a_714_);
lean_dec(v___x_712_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v___x_719_; 
if (v_isShared_717_ == 0)
{
v___x_719_ = v___x_716_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v_a_714_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
}
}
v___jp_726_:
{
uint8_t v_result_729_; lean_object* v___x_730_; lean_object* v___x_731_; double v___x_732_; lean_object* v_data_733_; 
v_result_729_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5_spec__5(v_fst_706_);
v___x_730_ = lean_box(v_result_729_);
v___x_731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_731_, 0, v___x_730_);
v___x_732_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__0);
lean_inc_ref(v_tag_692_);
lean_inc_ref(v___x_731_);
lean_inc(v_cls_690_);
v_data_733_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_733_, 0, v_cls_690_);
lean_ctor_set(v_data_733_, 1, v___x_731_);
lean_ctor_set(v_data_733_, 2, v_tag_692_);
lean_ctor_set_float(v_data_733_, sizeof(void*)*3, v___x_732_);
lean_ctor_set_float(v_data_733_, sizeof(void*)*3 + 8, v___x_732_);
lean_ctor_set_uint8(v_data_733_, sizeof(void*)*3 + 16, v_collapsed_691_);
if (v___x_725_ == 0)
{
lean_dec_ref_known(v___x_731_, 1);
lean_dec(v_snd_723_);
lean_dec(v_fst_722_);
lean_dec_ref(v_tag_692_);
lean_dec(v_cls_690_);
v___y_709_ = v___y_727_;
v___y_710_ = v_a_728_;
v_data_711_ = v_data_733_;
goto v___jp_708_;
}
else
{
lean_object* v_data_734_; double v___x_735_; double v___x_736_; 
lean_dec_ref_known(v_data_733_, 3);
v_data_734_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_734_, 0, v_cls_690_);
lean_ctor_set(v_data_734_, 1, v___x_731_);
lean_ctor_set(v_data_734_, 2, v_tag_692_);
v___x_735_ = lean_unbox_float(v_fst_722_);
lean_dec(v_fst_722_);
lean_ctor_set_float(v_data_734_, sizeof(void*)*3, v___x_735_);
v___x_736_ = lean_unbox_float(v_snd_723_);
lean_dec(v_snd_723_);
lean_ctor_set_float(v_data_734_, sizeof(void*)*3 + 8, v___x_736_);
lean_ctor_set_uint8(v_data_734_, sizeof(void*)*3 + 16, v_collapsed_691_);
v___y_709_ = v___y_727_;
v___y_710_ = v_a_728_;
v_data_711_ = v_data_734_;
goto v___jp_708_;
}
}
v___jp_737_:
{
lean_object* v_ref_738_; lean_object* v___x_739_; 
v_ref_738_ = lean_ctor_get(v___y_703_, 5);
lean_inc(v___y_704_);
lean_inc_ref(v___y_703_);
lean_inc(v___y_702_);
lean_inc_ref(v___y_701_);
lean_inc(v___y_700_);
lean_inc_ref(v___y_699_);
lean_inc(v___y_698_);
lean_inc(v_fst_706_);
v___x_739_ = lean_apply_9(v_msg_696_, v_fst_706_, v___y_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, lean_box(0));
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_a_740_);
lean_dec_ref_known(v___x_739_, 1);
v___y_727_ = v_ref_738_;
v_a_728_ = v_a_740_;
goto v___jp_726_;
}
else
{
lean_object* v___x_741_; 
lean_dec_ref_known(v___x_739_, 1);
v___x_741_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___closed__2);
v___y_727_ = v_ref_738_;
v_a_728_ = v___x_741_;
goto v___jp_726_;
}
}
v___jp_742_:
{
if (v_clsEnabled_694_ == 0)
{
if (v___y_743_ == 0)
{
lean_object* v___x_744_; lean_object* v_traceState_745_; lean_object* v_env_746_; lean_object* v_nextMacroScope_747_; lean_object* v_ngen_748_; lean_object* v_auxDeclNGen_749_; lean_object* v_cache_750_; lean_object* v_messages_751_; lean_object* v_infoState_752_; lean_object* v_snapshotTasks_753_; lean_object* v___x_755_; uint8_t v_isShared_756_; uint8_t v_isSharedCheck_772_; 
lean_dec(v_snd_723_);
lean_dec(v_fst_722_);
lean_dec_ref(v_msg_696_);
lean_dec_ref(v_tag_692_);
lean_dec(v_cls_690_);
v___x_744_ = lean_st_ref_take(v___y_704_);
v_traceState_745_ = lean_ctor_get(v___x_744_, 4);
v_env_746_ = lean_ctor_get(v___x_744_, 0);
v_nextMacroScope_747_ = lean_ctor_get(v___x_744_, 1);
v_ngen_748_ = lean_ctor_get(v___x_744_, 2);
v_auxDeclNGen_749_ = lean_ctor_get(v___x_744_, 3);
v_cache_750_ = lean_ctor_get(v___x_744_, 5);
v_messages_751_ = lean_ctor_get(v___x_744_, 6);
v_infoState_752_ = lean_ctor_get(v___x_744_, 7);
v_snapshotTasks_753_ = lean_ctor_get(v___x_744_, 8);
v_isSharedCheck_772_ = !lean_is_exclusive(v___x_744_);
if (v_isSharedCheck_772_ == 0)
{
v___x_755_ = v___x_744_;
v_isShared_756_ = v_isSharedCheck_772_;
goto v_resetjp_754_;
}
else
{
lean_inc(v_snapshotTasks_753_);
lean_inc(v_infoState_752_);
lean_inc(v_messages_751_);
lean_inc(v_cache_750_);
lean_inc(v_traceState_745_);
lean_inc(v_auxDeclNGen_749_);
lean_inc(v_ngen_748_);
lean_inc(v_nextMacroScope_747_);
lean_inc(v_env_746_);
lean_dec(v___x_744_);
v___x_755_ = lean_box(0);
v_isShared_756_ = v_isSharedCheck_772_;
goto v_resetjp_754_;
}
v_resetjp_754_:
{
uint64_t v_tid_757_; lean_object* v_traces_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_771_; 
v_tid_757_ = lean_ctor_get_uint64(v_traceState_745_, sizeof(void*)*1);
v_traces_758_ = lean_ctor_get(v_traceState_745_, 0);
v_isSharedCheck_771_ = !lean_is_exclusive(v_traceState_745_);
if (v_isSharedCheck_771_ == 0)
{
v___x_760_ = v_traceState_745_;
v_isShared_761_ = v_isSharedCheck_771_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_traces_758_);
lean_dec(v_traceState_745_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_771_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v___x_762_; lean_object* v___x_764_; 
v___x_762_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_695_, v_traces_758_);
lean_dec_ref(v_traces_758_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 0, v___x_762_);
v___x_764_ = v___x_760_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_770_; 
v_reuseFailAlloc_770_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_770_, 0, v___x_762_);
lean_ctor_set_uint64(v_reuseFailAlloc_770_, sizeof(void*)*1, v_tid_757_);
v___x_764_ = v_reuseFailAlloc_770_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
lean_object* v___x_766_; 
if (v_isShared_756_ == 0)
{
lean_ctor_set(v___x_755_, 4, v___x_764_);
v___x_766_ = v___x_755_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_env_746_);
lean_ctor_set(v_reuseFailAlloc_769_, 1, v_nextMacroScope_747_);
lean_ctor_set(v_reuseFailAlloc_769_, 2, v_ngen_748_);
lean_ctor_set(v_reuseFailAlloc_769_, 3, v_auxDeclNGen_749_);
lean_ctor_set(v_reuseFailAlloc_769_, 4, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_769_, 5, v_cache_750_);
lean_ctor_set(v_reuseFailAlloc_769_, 6, v_messages_751_);
lean_ctor_set(v_reuseFailAlloc_769_, 7, v_infoState_752_);
lean_ctor_set(v_reuseFailAlloc_769_, 8, v_snapshotTasks_753_);
v___x_766_ = v_reuseFailAlloc_769_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_767_ = lean_st_ref_set(v___y_704_, v___x_766_);
v___x_768_ = l_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__4_spec__5___redArg(v_fst_706_);
return v___x_768_;
}
}
}
}
}
else
{
goto v___jp_737_;
}
}
else
{
goto v___jp_737_;
}
}
v___jp_773_:
{
double v___x_775_; double v___x_776_; double v___x_777_; uint8_t v___x_778_; 
v___x_775_ = lean_unbox_float(v_snd_723_);
v___x_776_ = lean_unbox_float(v_fst_722_);
v___x_777_ = lean_float_sub(v___x_775_, v___x_776_);
v___x_778_ = lean_float_decLt(v___y_774_, v___x_777_);
v___y_743_ = v___x_778_;
goto v___jp_742_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5___boxed(lean_object* v_cls_789_, lean_object* v_collapsed_790_, lean_object* v_tag_791_, lean_object* v_opts_792_, lean_object* v_clsEnabled_793_, lean_object* v_oldTraces_794_, lean_object* v_msg_795_, lean_object* v_resStartStop_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_){
_start:
{
uint8_t v_collapsed_boxed_805_; uint8_t v_clsEnabled_boxed_806_; lean_object* v_res_807_; 
v_collapsed_boxed_805_ = lean_unbox(v_collapsed_790_);
v_clsEnabled_boxed_806_ = lean_unbox(v_clsEnabled_793_);
v_res_807_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5(v_cls_789_, v_collapsed_boxed_805_, v_tag_791_, v_opts_792_, v_clsEnabled_boxed_806_, v_oldTraces_794_, v_msg_795_, v_resStartStop_796_, v___y_797_, v___y_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
lean_dec(v___y_799_);
lean_dec_ref(v___y_798_);
lean_dec(v___y_797_);
lean_dec_ref(v_opts_792_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(lean_object* v_x_808_, lean_object* v_x_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
if (lean_obj_tag(v_x_809_) == 0)
{
lean_object* v___x_815_; 
v___x_815_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_815_, 0, v_x_808_);
return v___x_815_;
}
else
{
lean_object* v_head_816_; lean_object* v_tail_817_; uint8_t v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; 
v_head_816_ = lean_ctor_get(v_x_809_, 0);
lean_inc(v_head_816_);
v_tail_817_ = lean_ctor_get(v_x_809_, 1);
lean_inc(v_tail_817_);
lean_dec_ref_known(v_x_809_, 2);
v___x_818_ = 0;
v___x_819_ = lean_unsigned_to_nat(1000u);
v___x_820_ = l_Lean_Meta_SimpTheorems_addConst(v_x_808_, v_head_816_, v___x_818_, v___x_818_, v___x_819_, v___y_810_, v___y_811_, v___y_812_, v___y_813_);
if (lean_obj_tag(v___x_820_) == 0)
{
lean_object* v_a_821_; 
v_a_821_ = lean_ctor_get(v___x_820_, 0);
lean_inc(v_a_821_);
lean_dec_ref_known(v___x_820_, 1);
v_x_808_ = v_a_821_;
v_x_809_ = v_tail_817_;
goto _start;
}
else
{
lean_dec(v_tail_817_);
return v___x_820_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg___boxed(lean_object* v_x_823_, lean_object* v_x_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(v_x_823_, v_x_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
lean_dec(v___y_828_);
lean_dec_ref(v___y_827_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
return v_res_830_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22(void){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_875_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23(void){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__2(lean_box(0));
return v___x_876_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24(void){
_start:
{
lean_object* v___x_877_; 
v___x_877_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_FieldSimp_discharge_spec__3(lean_box(0));
return v___x_877_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25(void){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_878_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26(void){
_start:
{
lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_879_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__25);
v___x_880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_880_, 0, v___x_879_);
return v___x_880_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27(void){
_start:
{
lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_881_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__26);
v___x_882_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__24);
v___x_883_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__23);
v___x_884_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__22);
v___x_885_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_885_, 0, v___x_884_);
lean_ctor_set(v___x_885_, 1, v___x_884_);
lean_ctor_set(v___x_885_, 2, v___x_883_);
lean_ctor_set(v___x_885_, 3, v___x_882_);
lean_ctor_set(v___x_885_, 4, v___x_883_);
lean_ctor_set(v___x_885_, 5, v___x_881_);
return v___x_885_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30(void){
_start:
{
lean_object* v___x_889_; lean_object* v___x_890_; 
v___x_889_ = lean_box(0);
v___x_890_ = l_Lean_Expr_sort___override(v___x_889_);
return v___x_890_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37(void){
_start:
{
lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_900_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33));
v___x_901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__36));
v___x_902_ = l_Lean_Name_append(v___x_901_, v___x_900_);
return v___x_902_;
}
}
static double _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38(void){
_start:
{
lean_object* v___x_903_; double v___x_904_; 
v___x_903_ = lean_unsigned_to_nat(1000000000u);
v___x_904_ = lean_float_of_nat(v___x_903_);
return v___x_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed(lean_object* v_prop_905_, lean_object* v_a_906_, lean_object* v_a_907_, lean_object* v_a_908_, lean_object* v_a_909_, lean_object* v_a_910_, lean_object* v_a_911_, lean_object* v_a_912_, lean_object* v_a_913_){
_start:
{
lean_object* v_res_914_; 
v_res_914_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge(v_prop_905_, v_a_906_, v_a_907_, v_a_908_, v_a_909_, v_a_910_, v_a_911_, v_a_912_);
lean_dec(v_a_912_);
lean_dec_ref(v_a_911_);
lean_dec(v_a_910_);
lean_dec_ref(v_a_909_);
lean_dec(v_a_908_);
lean_dec_ref(v_a_907_);
lean_dec(v_a_906_);
return v_res_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp_discharge(lean_object* v_prop_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_, lean_object* v_a_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_){
_start:
{
lean_object* v_r_925_; lean_object* v___y_929_; lean_object* v___y_930_; uint8_t v___y_931_; lean_object* v___y_935_; lean_object* v_a_936_; lean_object* v___y_940_; lean_object* v___y_941_; lean_object* v___y_942_; lean_object* v___y_943_; lean_object* v___y_944_; lean_object* v___y_945_; lean_object* v___y_946_; lean_object* v___y_947_; lean_object* v___y_948_; uint8_t v___y_949_; lean_object* v___y_1059_; lean_object* v___y_1060_; lean_object* v___y_1061_; lean_object* v___y_1062_; lean_object* v___y_1063_; lean_object* v___y_1064_; lean_object* v___y_1065_; lean_object* v___y_1066_; lean_object* v___y_1080_; lean_object* v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; lean_object* v___y_1085_; lean_object* v___y_1086_; lean_object* v___y_1087_; lean_object* v___y_1088_; uint8_t v___y_1089_; lean_object* v_options_1090_; lean_object* v_inheritedTraceOptions_1091_; uint8_t v_hasTrace_1092_; lean_object* v___f_1093_; lean_object* v_____do__lift_1095_; lean_object* v___y_1096_; lean_object* v___y_1097_; lean_object* v___y_1098_; lean_object* v___y_1099_; lean_object* v___y_1100_; lean_object* v___y_1101_; lean_object* v___y_1102_; 
v_options_1090_ = lean_ctor_get(v_a_921_, 2);
v_inheritedTraceOptions_1091_ = lean_ctor_get(v_a_921_, 13);
v_hasTrace_1092_ = lean_ctor_get_uint8(v_options_1090_, sizeof(void*)*1);
lean_inc_ref(v_prop_915_);
v___f_1093_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__0___boxed), 6, 1);
lean_closure_set(v___f_1093_, 0, v_prop_915_);
if (v_hasTrace_1092_ == 0)
{
lean_object* v___x_1145_; 
lean_inc_ref(v_prop_915_);
v___x_1145_ = l___private_Lean_Meta_Tactic_Simp_Rewrite_0__Lean_Meta_Simp_dischargeUsingAssumption_x3f(v_prop_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1145_) == 0)
{
lean_object* v_a_1146_; 
v_a_1146_ = lean_ctor_get(v___x_1145_, 0);
lean_inc(v_a_1146_);
lean_dec_ref_known(v___x_1145_, 1);
v_____do__lift_1095_ = v_a_1146_;
v___y_1096_ = v_a_916_;
v___y_1097_ = v_a_917_;
v___y_1098_ = v_a_918_;
v___y_1099_ = v_a_919_;
v___y_1100_ = v_a_920_;
v___y_1101_ = v_a_921_;
v___y_1102_ = v_a_922_;
goto v___jp_1094_;
}
else
{
lean_dec_ref(v___f_1093_);
lean_dec_ref(v_prop_915_);
return v___x_1145_;
}
}
else
{
lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; uint8_t v___x_1151_; lean_object* v___y_1153_; lean_object* v___y_1154_; lean_object* v_a_1155_; lean_object* v___y_1165_; lean_object* v___y_1166_; lean_object* v_a_1167_; lean_object* v___y_1170_; lean_object* v___y_1171_; lean_object* v_a_1172_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v___y_1181_; lean_object* v___y_1182_; lean_object* v___y_1183_; lean_object* v___y_1184_; uint8_t v___y_1185_; lean_object* v___y_1187_; lean_object* v___y_1188_; lean_object* v___y_1189_; lean_object* v_a_1190_; lean_object* v___y_1194_; lean_object* v___y_1195_; lean_object* v___y_1196_; lean_object* v___y_1197_; uint8_t v___y_1198_; lean_object* v___y_1279_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1289_; lean_object* v___y_1290_; lean_object* v___y_1291_; lean_object* v___y_1292_; uint8_t v___y_1293_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v_a_1297_; lean_object* v___y_1310_; lean_object* v___y_1311_; lean_object* v_a_1312_; lean_object* v___y_1315_; lean_object* v___y_1316_; lean_object* v_a_1317_; lean_object* v___y_1320_; lean_object* v___y_1321_; lean_object* v___y_1322_; lean_object* v___y_1326_; lean_object* v___y_1327_; lean_object* v___y_1328_; lean_object* v___y_1329_; uint8_t v___y_1330_; lean_object* v___y_1332_; lean_object* v___y_1333_; lean_object* v___y_1334_; lean_object* v_a_1335_; lean_object* v___y_1339_; uint8_t v___y_1340_; lean_object* v___y_1341_; lean_object* v___y_1342_; lean_object* v___y_1343_; uint8_t v___y_1344_; lean_object* v___y_1425_; uint8_t v___y_1426_; lean_object* v___y_1427_; lean_object* v___y_1428_; lean_object* v___y_1436_; uint8_t v___y_1437_; lean_object* v___y_1438_; lean_object* v___y_1439_; lean_object* v___y_1440_; uint8_t v___y_1441_; 
v___x_1147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__33));
lean_inc_ref(v_prop_915_);
v___x_1148_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FieldSimp_Discharger_0__Mathlib_Tactic_FieldSimp_dischargerTraceMessage___boxed), 11, 2);
lean_closure_set(v___x_1148_, 0, lean_box(0));
lean_closure_set(v___x_1148_, 1, v_prop_915_);
v___x_1149_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__34));
v___x_1150_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__37);
v___x_1151_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1091_, v_options_1090_, v___x_1150_);
if (v___x_1151_ == 0)
{
lean_object* v___x_1521_; uint8_t v___x_1522_; 
v___x_1521_ = l_Lean_trace_profiler;
v___x_1522_ = l_Lean_Option_get___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__3(v_options_1090_, v___x_1521_);
if (v___x_1522_ == 0)
{
lean_object* v___x_1523_; 
lean_dec_ref(v___x_1148_);
lean_inc_ref(v_prop_915_);
v___x_1523_ = l___private_Lean_Meta_Tactic_Simp_Rewrite_0__Lean_Meta_Simp_dischargeUsingAssumption_x3f(v_prop_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1523_) == 0)
{
lean_object* v_a_1524_; 
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc(v_a_1524_);
lean_dec_ref_known(v___x_1523_, 1);
v_____do__lift_1095_ = v_a_1524_;
v___y_1096_ = v_a_916_;
v___y_1097_ = v_a_917_;
v___y_1098_ = v_a_918_;
v___y_1099_ = v_a_919_;
v___y_1100_ = v_a_920_;
v___y_1101_ = v_a_921_;
v___y_1102_ = v_a_922_;
goto v___jp_1094_;
}
else
{
lean_dec_ref(v___f_1093_);
lean_dec_ref(v_prop_915_);
return v___x_1523_;
}
}
else
{
lean_dec_ref(v___f_1093_);
goto v___jp_1442_;
}
}
else
{
lean_dec_ref(v___f_1093_);
goto v___jp_1442_;
}
v___jp_1152_:
{
lean_object* v___x_1156_; double v___x_1157_; double v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1156_ = lean_io_get_num_heartbeats();
v___x_1157_ = lean_float_of_nat(v___y_1154_);
v___x_1158_ = lean_float_of_nat(v___x_1156_);
v___x_1159_ = lean_box_float(v___x_1157_);
v___x_1160_ = lean_box_float(v___x_1158_);
v___x_1161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1159_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
v___x_1162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1162_, 0, v_a_1155_);
lean_ctor_set(v___x_1162_, 1, v___x_1161_);
v___x_1163_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5(v___x_1147_, v_hasTrace_1092_, v___x_1149_, v_options_1090_, v___x_1151_, v___y_1153_, v___x_1148_, v___x_1162_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
return v___x_1163_;
}
v___jp_1164_:
{
lean_object* v___x_1168_; 
v___x_1168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1168_, 0, v_a_1167_);
v___y_1153_ = v___y_1165_;
v___y_1154_ = v___y_1166_;
v_a_1155_ = v___x_1168_;
goto v___jp_1152_;
}
v___jp_1169_:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1173_, 0, v_a_1172_);
v___y_1153_ = v___y_1170_;
v___y_1154_ = v___y_1171_;
v_a_1155_ = v___x_1173_;
goto v___jp_1152_;
}
v___jp_1174_:
{
if (lean_obj_tag(v___y_1177_) == 0)
{
lean_object* v_a_1178_; 
v_a_1178_ = lean_ctor_get(v___y_1177_, 0);
lean_inc(v_a_1178_);
lean_dec_ref_known(v___y_1177_, 1);
v___y_1165_ = v___y_1175_;
v___y_1166_ = v___y_1176_;
v_a_1167_ = v_a_1178_;
goto v___jp_1164_;
}
else
{
lean_object* v_a_1179_; 
v_a_1179_ = lean_ctor_get(v___y_1177_, 0);
lean_inc(v_a_1179_);
lean_dec_ref_known(v___y_1177_, 1);
v___y_1170_ = v___y_1175_;
v___y_1171_ = v___y_1176_;
v_a_1172_ = v_a_1179_;
goto v___jp_1169_;
}
}
v___jp_1180_:
{
if (v___y_1185_ == 0)
{
lean_dec_ref(v___y_1183_);
lean_inc(v___y_1182_);
v___y_1165_ = v___y_1181_;
v___y_1166_ = v___y_1184_;
v_a_1167_ = v___y_1182_;
goto v___jp_1164_;
}
else
{
v___y_1170_ = v___y_1181_;
v___y_1171_ = v___y_1184_;
v_a_1172_ = v___y_1183_;
goto v___jp_1169_;
}
}
v___jp_1186_:
{
uint8_t v___x_1191_; 
v___x_1191_ = l_Lean_Exception_isInterrupt(v_a_1190_);
if (v___x_1191_ == 0)
{
uint8_t v___x_1192_; 
lean_inc_ref(v_a_1190_);
v___x_1192_ = l_Lean_Exception_isRuntime(v_a_1190_);
v___y_1181_ = v___y_1187_;
v___y_1182_ = v___y_1188_;
v___y_1183_ = v_a_1190_;
v___y_1184_ = v___y_1189_;
v___y_1185_ = v___x_1192_;
goto v___jp_1180_;
}
else
{
v___y_1181_ = v___y_1187_;
v___y_1182_ = v___y_1188_;
v___y_1183_ = v_a_1190_;
v___y_1184_ = v___y_1189_;
v___y_1185_ = v___x_1191_;
goto v___jp_1180_;
}
}
v___jp_1193_:
{
if (v___y_1198_ == 0)
{
lean_object* v_config_1199_; lean_object* v_userConfig_1200_; lean_object* v_zetaDeltaSet_1201_; lean_object* v_initUsedZetaDelta_1202_; lean_object* v_metaConfig_1203_; lean_object* v_indexConfig_1204_; uint32_t v_maxDischargeDepth_1205_; lean_object* v_simpTheorems_1206_; lean_object* v_congrTheorems_1207_; lean_object* v_parent_x3f_1208_; uint32_t v_dischargeDepth_1209_; lean_object* v_lctxInitIndices_1210_; uint8_t v_inDSimp_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; 
lean_dec_ref(v___y_1197_);
v_config_1199_ = lean_ctor_get(v_a_917_, 0);
v_userConfig_1200_ = lean_ctor_get(v_a_917_, 1);
v_zetaDeltaSet_1201_ = lean_ctor_get(v_a_917_, 2);
v_initUsedZetaDelta_1202_ = lean_ctor_get(v_a_917_, 3);
v_metaConfig_1203_ = lean_ctor_get(v_a_917_, 4);
v_indexConfig_1204_ = lean_ctor_get(v_a_917_, 5);
v_maxDischargeDepth_1205_ = lean_ctor_get_uint32(v_a_917_, sizeof(void*)*10);
v_simpTheorems_1206_ = lean_ctor_get(v_a_917_, 6);
v_congrTheorems_1207_ = lean_ctor_get(v_a_917_, 7);
v_parent_x3f_1208_ = lean_ctor_get(v_a_917_, 8);
v_dischargeDepth_1209_ = lean_ctor_get_uint32(v_a_917_, sizeof(void*)*10 + 4);
v_lctxInitIndices_1210_ = lean_ctor_get(v_a_917_, 9);
v_inDSimp_1211_ = lean_ctor_get_uint8(v_a_917_, sizeof(void*)*10 + 8);
v___x_1212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21));
v___x_1213_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27);
v___x_1214_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(v___x_1213_, v___x_1212_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1214_) == 0)
{
lean_object* v_a_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; 
v_a_1215_ = lean_ctor_get(v___x_1214_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___x_1214_, 1);
v___x_1216_ = lean_st_ref_get(v_a_918_);
v___x_1217_ = l_Lean_Meta_Simp_getSimprocs___redArg(v_a_922_);
if (lean_obj_tag(v___x_1217_) == 0)
{
lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1275_; 
v_a_1218_ = lean_ctor_get(v___x_1217_, 0);
v_isSharedCheck_1275_ = !lean_is_exclusive(v___x_1217_);
if (v_isSharedCheck_1275_ == 0)
{
v___x_1220_ = v___x_1217_;
v_isShared_1221_ = v_isSharedCheck_1275_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1217_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1275_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v_usedTheorems_1222_; lean_object* v_diag_1223_; uint32_t v___x_1224_; uint32_t v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1235_; 
v_usedTheorems_1222_ = lean_ctor_get(v___x_1216_, 3);
lean_inc_ref(v_usedTheorems_1222_);
v_diag_1223_ = lean_ctor_get(v___x_1216_, 5);
lean_inc_ref(v_diag_1223_);
lean_dec(v___x_1216_);
v___x_1224_ = 1;
v___x_1225_ = lean_uint32_add(v_dischargeDepth_1209_, v___x_1224_);
lean_inc(v_lctxInitIndices_1210_);
lean_inc(v_parent_x3f_1208_);
lean_inc_ref(v_congrTheorems_1207_);
lean_inc_ref_n(v_simpTheorems_1206_, 2);
lean_inc_ref(v_indexConfig_1204_);
lean_inc_ref(v_metaConfig_1203_);
lean_inc(v_initUsedZetaDelta_1202_);
lean_inc(v_zetaDeltaSet_1201_);
lean_inc_ref(v_userConfig_1200_);
lean_inc_ref(v_config_1199_);
v___x_1226_ = lean_alloc_ctor(0, 10, 9);
lean_ctor_set(v___x_1226_, 0, v_config_1199_);
lean_ctor_set(v___x_1226_, 1, v_userConfig_1200_);
lean_ctor_set(v___x_1226_, 2, v_zetaDeltaSet_1201_);
lean_ctor_set(v___x_1226_, 3, v_initUsedZetaDelta_1202_);
lean_ctor_set(v___x_1226_, 4, v_metaConfig_1203_);
lean_ctor_set(v___x_1226_, 5, v_indexConfig_1204_);
lean_ctor_set(v___x_1226_, 6, v_simpTheorems_1206_);
lean_ctor_set(v___x_1226_, 7, v_congrTheorems_1207_);
lean_ctor_set(v___x_1226_, 8, v_parent_x3f_1208_);
lean_ctor_set(v___x_1226_, 9, v_lctxInitIndices_1210_);
lean_ctor_set_uint32(v___x_1226_, sizeof(void*)*10, v_maxDischargeDepth_1205_);
lean_ctor_set_uint32(v___x_1226_, sizeof(void*)*10 + 4, v___x_1225_);
lean_ctor_set_uint8(v___x_1226_, sizeof(void*)*10 + 8, v_inDSimp_1211_);
v___x_1227_ = lean_array_push(v_simpTheorems_1206_, v_a_1215_);
v___x_1228_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v___x_1226_, v___x_1227_);
v___x_1229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1229_, 0, v_usedTheorems_1222_);
lean_ctor_set(v___x_1229_, 1, v_diag_1223_);
v___x_1230_ = lean_unsigned_to_nat(1u);
v___x_1231_ = lean_mk_empty_array_with_capacity(v___x_1230_);
v___x_1232_ = lean_array_push(v___x_1231_, v_a_1218_);
v___x_1233_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed), 9, 0);
if (v_isShared_1221_ == 0)
{
lean_ctor_set_tag(v___x_1220_, 1);
lean_ctor_set(v___x_1220_, 0, v___x_1233_);
v___x_1235_ = v___x_1220_;
goto v_reusejp_1234_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v___x_1233_);
v___x_1235_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1234_;
}
v_reusejp_1234_:
{
lean_object* v___x_1236_; 
v___x_1236_ = l_Lean_Meta_simp(v_prop_915_, v___x_1228_, v___x_1232_, v___x_1235_, v___x_1229_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
lean_dec_ref_known(v___x_1229_, 2);
if (lean_obj_tag(v___x_1236_) == 0)
{
lean_object* v_a_1237_; lean_object* v_fst_1238_; lean_object* v_snd_1239_; lean_object* v___x_1240_; lean_object* v_cache_1241_; lean_object* v_congrCache_1242_; lean_object* v_dsimpCache_1243_; lean_object* v_numSteps_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1270_; 
v_a_1237_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1237_);
lean_dec_ref_known(v___x_1236_, 1);
v_fst_1238_ = lean_ctor_get(v_a_1237_, 0);
lean_inc(v_fst_1238_);
v_snd_1239_ = lean_ctor_get(v_a_1237_, 1);
lean_inc(v_snd_1239_);
lean_dec(v_a_1237_);
v___x_1240_ = lean_st_ref_get(v_a_918_);
v_cache_1241_ = lean_ctor_get(v___x_1240_, 0);
v_congrCache_1242_ = lean_ctor_get(v___x_1240_, 1);
v_dsimpCache_1243_ = lean_ctor_get(v___x_1240_, 2);
v_numSteps_1244_ = lean_ctor_get(v___x_1240_, 4);
v_isSharedCheck_1270_ = !lean_is_exclusive(v___x_1240_);
if (v_isSharedCheck_1270_ == 0)
{
lean_object* v_unused_1271_; lean_object* v_unused_1272_; 
v_unused_1271_ = lean_ctor_get(v___x_1240_, 5);
lean_dec(v_unused_1271_);
v_unused_1272_ = lean_ctor_get(v___x_1240_, 3);
lean_dec(v_unused_1272_);
v___x_1246_ = v___x_1240_;
v_isShared_1247_ = v_isSharedCheck_1270_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_numSteps_1244_);
lean_inc(v_dsimpCache_1243_);
lean_inc(v_congrCache_1242_);
lean_inc(v_cache_1241_);
lean_dec(v___x_1240_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1270_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v_usedTheorems_1248_; lean_object* v_diag_1249_; lean_object* v___x_1251_; 
v_usedTheorems_1248_ = lean_ctor_get(v_snd_1239_, 0);
lean_inc_ref(v_usedTheorems_1248_);
v_diag_1249_ = lean_ctor_get(v_snd_1239_, 1);
lean_inc_ref(v_diag_1249_);
lean_dec(v_snd_1239_);
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 5, v_diag_1249_);
lean_ctor_set(v___x_1246_, 3, v_usedTheorems_1248_);
v___x_1251_ = v___x_1246_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1269_; 
v_reuseFailAlloc_1269_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1269_, 0, v_cache_1241_);
lean_ctor_set(v_reuseFailAlloc_1269_, 1, v_congrCache_1242_);
lean_ctor_set(v_reuseFailAlloc_1269_, 2, v_dsimpCache_1243_);
lean_ctor_set(v_reuseFailAlloc_1269_, 3, v_usedTheorems_1248_);
lean_ctor_set(v_reuseFailAlloc_1269_, 4, v_numSteps_1244_);
lean_ctor_set(v_reuseFailAlloc_1269_, 5, v_diag_1249_);
v___x_1251_ = v_reuseFailAlloc_1269_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
lean_object* v___x_1252_; lean_object* v_expr_1253_; lean_object* v___x_1254_; uint8_t v___x_1255_; 
v___x_1252_ = lean_st_ref_set(v_a_918_, v___x_1251_);
v_expr_1253_ = lean_ctor_get(v_fst_1238_, 0);
v___x_1254_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29));
v___x_1255_ = l_Lean_Expr_isConstOf(v_expr_1253_, v___x_1254_);
if (v___x_1255_ == 0)
{
lean_dec(v_fst_1238_);
lean_inc(v___y_1195_);
v___y_1165_ = v___y_1194_;
v___y_1166_ = v___y_1196_;
v_a_1167_ = v___y_1195_;
goto v___jp_1164_;
}
else
{
lean_object* v___x_1256_; 
v___x_1256_ = l_Lean_Meta_Simp_Result_getProof(v_fst_1238_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1256_) == 0)
{
lean_object* v_a_1257_; lean_object* v___x_1258_; 
v_a_1257_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_a_1257_);
lean_dec_ref_known(v___x_1256_, 1);
v___x_1258_ = l_Lean_Meta_mkOfEqTrue(v_a_1257_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; lean_object* v___x_1261_; uint8_t v_isShared_1262_; uint8_t v_isSharedCheck_1266_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1266_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1266_ == 0)
{
v___x_1261_ = v___x_1258_;
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
else
{
lean_inc(v_a_1259_);
lean_dec(v___x_1258_);
v___x_1261_ = lean_box(0);
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
v_resetjp_1260_:
{
lean_object* v___x_1264_; 
if (v_isShared_1262_ == 0)
{
lean_ctor_set_tag(v___x_1261_, 1);
v___x_1264_ = v___x_1261_;
goto v_reusejp_1263_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v_a_1259_);
v___x_1264_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1263_;
}
v_reusejp_1263_:
{
v___y_1165_ = v___y_1194_;
v___y_1166_ = v___y_1196_;
v_a_1167_ = v___x_1264_;
goto v___jp_1164_;
}
}
}
else
{
lean_object* v_a_1267_; 
v_a_1267_ = lean_ctor_get(v___x_1258_, 0);
lean_inc(v_a_1267_);
lean_dec_ref_known(v___x_1258_, 1);
v___y_1187_ = v___y_1194_;
v___y_1188_ = v___y_1195_;
v___y_1189_ = v___y_1196_;
v_a_1190_ = v_a_1267_;
goto v___jp_1186_;
}
}
else
{
lean_object* v_a_1268_; 
v_a_1268_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_a_1268_);
lean_dec_ref_known(v___x_1256_, 1);
v___y_1187_ = v___y_1194_;
v___y_1188_ = v___y_1195_;
v___y_1189_ = v___y_1196_;
v_a_1190_ = v_a_1268_;
goto v___jp_1186_;
}
}
}
}
}
else
{
lean_object* v_a_1273_; 
v_a_1273_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1273_);
lean_dec_ref_known(v___x_1236_, 1);
v___y_1170_ = v___y_1194_;
v___y_1171_ = v___y_1196_;
v_a_1172_ = v_a_1273_;
goto v___jp_1169_;
}
}
}
}
else
{
lean_object* v_a_1276_; 
lean_dec(v___x_1216_);
lean_dec(v_a_1215_);
lean_dec_ref(v_prop_915_);
v_a_1276_ = lean_ctor_get(v___x_1217_, 0);
lean_inc(v_a_1276_);
lean_dec_ref_known(v___x_1217_, 1);
v___y_1170_ = v___y_1194_;
v___y_1171_ = v___y_1196_;
v_a_1172_ = v_a_1276_;
goto v___jp_1169_;
}
}
else
{
lean_object* v_a_1277_; 
lean_dec_ref(v_prop_915_);
v_a_1277_ = lean_ctor_get(v___x_1214_, 0);
lean_inc(v_a_1277_);
lean_dec_ref_known(v___x_1214_, 1);
v___y_1170_ = v___y_1194_;
v___y_1171_ = v___y_1196_;
v_a_1172_ = v_a_1277_;
goto v___jp_1169_;
}
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1170_ = v___y_1194_;
v___y_1171_ = v___y_1196_;
v_a_1172_ = v___y_1197_;
goto v___jp_1169_;
}
}
v___jp_1278_:
{
lean_object* v___x_1282_; 
lean_inc_ref(v_prop_915_);
v___x_1282_ = lp_mathlib_Mathlib_Meta_Positivity_solve(v_prop_915_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1282_) == 0)
{
lean_object* v_a_1283_; lean_object* v___x_1284_; 
lean_dec_ref(v_prop_915_);
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref_known(v___x_1282_, 1);
v___x_1284_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(v_a_1283_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
v___y_1175_ = v___y_1279_;
v___y_1176_ = v___y_1281_;
v___y_1177_ = v___x_1284_;
goto v___jp_1174_;
}
else
{
lean_object* v_a_1285_; uint8_t v___x_1286_; 
v_a_1285_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1285_);
lean_dec_ref_known(v___x_1282_, 1);
v___x_1286_ = l_Lean_Exception_isInterrupt(v_a_1285_);
if (v___x_1286_ == 0)
{
uint8_t v___x_1287_; 
lean_inc(v_a_1285_);
v___x_1287_ = l_Lean_Exception_isRuntime(v_a_1285_);
v___y_1194_ = v___y_1279_;
v___y_1195_ = v___y_1280_;
v___y_1196_ = v___y_1281_;
v___y_1197_ = v_a_1285_;
v___y_1198_ = v___x_1287_;
goto v___jp_1193_;
}
else
{
v___y_1194_ = v___y_1279_;
v___y_1195_ = v___y_1280_;
v___y_1196_ = v___y_1281_;
v___y_1197_ = v_a_1285_;
v___y_1198_ = v___x_1286_;
goto v___jp_1193_;
}
}
}
v___jp_1288_:
{
if (v___y_1293_ == 0)
{
lean_dec_ref(v___y_1289_);
v___y_1279_ = v___y_1290_;
v___y_1280_ = v___y_1291_;
v___y_1281_ = v___y_1292_;
goto v___jp_1278_;
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1170_ = v___y_1290_;
v___y_1171_ = v___y_1292_;
v_a_1172_ = v___y_1289_;
goto v___jp_1169_;
}
}
v___jp_1294_:
{
lean_object* v___x_1298_; double v___x_1299_; double v___x_1300_; double v___x_1301_; double v___x_1302_; double v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; 
v___x_1298_ = lean_io_mono_nanos_now();
v___x_1299_ = lean_float_of_nat(v___y_1295_);
v___x_1300_ = lean_float_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__38);
v___x_1301_ = lean_float_div(v___x_1299_, v___x_1300_);
v___x_1302_ = lean_float_of_nat(v___x_1298_);
v___x_1303_ = lean_float_div(v___x_1302_, v___x_1300_);
v___x_1304_ = lean_box_float(v___x_1301_);
v___x_1305_ = lean_box_float(v___x_1303_);
v___x_1306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1306_, 0, v___x_1304_);
lean_ctor_set(v___x_1306_, 1, v___x_1305_);
v___x_1307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1307_, 0, v_a_1297_);
lean_ctor_set(v___x_1307_, 1, v___x_1306_);
v___x_1308_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_FieldSimp_discharge_spec__5(v___x_1147_, v_hasTrace_1092_, v___x_1149_, v_options_1090_, v___x_1151_, v___y_1296_, v___x_1148_, v___x_1307_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
return v___x_1308_;
}
v___jp_1309_:
{
lean_object* v___x_1313_; 
v___x_1313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1313_, 0, v_a_1312_);
v___y_1295_ = v___y_1310_;
v___y_1296_ = v___y_1311_;
v_a_1297_ = v___x_1313_;
goto v___jp_1294_;
}
v___jp_1314_:
{
lean_object* v___x_1318_; 
v___x_1318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1318_, 0, v_a_1317_);
v___y_1295_ = v___y_1315_;
v___y_1296_ = v___y_1316_;
v_a_1297_ = v___x_1318_;
goto v___jp_1294_;
}
v___jp_1319_:
{
if (lean_obj_tag(v___y_1322_) == 0)
{
lean_object* v_a_1323_; 
v_a_1323_ = lean_ctor_get(v___y_1322_, 0);
lean_inc(v_a_1323_);
lean_dec_ref_known(v___y_1322_, 1);
v___y_1315_ = v___y_1320_;
v___y_1316_ = v___y_1321_;
v_a_1317_ = v_a_1323_;
goto v___jp_1314_;
}
else
{
lean_object* v_a_1324_; 
v_a_1324_ = lean_ctor_get(v___y_1322_, 0);
lean_inc(v_a_1324_);
lean_dec_ref_known(v___y_1322_, 1);
v___y_1310_ = v___y_1320_;
v___y_1311_ = v___y_1321_;
v_a_1312_ = v_a_1324_;
goto v___jp_1309_;
}
}
v___jp_1325_:
{
if (v___y_1330_ == 0)
{
lean_dec_ref(v___y_1326_);
lean_inc(v___y_1329_);
v___y_1315_ = v___y_1327_;
v___y_1316_ = v___y_1328_;
v_a_1317_ = v___y_1329_;
goto v___jp_1314_;
}
else
{
v___y_1310_ = v___y_1327_;
v___y_1311_ = v___y_1328_;
v_a_1312_ = v___y_1326_;
goto v___jp_1309_;
}
}
v___jp_1331_:
{
uint8_t v___x_1336_; 
v___x_1336_ = l_Lean_Exception_isInterrupt(v_a_1335_);
if (v___x_1336_ == 0)
{
uint8_t v___x_1337_; 
lean_inc_ref(v_a_1335_);
v___x_1337_ = l_Lean_Exception_isRuntime(v_a_1335_);
v___y_1326_ = v_a_1335_;
v___y_1327_ = v___y_1332_;
v___y_1328_ = v___y_1334_;
v___y_1329_ = v___y_1333_;
v___y_1330_ = v___x_1337_;
goto v___jp_1325_;
}
else
{
v___y_1326_ = v_a_1335_;
v___y_1327_ = v___y_1332_;
v___y_1328_ = v___y_1334_;
v___y_1329_ = v___y_1333_;
v___y_1330_ = v___x_1336_;
goto v___jp_1325_;
}
}
v___jp_1338_:
{
if (v___y_1344_ == 0)
{
lean_object* v_config_1345_; lean_object* v_userConfig_1346_; lean_object* v_zetaDeltaSet_1347_; lean_object* v_initUsedZetaDelta_1348_; lean_object* v_metaConfig_1349_; lean_object* v_indexConfig_1350_; uint32_t v_maxDischargeDepth_1351_; lean_object* v_simpTheorems_1352_; lean_object* v_congrTheorems_1353_; lean_object* v_parent_x3f_1354_; uint32_t v_dischargeDepth_1355_; lean_object* v_lctxInitIndices_1356_; uint8_t v_inDSimp_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; 
lean_dec_ref(v___y_1343_);
v_config_1345_ = lean_ctor_get(v_a_917_, 0);
v_userConfig_1346_ = lean_ctor_get(v_a_917_, 1);
v_zetaDeltaSet_1347_ = lean_ctor_get(v_a_917_, 2);
v_initUsedZetaDelta_1348_ = lean_ctor_get(v_a_917_, 3);
v_metaConfig_1349_ = lean_ctor_get(v_a_917_, 4);
v_indexConfig_1350_ = lean_ctor_get(v_a_917_, 5);
v_maxDischargeDepth_1351_ = lean_ctor_get_uint32(v_a_917_, sizeof(void*)*10);
v_simpTheorems_1352_ = lean_ctor_get(v_a_917_, 6);
v_congrTheorems_1353_ = lean_ctor_get(v_a_917_, 7);
v_parent_x3f_1354_ = lean_ctor_get(v_a_917_, 8);
v_dischargeDepth_1355_ = lean_ctor_get_uint32(v_a_917_, sizeof(void*)*10 + 4);
v_lctxInitIndices_1356_ = lean_ctor_get(v_a_917_, 9);
v_inDSimp_1357_ = lean_ctor_get_uint8(v_a_917_, sizeof(void*)*10 + 8);
v___x_1358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21));
v___x_1359_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27);
v___x_1360_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg(v___y_1340_, v___x_1359_, v___x_1358_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_object* v_a_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; 
v_a_1361_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_a_1361_);
lean_dec_ref_known(v___x_1360_, 1);
v___x_1362_ = lean_st_ref_get(v_a_918_);
v___x_1363_ = l_Lean_Meta_Simp_getSimprocs___redArg(v_a_922_);
if (lean_obj_tag(v___x_1363_) == 0)
{
lean_object* v_a_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1421_; 
v_a_1364_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1366_ = v___x_1363_;
v_isShared_1367_ = v_isSharedCheck_1421_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_a_1364_);
lean_dec(v___x_1363_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1421_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
lean_object* v_usedTheorems_1368_; lean_object* v_diag_1369_; uint32_t v___x_1370_; uint32_t v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1381_; 
v_usedTheorems_1368_ = lean_ctor_get(v___x_1362_, 3);
lean_inc_ref(v_usedTheorems_1368_);
v_diag_1369_ = lean_ctor_get(v___x_1362_, 5);
lean_inc_ref(v_diag_1369_);
lean_dec(v___x_1362_);
v___x_1370_ = 1;
v___x_1371_ = lean_uint32_add(v_dischargeDepth_1355_, v___x_1370_);
lean_inc(v_lctxInitIndices_1356_);
lean_inc(v_parent_x3f_1354_);
lean_inc_ref(v_congrTheorems_1353_);
lean_inc_ref_n(v_simpTheorems_1352_, 2);
lean_inc_ref(v_indexConfig_1350_);
lean_inc_ref(v_metaConfig_1349_);
lean_inc(v_initUsedZetaDelta_1348_);
lean_inc(v_zetaDeltaSet_1347_);
lean_inc_ref(v_userConfig_1346_);
lean_inc_ref(v_config_1345_);
v___x_1372_ = lean_alloc_ctor(0, 10, 9);
lean_ctor_set(v___x_1372_, 0, v_config_1345_);
lean_ctor_set(v___x_1372_, 1, v_userConfig_1346_);
lean_ctor_set(v___x_1372_, 2, v_zetaDeltaSet_1347_);
lean_ctor_set(v___x_1372_, 3, v_initUsedZetaDelta_1348_);
lean_ctor_set(v___x_1372_, 4, v_metaConfig_1349_);
lean_ctor_set(v___x_1372_, 5, v_indexConfig_1350_);
lean_ctor_set(v___x_1372_, 6, v_simpTheorems_1352_);
lean_ctor_set(v___x_1372_, 7, v_congrTheorems_1353_);
lean_ctor_set(v___x_1372_, 8, v_parent_x3f_1354_);
lean_ctor_set(v___x_1372_, 9, v_lctxInitIndices_1356_);
lean_ctor_set_uint32(v___x_1372_, sizeof(void*)*10, v_maxDischargeDepth_1351_);
lean_ctor_set_uint32(v___x_1372_, sizeof(void*)*10 + 4, v___x_1371_);
lean_ctor_set_uint8(v___x_1372_, sizeof(void*)*10 + 8, v_inDSimp_1357_);
v___x_1373_ = lean_array_push(v_simpTheorems_1352_, v_a_1361_);
v___x_1374_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v___x_1372_, v___x_1373_);
v___x_1375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1375_, 0, v_usedTheorems_1368_);
lean_ctor_set(v___x_1375_, 1, v_diag_1369_);
v___x_1376_ = lean_unsigned_to_nat(1u);
v___x_1377_ = lean_mk_empty_array_with_capacity(v___x_1376_);
v___x_1378_ = lean_array_push(v___x_1377_, v_a_1364_);
v___x_1379_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed), 9, 0);
if (v_isShared_1367_ == 0)
{
lean_ctor_set_tag(v___x_1366_, 1);
lean_ctor_set(v___x_1366_, 0, v___x_1379_);
v___x_1381_ = v___x_1366_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v___x_1379_);
v___x_1381_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
lean_object* v___x_1382_; 
v___x_1382_ = l_Lean_Meta_simp(v_prop_915_, v___x_1374_, v___x_1378_, v___x_1381_, v___x_1375_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
lean_dec_ref_known(v___x_1375_, 2);
if (lean_obj_tag(v___x_1382_) == 0)
{
lean_object* v_a_1383_; lean_object* v_fst_1384_; lean_object* v_snd_1385_; lean_object* v___x_1386_; lean_object* v_cache_1387_; lean_object* v_congrCache_1388_; lean_object* v_dsimpCache_1389_; lean_object* v_numSteps_1390_; lean_object* v___x_1392_; uint8_t v_isShared_1393_; uint8_t v_isSharedCheck_1416_; 
v_a_1383_ = lean_ctor_get(v___x_1382_, 0);
lean_inc(v_a_1383_);
lean_dec_ref_known(v___x_1382_, 1);
v_fst_1384_ = lean_ctor_get(v_a_1383_, 0);
lean_inc(v_fst_1384_);
v_snd_1385_ = lean_ctor_get(v_a_1383_, 1);
lean_inc(v_snd_1385_);
lean_dec(v_a_1383_);
v___x_1386_ = lean_st_ref_get(v_a_918_);
v_cache_1387_ = lean_ctor_get(v___x_1386_, 0);
v_congrCache_1388_ = lean_ctor_get(v___x_1386_, 1);
v_dsimpCache_1389_ = lean_ctor_get(v___x_1386_, 2);
v_numSteps_1390_ = lean_ctor_get(v___x_1386_, 4);
v_isSharedCheck_1416_ = !lean_is_exclusive(v___x_1386_);
if (v_isSharedCheck_1416_ == 0)
{
lean_object* v_unused_1417_; lean_object* v_unused_1418_; 
v_unused_1417_ = lean_ctor_get(v___x_1386_, 5);
lean_dec(v_unused_1417_);
v_unused_1418_ = lean_ctor_get(v___x_1386_, 3);
lean_dec(v_unused_1418_);
v___x_1392_ = v___x_1386_;
v_isShared_1393_ = v_isSharedCheck_1416_;
goto v_resetjp_1391_;
}
else
{
lean_inc(v_numSteps_1390_);
lean_inc(v_dsimpCache_1389_);
lean_inc(v_congrCache_1388_);
lean_inc(v_cache_1387_);
lean_dec(v___x_1386_);
v___x_1392_ = lean_box(0);
v_isShared_1393_ = v_isSharedCheck_1416_;
goto v_resetjp_1391_;
}
v_resetjp_1391_:
{
lean_object* v_usedTheorems_1394_; lean_object* v_diag_1395_; lean_object* v___x_1397_; 
v_usedTheorems_1394_ = lean_ctor_get(v_snd_1385_, 0);
lean_inc_ref(v_usedTheorems_1394_);
v_diag_1395_ = lean_ctor_get(v_snd_1385_, 1);
lean_inc_ref(v_diag_1395_);
lean_dec(v_snd_1385_);
if (v_isShared_1393_ == 0)
{
lean_ctor_set(v___x_1392_, 5, v_diag_1395_);
lean_ctor_set(v___x_1392_, 3, v_usedTheorems_1394_);
v___x_1397_ = v___x_1392_;
goto v_reusejp_1396_;
}
else
{
lean_object* v_reuseFailAlloc_1415_; 
v_reuseFailAlloc_1415_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1415_, 0, v_cache_1387_);
lean_ctor_set(v_reuseFailAlloc_1415_, 1, v_congrCache_1388_);
lean_ctor_set(v_reuseFailAlloc_1415_, 2, v_dsimpCache_1389_);
lean_ctor_set(v_reuseFailAlloc_1415_, 3, v_usedTheorems_1394_);
lean_ctor_set(v_reuseFailAlloc_1415_, 4, v_numSteps_1390_);
lean_ctor_set(v_reuseFailAlloc_1415_, 5, v_diag_1395_);
v___x_1397_ = v_reuseFailAlloc_1415_;
goto v_reusejp_1396_;
}
v_reusejp_1396_:
{
lean_object* v___x_1398_; lean_object* v_expr_1399_; lean_object* v___x_1400_; uint8_t v___x_1401_; 
v___x_1398_ = lean_st_ref_set(v_a_918_, v___x_1397_);
v_expr_1399_ = lean_ctor_get(v_fst_1384_, 0);
v___x_1400_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29));
v___x_1401_ = l_Lean_Expr_isConstOf(v_expr_1399_, v___x_1400_);
if (v___x_1401_ == 0)
{
lean_dec(v_fst_1384_);
lean_inc(v___y_1342_);
v___y_1315_ = v___y_1339_;
v___y_1316_ = v___y_1341_;
v_a_1317_ = v___y_1342_;
goto v___jp_1314_;
}
else
{
lean_object* v___x_1402_; 
v___x_1402_ = l_Lean_Meta_Simp_Result_getProof(v_fst_1384_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1402_) == 0)
{
lean_object* v_a_1403_; lean_object* v___x_1404_; 
v_a_1403_ = lean_ctor_get(v___x_1402_, 0);
lean_inc(v_a_1403_);
lean_dec_ref_known(v___x_1402_, 1);
v___x_1404_ = l_Lean_Meta_mkOfEqTrue(v_a_1403_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
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
v___y_1315_ = v___y_1339_;
v___y_1316_ = v___y_1341_;
v_a_1317_ = v___x_1410_;
goto v___jp_1314_;
}
}
}
else
{
lean_object* v_a_1413_; 
v_a_1413_ = lean_ctor_get(v___x_1404_, 0);
lean_inc(v_a_1413_);
lean_dec_ref_known(v___x_1404_, 1);
v___y_1332_ = v___y_1339_;
v___y_1333_ = v___y_1342_;
v___y_1334_ = v___y_1341_;
v_a_1335_ = v_a_1413_;
goto v___jp_1331_;
}
}
else
{
lean_object* v_a_1414_; 
v_a_1414_ = lean_ctor_get(v___x_1402_, 0);
lean_inc(v_a_1414_);
lean_dec_ref_known(v___x_1402_, 1);
v___y_1332_ = v___y_1339_;
v___y_1333_ = v___y_1342_;
v___y_1334_ = v___y_1341_;
v_a_1335_ = v_a_1414_;
goto v___jp_1331_;
}
}
}
}
}
else
{
lean_object* v_a_1419_; 
v_a_1419_ = lean_ctor_get(v___x_1382_, 0);
lean_inc(v_a_1419_);
lean_dec_ref_known(v___x_1382_, 1);
v___y_1310_ = v___y_1339_;
v___y_1311_ = v___y_1341_;
v_a_1312_ = v_a_1419_;
goto v___jp_1309_;
}
}
}
}
else
{
lean_object* v_a_1422_; 
lean_dec(v___x_1362_);
lean_dec(v_a_1361_);
lean_dec_ref(v_prop_915_);
v_a_1422_ = lean_ctor_get(v___x_1363_, 0);
lean_inc(v_a_1422_);
lean_dec_ref_known(v___x_1363_, 1);
v___y_1310_ = v___y_1339_;
v___y_1311_ = v___y_1341_;
v_a_1312_ = v_a_1422_;
goto v___jp_1309_;
}
}
else
{
lean_object* v_a_1423_; 
lean_dec_ref(v_prop_915_);
v_a_1423_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_a_1423_);
lean_dec_ref_known(v___x_1360_, 1);
v___y_1310_ = v___y_1339_;
v___y_1311_ = v___y_1341_;
v_a_1312_ = v_a_1423_;
goto v___jp_1309_;
}
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1310_ = v___y_1339_;
v___y_1311_ = v___y_1341_;
v_a_1312_ = v___y_1343_;
goto v___jp_1309_;
}
}
v___jp_1424_:
{
lean_object* v___x_1429_; 
lean_inc_ref(v_prop_915_);
v___x_1429_ = lp_mathlib_Mathlib_Meta_Positivity_solve(v_prop_915_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1429_) == 0)
{
lean_object* v_a_1430_; lean_object* v___x_1431_; 
lean_dec_ref(v_prop_915_);
v_a_1430_ = lean_ctor_get(v___x_1429_, 0);
lean_inc(v_a_1430_);
lean_dec_ref_known(v___x_1429_, 1);
v___x_1431_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(v_a_1430_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
v___y_1320_ = v___y_1425_;
v___y_1321_ = v___y_1428_;
v___y_1322_ = v___x_1431_;
goto v___jp_1319_;
}
else
{
lean_object* v_a_1432_; uint8_t v___x_1433_; 
v_a_1432_ = lean_ctor_get(v___x_1429_, 0);
lean_inc(v_a_1432_);
lean_dec_ref_known(v___x_1429_, 1);
v___x_1433_ = l_Lean_Exception_isInterrupt(v_a_1432_);
if (v___x_1433_ == 0)
{
uint8_t v___x_1434_; 
lean_inc(v_a_1432_);
v___x_1434_ = l_Lean_Exception_isRuntime(v_a_1432_);
v___y_1339_ = v___y_1425_;
v___y_1340_ = v___y_1426_;
v___y_1341_ = v___y_1428_;
v___y_1342_ = v___y_1427_;
v___y_1343_ = v_a_1432_;
v___y_1344_ = v___x_1434_;
goto v___jp_1338_;
}
else
{
v___y_1339_ = v___y_1425_;
v___y_1340_ = v___y_1426_;
v___y_1341_ = v___y_1428_;
v___y_1342_ = v___y_1427_;
v___y_1343_ = v_a_1432_;
v___y_1344_ = v___x_1433_;
goto v___jp_1338_;
}
}
}
v___jp_1435_:
{
if (v___y_1441_ == 0)
{
lean_dec_ref(v___y_1440_);
v___y_1425_ = v___y_1436_;
v___y_1426_ = v___y_1437_;
v___y_1427_ = v___y_1439_;
v___y_1428_ = v___y_1438_;
goto v___jp_1424_;
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1310_ = v___y_1436_;
v___y_1311_ = v___y_1438_;
v_a_1312_ = v___y_1440_;
goto v___jp_1309_;
}
}
v___jp_1442_:
{
lean_object* v___x_1443_; 
v___x_1443_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__2___redArg(v_a_922_);
if (lean_obj_tag(v___x_1443_) == 0)
{
lean_object* v_a_1444_; lean_object* v___x_1445_; uint8_t v___x_1446_; 
v_a_1444_ = lean_ctor_get(v___x_1443_, 0);
lean_inc(v_a_1444_);
lean_dec_ref_known(v___x_1443_, 1);
v___x_1445_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1446_ = l_Lean_Option_get___at___00Lean_Meta_Simp_discharge_x3f_x27_spec__3(v_options_1090_, v___x_1445_);
if (v___x_1446_ == 0)
{
lean_object* v___x_1447_; lean_object* v___x_1448_; 
v___x_1447_ = lean_io_mono_nanos_now();
lean_inc_ref(v_prop_915_);
v___x_1448_ = l___private_Lean_Meta_Tactic_Simp_Rewrite_0__Lean_Meta_Simp_dischargeUsingAssumption_x3f(v_prop_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1448_) == 0)
{
lean_object* v_a_1449_; 
v_a_1449_ = lean_ctor_get(v___x_1448_, 0);
lean_inc(v_a_1449_);
lean_dec_ref_known(v___x_1448_, 1);
if (lean_obj_tag(v_a_1449_) == 1)
{
lean_object* v_val_1450_; lean_object* v___x_1451_; 
lean_dec_ref(v_prop_915_);
v_val_1450_ = lean_ctor_get(v_a_1449_, 0);
lean_inc(v_val_1450_);
lean_dec_ref_known(v_a_1449_, 1);
v___x_1451_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(v_val_1450_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
v___y_1320_ = v___x_1447_;
v___y_1321_ = v_a_1444_;
v___y_1322_ = v___x_1451_;
goto v___jp_1319_;
}
else
{
lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___f_1454_; lean_object* v___x_1455_; 
lean_dec(v_a_1449_);
v___x_1452_ = lean_box(v___x_1446_);
v___x_1453_ = lean_box(v_hasTrace_1092_);
lean_inc_ref(v_prop_915_);
v___f_1454_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__3___boxed), 8, 3);
lean_closure_set(v___f_1454_, 0, v_prop_915_);
lean_closure_set(v___f_1454_, 1, v___x_1452_);
lean_closure_set(v___f_1454_, 2, v___x_1453_);
v___x_1455_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(v___f_1454_, v___x_1446_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1455_) == 0)
{
lean_object* v_a_1456_; lean_object* v_snd_1457_; lean_object* v_snd_1458_; lean_object* v_snd_1459_; lean_object* v_snd_1460_; lean_object* v___x_1461_; uint8_t v___x_1462_; 
v_a_1456_ = lean_ctor_get(v___x_1455_, 0);
lean_inc(v_a_1456_);
lean_dec_ref_known(v___x_1455_, 1);
v_snd_1457_ = lean_ctor_get(v_a_1456_, 1);
lean_inc(v_snd_1457_);
lean_dec(v_a_1456_);
v_snd_1458_ = lean_ctor_get(v_snd_1457_, 1);
lean_inc(v_snd_1458_);
lean_dec(v_snd_1457_);
v_snd_1459_ = lean_ctor_get(v_snd_1458_, 1);
lean_inc(v_snd_1459_);
lean_dec(v_snd_1458_);
v_snd_1460_ = lean_ctor_get(v_snd_1459_, 1);
lean_inc(v_snd_1460_);
lean_dec(v_snd_1459_);
v___x_1461_ = lean_box(0);
v___x_1462_ = lean_unbox(v_snd_1460_);
lean_dec(v_snd_1460_);
if (v___x_1462_ == 0)
{
v___y_1425_ = v___x_1447_;
v___y_1426_ = v___x_1446_;
v___y_1427_ = v___x_1461_;
v___y_1428_ = v_a_1444_;
goto v___jp_1424_;
}
else
{
lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; 
v___x_1463_ = lean_box(0);
v___x_1464_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30);
lean_inc_ref(v_prop_915_);
v___x_1465_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1463_, v___x_1464_, v_prop_915_, v___x_1446_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1465_) == 0)
{
lean_object* v_a_1466_; lean_object* v___x_1468_; uint8_t v_isShared_1469_; uint8_t v_isSharedCheck_1475_; 
v_a_1466_ = lean_ctor_get(v___x_1465_, 0);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1465_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1468_ = v___x_1465_;
v_isShared_1469_ = v_isSharedCheck_1475_;
goto v_resetjp_1467_;
}
else
{
lean_inc(v_a_1466_);
lean_dec(v___x_1465_);
v___x_1468_ = lean_box(0);
v_isShared_1469_ = v_isSharedCheck_1475_;
goto v_resetjp_1467_;
}
v_resetjp_1467_:
{
if (lean_obj_tag(v_a_1466_) == 0)
{
uint8_t v_val_1470_; 
v_val_1470_ = lean_ctor_get_uint8(v_a_1466_, sizeof(void*)*1);
if (v_val_1470_ == 1)
{
lean_object* v_proof_1471_; lean_object* v___x_1473_; 
lean_dec_ref(v_prop_915_);
v_proof_1471_ = lean_ctor_get(v_a_1466_, 0);
lean_inc_ref(v_proof_1471_);
lean_dec_ref_known(v_a_1466_, 1);
if (v_isShared_1469_ == 0)
{
lean_ctor_set_tag(v___x_1468_, 1);
lean_ctor_set(v___x_1468_, 0, v_proof_1471_);
v___x_1473_ = v___x_1468_;
goto v_reusejp_1472_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v_proof_1471_);
v___x_1473_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1472_;
}
v_reusejp_1472_:
{
v___y_1315_ = v___x_1447_;
v___y_1316_ = v_a_1444_;
v_a_1317_ = v___x_1473_;
goto v___jp_1314_;
}
}
else
{
lean_dec_ref_known(v_a_1466_, 1);
lean_del_object(v___x_1468_);
v___y_1425_ = v___x_1447_;
v___y_1426_ = v___x_1446_;
v___y_1427_ = v___x_1461_;
v___y_1428_ = v_a_1444_;
goto v___jp_1424_;
}
}
else
{
lean_del_object(v___x_1468_);
lean_dec(v_a_1466_);
v___y_1425_ = v___x_1447_;
v___y_1426_ = v___x_1446_;
v___y_1427_ = v___x_1461_;
v___y_1428_ = v_a_1444_;
goto v___jp_1424_;
}
}
}
else
{
lean_object* v_a_1476_; uint8_t v___x_1477_; 
v_a_1476_ = lean_ctor_get(v___x_1465_, 0);
lean_inc(v_a_1476_);
lean_dec_ref_known(v___x_1465_, 1);
v___x_1477_ = l_Lean_Exception_isInterrupt(v_a_1476_);
if (v___x_1477_ == 0)
{
uint8_t v___x_1478_; 
lean_inc(v_a_1476_);
v___x_1478_ = l_Lean_Exception_isRuntime(v_a_1476_);
v___y_1436_ = v___x_1447_;
v___y_1437_ = v___x_1446_;
v___y_1438_ = v_a_1444_;
v___y_1439_ = v___x_1461_;
v___y_1440_ = v_a_1476_;
v___y_1441_ = v___x_1478_;
goto v___jp_1435_;
}
else
{
v___y_1436_ = v___x_1447_;
v___y_1437_ = v___x_1446_;
v___y_1438_ = v_a_1444_;
v___y_1439_ = v___x_1461_;
v___y_1440_ = v_a_1476_;
v___y_1441_ = v___x_1477_;
goto v___jp_1435_;
}
}
}
}
else
{
lean_object* v_a_1479_; 
lean_dec_ref(v_prop_915_);
v_a_1479_ = lean_ctor_get(v___x_1455_, 0);
lean_inc(v_a_1479_);
lean_dec_ref_known(v___x_1455_, 1);
v___y_1310_ = v___x_1447_;
v___y_1311_ = v_a_1444_;
v_a_1312_ = v_a_1479_;
goto v___jp_1309_;
}
}
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1320_ = v___x_1447_;
v___y_1321_ = v_a_1444_;
v___y_1322_ = v___x_1448_;
goto v___jp_1319_;
}
}
else
{
lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1480_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_prop_915_);
v___x_1481_ = l___private_Lean_Meta_Tactic_Simp_Rewrite_0__Lean_Meta_Simp_dischargeUsingAssumption_x3f(v_prop_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1481_) == 0)
{
lean_object* v_a_1482_; 
v_a_1482_ = lean_ctor_get(v___x_1481_, 0);
lean_inc(v_a_1482_);
lean_dec_ref_known(v___x_1481_, 1);
if (lean_obj_tag(v_a_1482_) == 1)
{
lean_object* v_val_1483_; lean_object* v___x_1484_; 
lean_dec_ref(v_prop_915_);
v_val_1483_ = lean_ctor_get(v_a_1482_, 0);
lean_inc(v_val_1483_);
lean_dec_ref_known(v_a_1482_, 1);
v___x_1484_ = lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__1(v_val_1483_, v_a_916_, v_a_917_, v_a_918_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
v___y_1175_ = v_a_1444_;
v___y_1176_ = v___x_1480_;
v___y_1177_ = v___x_1484_;
goto v___jp_1174_;
}
else
{
lean_object* v___x_1485_; lean_object* v___f_1486_; uint8_t v___x_1487_; lean_object* v___x_1488_; 
lean_dec(v_a_1482_);
v___x_1485_ = lean_box(v___x_1446_);
lean_inc_ref(v_prop_915_);
v___f_1486_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___lam__2___boxed), 7, 2);
lean_closure_set(v___f_1486_, 0, v_prop_915_);
lean_closure_set(v___f_1486_, 1, v___x_1485_);
v___x_1487_ = 0;
v___x_1488_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(v___f_1486_, v___x_1487_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1488_) == 0)
{
lean_object* v_a_1489_; lean_object* v_snd_1490_; lean_object* v_snd_1491_; lean_object* v_snd_1492_; lean_object* v_snd_1493_; lean_object* v___x_1494_; uint8_t v___x_1495_; 
v_a_1489_ = lean_ctor_get(v___x_1488_, 0);
lean_inc(v_a_1489_);
lean_dec_ref_known(v___x_1488_, 1);
v_snd_1490_ = lean_ctor_get(v_a_1489_, 1);
lean_inc(v_snd_1490_);
lean_dec(v_a_1489_);
v_snd_1491_ = lean_ctor_get(v_snd_1490_, 1);
lean_inc(v_snd_1491_);
lean_dec(v_snd_1490_);
v_snd_1492_ = lean_ctor_get(v_snd_1491_, 1);
lean_inc(v_snd_1492_);
lean_dec(v_snd_1491_);
v_snd_1493_ = lean_ctor_get(v_snd_1492_, 1);
lean_inc(v_snd_1493_);
lean_dec(v_snd_1492_);
v___x_1494_ = lean_box(0);
v___x_1495_ = lean_unbox(v_snd_1493_);
lean_dec(v_snd_1493_);
if (v___x_1495_ == 0)
{
v___y_1279_ = v_a_1444_;
v___y_1280_ = v___x_1494_;
v___y_1281_ = v___x_1480_;
goto v___jp_1278_;
}
else
{
lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___x_1496_ = lean_box(0);
v___x_1497_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30);
lean_inc_ref(v_prop_915_);
v___x_1498_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1496_, v___x_1497_, v_prop_915_, v___x_1487_, v_a_919_, v_a_920_, v_a_921_, v_a_922_);
if (lean_obj_tag(v___x_1498_) == 0)
{
lean_object* v_a_1499_; lean_object* v___x_1501_; uint8_t v_isShared_1502_; uint8_t v_isSharedCheck_1508_; 
v_a_1499_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1508_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1501_ = v___x_1498_;
v_isShared_1502_ = v_isSharedCheck_1508_;
goto v_resetjp_1500_;
}
else
{
lean_inc(v_a_1499_);
lean_dec(v___x_1498_);
v___x_1501_ = lean_box(0);
v_isShared_1502_ = v_isSharedCheck_1508_;
goto v_resetjp_1500_;
}
v_resetjp_1500_:
{
if (lean_obj_tag(v_a_1499_) == 0)
{
uint8_t v_val_1503_; 
v_val_1503_ = lean_ctor_get_uint8(v_a_1499_, sizeof(void*)*1);
if (v_val_1503_ == 1)
{
lean_object* v_proof_1504_; lean_object* v___x_1506_; 
lean_dec_ref(v_prop_915_);
v_proof_1504_ = lean_ctor_get(v_a_1499_, 0);
lean_inc_ref(v_proof_1504_);
lean_dec_ref_known(v_a_1499_, 1);
if (v_isShared_1502_ == 0)
{
lean_ctor_set_tag(v___x_1501_, 1);
lean_ctor_set(v___x_1501_, 0, v_proof_1504_);
v___x_1506_ = v___x_1501_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v_proof_1504_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
v___y_1165_ = v_a_1444_;
v___y_1166_ = v___x_1480_;
v_a_1167_ = v___x_1506_;
goto v___jp_1164_;
}
}
else
{
lean_dec_ref_known(v_a_1499_, 1);
lean_del_object(v___x_1501_);
v___y_1279_ = v_a_1444_;
v___y_1280_ = v___x_1494_;
v___y_1281_ = v___x_1480_;
goto v___jp_1278_;
}
}
else
{
lean_del_object(v___x_1501_);
lean_dec(v_a_1499_);
v___y_1279_ = v_a_1444_;
v___y_1280_ = v___x_1494_;
v___y_1281_ = v___x_1480_;
goto v___jp_1278_;
}
}
}
else
{
lean_object* v_a_1509_; uint8_t v___x_1510_; 
v_a_1509_ = lean_ctor_get(v___x_1498_, 0);
lean_inc(v_a_1509_);
lean_dec_ref_known(v___x_1498_, 1);
v___x_1510_ = l_Lean_Exception_isInterrupt(v_a_1509_);
if (v___x_1510_ == 0)
{
uint8_t v___x_1511_; 
lean_inc(v_a_1509_);
v___x_1511_ = l_Lean_Exception_isRuntime(v_a_1509_);
v___y_1289_ = v_a_1509_;
v___y_1290_ = v_a_1444_;
v___y_1291_ = v___x_1494_;
v___y_1292_ = v___x_1480_;
v___y_1293_ = v___x_1511_;
goto v___jp_1288_;
}
else
{
v___y_1289_ = v_a_1509_;
v___y_1290_ = v_a_1444_;
v___y_1291_ = v___x_1494_;
v___y_1292_ = v___x_1480_;
v___y_1293_ = v___x_1510_;
goto v___jp_1288_;
}
}
}
}
else
{
lean_object* v_a_1512_; 
lean_dec_ref(v_prop_915_);
v_a_1512_ = lean_ctor_get(v___x_1488_, 0);
lean_inc(v_a_1512_);
lean_dec_ref_known(v___x_1488_, 1);
v___y_1170_ = v_a_1444_;
v___y_1171_ = v___x_1480_;
v_a_1172_ = v_a_1512_;
goto v___jp_1169_;
}
}
}
else
{
lean_dec_ref(v_prop_915_);
v___y_1175_ = v_a_1444_;
v___y_1176_ = v___x_1480_;
v___y_1177_ = v___x_1481_;
goto v___jp_1174_;
}
}
}
else
{
lean_object* v_a_1513_; lean_object* v___x_1515_; uint8_t v_isShared_1516_; uint8_t v_isSharedCheck_1520_; 
lean_dec_ref(v___x_1148_);
lean_dec_ref(v_prop_915_);
v_a_1513_ = lean_ctor_get(v___x_1443_, 0);
v_isSharedCheck_1520_ = !lean_is_exclusive(v___x_1443_);
if (v_isSharedCheck_1520_ == 0)
{
v___x_1515_ = v___x_1443_;
v_isShared_1516_ = v_isSharedCheck_1520_;
goto v_resetjp_1514_;
}
else
{
lean_inc(v_a_1513_);
lean_dec(v___x_1443_);
v___x_1515_ = lean_box(0);
v_isShared_1516_ = v_isSharedCheck_1520_;
goto v_resetjp_1514_;
}
v_resetjp_1514_:
{
lean_object* v___x_1518_; 
if (v_isShared_1516_ == 0)
{
v___x_1518_ = v___x_1515_;
goto v_reusejp_1517_;
}
else
{
lean_object* v_reuseFailAlloc_1519_; 
v_reuseFailAlloc_1519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1519_, 0, v_a_1513_);
v___x_1518_ = v_reuseFailAlloc_1519_;
goto v_reusejp_1517_;
}
v_reusejp_1517_:
{
return v___x_1518_;
}
}
}
}
}
v___jp_924_:
{
lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_926_, 0, v_r_925_);
v___x_927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_927_, 0, v___x_926_);
return v___x_927_;
}
v___jp_928_:
{
if (v___y_931_ == 0)
{
lean_object* v___x_932_; 
lean_dec_ref(v___y_930_);
lean_inc(v___y_929_);
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v___y_929_);
return v___x_932_;
}
else
{
lean_object* v___x_933_; 
v___x_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_933_, 0, v___y_930_);
return v___x_933_;
}
}
v___jp_934_:
{
uint8_t v___x_937_; 
v___x_937_ = l_Lean_Exception_isInterrupt(v_a_936_);
if (v___x_937_ == 0)
{
uint8_t v___x_938_; 
lean_inc_ref(v_a_936_);
v___x_938_ = l_Lean_Exception_isRuntime(v_a_936_);
v___y_929_ = v___y_935_;
v___y_930_ = v_a_936_;
v___y_931_ = v___x_938_;
goto v___jp_928_;
}
else
{
v___y_929_ = v___y_935_;
v___y_930_ = v_a_936_;
v___y_931_ = v___x_937_;
goto v___jp_928_;
}
}
v___jp_939_:
{
if (v___y_949_ == 0)
{
lean_object* v_config_950_; lean_object* v_userConfig_951_; lean_object* v_zetaDeltaSet_952_; lean_object* v_initUsedZetaDelta_953_; lean_object* v_metaConfig_954_; lean_object* v_indexConfig_955_; uint32_t v_maxDischargeDepth_956_; lean_object* v_simpTheorems_957_; lean_object* v_congrTheorems_958_; lean_object* v_parent_x3f_959_; uint32_t v_dischargeDepth_960_; lean_object* v_lctxInitIndices_961_; uint8_t v_inDSimp_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
lean_dec_ref(v___y_946_);
v_config_950_ = lean_ctor_get(v___y_943_, 0);
v_userConfig_951_ = lean_ctor_get(v___y_943_, 1);
v_zetaDeltaSet_952_ = lean_ctor_get(v___y_943_, 2);
v_initUsedZetaDelta_953_ = lean_ctor_get(v___y_943_, 3);
v_metaConfig_954_ = lean_ctor_get(v___y_943_, 4);
v_indexConfig_955_ = lean_ctor_get(v___y_943_, 5);
v_maxDischargeDepth_956_ = lean_ctor_get_uint32(v___y_943_, sizeof(void*)*10);
v_simpTheorems_957_ = lean_ctor_get(v___y_943_, 6);
v_congrTheorems_958_ = lean_ctor_get(v___y_943_, 7);
v_parent_x3f_959_ = lean_ctor_get(v___y_943_, 8);
v_dischargeDepth_960_ = lean_ctor_get_uint32(v___y_943_, sizeof(void*)*10 + 4);
v_lctxInitIndices_961_ = lean_ctor_get(v___y_943_, 9);
v_inDSimp_962_ = lean_ctor_get_uint8(v___y_943_, sizeof(void*)*10 + 8);
v___x_963_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__21));
v___x_964_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__27);
v___x_965_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(v___x_964_, v___x_963_, v___y_944_, v___y_942_, v___y_941_, v___y_945_);
if (lean_obj_tag(v___x_965_) == 0)
{
lean_object* v_a_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
v_a_966_ = lean_ctor_get(v___x_965_, 0);
lean_inc(v_a_966_);
lean_dec_ref_known(v___x_965_, 1);
v___x_967_ = lean_st_ref_get(v___y_948_);
v___x_968_ = l_Lean_Meta_Simp_getSimprocs___redArg(v___y_945_);
if (lean_obj_tag(v___x_968_) == 0)
{
lean_object* v_a_969_; lean_object* v_usedTheorems_970_; lean_object* v_diag_971_; uint32_t v___x_972_; uint32_t v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v_a_969_ = lean_ctor_get(v___x_968_, 0);
lean_inc(v_a_969_);
lean_dec_ref_known(v___x_968_, 1);
v_usedTheorems_970_ = lean_ctor_get(v___x_967_, 3);
lean_inc_ref(v_usedTheorems_970_);
v_diag_971_ = lean_ctor_get(v___x_967_, 5);
lean_inc_ref(v_diag_971_);
lean_dec(v___x_967_);
v___x_972_ = 1;
v___x_973_ = lean_uint32_add(v_dischargeDepth_960_, v___x_972_);
lean_inc(v_lctxInitIndices_961_);
lean_inc(v_parent_x3f_959_);
lean_inc_ref(v_congrTheorems_958_);
lean_inc_ref_n(v_simpTheorems_957_, 2);
lean_inc_ref(v_indexConfig_955_);
lean_inc_ref(v_metaConfig_954_);
lean_inc(v_initUsedZetaDelta_953_);
lean_inc(v_zetaDeltaSet_952_);
lean_inc_ref(v_userConfig_951_);
lean_inc_ref(v_config_950_);
v___x_974_ = lean_alloc_ctor(0, 10, 9);
lean_ctor_set(v___x_974_, 0, v_config_950_);
lean_ctor_set(v___x_974_, 1, v_userConfig_951_);
lean_ctor_set(v___x_974_, 2, v_zetaDeltaSet_952_);
lean_ctor_set(v___x_974_, 3, v_initUsedZetaDelta_953_);
lean_ctor_set(v___x_974_, 4, v_metaConfig_954_);
lean_ctor_set(v___x_974_, 5, v_indexConfig_955_);
lean_ctor_set(v___x_974_, 6, v_simpTheorems_957_);
lean_ctor_set(v___x_974_, 7, v_congrTheorems_958_);
lean_ctor_set(v___x_974_, 8, v_parent_x3f_959_);
lean_ctor_set(v___x_974_, 9, v_lctxInitIndices_961_);
lean_ctor_set_uint32(v___x_974_, sizeof(void*)*10, v_maxDischargeDepth_956_);
lean_ctor_set_uint32(v___x_974_, sizeof(void*)*10 + 4, v___x_973_);
lean_ctor_set_uint8(v___x_974_, sizeof(void*)*10 + 8, v_inDSimp_962_);
v___x_975_ = lean_array_push(v_simpTheorems_957_, v_a_966_);
v___x_976_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v___x_974_, v___x_975_);
v___x_977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_977_, 0, v_usedTheorems_970_);
lean_ctor_set(v___x_977_, 1, v_diag_971_);
v___x_978_ = lean_unsigned_to_nat(1u);
v___x_979_ = lean_mk_empty_array_with_capacity(v___x_978_);
v___x_980_ = lean_array_push(v___x_979_, v_a_969_);
v___x_981_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___boxed), 9, 0);
v___x_982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_982_, 0, v___x_981_);
v___x_983_ = l_Lean_Meta_simp(v_prop_915_, v___x_976_, v___x_980_, v___x_982_, v___x_977_, v___y_944_, v___y_942_, v___y_941_, v___y_945_);
lean_dec_ref_known(v___x_977_, 2);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_1033_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_986_ = v___x_983_;
v_isShared_987_ = v_isSharedCheck_1033_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_a_984_);
lean_dec(v___x_983_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_1033_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v_fst_988_; lean_object* v_snd_989_; lean_object* v___x_990_; lean_object* v_cache_991_; lean_object* v_congrCache_992_; lean_object* v_dsimpCache_993_; lean_object* v_numSteps_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1030_; 
v_fst_988_ = lean_ctor_get(v_a_984_, 0);
lean_inc(v_fst_988_);
v_snd_989_ = lean_ctor_get(v_a_984_, 1);
lean_inc(v_snd_989_);
lean_dec(v_a_984_);
v___x_990_ = lean_st_ref_get(v___y_948_);
v_cache_991_ = lean_ctor_get(v___x_990_, 0);
v_congrCache_992_ = lean_ctor_get(v___x_990_, 1);
v_dsimpCache_993_ = lean_ctor_get(v___x_990_, 2);
v_numSteps_994_ = lean_ctor_get(v___x_990_, 4);
v_isSharedCheck_1030_ = !lean_is_exclusive(v___x_990_);
if (v_isSharedCheck_1030_ == 0)
{
lean_object* v_unused_1031_; lean_object* v_unused_1032_; 
v_unused_1031_ = lean_ctor_get(v___x_990_, 5);
lean_dec(v_unused_1031_);
v_unused_1032_ = lean_ctor_get(v___x_990_, 3);
lean_dec(v_unused_1032_);
v___x_996_ = v___x_990_;
v_isShared_997_ = v_isSharedCheck_1030_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_numSteps_994_);
lean_inc(v_dsimpCache_993_);
lean_inc(v_congrCache_992_);
lean_inc(v_cache_991_);
lean_dec(v___x_990_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1030_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v_usedTheorems_998_; lean_object* v_diag_999_; lean_object* v___x_1001_; 
v_usedTheorems_998_ = lean_ctor_get(v_snd_989_, 0);
lean_inc_ref(v_usedTheorems_998_);
v_diag_999_ = lean_ctor_get(v_snd_989_, 1);
lean_inc_ref(v_diag_999_);
lean_dec(v_snd_989_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 5, v_diag_999_);
lean_ctor_set(v___x_996_, 3, v_usedTheorems_998_);
v___x_1001_ = v___x_996_;
goto v_reusejp_1000_;
}
else
{
lean_object* v_reuseFailAlloc_1029_; 
v_reuseFailAlloc_1029_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1029_, 0, v_cache_991_);
lean_ctor_set(v_reuseFailAlloc_1029_, 1, v_congrCache_992_);
lean_ctor_set(v_reuseFailAlloc_1029_, 2, v_dsimpCache_993_);
lean_ctor_set(v_reuseFailAlloc_1029_, 3, v_usedTheorems_998_);
lean_ctor_set(v_reuseFailAlloc_1029_, 4, v_numSteps_994_);
lean_ctor_set(v_reuseFailAlloc_1029_, 5, v_diag_999_);
v___x_1001_ = v_reuseFailAlloc_1029_;
goto v_reusejp_1000_;
}
v_reusejp_1000_:
{
lean_object* v___x_1002_; lean_object* v_expr_1003_; lean_object* v___x_1004_; uint8_t v___x_1005_; 
v___x_1002_ = lean_st_ref_set(v___y_948_, v___x_1001_);
v_expr_1003_ = lean_ctor_get(v_fst_988_, 0);
v___x_1004_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__29));
v___x_1005_ = l_Lean_Expr_isConstOf(v_expr_1003_, v___x_1004_);
if (v___x_1005_ == 0)
{
lean_object* v___x_1007_; 
lean_dec(v_fst_988_);
lean_inc(v___y_940_);
if (v_isShared_987_ == 0)
{
lean_ctor_set(v___x_986_, 0, v___y_940_);
v___x_1007_ = v___x_986_;
goto v_reusejp_1006_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v___y_940_);
v___x_1007_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1006_;
}
v_reusejp_1006_:
{
return v___x_1007_;
}
}
else
{
lean_object* v___x_1009_; 
lean_del_object(v___x_986_);
v___x_1009_ = l_Lean_Meta_Simp_Result_getProof(v_fst_988_, v___y_944_, v___y_942_, v___y_941_, v___y_945_);
if (lean_obj_tag(v___x_1009_) == 0)
{
lean_object* v_a_1010_; lean_object* v___x_1012_; uint8_t v_isShared_1013_; uint8_t v_isSharedCheck_1027_; 
v_a_1010_ = lean_ctor_get(v___x_1009_, 0);
v_isSharedCheck_1027_ = !lean_is_exclusive(v___x_1009_);
if (v_isSharedCheck_1027_ == 0)
{
v___x_1012_ = v___x_1009_;
v_isShared_1013_ = v_isSharedCheck_1027_;
goto v_resetjp_1011_;
}
else
{
lean_inc(v_a_1010_);
lean_dec(v___x_1009_);
v___x_1012_ = lean_box(0);
v_isShared_1013_ = v_isSharedCheck_1027_;
goto v_resetjp_1011_;
}
v_resetjp_1011_:
{
lean_object* v___x_1014_; 
v___x_1014_ = l_Lean_Meta_mkOfEqTrue(v_a_1010_, v___y_944_, v___y_942_, v___y_941_, v___y_945_);
if (lean_obj_tag(v___x_1014_) == 0)
{
lean_object* v_a_1015_; lean_object* v___x_1017_; uint8_t v_isShared_1018_; uint8_t v_isSharedCheck_1025_; 
v_a_1015_ = lean_ctor_get(v___x_1014_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_1014_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1017_ = v___x_1014_;
v_isShared_1018_ = v_isSharedCheck_1025_;
goto v_resetjp_1016_;
}
else
{
lean_inc(v_a_1015_);
lean_dec(v___x_1014_);
v___x_1017_ = lean_box(0);
v_isShared_1018_ = v_isSharedCheck_1025_;
goto v_resetjp_1016_;
}
v_resetjp_1016_:
{
lean_object* v___x_1020_; 
if (v_isShared_1013_ == 0)
{
lean_ctor_set_tag(v___x_1012_, 1);
lean_ctor_set(v___x_1012_, 0, v_a_1015_);
v___x_1020_ = v___x_1012_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v_a_1015_);
v___x_1020_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
lean_object* v___x_1022_; 
if (v_isShared_1018_ == 0)
{
lean_ctor_set(v___x_1017_, 0, v___x_1020_);
v___x_1022_ = v___x_1017_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_1020_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
return v___x_1022_;
}
}
}
}
else
{
lean_object* v_a_1026_; 
lean_del_object(v___x_1012_);
v_a_1026_ = lean_ctor_get(v___x_1014_, 0);
lean_inc(v_a_1026_);
lean_dec_ref_known(v___x_1014_, 1);
v___y_935_ = v___y_940_;
v_a_936_ = v_a_1026_;
goto v___jp_934_;
}
}
}
else
{
lean_object* v_a_1028_; 
v_a_1028_ = lean_ctor_get(v___x_1009_, 0);
lean_inc(v_a_1028_);
lean_dec_ref_known(v___x_1009_, 1);
v___y_935_ = v___y_940_;
v_a_936_ = v_a_1028_;
goto v___jp_934_;
}
}
}
}
}
}
else
{
lean_object* v_a_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1041_; 
v_a_1034_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1036_ = v___x_983_;
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_a_1034_);
lean_dec(v___x_983_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v___x_1039_; 
if (v_isShared_1037_ == 0)
{
v___x_1039_ = v___x_1036_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v_a_1034_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
return v___x_1039_;
}
}
}
}
else
{
lean_object* v_a_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1049_; 
lean_dec(v___x_967_);
lean_dec(v_a_966_);
lean_dec_ref(v_prop_915_);
v_a_1042_ = lean_ctor_get(v___x_968_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_968_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1044_ = v___x_968_;
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_a_1042_);
lean_dec(v___x_968_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v___x_1047_; 
if (v_isShared_1045_ == 0)
{
v___x_1047_ = v___x_1044_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v_a_1042_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
}
else
{
lean_object* v_a_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1057_; 
lean_dec_ref(v_prop_915_);
v_a_1050_ = lean_ctor_get(v___x_965_, 0);
v_isSharedCheck_1057_ = !lean_is_exclusive(v___x_965_);
if (v_isSharedCheck_1057_ == 0)
{
v___x_1052_ = v___x_965_;
v_isShared_1053_ = v_isSharedCheck_1057_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_a_1050_);
lean_dec(v___x_965_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1057_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1055_; 
if (v_isShared_1053_ == 0)
{
v___x_1055_ = v___x_1052_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1056_; 
v_reuseFailAlloc_1056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1056_, 0, v_a_1050_);
v___x_1055_ = v_reuseFailAlloc_1056_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
return v___x_1055_;
}
}
}
}
else
{
lean_dec_ref(v_prop_915_);
return v___y_946_;
}
}
v___jp_1058_:
{
lean_object* v___x_1067_; 
lean_inc_ref(v_prop_915_);
v___x_1067_ = lp_mathlib_Mathlib_Meta_Positivity_solve(v_prop_915_, v___y_1063_, v___y_1061_, v___y_1060_, v___y_1064_);
if (lean_obj_tag(v___x_1067_) == 0)
{
lean_object* v_a_1068_; 
lean_dec_ref(v_prop_915_);
v_a_1068_ = lean_ctor_get(v___x_1067_, 0);
lean_inc(v_a_1068_);
lean_dec_ref_known(v___x_1067_, 1);
v_r_925_ = v_a_1068_;
goto v___jp_924_;
}
else
{
lean_object* v_a_1069_; lean_object* v___x_1071_; uint8_t v_isShared_1072_; uint8_t v_isSharedCheck_1078_; 
v_a_1069_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1071_ = v___x_1067_;
v_isShared_1072_ = v_isSharedCheck_1078_;
goto v_resetjp_1070_;
}
else
{
lean_inc(v_a_1069_);
lean_dec(v___x_1067_);
v___x_1071_ = lean_box(0);
v_isShared_1072_ = v_isSharedCheck_1078_;
goto v_resetjp_1070_;
}
v_resetjp_1070_:
{
lean_object* v___x_1074_; 
lean_inc(v_a_1069_);
if (v_isShared_1072_ == 0)
{
v___x_1074_ = v___x_1071_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1069_);
v___x_1074_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
uint8_t v___x_1075_; 
v___x_1075_ = l_Lean_Exception_isInterrupt(v_a_1069_);
if (v___x_1075_ == 0)
{
uint8_t v___x_1076_; 
v___x_1076_ = l_Lean_Exception_isRuntime(v_a_1069_);
v___y_940_ = v___y_1059_;
v___y_941_ = v___y_1060_;
v___y_942_ = v___y_1061_;
v___y_943_ = v___y_1062_;
v___y_944_ = v___y_1063_;
v___y_945_ = v___y_1064_;
v___y_946_ = v___x_1074_;
v___y_947_ = v___y_1065_;
v___y_948_ = v___y_1066_;
v___y_949_ = v___x_1076_;
goto v___jp_939_;
}
else
{
lean_dec(v_a_1069_);
v___y_940_ = v___y_1059_;
v___y_941_ = v___y_1060_;
v___y_942_ = v___y_1061_;
v___y_943_ = v___y_1062_;
v___y_944_ = v___y_1063_;
v___y_945_ = v___y_1064_;
v___y_946_ = v___x_1074_;
v___y_947_ = v___y_1065_;
v___y_948_ = v___y_1066_;
v___y_949_ = v___x_1075_;
goto v___jp_939_;
}
}
}
}
}
v___jp_1079_:
{
if (v___y_1089_ == 0)
{
lean_dec_ref(v___y_1085_);
v___y_1059_ = v___y_1082_;
v___y_1060_ = v___y_1081_;
v___y_1061_ = v___y_1080_;
v___y_1062_ = v___y_1083_;
v___y_1063_ = v___y_1084_;
v___y_1064_ = v___y_1086_;
v___y_1065_ = v___y_1087_;
v___y_1066_ = v___y_1088_;
goto v___jp_1058_;
}
else
{
lean_dec_ref(v_prop_915_);
return v___y_1085_;
}
}
v___jp_1094_:
{
if (lean_obj_tag(v_____do__lift_1095_) == 1)
{
lean_object* v_val_1103_; 
lean_dec_ref(v___f_1093_);
lean_dec_ref(v_prop_915_);
v_val_1103_ = lean_ctor_get(v_____do__lift_1095_, 0);
lean_inc(v_val_1103_);
lean_dec_ref_known(v_____do__lift_1095_, 1);
v_r_925_ = v_val_1103_;
goto v___jp_924_;
}
else
{
uint8_t v___x_1104_; lean_object* v___x_1105_; 
lean_dec(v_____do__lift_1095_);
v___x_1104_ = 0;
v___x_1105_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_FieldSimp_discharge_spec__1___redArg(v___f_1093_, v___x_1104_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_);
if (lean_obj_tag(v___x_1105_) == 0)
{
lean_object* v_a_1106_; lean_object* v_snd_1107_; lean_object* v_snd_1108_; lean_object* v_snd_1109_; lean_object* v_snd_1110_; lean_object* v___x_1111_; uint8_t v___x_1112_; 
v_a_1106_ = lean_ctor_get(v___x_1105_, 0);
lean_inc(v_a_1106_);
lean_dec_ref_known(v___x_1105_, 1);
v_snd_1107_ = lean_ctor_get(v_a_1106_, 1);
lean_inc(v_snd_1107_);
lean_dec(v_a_1106_);
v_snd_1108_ = lean_ctor_get(v_snd_1107_, 1);
lean_inc(v_snd_1108_);
lean_dec(v_snd_1107_);
v_snd_1109_ = lean_ctor_get(v_snd_1108_, 1);
lean_inc(v_snd_1109_);
lean_dec(v_snd_1108_);
v_snd_1110_ = lean_ctor_get(v_snd_1109_, 1);
lean_inc(v_snd_1110_);
lean_dec(v_snd_1109_);
v___x_1111_ = lean_box(0);
v___x_1112_ = lean_unbox(v_snd_1110_);
lean_dec(v_snd_1110_);
if (v___x_1112_ == 0)
{
v___y_1059_ = v___x_1111_;
v___y_1060_ = v___y_1101_;
v___y_1061_ = v___y_1100_;
v___y_1062_ = v___y_1097_;
v___y_1063_ = v___y_1099_;
v___y_1064_ = v___y_1102_;
v___y_1065_ = v___y_1096_;
v___y_1066_ = v___y_1098_;
goto v___jp_1058_;
}
else
{
lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; 
v___x_1113_ = lean_box(0);
v___x_1114_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30, &lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_FieldSimp_discharge___closed__30);
lean_inc_ref(v_prop_915_);
v___x_1115_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1113_, v___x_1114_, v_prop_915_, v___x_1104_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_);
if (lean_obj_tag(v___x_1115_) == 0)
{
lean_object* v_a_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1126_; 
v_a_1116_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1118_ = v___x_1115_;
v_isShared_1119_ = v_isSharedCheck_1126_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_a_1116_);
lean_dec(v___x_1115_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1126_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
if (lean_obj_tag(v_a_1116_) == 0)
{
uint8_t v_val_1120_; 
v_val_1120_ = lean_ctor_get_uint8(v_a_1116_, sizeof(void*)*1);
if (v_val_1120_ == 1)
{
lean_object* v_proof_1121_; lean_object* v___x_1122_; lean_object* v___x_1124_; 
lean_dec_ref(v_prop_915_);
v_proof_1121_ = lean_ctor_get(v_a_1116_, 0);
lean_inc_ref(v_proof_1121_);
lean_dec_ref_known(v_a_1116_, 1);
v___x_1122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1122_, 0, v_proof_1121_);
if (v_isShared_1119_ == 0)
{
lean_ctor_set(v___x_1118_, 0, v___x_1122_);
v___x_1124_ = v___x_1118_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v___x_1122_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
else
{
lean_dec_ref_known(v_a_1116_, 1);
lean_del_object(v___x_1118_);
v___y_1059_ = v___x_1111_;
v___y_1060_ = v___y_1101_;
v___y_1061_ = v___y_1100_;
v___y_1062_ = v___y_1097_;
v___y_1063_ = v___y_1099_;
v___y_1064_ = v___y_1102_;
v___y_1065_ = v___y_1096_;
v___y_1066_ = v___y_1098_;
goto v___jp_1058_;
}
}
else
{
lean_del_object(v___x_1118_);
lean_dec(v_a_1116_);
v___y_1059_ = v___x_1111_;
v___y_1060_ = v___y_1101_;
v___y_1061_ = v___y_1100_;
v___y_1062_ = v___y_1097_;
v___y_1063_ = v___y_1099_;
v___y_1064_ = v___y_1102_;
v___y_1065_ = v___y_1096_;
v___y_1066_ = v___y_1098_;
goto v___jp_1058_;
}
}
}
else
{
lean_object* v_a_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1136_; 
v_a_1127_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1136_ == 0)
{
v___x_1129_ = v___x_1115_;
v_isShared_1130_ = v_isSharedCheck_1136_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_a_1127_);
lean_dec(v___x_1115_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1136_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1132_; 
lean_inc(v_a_1127_);
if (v_isShared_1130_ == 0)
{
v___x_1132_ = v___x_1129_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v_a_1127_);
v___x_1132_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
uint8_t v___x_1133_; 
v___x_1133_ = l_Lean_Exception_isInterrupt(v_a_1127_);
if (v___x_1133_ == 0)
{
uint8_t v___x_1134_; 
v___x_1134_ = l_Lean_Exception_isRuntime(v_a_1127_);
v___y_1080_ = v___y_1100_;
v___y_1081_ = v___y_1101_;
v___y_1082_ = v___x_1111_;
v___y_1083_ = v___y_1097_;
v___y_1084_ = v___y_1099_;
v___y_1085_ = v___x_1132_;
v___y_1086_ = v___y_1102_;
v___y_1087_ = v___y_1096_;
v___y_1088_ = v___y_1098_;
v___y_1089_ = v___x_1134_;
goto v___jp_1079_;
}
else
{
lean_dec(v_a_1127_);
v___y_1080_ = v___y_1100_;
v___y_1081_ = v___y_1101_;
v___y_1082_ = v___x_1111_;
v___y_1083_ = v___y_1097_;
v___y_1084_ = v___y_1099_;
v___y_1085_ = v___x_1132_;
v___y_1086_ = v___y_1102_;
v___y_1087_ = v___y_1096_;
v___y_1088_ = v___y_1098_;
v___y_1089_ = v___x_1133_;
goto v___jp_1079_;
}
}
}
}
}
}
else
{
lean_object* v_a_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1144_; 
lean_dec_ref(v_prop_915_);
v_a_1137_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1144_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1144_ == 0)
{
v___x_1139_ = v___x_1105_;
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_a_1137_);
lean_dec(v___x_1105_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1142_; 
if (v_isShared_1140_ == 0)
{
v___x_1142_ = v___x_1139_;
goto v_reusejp_1141_;
}
else
{
lean_object* v_reuseFailAlloc_1143_; 
v_reuseFailAlloc_1143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1143_, 0, v_a_1137_);
v___x_1142_ = v_reuseFailAlloc_1143_;
goto v_reusejp_1141_;
}
v_reusejp_1141_:
{
return v___x_1142_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4(lean_object* v_x_1525_, lean_object* v_x_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_){
_start:
{
lean_object* v___x_1535_; 
v___x_1535_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___redArg(v_x_1525_, v_x_1526_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_);
return v___x_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4___boxed(lean_object* v_x_1536_, lean_object* v_x_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_){
_start:
{
lean_object* v_res_1546_; 
v_res_1546_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__4(v_x_1536_, v_x_1537_, v___y_1538_, v___y_1539_, v___y_1540_, v___y_1541_, v___y_1542_, v___y_1543_, v___y_1544_);
lean_dec(v___y_1544_);
lean_dec_ref(v___y_1543_);
lean_dec(v___y_1542_);
lean_dec_ref(v___y_1541_);
lean_dec(v___y_1540_);
lean_dec_ref(v___y_1539_);
lean_dec(v___y_1538_);
return v_res_1546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6(uint8_t v___x_1547_, lean_object* v_x_1548_, lean_object* v_x_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v___x_1558_; 
v___x_1558_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___redArg(v___x_1547_, v_x_1548_, v_x_1549_, v___y_1553_, v___y_1554_, v___y_1555_, v___y_1556_);
return v___x_1558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6___boxed(lean_object* v___x_1559_, lean_object* v_x_1560_, lean_object* v_x_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_){
_start:
{
uint8_t v___x_119844__boxed_1570_; lean_object* v_res_1571_; 
v___x_119844__boxed_1570_ = lean_unbox(v___x_1559_);
v_res_1571_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_FieldSimp_discharge_spec__6(v___x_119844__boxed_1570_, v_x_1560_, v_x_1561_, v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_);
lean_dec(v___y_1568_);
lean_dec_ref(v___y_1567_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
lean_dec(v___y_1564_);
lean_dec_ref(v___y_1563_);
lean_dec(v___y_1562_);
return v_res_1571_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v___x_1589_ = lean_box(0);
v___x_1590_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1591_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1591_, 0, v___x_1590_);
lean_ctor_set(v___x_1591_, 1, v___x_1589_);
return v___x_1591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1593_; lean_object* v___x_1594_; 
v___x_1593_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___closed__0);
v___x_1594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1593_);
return v___x_1594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg___boxed(lean_object* v___y_1595_){
_start:
{
lean_object* v_res_1596_; 
v_res_1596_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg();
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0(lean_object* v_00_u03b1_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_){
_start:
{
lean_object* v___x_1607_; 
v___x_1607_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg();
return v___x_1607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___boxed(lean_object* v_00_u03b1_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_){
_start:
{
lean_object* v_res_1618_; 
v_res_1618_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0(v_00_u03b1_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_, v___y_1615_, v___y_1616_);
lean_dec(v___y_1616_);
lean_dec_ref(v___y_1615_);
lean_dec(v___y_1614_);
lean_dec_ref(v___y_1613_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
return v_res_1618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1(lean_object* v_x_1620_, lean_object* v_a_1621_, lean_object* v_a_1622_, lean_object* v_a_1623_, lean_object* v_a_1624_, lean_object* v_a_1625_, lean_object* v_a_1626_, lean_object* v_a_1627_, lean_object* v_a_1628_){
_start:
{
lean_object* v___x_1630_; uint8_t v___x_1631_; 
v___x_1630_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp_tacticField__simp__discharge___closed__3));
v___x_1631_ = l_Lean_Syntax_isOfKind(v_x_1620_, v___x_1630_);
if (v___x_1631_ == 0)
{
lean_object* v___x_1632_; 
v___x_1632_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1_spec__0___redArg();
return v___x_1632_;
}
else
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___closed__0));
v___x_1634_ = lp_mathlib_wrapSimpDischarger(v___x_1633_, v_a_1621_, v_a_1622_, v_a_1623_, v_a_1624_, v_a_1625_, v_a_1626_, v_a_1627_, v_a_1628_);
return v___x_1634_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1___boxed(lean_object* v_x_1635_, lean_object* v_a_1636_, lean_object* v_a_1637_, lean_object* v_a_1638_, lean_object* v_a_1639_, lean_object* v_a_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_, lean_object* v_a_1643_, lean_object* v_a_1644_){
_start:
{
lean_object* v_res_1645_; 
v_res_1645_ = lp_mathlib_Mathlib_Tactic_FieldSimp___aux__Mathlib__Tactic__FieldSimp__Discharger______elabRules__Mathlib__Tactic__FieldSimp__tacticField__simp__discharge__1(v_x_1635_, v_a_1636_, v_a_1637_, v_a_1638_, v_a_1639_, v_a_1640_, v_a_1641_, v_a_1642_, v_a_1643_);
lean_dec(v_a_1643_);
lean_dec_ref(v_a_1642_);
lean_dec(v_a_1641_);
lean_dec_ref(v_a_1640_);
lean_dec(v_a_1639_);
lean_dec_ref(v_a_1638_);
lean_dec(v_a_1637_);
lean_dec_ref(v_a_1636_);
return v_res_1645_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Rewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_DischargerAsTactic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_DischargerAsTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Simp_Rewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Positivity_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_DischargerAsTactic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Positivity_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_DischargerAsTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FieldSimp_Discharger(builtin);
}
#ifdef __cplusplus
}
#endif
