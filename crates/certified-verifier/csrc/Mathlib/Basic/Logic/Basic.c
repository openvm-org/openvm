// Lean compiler output
// Module: Mathlib.Basic.Logic.Basic
// Imports: public import Init public meta import Init public import Mathlib.Lean.Meta.Simp public import Batteries.Logic public import Batteries.Util.LibraryNote public import Mathlib.Tactic.Attr.Register
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_of_nat(lean_object*);
double lean_float_div(double, double);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00eqComm_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_eqComm___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "commuting equality: "};
static const lean_object* lp_mathlib_eqComm___lam__0___closed__0 = (const lean_object*)&lp_mathlib_eqComm___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_eqComm___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_eqComm___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4___boxed(lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0;
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_eqComm___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_eqComm___lam__1___closed__0 = (const lean_object*)&lp_mathlib_eqComm___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_eqComm___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_eqComm___lam__1___closed__1 = (const lean_object*)&lp_mathlib_eqComm___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_eqComm___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_eqComm___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_eqComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_eqComm___closed__0 = (const lean_object*)&lp_mathlib_eqComm___closed__0_value;
static const lean_string_object lp_mathlib_eqComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_eqComm___closed__1 = (const lean_object*)&lp_mathlib_eqComm___closed__1_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_eqComm___closed__2 = (const lean_object*)&lp_mathlib_eqComm___closed__2_value;
static const lean_string_object lp_mathlib_eqComm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "eqComm"};
static const lean_object* lp_mathlib_eqComm___closed__3 = (const lean_object*)&lp_mathlib_eqComm___closed__3_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__3_value),LEAN_SCALAR_PTR_LITERAL(102, 71, 228, 53, 117, 226, 252, 252)}};
static const lean_object* lp_mathlib_eqComm___closed__4 = (const lean_object*)&lp_mathlib_eqComm___closed__4_value;
static const lean_string_object lp_mathlib_eqComm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_comm"};
static const lean_object* lp_mathlib_eqComm___closed__5 = (const lean_object*)&lp_mathlib_eqComm___closed__5_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__5_value),LEAN_SCALAR_PTR_LITERAL(167, 239, 253, 155, 14, 133, 114, 108)}};
static const lean_object* lp_mathlib_eqComm___closed__6 = (const lean_object*)&lp_mathlib_eqComm___closed__6_value;
static const lean_string_object lp_mathlib_eqComm___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib_eqComm___closed__7 = (const lean_object*)&lp_mathlib_eqComm___closed__7_value;
static const lean_string_object lp_mathlib_eqComm___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not_eq_eq_eq_not"};
static const lean_object* lp_mathlib_eqComm___closed__8 = (const lean_object*)&lp_mathlib_eqComm___closed__8_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__7_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__9_value_aux_0),((lean_object*)&lp_mathlib_eqComm___closed__8_value),LEAN_SCALAR_PTR_LITERAL(76, 35, 121, 232, 10, 25, 250, 193)}};
static const lean_object* lp_mathlib_eqComm___closed__9 = (const lean_object*)&lp_mathlib_eqComm___closed__9_value;
static const lean_string_object lp_mathlib_eqComm___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "inv_eq_iff_eq_inv"};
static const lean_object* lp_mathlib_eqComm___closed__10 = (const lean_object*)&lp_mathlib_eqComm___closed__10_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__10_value),LEAN_SCALAR_PTR_LITERAL(136, 88, 249, 140, 166, 92, 138, 227)}};
static const lean_object* lp_mathlib_eqComm___closed__11 = (const lean_object*)&lp_mathlib_eqComm___closed__11_value;
static const lean_string_object lp_mathlib_eqComm___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "eq_inv_mul_iff_mul_eq"};
static const lean_object* lp_mathlib_eqComm___closed__12 = (const lean_object*)&lp_mathlib_eqComm___closed__12_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__12_value),LEAN_SCALAR_PTR_LITERAL(88, 139, 227, 78, 124, 35, 104, 107)}};
static const lean_object* lp_mathlib_eqComm___closed__13 = (const lean_object*)&lp_mathlib_eqComm___closed__13_value;
static const lean_string_object lp_mathlib_eqComm___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "eq_mul_inv_iff_mul_eq"};
static const lean_object* lp_mathlib_eqComm___closed__14 = (const lean_object*)&lp_mathlib_eqComm___closed__14_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__14_value),LEAN_SCALAR_PTR_LITERAL(158, 111, 7, 87, 208, 26, 115, 116)}};
static const lean_object* lp_mathlib_eqComm___closed__15 = (const lean_object*)&lp_mathlib_eqComm___closed__15_value;
static const lean_string_object lp_mathlib_eqComm___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "neg_eq_iff_eq_neg"};
static const lean_object* lp_mathlib_eqComm___closed__16 = (const lean_object*)&lp_mathlib_eqComm___closed__16_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__16_value),LEAN_SCALAR_PTR_LITERAL(114, 239, 58, 3, 142, 29, 163, 82)}};
static const lean_object* lp_mathlib_eqComm___closed__17 = (const lean_object*)&lp_mathlib_eqComm___closed__17_value;
static const lean_string_object lp_mathlib_eqComm___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_eqComm___closed__18 = (const lean_object*)&lp_mathlib_eqComm___closed__18_value;
static const lean_string_object lp_mathlib_eqComm___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Involutive"};
static const lean_object* lp_mathlib_eqComm___closed__19 = (const lean_object*)&lp_mathlib_eqComm___closed__19_value;
static const lean_string_object lp_mathlib_eqComm___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "eq_iff"};
static const lean_object* lp_mathlib_eqComm___closed__20 = (const lean_object*)&lp_mathlib_eqComm___closed__20_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__18_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__21_value_aux_0),((lean_object*)&lp_mathlib_eqComm___closed__19_value),LEAN_SCALAR_PTR_LITERAL(93, 104, 116, 223, 223, 24, 26, 175)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__21_value_aux_1),((lean_object*)&lp_mathlib_eqComm___closed__20_value),LEAN_SCALAR_PTR_LITERAL(80, 49, 111, 244, 11, 58, 169, 152)}};
static const lean_object* lp_mathlib_eqComm___closed__21 = (const lean_object*)&lp_mathlib_eqComm___closed__21_value;
static const lean_string_object lp_mathlib_eqComm___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "vadd_eq_iff_eq_neg_vadd"};
static const lean_object* lp_mathlib_eqComm___closed__22 = (const lean_object*)&lp_mathlib_eqComm___closed__22_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__22_value),LEAN_SCALAR_PTR_LITERAL(114, 210, 127, 166, 106, 159, 182, 67)}};
static const lean_object* lp_mathlib_eqComm___closed__23 = (const lean_object*)&lp_mathlib_eqComm___closed__23_value;
static const lean_string_object lp_mathlib_eqComm___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Equiv"};
static const lean_object* lp_mathlib_eqComm___closed__24 = (const lean_object*)&lp_mathlib_eqComm___closed__24_value;
static const lean_string_object lp_mathlib_eqComm___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "eq_symm_apply"};
static const lean_object* lp_mathlib_eqComm___closed__25 = (const lean_object*)&lp_mathlib_eqComm___closed__25_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__24_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__26_value_aux_0),((lean_object*)&lp_mathlib_eqComm___closed__25_value),LEAN_SCALAR_PTR_LITERAL(251, 21, 201, 48, 79, 70, 63, 249)}};
static const lean_object* lp_mathlib_eqComm___closed__26 = (const lean_object*)&lp_mathlib_eqComm___closed__26_value;
static const lean_string_object lp_mathlib_eqComm___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "beq_iff_eq"};
static const lean_object* lp_mathlib_eqComm___closed__27 = (const lean_object*)&lp_mathlib_eqComm___closed__27_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__27_value),LEAN_SCALAR_PTR_LITERAL(114, 143, 164, 116, 169, 150, 51, 57)}};
static const lean_object* lp_mathlib_eqComm___closed__28 = (const lean_object*)&lp_mathlib_eqComm___closed__28_value;
static const lean_string_object lp_mathlib_eqComm___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "funext_iff"};
static const lean_object* lp_mathlib_eqComm___closed__29 = (const lean_object*)&lp_mathlib_eqComm___closed__29_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__29_value),LEAN_SCALAR_PTR_LITERAL(143, 181, 28, 9, 0, 242, 19, 62)}};
static const lean_object* lp_mathlib_eqComm___closed__30 = (const lean_object*)&lp_mathlib_eqComm___closed__30_value;
static const lean_string_object lp_mathlib_eqComm___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_iff_iff"};
static const lean_object* lp_mathlib_eqComm___closed__31 = (const lean_object*)&lp_mathlib_eqComm___closed__31_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__31_value),LEAN_SCALAR_PTR_LITERAL(31, 202, 150, 50, 184, 133, 187, 239)}};
static const lean_object* lp_mathlib_eqComm___closed__32 = (const lean_object*)&lp_mathlib_eqComm___closed__32_value;
static const lean_string_object lp_mathlib_eqComm___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_mathlib_eqComm___closed__33 = (const lean_object*)&lp_mathlib_eqComm___closed__33_value;
static const lean_string_object lp_mathlib_eqComm___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "swap_eq_iff_eq_swap"};
static const lean_object* lp_mathlib_eqComm___closed__34 = (const lean_object*)&lp_mathlib_eqComm___closed__34_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__33_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__35_value_aux_0),((lean_object*)&lp_mathlib_eqComm___closed__34_value),LEAN_SCALAR_PTR_LITERAL(125, 169, 87, 104, 134, 49, 99, 40)}};
static const lean_object* lp_mathlib_eqComm___closed__35 = (const lean_object*)&lp_mathlib_eqComm___closed__35_value;
static const lean_string_object lp_mathlib_eqComm___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "left_eq_dite_iff"};
static const lean_object* lp_mathlib_eqComm___closed__36 = (const lean_object*)&lp_mathlib_eqComm___closed__36_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__36_value),LEAN_SCALAR_PTR_LITERAL(217, 97, 169, 9, 99, 161, 59, 105)}};
static const lean_object* lp_mathlib_eqComm___closed__37 = (const lean_object*)&lp_mathlib_eqComm___closed__37_value;
static const lean_string_object lp_mathlib_eqComm___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "right_eq_dite_iff"};
static const lean_object* lp_mathlib_eqComm___closed__38 = (const lean_object*)&lp_mathlib_eqComm___closed__38_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__38_value),LEAN_SCALAR_PTR_LITERAL(111, 62, 244, 124, 120, 196, 116, 84)}};
static const lean_object* lp_mathlib_eqComm___closed__39 = (const lean_object*)&lp_mathlib_eqComm___closed__39_value;
static const lean_array_object lp_mathlib_eqComm___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*16, .m_other = 0, .m_tag = 246}, .m_size = 16, .m_capacity = 16, .m_data = {((lean_object*)&lp_mathlib_eqComm___closed__4_value),((lean_object*)&lp_mathlib_eqComm___closed__6_value),((lean_object*)&lp_mathlib_eqComm___closed__9_value),((lean_object*)&lp_mathlib_eqComm___closed__11_value),((lean_object*)&lp_mathlib_eqComm___closed__13_value),((lean_object*)&lp_mathlib_eqComm___closed__15_value),((lean_object*)&lp_mathlib_eqComm___closed__17_value),((lean_object*)&lp_mathlib_eqComm___closed__21_value),((lean_object*)&lp_mathlib_eqComm___closed__23_value),((lean_object*)&lp_mathlib_eqComm___closed__26_value),((lean_object*)&lp_mathlib_eqComm___closed__28_value),((lean_object*)&lp_mathlib_eqComm___closed__30_value),((lean_object*)&lp_mathlib_eqComm___closed__32_value),((lean_object*)&lp_mathlib_eqComm___closed__35_value),((lean_object*)&lp_mathlib_eqComm___closed__37_value),((lean_object*)&lp_mathlib_eqComm___closed__39_value)}};
static const lean_object* lp_mathlib_eqComm___closed__40 = (const lean_object*)&lp_mathlib_eqComm___closed__40_value;
static const lean_string_object lp_mathlib_eqComm___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_eqComm___closed__41 = (const lean_object*)&lp_mathlib_eqComm___closed__41_value;
static const lean_string_object lp_mathlib_eqComm___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_eqComm___closed__42 = (const lean_object*)&lp_mathlib_eqComm___closed__42_value;
static const lean_string_object lp_mathlib_eqComm___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_eqComm___closed__43 = (const lean_object*)&lp_mathlib_eqComm___closed__43_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__41_value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__44_value_aux_0),((lean_object*)&lp_mathlib_eqComm___closed__42_value),LEAN_SCALAR_PTR_LITERAL(194, 95, 140, 15, 16, 100, 236, 219)}};
static const lean_ctor_object lp_mathlib_eqComm___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqComm___closed__44_value_aux_1),((lean_object*)&lp_mathlib_eqComm___closed__43_value),LEAN_SCALAR_PTR_LITERAL(166, 18, 104, 2, 176, 25, 65, 55)}};
static const lean_object* lp_mathlib_eqComm___closed__44 = (const lean_object*)&lp_mathlib_eqComm___closed__44_value;
static const lean_string_object lp_mathlib_eqComm___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_eqComm___closed__45 = (const lean_object*)&lp_mathlib_eqComm___closed__45_value;
static const lean_string_object lp_mathlib_eqComm___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eq_comm_eq"};
static const lean_object* lp_mathlib_eqComm___closed__46 = (const lean_object*)&lp_mathlib_eqComm___closed__46_value;
static const lean_ctor_object lp_mathlib_eqComm___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__46_value),LEAN_SCALAR_PTR_LITERAL(145, 130, 75, 174, 246, 192, 200, 144)}};
static const lean_object* lp_mathlib_eqComm___closed__47 = (const lean_object*)&lp_mathlib_eqComm___closed__47_value;
LEAN_EXPORT lean_object* lp_mathlib_eqComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_eqComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_iffComm___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "commuting iff: "};
static const lean_object* lp_mathlib_iffComm___lam__0___closed__0 = (const lean_object*)&lp_mathlib_iffComm___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_iffComm___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_iffComm___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_iffComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_iffComm___closed__0 = (const lean_object*)&lp_mathlib_iffComm___closed__0_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_iffComm___closed__1 = (const lean_object*)&lp_mathlib_iffComm___closed__1_value;
static lean_once_cell_t lp_mathlib_iffComm___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_iffComm___closed__2;
static const lean_string_object lp_mathlib_iffComm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "iffComm"};
static const lean_object* lp_mathlib_iffComm___closed__3 = (const lean_object*)&lp_mathlib_iffComm___closed__3_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__3_value),LEAN_SCALAR_PTR_LITERAL(180, 210, 42, 24, 172, 190, 199, 249)}};
static const lean_object* lp_mathlib_iffComm___closed__4 = (const lean_object*)&lp_mathlib_iffComm___closed__4_value;
static const lean_string_object lp_mathlib_iffComm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comm"};
static const lean_object* lp_mathlib_iffComm___closed__5 = (const lean_object*)&lp_mathlib_iffComm___closed__5_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_iffComm___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iffComm___closed__6_value_aux_0),((lean_object*)&lp_mathlib_iffComm___closed__5_value),LEAN_SCALAR_PTR_LITERAL(157, 174, 211, 99, 78, 51, 143, 186)}};
static const lean_object* lp_mathlib_iffComm___closed__6 = (const lean_object*)&lp_mathlib_iffComm___closed__6_value;
static const lean_string_object lp_mathlib_iffComm___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "and_congr_left_iff"};
static const lean_object* lp_mathlib_iffComm___closed__7 = (const lean_object*)&lp_mathlib_iffComm___closed__7_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__7_value),LEAN_SCALAR_PTR_LITERAL(19, 71, 111, 200, 241, 177, 72, 164)}};
static const lean_object* lp_mathlib_iffComm___closed__8 = (const lean_object*)&lp_mathlib_iffComm___closed__8_value;
static const lean_string_object lp_mathlib_iffComm___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "and_congr_right_iff"};
static const lean_object* lp_mathlib_iffComm___closed__9 = (const lean_object*)&lp_mathlib_iffComm___closed__9_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__9_value),LEAN_SCALAR_PTR_LITERAL(14, 48, 248, 163, 187, 58, 133, 97)}};
static const lean_object* lp_mathlib_iffComm___closed__10 = (const lean_object*)&lp_mathlib_iffComm___closed__10_value;
static const lean_string_object lp_mathlib_iffComm___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "iff_def"};
static const lean_object* lp_mathlib_iffComm___closed__11 = (const lean_object*)&lp_mathlib_iffComm___closed__11_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__11_value),LEAN_SCALAR_PTR_LITERAL(158, 82, 239, 34, 70, 18, 25, 83)}};
static const lean_object* lp_mathlib_iffComm___closed__12 = (const lean_object*)&lp_mathlib_iffComm___closed__12_value;
static const lean_string_object lp_mathlib_iffComm___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "iff_def'"};
static const lean_object* lp_mathlib_iffComm___closed__13 = (const lean_object*)&lp_mathlib_iffComm___closed__13_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__13_value),LEAN_SCALAR_PTR_LITERAL(158, 55, 67, 26, 43, 186, 27, 174)}};
static const lean_object* lp_mathlib_iffComm___closed__14 = (const lean_object*)&lp_mathlib_iffComm___closed__14_value;
static const lean_string_object lp_mathlib_iffComm___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "iff_iff_implies_and_implies"};
static const lean_object* lp_mathlib_iffComm___closed__15 = (const lean_object*)&lp_mathlib_iffComm___closed__15_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__15_value),LEAN_SCALAR_PTR_LITERAL(65, 98, 125, 184, 35, 143, 136, 103)}};
static const lean_object* lp_mathlib_iffComm___closed__16 = (const lean_object*)&lp_mathlib_iffComm___closed__16_value;
static const lean_string_object lp_mathlib_iffComm___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "coe_iff_coe"};
static const lean_object* lp_mathlib_iffComm___closed__17 = (const lean_object*)&lp_mathlib_iffComm___closed__17_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqComm___closed__7_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_iffComm___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iffComm___closed__18_value_aux_0),((lean_object*)&lp_mathlib_iffComm___closed__17_value),LEAN_SCALAR_PTR_LITERAL(101, 121, 117, 111, 59, 221, 161, 62)}};
static const lean_object* lp_mathlib_iffComm___closed__18 = (const lean_object*)&lp_mathlib_iffComm___closed__18_value;
static const lean_array_object lp_mathlib_iffComm___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*8, .m_other = 0, .m_tag = 246}, .m_size = 8, .m_capacity = 8, .m_data = {((lean_object*)&lp_mathlib_iffComm___closed__4_value),((lean_object*)&lp_mathlib_iffComm___closed__6_value),((lean_object*)&lp_mathlib_iffComm___closed__8_value),((lean_object*)&lp_mathlib_iffComm___closed__10_value),((lean_object*)&lp_mathlib_iffComm___closed__12_value),((lean_object*)&lp_mathlib_iffComm___closed__14_value),((lean_object*)&lp_mathlib_iffComm___closed__16_value),((lean_object*)&lp_mathlib_iffComm___closed__18_value)}};
static const lean_object* lp_mathlib_iffComm___closed__19 = (const lean_object*)&lp_mathlib_iffComm___closed__19_value;
static const lean_string_object lp_mathlib_iffComm___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "iff_comm_eq"};
static const lean_object* lp_mathlib_iffComm___closed__20 = (const lean_object*)&lp_mathlib_iffComm___closed__20_value;
static const lean_ctor_object lp_mathlib_iffComm___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iffComm___closed__20_value),LEAN_SCALAR_PTR_LITERAL(249, 214, 122, 210, 168, 117, 21, 31)}};
static const lean_object* lp_mathlib_iffComm___closed__21 = (const lean_object*)&lp_mathlib_iffComm___closed__21_value;
LEAN_EXPORT lean_object* lp_mathlib_iffComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iffComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_hidden___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_hidden___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_hidden(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_hidden___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_decidableEq__of__subsingleton(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decidableEq__of__subsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_fact__non_x2dinstances;
LEAN_EXPORT uint8_t lp_mathlib_instDecidableFact___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableFact___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableFact(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableFact___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_swap_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_swap_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_decidable__namespace;
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_decidable__arguments;
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___aux__1___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___aux__1(lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___redArg(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor(lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Classical_choice__of__byContradiction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Classical_choice__of__byContradiction_x27(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_unsigned_to_nat(32u);
v___x_2_ = lean_mk_empty_array_with_capacity(v___x_1_);
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_4_ = ((size_t)5ULL);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_unsigned_to_nat(32u);
v___x_7_ = lean_mk_empty_array_with_capacity(v___x_6_);
v___x_8_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__0);
v___x_9_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
lean_ctor_set(v___x_9_, 2, v___x_5_);
lean_ctor_set(v___x_9_, 3, v___x_5_);
lean_ctor_set_usize(v___x_9_, 4, v___x_4_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(lean_object* v___y_10_){
_start:
{
lean_object* v___x_12_; lean_object* v_traceState_13_; lean_object* v_traces_14_; lean_object* v___x_15_; lean_object* v_traceState_16_; lean_object* v_env_17_; lean_object* v_nextMacroScope_18_; lean_object* v_ngen_19_; lean_object* v_auxDeclNGen_20_; lean_object* v_cache_21_; lean_object* v_messages_22_; lean_object* v_infoState_23_; lean_object* v_snapshotTasks_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_43_; 
v___x_12_ = lean_st_ref_get(v___y_10_);
v_traceState_13_ = lean_ctor_get(v___x_12_, 4);
lean_inc_ref(v_traceState_13_);
lean_dec(v___x_12_);
v_traces_14_ = lean_ctor_get(v_traceState_13_, 0);
lean_inc_ref(v_traces_14_);
lean_dec_ref(v_traceState_13_);
v___x_15_ = lean_st_ref_take(v___y_10_);
v_traceState_16_ = lean_ctor_get(v___x_15_, 4);
v_env_17_ = lean_ctor_get(v___x_15_, 0);
v_nextMacroScope_18_ = lean_ctor_get(v___x_15_, 1);
v_ngen_19_ = lean_ctor_get(v___x_15_, 2);
v_auxDeclNGen_20_ = lean_ctor_get(v___x_15_, 3);
v_cache_21_ = lean_ctor_get(v___x_15_, 5);
v_messages_22_ = lean_ctor_get(v___x_15_, 6);
v_infoState_23_ = lean_ctor_get(v___x_15_, 7);
v_snapshotTasks_24_ = lean_ctor_get(v___x_15_, 8);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_43_ == 0)
{
v___x_26_ = v___x_15_;
v_isShared_27_ = v_isSharedCheck_43_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_snapshotTasks_24_);
lean_inc(v_infoState_23_);
lean_inc(v_messages_22_);
lean_inc(v_cache_21_);
lean_inc(v_traceState_16_);
lean_inc(v_auxDeclNGen_20_);
lean_inc(v_ngen_19_);
lean_inc(v_nextMacroScope_18_);
lean_inc(v_env_17_);
lean_dec(v___x_15_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_43_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
uint64_t v_tid_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_41_; 
v_tid_28_ = lean_ctor_get_uint64(v_traceState_16_, sizeof(void*)*1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_traceState_16_);
if (v_isSharedCheck_41_ == 0)
{
lean_object* v_unused_42_; 
v_unused_42_ = lean_ctor_get(v_traceState_16_, 0);
lean_dec(v_unused_42_);
v___x_30_ = v_traceState_16_;
v_isShared_31_ = v_isSharedCheck_41_;
goto v_resetjp_29_;
}
else
{
lean_dec(v_traceState_16_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_41_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_32_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___closed__1);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_32_);
v___x_34_ = v___x_30_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v___x_32_);
lean_ctor_set_uint64(v_reuseFailAlloc_40_, sizeof(void*)*1, v_tid_28_);
v___x_34_ = v_reuseFailAlloc_40_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
lean_object* v___x_36_; 
if (v_isShared_27_ == 0)
{
lean_ctor_set(v___x_26_, 4, v___x_34_);
v___x_36_ = v___x_26_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v_env_17_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v_nextMacroScope_18_);
lean_ctor_set(v_reuseFailAlloc_39_, 2, v_ngen_19_);
lean_ctor_set(v_reuseFailAlloc_39_, 3, v_auxDeclNGen_20_);
lean_ctor_set(v_reuseFailAlloc_39_, 4, v___x_34_);
lean_ctor_set(v_reuseFailAlloc_39_, 5, v_cache_21_);
lean_ctor_set(v_reuseFailAlloc_39_, 6, v_messages_22_);
lean_ctor_set(v_reuseFailAlloc_39_, 7, v_infoState_23_);
lean_ctor_set(v_reuseFailAlloc_39_, 8, v_snapshotTasks_24_);
v___x_36_ = v_reuseFailAlloc_39_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = lean_st_ref_set(v___y_10_, v___x_36_);
v___x_38_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_38_, 0, v_traces_14_);
return v___x_38_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg___boxed(lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(v___y_44_);
lean_dec(v___y_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0(lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(v___y_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___boxed(lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0(v___y_56_, v___y_57_, v___y_58_, v___y_59_, v___y_60_, v___y_61_, v___y_62_);
lean_dec(v___y_62_);
lean_dec_ref(v___y_61_);
lean_dec(v___y_60_);
lean_dec_ref(v___y_59_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
return v_res_64_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(lean_object* v_opts_65_, lean_object* v_opt_66_){
_start:
{
lean_object* v_name_67_; lean_object* v_defValue_68_; lean_object* v_map_69_; lean_object* v___x_70_; 
v_name_67_ = lean_ctor_get(v_opt_66_, 0);
v_defValue_68_ = lean_ctor_get(v_opt_66_, 1);
v_map_69_ = lean_ctor_get(v_opts_65_, 0);
v___x_70_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_69_, v_name_67_);
if (lean_obj_tag(v___x_70_) == 0)
{
uint8_t v___x_71_; 
v___x_71_ = lean_unbox(v_defValue_68_);
return v___x_71_;
}
else
{
lean_object* v_val_72_; 
v_val_72_ = lean_ctor_get(v___x_70_, 0);
lean_inc(v_val_72_);
lean_dec_ref_known(v___x_70_, 1);
if (lean_obj_tag(v_val_72_) == 1)
{
uint8_t v_v_73_; 
v_v_73_ = lean_ctor_get_uint8(v_val_72_, 0);
lean_dec_ref_known(v_val_72_, 0);
return v_v_73_;
}
else
{
uint8_t v___x_74_; 
lean_dec(v_val_72_);
v___x_74_ = lean_unbox(v_defValue_68_);
return v___x_74_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00eqComm_spec__1___boxed(lean_object* v_opts_75_, lean_object* v_opt_76_){
_start:
{
uint8_t v_res_77_; lean_object* v_r_78_; 
v_res_77_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_opts_75_, v_opt_76_);
lean_dec_ref(v_opt_76_);
lean_dec_ref(v_opts_75_);
v_r_78_ = lean_box(v_res_77_);
return v_r_78_;
}
}
static lean_object* _init_lp_mathlib_eqComm___lam__0___closed__1(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = ((lean_object*)(lp_mathlib_eqComm___lam__0___closed__0));
v___x_81_ = l_Lean_stringToMessageData(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__0(lean_object* v_e_82_, lean_object* v_x_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_92_ = lean_obj_once(&lp_mathlib_eqComm___lam__0___closed__1, &lp_mathlib_eqComm___lam__0___closed__1_once, _init_lp_mathlib_eqComm___lam__0___closed__1);
v___x_93_ = l_Lean_MessageData_ofExpr(v_e_82_);
v___x_94_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__0___boxed(lean_object* v_e_96_, lean_object* v_x_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_eqComm___lam__0(v_e_96_, v_x_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
lean_dec(v___y_98_);
lean_dec_ref(v_x_97_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5(lean_object* v_opts_107_, lean_object* v_opt_108_){
_start:
{
lean_object* v_name_109_; lean_object* v_defValue_110_; lean_object* v_map_111_; lean_object* v___x_112_; 
v_name_109_ = lean_ctor_get(v_opt_108_, 0);
v_defValue_110_ = lean_ctor_get(v_opt_108_, 1);
v_map_111_ = lean_ctor_get(v_opts_107_, 0);
v___x_112_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_111_, v_name_109_);
if (lean_obj_tag(v___x_112_) == 0)
{
lean_inc(v_defValue_110_);
return v_defValue_110_;
}
else
{
lean_object* v_val_113_; 
v_val_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc(v_val_113_);
lean_dec_ref_known(v___x_112_, 1);
if (lean_obj_tag(v_val_113_) == 3)
{
lean_object* v_v_114_; 
v_v_114_ = lean_ctor_get(v_val_113_, 0);
lean_inc(v_v_114_);
lean_dec_ref_known(v_val_113_, 1);
return v_v_114_;
}
else
{
lean_dec(v_val_113_);
lean_inc(v_defValue_110_);
return v_defValue_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5___boxed(lean_object* v_opts_115_, lean_object* v_opt_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5(v_opts_115_, v_opt_116_);
lean_dec_ref(v_opt_116_);
lean_dec_ref(v_opts_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4(lean_object* v_msgData_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___x_124_; lean_object* v_env_125_; lean_object* v___x_126_; lean_object* v_mctx_127_; lean_object* v_lctx_128_; lean_object* v_options_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_124_ = lean_st_ref_get(v___y_122_);
v_env_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc_ref(v_env_125_);
lean_dec(v___x_124_);
v___x_126_ = lean_st_ref_get(v___y_120_);
v_mctx_127_ = lean_ctor_get(v___x_126_, 0);
lean_inc_ref(v_mctx_127_);
lean_dec(v___x_126_);
v_lctx_128_ = lean_ctor_get(v___y_119_, 2);
v_options_129_ = lean_ctor_get(v___y_121_, 2);
lean_inc_ref(v_options_129_);
lean_inc_ref(v_lctx_128_);
v___x_130_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_130_, 0, v_env_125_);
lean_ctor_set(v___x_130_, 1, v_mctx_127_);
lean_ctor_set(v___x_130_, 2, v_lctx_128_);
lean_ctor_set(v___x_130_, 3, v_options_129_);
v___x_131_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v_msgData_118_);
v___x_132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4___boxed(lean_object* v_msgData_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4(v_msgData_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3(size_t v_sz_140_, size_t v_i_141_, lean_object* v_bs_142_){
_start:
{
uint8_t v___x_143_; 
v___x_143_ = lean_usize_dec_lt(v_i_141_, v_sz_140_);
if (v___x_143_ == 0)
{
return v_bs_142_;
}
else
{
lean_object* v_v_144_; lean_object* v_msg_145_; lean_object* v___x_146_; lean_object* v_bs_x27_147_; size_t v___x_148_; size_t v___x_149_; lean_object* v___x_150_; 
v_v_144_ = lean_array_uget_borrowed(v_bs_142_, v_i_141_);
v_msg_145_ = lean_ctor_get(v_v_144_, 1);
lean_inc_ref(v_msg_145_);
v___x_146_ = lean_unsigned_to_nat(0u);
v_bs_x27_147_ = lean_array_uset(v_bs_142_, v_i_141_, v___x_146_);
v___x_148_ = ((size_t)1ULL);
v___x_149_ = lean_usize_add(v_i_141_, v___x_148_);
v___x_150_ = lean_array_uset(v_bs_x27_147_, v_i_141_, v_msg_145_);
v_i_141_ = v___x_149_;
v_bs_142_ = v___x_150_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3___boxed(lean_object* v_sz_152_, lean_object* v_i_153_, lean_object* v_bs_154_){
_start:
{
size_t v_sz_boxed_155_; size_t v_i_boxed_156_; lean_object* v_res_157_; 
v_sz_boxed_155_ = lean_unbox_usize(v_sz_152_);
lean_dec(v_sz_152_);
v_i_boxed_156_ = lean_unbox_usize(v_i_153_);
lean_dec(v_i_153_);
v_res_157_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3(v_sz_boxed_155_, v_i_boxed_156_, v_bs_154_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg(lean_object* v_oldTraces_158_, lean_object* v_data_159_, lean_object* v_ref_160_, lean_object* v_msg_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v_fileName_167_; lean_object* v_fileMap_168_; lean_object* v_options_169_; lean_object* v_currRecDepth_170_; lean_object* v_maxRecDepth_171_; lean_object* v_ref_172_; lean_object* v_currNamespace_173_; lean_object* v_openDecls_174_; lean_object* v_initHeartbeats_175_; lean_object* v_maxHeartbeats_176_; lean_object* v_quotContext_177_; lean_object* v_currMacroScope_178_; uint8_t v_diag_179_; lean_object* v_cancelTk_x3f_180_; uint8_t v_suppressElabErrors_181_; lean_object* v_inheritedTraceOptions_182_; lean_object* v___x_183_; lean_object* v_traceState_184_; lean_object* v_traces_185_; lean_object* v_ref_186_; lean_object* v___x_187_; lean_object* v___x_188_; size_t v_sz_189_; size_t v___x_190_; lean_object* v___x_191_; lean_object* v_msg_192_; lean_object* v___x_193_; lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_231_; 
v_fileName_167_ = lean_ctor_get(v___y_164_, 0);
v_fileMap_168_ = lean_ctor_get(v___y_164_, 1);
v_options_169_ = lean_ctor_get(v___y_164_, 2);
v_currRecDepth_170_ = lean_ctor_get(v___y_164_, 3);
v_maxRecDepth_171_ = lean_ctor_get(v___y_164_, 4);
v_ref_172_ = lean_ctor_get(v___y_164_, 5);
v_currNamespace_173_ = lean_ctor_get(v___y_164_, 6);
v_openDecls_174_ = lean_ctor_get(v___y_164_, 7);
v_initHeartbeats_175_ = lean_ctor_get(v___y_164_, 8);
v_maxHeartbeats_176_ = lean_ctor_get(v___y_164_, 9);
v_quotContext_177_ = lean_ctor_get(v___y_164_, 10);
v_currMacroScope_178_ = lean_ctor_get(v___y_164_, 11);
v_diag_179_ = lean_ctor_get_uint8(v___y_164_, sizeof(void*)*14);
v_cancelTk_x3f_180_ = lean_ctor_get(v___y_164_, 12);
v_suppressElabErrors_181_ = lean_ctor_get_uint8(v___y_164_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_182_ = lean_ctor_get(v___y_164_, 13);
v___x_183_ = lean_st_ref_get(v___y_165_);
v_traceState_184_ = lean_ctor_get(v___x_183_, 4);
lean_inc_ref(v_traceState_184_);
lean_dec(v___x_183_);
v_traces_185_ = lean_ctor_get(v_traceState_184_, 0);
lean_inc_ref(v_traces_185_);
lean_dec_ref(v_traceState_184_);
v_ref_186_ = l_Lean_replaceRef(v_ref_160_, v_ref_172_);
lean_inc_ref(v_inheritedTraceOptions_182_);
lean_inc(v_cancelTk_x3f_180_);
lean_inc(v_currMacroScope_178_);
lean_inc(v_quotContext_177_);
lean_inc(v_maxHeartbeats_176_);
lean_inc(v_initHeartbeats_175_);
lean_inc(v_openDecls_174_);
lean_inc(v_currNamespace_173_);
lean_inc(v_maxRecDepth_171_);
lean_inc(v_currRecDepth_170_);
lean_inc_ref(v_options_169_);
lean_inc_ref(v_fileMap_168_);
lean_inc_ref(v_fileName_167_);
v___x_187_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_187_, 0, v_fileName_167_);
lean_ctor_set(v___x_187_, 1, v_fileMap_168_);
lean_ctor_set(v___x_187_, 2, v_options_169_);
lean_ctor_set(v___x_187_, 3, v_currRecDepth_170_);
lean_ctor_set(v___x_187_, 4, v_maxRecDepth_171_);
lean_ctor_set(v___x_187_, 5, v_ref_186_);
lean_ctor_set(v___x_187_, 6, v_currNamespace_173_);
lean_ctor_set(v___x_187_, 7, v_openDecls_174_);
lean_ctor_set(v___x_187_, 8, v_initHeartbeats_175_);
lean_ctor_set(v___x_187_, 9, v_maxHeartbeats_176_);
lean_ctor_set(v___x_187_, 10, v_quotContext_177_);
lean_ctor_set(v___x_187_, 11, v_currMacroScope_178_);
lean_ctor_set(v___x_187_, 12, v_cancelTk_x3f_180_);
lean_ctor_set(v___x_187_, 13, v_inheritedTraceOptions_182_);
lean_ctor_set_uint8(v___x_187_, sizeof(void*)*14, v_diag_179_);
lean_ctor_set_uint8(v___x_187_, sizeof(void*)*14 + 1, v_suppressElabErrors_181_);
v___x_188_ = l_Lean_PersistentArray_toArray___redArg(v_traces_185_);
lean_dec_ref(v_traces_185_);
v_sz_189_ = lean_array_size(v___x_188_);
v___x_190_ = ((size_t)0ULL);
v___x_191_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__3(v_sz_189_, v___x_190_, v___x_188_);
v_msg_192_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_192_, 0, v_data_159_);
lean_ctor_set(v_msg_192_, 1, v_msg_161_);
lean_ctor_set(v_msg_192_, 2, v___x_191_);
v___x_193_ = lp_mathlib_Lean_addMessageContextFull___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2_spec__4(v_msg_192_, v___y_162_, v___y_163_, v___x_187_, v___y_165_);
lean_dec_ref_known(v___x_187_, 14);
v_a_194_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_231_ == 0)
{
v___x_196_ = v___x_193_;
v_isShared_197_ = v_isSharedCheck_231_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_dec(v___x_193_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_231_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_198_; lean_object* v_traceState_199_; lean_object* v_env_200_; lean_object* v_nextMacroScope_201_; lean_object* v_ngen_202_; lean_object* v_auxDeclNGen_203_; lean_object* v_cache_204_; lean_object* v_messages_205_; lean_object* v_infoState_206_; lean_object* v_snapshotTasks_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_230_; 
v___x_198_ = lean_st_ref_take(v___y_165_);
v_traceState_199_ = lean_ctor_get(v___x_198_, 4);
v_env_200_ = lean_ctor_get(v___x_198_, 0);
v_nextMacroScope_201_ = lean_ctor_get(v___x_198_, 1);
v_ngen_202_ = lean_ctor_get(v___x_198_, 2);
v_auxDeclNGen_203_ = lean_ctor_get(v___x_198_, 3);
v_cache_204_ = lean_ctor_get(v___x_198_, 5);
v_messages_205_ = lean_ctor_get(v___x_198_, 6);
v_infoState_206_ = lean_ctor_get(v___x_198_, 7);
v_snapshotTasks_207_ = lean_ctor_get(v___x_198_, 8);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_230_ == 0)
{
v___x_209_ = v___x_198_;
v_isShared_210_ = v_isSharedCheck_230_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_snapshotTasks_207_);
lean_inc(v_infoState_206_);
lean_inc(v_messages_205_);
lean_inc(v_cache_204_);
lean_inc(v_traceState_199_);
lean_inc(v_auxDeclNGen_203_);
lean_inc(v_ngen_202_);
lean_inc(v_nextMacroScope_201_);
lean_inc(v_env_200_);
lean_dec(v___x_198_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_230_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
uint64_t v_tid_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_228_; 
v_tid_211_ = lean_ctor_get_uint64(v_traceState_199_, sizeof(void*)*1);
v_isSharedCheck_228_ = !lean_is_exclusive(v_traceState_199_);
if (v_isSharedCheck_228_ == 0)
{
lean_object* v_unused_229_; 
v_unused_229_ = lean_ctor_get(v_traceState_199_, 0);
lean_dec(v_unused_229_);
v___x_213_ = v_traceState_199_;
v_isShared_214_ = v_isSharedCheck_228_;
goto v_resetjp_212_;
}
else
{
lean_dec(v_traceState_199_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_228_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_218_; 
v___x_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_215_, 0, v_ref_160_);
lean_ctor_set(v___x_215_, 1, v_a_194_);
v___x_216_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_158_, v___x_215_);
if (v_isShared_214_ == 0)
{
lean_ctor_set(v___x_213_, 0, v___x_216_);
v___x_218_ = v___x_213_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v___x_216_);
lean_ctor_set_uint64(v_reuseFailAlloc_227_, sizeof(void*)*1, v_tid_211_);
v___x_218_ = v_reuseFailAlloc_227_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
lean_object* v___x_220_; 
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 4, v___x_218_);
v___x_220_ = v___x_209_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v_env_200_);
lean_ctor_set(v_reuseFailAlloc_226_, 1, v_nextMacroScope_201_);
lean_ctor_set(v_reuseFailAlloc_226_, 2, v_ngen_202_);
lean_ctor_set(v_reuseFailAlloc_226_, 3, v_auxDeclNGen_203_);
lean_ctor_set(v_reuseFailAlloc_226_, 4, v___x_218_);
lean_ctor_set(v_reuseFailAlloc_226_, 5, v_cache_204_);
lean_ctor_set(v_reuseFailAlloc_226_, 6, v_messages_205_);
lean_ctor_set(v_reuseFailAlloc_226_, 7, v_infoState_206_);
lean_ctor_set(v_reuseFailAlloc_226_, 8, v_snapshotTasks_207_);
v___x_220_ = v_reuseFailAlloc_226_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_224_; 
v___x_221_ = lean_st_ref_set(v___y_165_, v___x_220_);
v___x_222_ = lean_box(0);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 0, v___x_222_);
v___x_224_ = v___x_196_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v___x_222_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg___boxed(lean_object* v_oldTraces_232_, lean_object* v_data_233_, lean_object* v_ref_234_, lean_object* v_msg_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg(v_oldTraces_232_, v_data_233_, v_ref_234_, v_msg_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
lean_dec(v___y_239_);
lean_dec_ref(v___y_238_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(lean_object* v_x_242_){
_start:
{
if (lean_obj_tag(v_x_242_) == 0)
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_251_; 
v_a_244_ = lean_ctor_get(v_x_242_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v_x_242_);
if (v_isSharedCheck_251_ == 0)
{
v___x_246_ = v_x_242_;
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v_x_242_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_249_; 
if (v_isShared_247_ == 0)
{
lean_ctor_set_tag(v___x_246_, 1);
v___x_249_ = v___x_246_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_a_244_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
v_a_252_ = lean_ctor_get(v_x_242_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v_x_242_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v_x_242_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v_x_242_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
lean_ctor_set_tag(v___x_254_, 0);
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(0, 1, 0);
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
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg___boxed(lean_object* v_x_260_, lean_object* v___y_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(v_x_260_);
return v_res_262_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4(lean_object* v_e_263_){
_start:
{
if (lean_obj_tag(v_e_263_) == 0)
{
uint8_t v___x_264_; 
v___x_264_ = 2;
return v___x_264_;
}
else
{
uint8_t v___x_265_; 
v___x_265_ = 0;
return v___x_265_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4___boxed(lean_object* v_e_266_){
_start:
{
uint8_t v_res_267_; lean_object* v_r_268_; 
v_res_267_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4(v_e_266_);
lean_dec_ref(v_e_266_);
v_r_268_ = lean_box(v_res_267_);
return v_r_268_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0(void){
_start:
{
lean_object* v___x_269_; double v___x_270_; 
v___x_269_ = lean_unsigned_to_nat(0u);
v___x_270_ = lean_float_of_nat(v___x_269_);
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__1));
v___x_273_ = l_Lean_stringToMessageData(v___x_272_);
return v___x_273_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3(void){
_start:
{
lean_object* v___x_274_; double v___x_275_; 
v___x_274_ = lean_unsigned_to_nat(1000u);
v___x_275_ = lean_float_of_nat(v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(lean_object* v_cls_276_, uint8_t v_collapsed_277_, lean_object* v_tag_278_, lean_object* v_opts_279_, uint8_t v_clsEnabled_280_, lean_object* v_oldTraces_281_, lean_object* v_msg_282_, lean_object* v_resStartStop_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_){
_start:
{
lean_object* v_fst_292_; lean_object* v_snd_293_; lean_object* v___y_295_; lean_object* v___y_296_; lean_object* v_data_297_; lean_object* v_fst_308_; lean_object* v_snd_309_; lean_object* v___x_310_; uint8_t v___x_311_; lean_object* v___y_313_; lean_object* v_a_314_; uint8_t v___y_329_; double v___y_360_; 
v_fst_292_ = lean_ctor_get(v_resStartStop_283_, 0);
lean_inc(v_fst_292_);
v_snd_293_ = lean_ctor_get(v_resStartStop_283_, 1);
lean_inc(v_snd_293_);
lean_dec_ref(v_resStartStop_283_);
v_fst_308_ = lean_ctor_get(v_snd_293_, 0);
lean_inc(v_fst_308_);
v_snd_309_ = lean_ctor_get(v_snd_293_, 1);
lean_inc(v_snd_309_);
lean_dec(v_snd_293_);
v___x_310_ = l_Lean_trace_profiler;
v___x_311_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_opts_279_, v___x_310_);
if (v___x_311_ == 0)
{
v___y_329_ = v___x_311_;
goto v___jp_328_;
}
else
{
lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_365_ = l_Lean_trace_profiler_useHeartbeats;
v___x_366_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_opts_279_, v___x_365_);
if (v___x_366_ == 0)
{
lean_object* v___x_367_; lean_object* v___x_368_; double v___x_369_; double v___x_370_; double v___x_371_; 
v___x_367_ = l_Lean_trace_profiler_threshold;
v___x_368_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5(v_opts_279_, v___x_367_);
v___x_369_ = lean_float_of_nat(v___x_368_);
v___x_370_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__3);
v___x_371_ = lean_float_div(v___x_369_, v___x_370_);
v___y_360_ = v___x_371_;
goto v___jp_359_;
}
else
{
lean_object* v___x_372_; lean_object* v___x_373_; double v___x_374_; 
v___x_372_ = l_Lean_trace_profiler_threshold;
v___x_373_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__5(v_opts_279_, v___x_372_);
v___x_374_ = lean_float_of_nat(v___x_373_);
v___y_360_ = v___x_374_;
goto v___jp_359_;
}
}
v___jp_294_:
{
lean_object* v___x_298_; 
lean_inc(v___y_295_);
v___x_298_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg(v_oldTraces_281_, v_data_297_, v___y_295_, v___y_296_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v___x_299_; 
lean_dec_ref_known(v___x_298_, 1);
v___x_299_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(v_fst_292_);
return v___x_299_;
}
else
{
lean_object* v_a_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_307_; 
lean_dec(v_fst_292_);
v_a_300_ = lean_ctor_get(v___x_298_, 0);
v_isSharedCheck_307_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_307_ == 0)
{
v___x_302_ = v___x_298_;
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_a_300_);
lean_dec(v___x_298_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_305_; 
if (v_isShared_303_ == 0)
{
v___x_305_ = v___x_302_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v_a_300_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
}
v___jp_312_:
{
uint8_t v_result_315_; lean_object* v___x_316_; lean_object* v___x_317_; double v___x_318_; lean_object* v_data_319_; 
v_result_315_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__4(v_fst_292_);
v___x_316_ = lean_box(v_result_315_);
v___x_317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
v___x_318_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__0);
lean_inc_ref(v_tag_278_);
lean_inc_ref(v___x_317_);
lean_inc(v_cls_276_);
v_data_319_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_319_, 0, v_cls_276_);
lean_ctor_set(v_data_319_, 1, v___x_317_);
lean_ctor_set(v_data_319_, 2, v_tag_278_);
lean_ctor_set_float(v_data_319_, sizeof(void*)*3, v___x_318_);
lean_ctor_set_float(v_data_319_, sizeof(void*)*3 + 8, v___x_318_);
lean_ctor_set_uint8(v_data_319_, sizeof(void*)*3 + 16, v_collapsed_277_);
if (v___x_311_ == 0)
{
lean_dec_ref_known(v___x_317_, 1);
lean_dec(v_snd_309_);
lean_dec(v_fst_308_);
lean_dec_ref(v_tag_278_);
lean_dec(v_cls_276_);
v___y_295_ = v___y_313_;
v___y_296_ = v_a_314_;
v_data_297_ = v_data_319_;
goto v___jp_294_;
}
else
{
lean_object* v_data_320_; double v___x_321_; double v___x_322_; 
lean_dec_ref_known(v_data_319_, 3);
v_data_320_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_320_, 0, v_cls_276_);
lean_ctor_set(v_data_320_, 1, v___x_317_);
lean_ctor_set(v_data_320_, 2, v_tag_278_);
v___x_321_ = lean_unbox_float(v_fst_308_);
lean_dec(v_fst_308_);
lean_ctor_set_float(v_data_320_, sizeof(void*)*3, v___x_321_);
v___x_322_ = lean_unbox_float(v_snd_309_);
lean_dec(v_snd_309_);
lean_ctor_set_float(v_data_320_, sizeof(void*)*3 + 8, v___x_322_);
lean_ctor_set_uint8(v_data_320_, sizeof(void*)*3 + 16, v_collapsed_277_);
v___y_295_ = v___y_313_;
v___y_296_ = v_a_314_;
v_data_297_ = v_data_320_;
goto v___jp_294_;
}
}
v___jp_323_:
{
lean_object* v_ref_324_; lean_object* v___x_325_; 
v_ref_324_ = lean_ctor_get(v___y_289_, 5);
lean_inc(v___y_290_);
lean_inc_ref(v___y_289_);
lean_inc(v___y_288_);
lean_inc_ref(v___y_287_);
lean_inc(v___y_286_);
lean_inc_ref(v___y_285_);
lean_inc(v___y_284_);
lean_inc(v_fst_292_);
v___x_325_ = lean_apply_9(v_msg_282_, v_fst_292_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, lean_box(0));
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v_a_326_; 
v_a_326_ = lean_ctor_get(v___x_325_, 0);
lean_inc(v_a_326_);
lean_dec_ref_known(v___x_325_, 1);
v___y_313_ = v_ref_324_;
v_a_314_ = v_a_326_;
goto v___jp_312_;
}
else
{
lean_object* v___x_327_; 
lean_dec_ref_known(v___x_325_, 1);
v___x_327_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___closed__2);
v___y_313_ = v_ref_324_;
v_a_314_ = v___x_327_;
goto v___jp_312_;
}
}
v___jp_328_:
{
if (v_clsEnabled_280_ == 0)
{
if (v___y_329_ == 0)
{
lean_object* v___x_330_; lean_object* v_traceState_331_; lean_object* v_env_332_; lean_object* v_nextMacroScope_333_; lean_object* v_ngen_334_; lean_object* v_auxDeclNGen_335_; lean_object* v_cache_336_; lean_object* v_messages_337_; lean_object* v_infoState_338_; lean_object* v_snapshotTasks_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_358_; 
lean_dec(v_snd_309_);
lean_dec(v_fst_308_);
lean_dec_ref(v_msg_282_);
lean_dec_ref(v_tag_278_);
lean_dec(v_cls_276_);
v___x_330_ = lean_st_ref_take(v___y_290_);
v_traceState_331_ = lean_ctor_get(v___x_330_, 4);
v_env_332_ = lean_ctor_get(v___x_330_, 0);
v_nextMacroScope_333_ = lean_ctor_get(v___x_330_, 1);
v_ngen_334_ = lean_ctor_get(v___x_330_, 2);
v_auxDeclNGen_335_ = lean_ctor_get(v___x_330_, 3);
v_cache_336_ = lean_ctor_get(v___x_330_, 5);
v_messages_337_ = lean_ctor_get(v___x_330_, 6);
v_infoState_338_ = lean_ctor_get(v___x_330_, 7);
v_snapshotTasks_339_ = lean_ctor_get(v___x_330_, 8);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_358_ == 0)
{
v___x_341_ = v___x_330_;
v_isShared_342_ = v_isSharedCheck_358_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_snapshotTasks_339_);
lean_inc(v_infoState_338_);
lean_inc(v_messages_337_);
lean_inc(v_cache_336_);
lean_inc(v_traceState_331_);
lean_inc(v_auxDeclNGen_335_);
lean_inc(v_ngen_334_);
lean_inc(v_nextMacroScope_333_);
lean_inc(v_env_332_);
lean_dec(v___x_330_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_358_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
uint64_t v_tid_343_; lean_object* v_traces_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_357_; 
v_tid_343_ = lean_ctor_get_uint64(v_traceState_331_, sizeof(void*)*1);
v_traces_344_ = lean_ctor_get(v_traceState_331_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v_traceState_331_);
if (v_isSharedCheck_357_ == 0)
{
v___x_346_ = v_traceState_331_;
v_isShared_347_ = v_isSharedCheck_357_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_traces_344_);
lean_dec(v_traceState_331_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_357_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_350_; 
v___x_348_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_281_, v_traces_344_);
lean_dec_ref(v_traces_344_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 0, v___x_348_);
v___x_350_ = v___x_346_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_348_);
lean_ctor_set_uint64(v_reuseFailAlloc_356_, sizeof(void*)*1, v_tid_343_);
v___x_350_ = v_reuseFailAlloc_356_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
lean_object* v___x_352_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 4, v___x_350_);
v___x_352_ = v___x_341_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v_env_332_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v_nextMacroScope_333_);
lean_ctor_set(v_reuseFailAlloc_355_, 2, v_ngen_334_);
lean_ctor_set(v_reuseFailAlloc_355_, 3, v_auxDeclNGen_335_);
lean_ctor_set(v_reuseFailAlloc_355_, 4, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_355_, 5, v_cache_336_);
lean_ctor_set(v_reuseFailAlloc_355_, 6, v_messages_337_);
lean_ctor_set(v_reuseFailAlloc_355_, 7, v_infoState_338_);
lean_ctor_set(v_reuseFailAlloc_355_, 8, v_snapshotTasks_339_);
v___x_352_ = v_reuseFailAlloc_355_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_353_ = lean_st_ref_set(v___y_290_, v___x_352_);
v___x_354_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(v_fst_292_);
return v___x_354_;
}
}
}
}
}
else
{
goto v___jp_323_;
}
}
else
{
goto v___jp_323_;
}
}
v___jp_359_:
{
double v___x_361_; double v___x_362_; double v___x_363_; uint8_t v___x_364_; 
v___x_361_ = lean_unbox_float(v_snd_309_);
v___x_362_ = lean_unbox_float(v_fst_308_);
v___x_363_ = lean_float_sub(v___x_361_, v___x_362_);
v___x_364_ = lean_float_decLt(v___y_360_, v___x_363_);
v___y_329_ = v___x_364_;
goto v___jp_328_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2___boxed(lean_object* v_cls_375_, lean_object* v_collapsed_376_, lean_object* v_tag_377_, lean_object* v_opts_378_, lean_object* v_clsEnabled_379_, lean_object* v_oldTraces_380_, lean_object* v_msg_381_, lean_object* v_resStartStop_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_){
_start:
{
uint8_t v_collapsed_boxed_391_; uint8_t v_clsEnabled_boxed_392_; lean_object* v_res_393_; 
v_collapsed_boxed_391_ = lean_unbox(v_collapsed_376_);
v_clsEnabled_boxed_392_ = lean_unbox(v_clsEnabled_379_);
v_res_393_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(v_cls_375_, v_collapsed_boxed_391_, v_tag_377_, v_opts_378_, v_clsEnabled_boxed_392_, v_oldTraces_380_, v_msg_381_, v_resStartStop_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_, v___y_388_, v___y_389_);
lean_dec(v___y_389_);
lean_dec_ref(v___y_388_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec(v___y_383_);
lean_dec_ref(v_opts_378_);
return v_res_393_;
}
}
static double _init_lp_mathlib_eqComm___lam__1___closed__2(void){
_start:
{
lean_object* v___x_397_; double v___x_398_; 
v___x_397_ = lean_unsigned_to_nat(1000000000u);
v___x_398_ = lean_float_of_nat(v___x_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__1(lean_object* v_a_399_, lean_object* v___x_400_, uint8_t v___x_401_, lean_object* v___x_402_, lean_object* v___f_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v_options_412_; uint8_t v_hasTrace_413_; 
v_options_412_ = lean_ctor_get(v___y_409_, 2);
v_hasTrace_413_ = lean_ctor_get_uint8(v_options_412_, sizeof(void*)*1);
if (v_hasTrace_413_ == 0)
{
lean_object* v___x_414_; 
lean_dec_ref(v___f_403_);
lean_dec_ref(v___x_402_);
lean_dec(v___x_400_);
lean_inc(v___y_410_);
lean_inc_ref(v___y_409_);
lean_inc(v___y_408_);
lean_inc_ref(v___y_407_);
lean_inc(v___y_406_);
lean_inc_ref(v___y_405_);
lean_inc(v___y_404_);
v___x_414_ = lean_simp(v_a_399_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_414_;
}
else
{
lean_object* v_inheritedTraceOptions_415_; lean_object* v___x_416_; lean_object* v___x_417_; uint8_t v___x_418_; lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v_a_422_; lean_object* v___y_435_; lean_object* v___y_436_; lean_object* v_a_437_; 
v_inheritedTraceOptions_415_ = lean_ctor_get(v___y_409_, 13);
v___x_416_ = ((lean_object*)(lp_mathlib_eqComm___lam__1___closed__1));
lean_inc(v___x_400_);
v___x_417_ = l_Lean_Name_append(v___x_416_, v___x_400_);
v___x_418_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_415_, v_options_412_, v___x_417_);
lean_dec(v___x_417_);
if (v___x_418_ == 0)
{
lean_object* v___x_487_; uint8_t v___x_488_; 
v___x_487_ = l_Lean_trace_profiler;
v___x_488_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_options_412_, v___x_487_);
if (v___x_488_ == 0)
{
lean_object* v___x_489_; 
lean_dec_ref(v___f_403_);
lean_dec_ref(v___x_402_);
lean_dec(v___x_400_);
lean_inc(v___y_410_);
lean_inc_ref(v___y_409_);
lean_inc(v___y_408_);
lean_inc_ref(v___y_407_);
lean_inc(v___y_406_);
lean_inc_ref(v___y_405_);
lean_inc(v___y_404_);
v___x_489_ = lean_simp(v_a_399_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_489_;
}
else
{
goto v___jp_446_;
}
}
else
{
goto v___jp_446_;
}
v___jp_419_:
{
lean_object* v___x_423_; double v___x_424_; double v___x_425_; double v___x_426_; double v___x_427_; double v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_423_ = lean_io_mono_nanos_now();
v___x_424_ = lean_float_of_nat(v___y_420_);
v___x_425_ = lean_float_once(&lp_mathlib_eqComm___lam__1___closed__2, &lp_mathlib_eqComm___lam__1___closed__2_once, _init_lp_mathlib_eqComm___lam__1___closed__2);
v___x_426_ = lean_float_div(v___x_424_, v___x_425_);
v___x_427_ = lean_float_of_nat(v___x_423_);
v___x_428_ = lean_float_div(v___x_427_, v___x_425_);
v___x_429_ = lean_box_float(v___x_426_);
v___x_430_ = lean_box_float(v___x_428_);
v___x_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_431_, 0, v___x_429_);
lean_ctor_set(v___x_431_, 1, v___x_430_);
v___x_432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_432_, 0, v_a_422_);
lean_ctor_set(v___x_432_, 1, v___x_431_);
v___x_433_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(v___x_400_, v___x_401_, v___x_402_, v_options_412_, v___x_418_, v___y_421_, v___f_403_, v___x_432_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_433_;
}
v___jp_434_:
{
lean_object* v___x_438_; double v___x_439_; double v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_438_ = lean_io_get_num_heartbeats();
v___x_439_ = lean_float_of_nat(v___y_435_);
v___x_440_ = lean_float_of_nat(v___x_438_);
v___x_441_ = lean_box_float(v___x_439_);
v___x_442_ = lean_box_float(v___x_440_);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_441_);
lean_ctor_set(v___x_443_, 1, v___x_442_);
v___x_444_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_444_, 0, v_a_437_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(v___x_400_, v___x_401_, v___x_402_, v_options_412_, v___x_418_, v___y_436_, v___f_403_, v___x_444_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_445_;
}
v___jp_446_:
{
lean_object* v___x_447_; lean_object* v_a_448_; lean_object* v___x_449_; uint8_t v___x_450_; 
v___x_447_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(v___y_410_);
v_a_448_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_448_);
lean_dec_ref(v___x_447_);
v___x_449_ = l_Lean_trace_profiler_useHeartbeats;
v___x_450_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_options_412_, v___x_449_);
if (v___x_450_ == 0)
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = lean_io_mono_nanos_now();
lean_inc(v___y_410_);
lean_inc_ref(v___y_409_);
lean_inc(v___y_408_);
lean_inc_ref(v___y_407_);
lean_inc(v___y_406_);
lean_inc_ref(v___y_405_);
lean_inc(v___y_404_);
v___x_452_ = lean_simp(v_a_399_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
if (lean_obj_tag(v___x_452_) == 0)
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
v_a_453_ = lean_ctor_get(v___x_452_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_452_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_452_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
lean_ctor_set_tag(v___x_455_, 1);
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
v___y_420_ = v___x_451_;
v___y_421_ = v_a_448_;
v_a_422_ = v___x_458_;
goto v___jp_419_;
}
}
}
else
{
lean_object* v_a_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_468_; 
v_a_461_ = lean_ctor_get(v___x_452_, 0);
v_isSharedCheck_468_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_468_ == 0)
{
v___x_463_ = v___x_452_;
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_a_461_);
lean_dec(v___x_452_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_468_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_466_; 
if (v_isShared_464_ == 0)
{
lean_ctor_set_tag(v___x_463_, 0);
v___x_466_ = v___x_463_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v_a_461_);
v___x_466_ = v_reuseFailAlloc_467_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
v___y_420_ = v___x_451_;
v___y_421_ = v_a_448_;
v_a_422_ = v___x_466_;
goto v___jp_419_;
}
}
}
}
else
{
lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_469_ = lean_io_get_num_heartbeats();
lean_inc(v___y_410_);
lean_inc_ref(v___y_409_);
lean_inc(v___y_408_);
lean_inc_ref(v___y_407_);
lean_inc(v___y_406_);
lean_inc_ref(v___y_405_);
lean_inc(v___y_404_);
v___x_470_ = lean_simp(v_a_399_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
if (lean_obj_tag(v___x_470_) == 0)
{
lean_object* v_a_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_478_; 
v_a_471_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_478_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_478_ == 0)
{
v___x_473_ = v___x_470_;
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_a_471_);
lean_dec(v___x_470_);
v___x_473_ = lean_box(0);
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
v_resetjp_472_:
{
lean_object* v___x_476_; 
if (v_isShared_474_ == 0)
{
lean_ctor_set_tag(v___x_473_, 1);
v___x_476_ = v___x_473_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_a_471_);
v___x_476_ = v_reuseFailAlloc_477_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
v___y_435_ = v___x_469_;
v___y_436_ = v_a_448_;
v_a_437_ = v___x_476_;
goto v___jp_434_;
}
}
}
else
{
lean_object* v_a_479_; lean_object* v___x_481_; uint8_t v_isShared_482_; uint8_t v_isSharedCheck_486_; 
v_a_479_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_486_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_486_ == 0)
{
v___x_481_ = v___x_470_;
v_isShared_482_ = v_isSharedCheck_486_;
goto v_resetjp_480_;
}
else
{
lean_inc(v_a_479_);
lean_dec(v___x_470_);
v___x_481_ = lean_box(0);
v_isShared_482_ = v_isSharedCheck_486_;
goto v_resetjp_480_;
}
v_resetjp_480_:
{
lean_object* v___x_484_; 
if (v_isShared_482_ == 0)
{
lean_ctor_set_tag(v___x_481_, 0);
v___x_484_ = v___x_481_;
goto v_reusejp_483_;
}
else
{
lean_object* v_reuseFailAlloc_485_; 
v_reuseFailAlloc_485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_485_, 0, v_a_479_);
v___x_484_ = v_reuseFailAlloc_485_;
goto v_reusejp_483_;
}
v_reusejp_483_:
{
v___y_435_ = v___x_469_;
v___y_436_ = v_a_448_;
v_a_437_ = v___x_484_;
goto v___jp_434_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm___lam__1___boxed(lean_object* v_a_490_, lean_object* v___x_491_, lean_object* v___x_492_, lean_object* v___x_493_, lean_object* v___f_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
uint8_t v___x_45327__boxed_503_; lean_object* v_res_504_; 
v___x_45327__boxed_503_ = lean_unbox(v___x_492_);
v_res_504_ = lp_mathlib_eqComm___lam__1(v_a_490_, v___x_491_, v___x_45327__boxed_503_, v___x_493_, v___f_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
lean_dec(v___y_495_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm(lean_object* v_e_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_){
_start:
{
lean_object* v___y_626_; lean_object* v___x_629_; uint8_t v___x_630_; 
lean_inc_ref(v_e_613_);
v___x_629_ = l_Lean_Expr_cleanupAnnotations(v_e_613_);
v___x_630_ = l_Lean_Expr_isApp(v___x_629_);
if (v___x_630_ == 0)
{
lean_dec_ref(v___x_629_);
lean_dec_ref(v_e_613_);
goto v___jp_622_;
}
else
{
lean_object* v_arg_631_; lean_object* v___x_632_; uint8_t v___x_633_; 
v_arg_631_ = lean_ctor_get(v___x_629_, 1);
lean_inc_ref(v_arg_631_);
v___x_632_ = l_Lean_Expr_appFnCleanup___redArg(v___x_629_);
v___x_633_ = l_Lean_Expr_isApp(v___x_632_);
if (v___x_633_ == 0)
{
lean_dec_ref(v___x_632_);
lean_dec_ref(v_arg_631_);
lean_dec_ref(v_e_613_);
goto v___jp_622_;
}
else
{
lean_object* v_arg_634_; lean_object* v___x_635_; uint8_t v___x_636_; 
v_arg_634_ = lean_ctor_get(v___x_632_, 1);
lean_inc_ref(v_arg_634_);
v___x_635_ = l_Lean_Expr_appFnCleanup___redArg(v___x_632_);
v___x_636_ = l_Lean_Expr_isApp(v___x_635_);
if (v___x_636_ == 0)
{
lean_dec_ref(v___x_635_);
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
lean_dec_ref(v_e_613_);
goto v___jp_622_;
}
else
{
lean_object* v___x_637_; lean_object* v___x_638_; uint8_t v___x_639_; 
v___x_637_ = l_Lean_Expr_appFnCleanup___redArg(v___x_635_);
v___x_638_ = ((lean_object*)(lp_mathlib_eqComm___closed__2));
v___x_639_ = l_Lean_Expr_isConstOf(v___x_637_, v___x_638_);
lean_dec_ref(v___x_637_);
if (v___x_639_ == 0)
{
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
lean_dec_ref(v_e_613_);
goto v___jp_622_;
}
else
{
lean_object* v___x_640_; 
lean_inc_ref(v_arg_634_);
lean_inc_ref(v_arg_631_);
v___x_640_ = l_Lean_Meta_mkEq(v_arg_631_, v_arg_634_, v_a_617_, v_a_618_, v_a_619_, v_a_620_);
if (lean_obj_tag(v___x_640_) == 0)
{
lean_object* v_a_641_; lean_object* v___f_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___f_647_; lean_object* v___x_648_; 
v_a_641_ = lean_ctor_get(v___x_640_, 0);
lean_inc_n(v_a_641_, 2);
lean_dec_ref_known(v___x_640_, 1);
v___f_642_ = lean_alloc_closure((void*)(lp_mathlib_eqComm___lam__0___boxed), 10, 1);
lean_closure_set(v___f_642_, 0, v_e_613_);
v___x_643_ = ((lean_object*)(lp_mathlib_eqComm___closed__40));
v___x_644_ = ((lean_object*)(lp_mathlib_eqComm___closed__44));
v___x_645_ = ((lean_object*)(lp_mathlib_eqComm___closed__45));
v___x_646_ = lean_box(v___x_639_);
v___f_647_ = lean_alloc_closure((void*)(lp_mathlib_eqComm___lam__1___boxed), 13, 5);
lean_closure_set(v___f_647_, 0, v_a_641_);
lean_closure_set(v___f_647_, 1, v___x_644_);
lean_closure_set(v___f_647_, 2, v___x_646_);
lean_closure_set(v___f_647_, 3, v___x_645_);
lean_closure_set(v___f_647_, 4, v___f_642_);
v___x_648_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(v___x_643_, v___f_647_, v_a_614_, v_a_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_);
if (lean_obj_tag(v___x_648_) == 0)
{
lean_object* v_a_649_; lean_object* v_expr_650_; lean_object* v___x_651_; 
v_a_649_ = lean_ctor_get(v___x_648_, 0);
lean_inc(v_a_649_);
lean_dec_ref_known(v___x_648_, 1);
v_expr_650_ = lean_ctor_get(v_a_649_, 0);
lean_inc_ref_n(v_expr_650_, 2);
v___x_651_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_expr_650_, v_a_618_);
if (lean_obj_tag(v___x_651_) == 0)
{
lean_object* v_a_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_772_; 
v_a_652_ = lean_ctor_get(v___x_651_, 0);
v_isSharedCheck_772_ = !lean_is_exclusive(v___x_651_);
if (v_isSharedCheck_772_ == 0)
{
v___x_654_ = v___x_651_;
v_isShared_655_ = v_isSharedCheck_772_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_a_652_);
lean_dec(v___x_651_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_772_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___y_657_; lean_object* v___y_658_; lean_object* v___y_659_; lean_object* v___y_660_; uint8_t v___y_751_; lean_object* v___x_756_; uint8_t v___x_757_; 
v___x_756_ = l_Lean_Expr_cleanupAnnotations(v_a_652_);
v___x_757_ = l_Lean_Expr_isApp(v___x_756_);
if (v___x_757_ == 0)
{
lean_dec_ref(v___x_756_);
lean_del_object(v___x_654_);
v___y_657_ = v_a_617_;
v___y_658_ = v_a_618_;
v___y_659_ = v_a_619_;
v___y_660_ = v_a_620_;
goto v___jp_656_;
}
else
{
lean_object* v_arg_758_; lean_object* v___x_759_; uint8_t v___x_760_; 
v_arg_758_ = lean_ctor_get(v___x_756_, 1);
lean_inc_ref(v_arg_758_);
v___x_759_ = l_Lean_Expr_appFnCleanup___redArg(v___x_756_);
v___x_760_ = l_Lean_Expr_isApp(v___x_759_);
if (v___x_760_ == 0)
{
lean_dec_ref(v___x_759_);
lean_dec_ref(v_arg_758_);
lean_del_object(v___x_654_);
v___y_657_ = v_a_617_;
v___y_658_ = v_a_618_;
v___y_659_ = v_a_619_;
v___y_660_ = v_a_620_;
goto v___jp_656_;
}
else
{
lean_object* v_arg_761_; uint8_t v___y_763_; lean_object* v___x_766_; uint8_t v___x_767_; 
v_arg_761_ = lean_ctor_get(v___x_759_, 1);
lean_inc_ref(v_arg_761_);
v___x_766_ = l_Lean_Expr_appFnCleanup___redArg(v___x_759_);
v___x_767_ = l_Lean_Expr_isApp(v___x_766_);
if (v___x_767_ == 0)
{
lean_dec_ref(v___x_766_);
lean_dec_ref(v_arg_761_);
lean_dec_ref(v_arg_758_);
lean_del_object(v___x_654_);
v___y_657_ = v_a_617_;
v___y_658_ = v_a_618_;
v___y_659_ = v_a_619_;
v___y_660_ = v_a_620_;
goto v___jp_656_;
}
else
{
lean_object* v___x_768_; uint8_t v___x_769_; 
v___x_768_ = l_Lean_Expr_appFnCleanup___redArg(v___x_766_);
v___x_769_ = l_Lean_Expr_isConstOf(v___x_768_, v___x_638_);
lean_dec_ref(v___x_768_);
if (v___x_769_ == 0)
{
lean_dec_ref(v_arg_761_);
lean_dec_ref(v_arg_758_);
lean_del_object(v___x_654_);
v___y_657_ = v_a_617_;
v___y_658_ = v_a_618_;
v___y_659_ = v_a_619_;
v___y_660_ = v_a_620_;
goto v___jp_656_;
}
else
{
uint8_t v___x_770_; 
v___x_770_ = lean_expr_eqv(v_arg_761_, v_arg_631_);
if (v___x_770_ == 0)
{
v___y_763_ = v___x_770_;
goto v___jp_762_;
}
else
{
uint8_t v___x_771_; 
v___x_771_ = lean_expr_eqv(v_arg_758_, v_arg_634_);
v___y_763_ = v___x_771_;
goto v___jp_762_;
}
}
}
v___jp_762_:
{
if (v___y_763_ == 0)
{
uint8_t v___x_764_; 
v___x_764_ = lean_expr_eqv(v_arg_761_, v_arg_634_);
lean_dec_ref(v_arg_761_);
if (v___x_764_ == 0)
{
lean_dec_ref(v_arg_758_);
v___y_751_ = v___x_764_;
goto v___jp_750_;
}
else
{
uint8_t v___x_765_; 
v___x_765_ = lean_expr_eqv(v_arg_758_, v_arg_631_);
lean_dec_ref(v_arg_758_);
v___y_751_ = v___x_765_;
goto v___jp_750_;
}
}
else
{
lean_dec_ref(v_arg_761_);
lean_dec_ref(v_arg_758_);
v___y_751_ = v___x_639_;
goto v___jp_750_;
}
}
}
}
v___jp_656_:
{
lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_661_ = ((lean_object*)(lp_mathlib_eqComm___closed__47));
v___x_662_ = lean_unsigned_to_nat(2u);
v___x_663_ = lean_mk_empty_array_with_capacity(v___x_662_);
lean_inc_ref(v___x_663_);
v___x_664_ = lean_array_push(v___x_663_, v_arg_634_);
v___x_665_ = lean_array_push(v___x_664_, v_arg_631_);
v___x_666_ = l_Lean_Meta_mkAppM(v___x_661_, v___x_665_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_666_) == 0)
{
lean_object* v_a_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
lean_inc(v_a_667_);
lean_dec_ref_known(v___x_666_, 1);
v___x_668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_668_, 0, v_a_667_);
v___x_669_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_669_, 0, v_a_641_);
lean_ctor_set(v___x_669_, 1, v___x_668_);
lean_ctor_set_uint8(v___x_669_, sizeof(void*)*2, v___x_639_);
v___x_670_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_669_, v_a_649_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_670_) == 0)
{
lean_object* v_a_671_; lean_object* v___x_672_; 
v_a_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_a_671_);
lean_dec_ref_known(v___x_670_, 1);
v___x_672_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_expr_650_, v___y_658_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v___x_674_; uint8_t v___x_675_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v___x_674_ = l_Lean_Expr_cleanupAnnotations(v_a_673_);
v___x_675_ = l_Lean_Expr_isApp(v___x_674_);
if (v___x_675_ == 0)
{
lean_dec_ref(v___x_674_);
lean_dec_ref(v___x_663_);
v___y_626_ = v_a_671_;
goto v___jp_625_;
}
else
{
lean_object* v_arg_676_; lean_object* v___x_677_; uint8_t v___x_678_; 
v_arg_676_ = lean_ctor_get(v___x_674_, 1);
lean_inc_ref(v_arg_676_);
v___x_677_ = l_Lean_Expr_appFnCleanup___redArg(v___x_674_);
v___x_678_ = l_Lean_Expr_isApp(v___x_677_);
if (v___x_678_ == 0)
{
lean_dec_ref(v___x_677_);
lean_dec_ref(v_arg_676_);
lean_dec_ref(v___x_663_);
v___y_626_ = v_a_671_;
goto v___jp_625_;
}
else
{
lean_object* v_arg_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v_arg_679_ = lean_ctor_get(v___x_677_, 1);
lean_inc_ref(v_arg_679_);
v___x_680_ = l_Lean_Expr_appFnCleanup___redArg(v___x_677_);
v___x_681_ = l_Lean_Expr_isApp(v___x_680_);
if (v___x_681_ == 0)
{
lean_dec_ref(v___x_680_);
lean_dec_ref(v_arg_679_);
lean_dec_ref(v_arg_676_);
lean_dec_ref(v___x_663_);
v___y_626_ = v_a_671_;
goto v___jp_625_;
}
else
{
lean_object* v___x_682_; uint8_t v___x_683_; 
v___x_682_ = l_Lean_Expr_appFnCleanup___redArg(v___x_680_);
v___x_683_ = l_Lean_Expr_isConstOf(v___x_682_, v___x_638_);
lean_dec_ref(v___x_682_);
if (v___x_683_ == 0)
{
lean_dec_ref(v_arg_679_);
lean_dec_ref(v_arg_676_);
lean_dec_ref(v___x_663_);
v___y_626_ = v_a_671_;
goto v___jp_625_;
}
else
{
lean_object* v___x_684_; 
lean_inc_ref(v_arg_679_);
lean_inc_ref(v_arg_676_);
v___x_684_ = l_Lean_Meta_mkEq(v_arg_676_, v_arg_679_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_684_) == 0)
{
lean_object* v_a_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; 
v_a_685_ = lean_ctor_get(v___x_684_, 0);
lean_inc(v_a_685_);
lean_dec_ref_known(v___x_684_, 1);
v___x_686_ = lean_array_push(v___x_663_, v_arg_679_);
v___x_687_ = lean_array_push(v___x_686_, v_arg_676_);
v___x_688_ = l_Lean_Meta_mkAppM(v___x_661_, v___x_687_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_688_) == 0)
{
lean_object* v_a_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; 
v_a_689_ = lean_ctor_get(v___x_688_, 0);
lean_inc(v_a_689_);
lean_dec_ref_known(v___x_688_, 1);
v___x_690_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_690_, 0, v_a_689_);
v___x_691_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_691_, 0, v_a_685_);
lean_ctor_set(v___x_691_, 1, v___x_690_);
lean_ctor_set_uint8(v___x_691_, sizeof(void*)*2, v___x_639_);
v___x_692_ = l_Lean_Meta_Simp_Result_mkEqTrans(v_a_671_, v___x_691_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_692_) == 0)
{
lean_object* v_a_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_701_; 
v_a_693_ = lean_ctor_get(v___x_692_, 0);
v_isSharedCheck_701_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_701_ == 0)
{
v___x_695_ = v___x_692_;
v_isShared_696_ = v_isSharedCheck_701_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_a_693_);
lean_dec(v___x_692_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_701_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v___x_697_; lean_object* v___x_699_; 
v___x_697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_697_, 0, v_a_693_);
if (v_isShared_696_ == 0)
{
lean_ctor_set(v___x_695_, 0, v___x_697_);
v___x_699_ = v___x_695_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v___x_697_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
else
{
lean_object* v_a_702_; lean_object* v___x_704_; uint8_t v_isShared_705_; uint8_t v_isSharedCheck_709_; 
v_a_702_ = lean_ctor_get(v___x_692_, 0);
v_isSharedCheck_709_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_709_ == 0)
{
v___x_704_ = v___x_692_;
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
else
{
lean_inc(v_a_702_);
lean_dec(v___x_692_);
v___x_704_ = lean_box(0);
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
v_resetjp_703_:
{
lean_object* v___x_707_; 
if (v_isShared_705_ == 0)
{
v___x_707_ = v___x_704_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v_a_702_);
v___x_707_ = v_reuseFailAlloc_708_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
return v___x_707_;
}
}
}
}
else
{
lean_object* v_a_710_; lean_object* v___x_712_; uint8_t v_isShared_713_; uint8_t v_isSharedCheck_717_; 
lean_dec(v_a_685_);
lean_dec(v_a_671_);
v_a_710_ = lean_ctor_get(v___x_688_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v___x_688_);
if (v_isSharedCheck_717_ == 0)
{
v___x_712_ = v___x_688_;
v_isShared_713_ = v_isSharedCheck_717_;
goto v_resetjp_711_;
}
else
{
lean_inc(v_a_710_);
lean_dec(v___x_688_);
v___x_712_ = lean_box(0);
v_isShared_713_ = v_isSharedCheck_717_;
goto v_resetjp_711_;
}
v_resetjp_711_:
{
lean_object* v___x_715_; 
if (v_isShared_713_ == 0)
{
v___x_715_ = v___x_712_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v_a_710_);
v___x_715_ = v_reuseFailAlloc_716_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
return v___x_715_;
}
}
}
}
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_725_; 
lean_dec_ref(v_arg_679_);
lean_dec_ref(v_arg_676_);
lean_dec(v_a_671_);
lean_dec_ref(v___x_663_);
v_a_718_ = lean_ctor_get(v___x_684_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_684_);
if (v_isSharedCheck_725_ == 0)
{
v___x_720_ = v___x_684_;
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_684_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_718_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
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
lean_object* v_a_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_733_; 
lean_dec(v_a_671_);
lean_dec_ref(v___x_663_);
v_a_726_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_733_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_733_ == 0)
{
v___x_728_ = v___x_672_;
v_isShared_729_ = v_isSharedCheck_733_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_a_726_);
lean_dec(v___x_672_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_733_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v___x_731_; 
if (v_isShared_729_ == 0)
{
v___x_731_ = v___x_728_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_a_726_);
v___x_731_ = v_reuseFailAlloc_732_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
return v___x_731_;
}
}
}
}
else
{
lean_object* v_a_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_741_; 
lean_dec_ref(v___x_663_);
lean_dec_ref(v_expr_650_);
v_a_734_ = lean_ctor_get(v___x_670_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_741_ == 0)
{
v___x_736_ = v___x_670_;
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_a_734_);
lean_dec(v___x_670_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v___x_739_; 
if (v_isShared_737_ == 0)
{
v___x_739_ = v___x_736_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_a_734_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_dec_ref(v___x_663_);
lean_dec_ref(v_expr_650_);
lean_dec(v_a_649_);
lean_dec(v_a_641_);
v_a_742_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_666_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_666_);
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
v___jp_750_:
{
if (v___y_751_ == 0)
{
lean_del_object(v___x_654_);
v___y_657_ = v_a_617_;
v___y_658_ = v_a_618_;
v___y_659_ = v_a_619_;
v___y_660_ = v_a_620_;
goto v___jp_656_;
}
else
{
lean_object* v___x_752_; lean_object* v___x_754_; 
lean_dec_ref(v_expr_650_);
lean_dec(v_a_649_);
lean_dec(v_a_641_);
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
v___x_752_ = ((lean_object*)(lp_mathlib_eqComm___closed__0));
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 0, v___x_752_);
v___x_754_ = v___x_654_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v___x_752_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
return v___x_754_;
}
}
}
}
}
else
{
lean_object* v_a_773_; lean_object* v___x_775_; uint8_t v_isShared_776_; uint8_t v_isSharedCheck_780_; 
lean_dec_ref(v_expr_650_);
lean_dec(v_a_649_);
lean_dec(v_a_641_);
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
v_a_773_ = lean_ctor_get(v___x_651_, 0);
v_isSharedCheck_780_ = !lean_is_exclusive(v___x_651_);
if (v_isSharedCheck_780_ == 0)
{
v___x_775_ = v___x_651_;
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
else
{
lean_inc(v_a_773_);
lean_dec(v___x_651_);
v___x_775_ = lean_box(0);
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
v_resetjp_774_:
{
lean_object* v___x_778_; 
if (v_isShared_776_ == 0)
{
v___x_778_ = v___x_775_;
goto v_reusejp_777_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v_a_773_);
v___x_778_ = v_reuseFailAlloc_779_;
goto v_reusejp_777_;
}
v_reusejp_777_:
{
return v___x_778_;
}
}
}
}
else
{
lean_object* v_a_781_; lean_object* v___x_783_; uint8_t v_isShared_784_; uint8_t v_isSharedCheck_788_; 
lean_dec(v_a_641_);
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
v_a_781_ = lean_ctor_get(v___x_648_, 0);
v_isSharedCheck_788_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_788_ == 0)
{
v___x_783_ = v___x_648_;
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
else
{
lean_inc(v_a_781_);
lean_dec(v___x_648_);
v___x_783_ = lean_box(0);
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
v_resetjp_782_:
{
lean_object* v___x_786_; 
if (v_isShared_784_ == 0)
{
v___x_786_ = v___x_783_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_a_781_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
}
}
else
{
lean_object* v_a_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_796_; 
lean_dec_ref(v_arg_634_);
lean_dec_ref(v_arg_631_);
lean_dec_ref(v_e_613_);
v_a_789_ = lean_ctor_get(v___x_640_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v___x_640_);
if (v_isSharedCheck_796_ == 0)
{
v___x_791_ = v___x_640_;
v_isShared_792_ = v_isSharedCheck_796_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_a_789_);
lean_dec(v___x_640_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_796_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_794_; 
if (v_isShared_792_ == 0)
{
v___x_794_ = v___x_791_;
goto v_reusejp_793_;
}
else
{
lean_object* v_reuseFailAlloc_795_; 
v_reuseFailAlloc_795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_795_, 0, v_a_789_);
v___x_794_ = v_reuseFailAlloc_795_;
goto v_reusejp_793_;
}
v_reusejp_793_:
{
return v___x_794_;
}
}
}
}
}
}
}
v___jp_622_:
{
lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_623_ = ((lean_object*)(lp_mathlib_eqComm___closed__0));
v___x_624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_624_, 0, v___x_623_);
return v___x_624_;
}
v___jp_625_:
{
lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_627_, 0, v___y_626_);
v___x_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
return v___x_628_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_eqComm___boxed(lean_object* v_e_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib_eqComm(v_e_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_, v_a_803_, v_a_804_);
lean_dec(v_a_804_);
lean_dec_ref(v_a_803_);
lean_dec(v_a_802_);
lean_dec_ref(v_a_801_);
lean_dec(v_a_800_);
lean_dec_ref(v_a_799_);
lean_dec(v_a_798_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3(lean_object* v_00_u03b1_807_, lean_object* v_x_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___redArg(v_x_808_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3___boxed(lean_object* v_00_u03b1_818_, lean_object* v_x_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_){
_start:
{
lean_object* v_res_828_; 
v_res_828_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__3(v_00_u03b1_818_, v_x_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
lean_dec(v___y_824_);
lean_dec_ref(v___y_823_);
lean_dec(v___y_822_);
lean_dec_ref(v___y_821_);
lean_dec(v___y_820_);
return v_res_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2(lean_object* v_oldTraces_829_, lean_object* v_data_830_, lean_object* v_ref_831_, lean_object* v_msg_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v___x_841_; 
v___x_841_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___redArg(v_oldTraces_829_, v_data_830_, v_ref_831_, v_msg_832_, v___y_836_, v___y_837_, v___y_838_, v___y_839_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2___boxed(lean_object* v_oldTraces_842_, lean_object* v_data_843_, lean_object* v_ref_844_, lean_object* v_msg_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2_spec__2(v_oldTraces_842_, v_data_843_, v_ref_844_, v_msg_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___y_846_);
return v_res_854_;
}
}
static lean_object* _init_lp_mathlib_iffComm___lam__0___closed__1(void){
_start:
{
lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_856_ = ((lean_object*)(lp_mathlib_iffComm___lam__0___closed__0));
v___x_857_ = l_Lean_stringToMessageData(v___x_856_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__0(lean_object* v_e_858_, lean_object* v_x_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; 
v___x_868_ = lean_obj_once(&lp_mathlib_iffComm___lam__0___closed__1, &lp_mathlib_iffComm___lam__0___closed__1_once, _init_lp_mathlib_iffComm___lam__0___closed__1);
v___x_869_ = l_Lean_MessageData_ofExpr(v_e_858_);
v___x_870_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_870_, 0, v___x_868_);
lean_ctor_set(v___x_870_, 1, v___x_869_);
v___x_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
return v___x_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__0___boxed(lean_object* v_e_872_, lean_object* v_x_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib_iffComm___lam__0(v_e_872_, v_x_873_, v___y_874_, v___y_875_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_);
lean_dec(v___y_880_);
lean_dec_ref(v___y_879_);
lean_dec(v___y_878_);
lean_dec_ref(v___y_877_);
lean_dec(v___y_876_);
lean_dec_ref(v___y_875_);
lean_dec(v___y_874_);
lean_dec_ref(v_x_873_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__1(lean_object* v_symmExpr_883_, lean_object* v___x_884_, uint8_t v___x_885_, lean_object* v___x_886_, lean_object* v___f_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_){
_start:
{
lean_object* v_options_896_; uint8_t v_hasTrace_897_; 
v_options_896_ = lean_ctor_get(v___y_893_, 2);
v_hasTrace_897_ = lean_ctor_get_uint8(v_options_896_, sizeof(void*)*1);
if (v_hasTrace_897_ == 0)
{
lean_object* v___x_898_; 
lean_dec_ref(v___f_887_);
lean_dec_ref(v___x_886_);
lean_dec(v___x_884_);
lean_inc(v___y_894_);
lean_inc_ref(v___y_893_);
lean_inc(v___y_892_);
lean_inc_ref(v___y_891_);
lean_inc(v___y_890_);
lean_inc_ref(v___y_889_);
lean_inc(v___y_888_);
v___x_898_ = lean_simp(v_symmExpr_883_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
return v___x_898_;
}
else
{
lean_object* v_inheritedTraceOptions_899_; lean_object* v___x_900_; lean_object* v___x_901_; uint8_t v___x_902_; lean_object* v___y_904_; lean_object* v___y_905_; lean_object* v_a_906_; lean_object* v___y_919_; lean_object* v___y_920_; lean_object* v_a_921_; 
v_inheritedTraceOptions_899_ = lean_ctor_get(v___y_893_, 13);
v___x_900_ = ((lean_object*)(lp_mathlib_eqComm___lam__1___closed__1));
lean_inc(v___x_884_);
v___x_901_ = l_Lean_Name_append(v___x_900_, v___x_884_);
v___x_902_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_899_, v_options_896_, v___x_901_);
lean_dec(v___x_901_);
if (v___x_902_ == 0)
{
lean_object* v___x_971_; uint8_t v___x_972_; 
v___x_971_ = l_Lean_trace_profiler;
v___x_972_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_options_896_, v___x_971_);
if (v___x_972_ == 0)
{
lean_object* v___x_973_; 
lean_dec_ref(v___f_887_);
lean_dec_ref(v___x_886_);
lean_dec(v___x_884_);
lean_inc(v___y_894_);
lean_inc_ref(v___y_893_);
lean_inc(v___y_892_);
lean_inc_ref(v___y_891_);
lean_inc(v___y_890_);
lean_inc_ref(v___y_889_);
lean_inc(v___y_888_);
v___x_973_ = lean_simp(v_symmExpr_883_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
return v___x_973_;
}
else
{
goto v___jp_930_;
}
}
else
{
goto v___jp_930_;
}
v___jp_903_:
{
lean_object* v___x_907_; double v___x_908_; double v___x_909_; double v___x_910_; double v___x_911_; double v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; 
v___x_907_ = lean_io_mono_nanos_now();
v___x_908_ = lean_float_of_nat(v___y_904_);
v___x_909_ = lean_float_once(&lp_mathlib_eqComm___lam__1___closed__2, &lp_mathlib_eqComm___lam__1___closed__2_once, _init_lp_mathlib_eqComm___lam__1___closed__2);
v___x_910_ = lean_float_div(v___x_908_, v___x_909_);
v___x_911_ = lean_float_of_nat(v___x_907_);
v___x_912_ = lean_float_div(v___x_911_, v___x_909_);
v___x_913_ = lean_box_float(v___x_910_);
v___x_914_ = lean_box_float(v___x_912_);
v___x_915_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_915_, 0, v___x_913_);
lean_ctor_set(v___x_915_, 1, v___x_914_);
v___x_916_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_916_, 0, v_a_906_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v___x_917_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(v___x_884_, v___x_885_, v___x_886_, v_options_896_, v___x_902_, v___y_905_, v___f_887_, v___x_916_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
return v___x_917_;
}
v___jp_918_:
{
lean_object* v___x_922_; double v___x_923_; double v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v___x_922_ = lean_io_get_num_heartbeats();
v___x_923_ = lean_float_of_nat(v___y_919_);
v___x_924_ = lean_float_of_nat(v___x_922_);
v___x_925_ = lean_box_float(v___x_923_);
v___x_926_ = lean_box_float(v___x_924_);
v___x_927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_927_, 0, v___x_925_);
lean_ctor_set(v___x_927_, 1, v___x_926_);
v___x_928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_928_, 0, v_a_921_);
lean_ctor_set(v___x_928_, 1, v___x_927_);
v___x_929_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00eqComm_spec__2(v___x_884_, v___x_885_, v___x_886_, v_options_896_, v___x_902_, v___y_920_, v___f_887_, v___x_928_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
return v___x_929_;
}
v___jp_930_:
{
lean_object* v___x_931_; lean_object* v_a_932_; lean_object* v___x_933_; uint8_t v___x_934_; 
v___x_931_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00eqComm_spec__0___redArg(v___y_894_);
v_a_932_ = lean_ctor_get(v___x_931_, 0);
lean_inc(v_a_932_);
lean_dec_ref(v___x_931_);
v___x_933_ = l_Lean_trace_profiler_useHeartbeats;
v___x_934_ = lp_mathlib_Lean_Option_get___at___00eqComm_spec__1(v_options_896_, v___x_933_);
if (v___x_934_ == 0)
{
lean_object* v___x_935_; lean_object* v___x_936_; 
v___x_935_ = lean_io_mono_nanos_now();
lean_inc(v___y_894_);
lean_inc_ref(v___y_893_);
lean_inc(v___y_892_);
lean_inc_ref(v___y_891_);
lean_inc(v___y_890_);
lean_inc_ref(v___y_889_);
lean_inc(v___y_888_);
v___x_936_ = lean_simp(v_symmExpr_883_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
if (lean_obj_tag(v___x_936_) == 0)
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_944_; 
v_a_937_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_944_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_944_ == 0)
{
v___x_939_ = v___x_936_;
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_936_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_942_; 
if (v_isShared_940_ == 0)
{
lean_ctor_set_tag(v___x_939_, 1);
v___x_942_ = v___x_939_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_a_937_);
v___x_942_ = v_reuseFailAlloc_943_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
v___y_904_ = v___x_935_;
v___y_905_ = v_a_932_;
v_a_906_ = v___x_942_;
goto v___jp_903_;
}
}
}
else
{
lean_object* v_a_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_952_; 
v_a_945_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_952_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_952_ == 0)
{
v___x_947_ = v___x_936_;
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_a_945_);
lean_dec(v___x_936_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_950_; 
if (v_isShared_948_ == 0)
{
lean_ctor_set_tag(v___x_947_, 0);
v___x_950_ = v___x_947_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v_a_945_);
v___x_950_ = v_reuseFailAlloc_951_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
v___y_904_ = v___x_935_;
v___y_905_ = v_a_932_;
v_a_906_ = v___x_950_;
goto v___jp_903_;
}
}
}
}
else
{
lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_953_ = lean_io_get_num_heartbeats();
lean_inc(v___y_894_);
lean_inc_ref(v___y_893_);
lean_inc(v___y_892_);
lean_inc_ref(v___y_891_);
lean_inc(v___y_890_);
lean_inc_ref(v___y_889_);
lean_inc(v___y_888_);
v___x_954_ = lean_simp(v_symmExpr_883_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_);
if (lean_obj_tag(v___x_954_) == 0)
{
lean_object* v_a_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_962_; 
v_a_955_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_962_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_962_ == 0)
{
v___x_957_ = v___x_954_;
v_isShared_958_ = v_isSharedCheck_962_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_a_955_);
lean_dec(v___x_954_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_962_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v___x_960_; 
if (v_isShared_958_ == 0)
{
lean_ctor_set_tag(v___x_957_, 1);
v___x_960_ = v___x_957_;
goto v_reusejp_959_;
}
else
{
lean_object* v_reuseFailAlloc_961_; 
v_reuseFailAlloc_961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_961_, 0, v_a_955_);
v___x_960_ = v_reuseFailAlloc_961_;
goto v_reusejp_959_;
}
v_reusejp_959_:
{
v___y_919_ = v___x_953_;
v___y_920_ = v_a_932_;
v_a_921_ = v___x_960_;
goto v___jp_918_;
}
}
}
else
{
lean_object* v_a_963_; lean_object* v___x_965_; uint8_t v_isShared_966_; uint8_t v_isSharedCheck_970_; 
v_a_963_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_970_ == 0)
{
v___x_965_ = v___x_954_;
v_isShared_966_ = v_isSharedCheck_970_;
goto v_resetjp_964_;
}
else
{
lean_inc(v_a_963_);
lean_dec(v___x_954_);
v___x_965_ = lean_box(0);
v_isShared_966_ = v_isSharedCheck_970_;
goto v_resetjp_964_;
}
v_resetjp_964_:
{
lean_object* v___x_968_; 
if (v_isShared_966_ == 0)
{
lean_ctor_set_tag(v___x_965_, 0);
v___x_968_ = v___x_965_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v_a_963_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
v___y_919_ = v___x_953_;
v___y_920_ = v_a_932_;
v_a_921_ = v___x_968_;
goto v___jp_918_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm___lam__1___boxed(lean_object* v_symmExpr_974_, lean_object* v___x_975_, lean_object* v___x_976_, lean_object* v___x_977_, lean_object* v___f_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_){
_start:
{
uint8_t v___x_20954__boxed_987_; lean_object* v_res_988_; 
v___x_20954__boxed_987_ = lean_unbox(v___x_976_);
v_res_988_ = lp_mathlib_iffComm___lam__1(v_symmExpr_974_, v___x_975_, v___x_20954__boxed_987_, v___x_977_, v___f_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_);
lean_dec(v___y_985_);
lean_dec_ref(v___y_984_);
lean_dec(v___y_983_);
lean_dec_ref(v___y_982_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec(v___y_979_);
return v_res_988_;
}
}
static lean_object* _init_lp_mathlib_iffComm___closed__2(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; 
v___x_992_ = lean_box(0);
v___x_993_ = ((lean_object*)(lp_mathlib_iffComm___closed__1));
v___x_994_ = l_Lean_Expr_const___override(v___x_993_, v___x_992_);
return v___x_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm(lean_object* v_e_1042_, lean_object* v_a_1043_, lean_object* v_a_1044_, lean_object* v_a_1045_, lean_object* v_a_1046_, lean_object* v_a_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_){
_start:
{
lean_object* v___y_1052_; lean_object* v___x_1058_; uint8_t v___x_1059_; 
lean_inc_ref(v_e_1042_);
v___x_1058_ = l_Lean_Expr_cleanupAnnotations(v_e_1042_);
v___x_1059_ = l_Lean_Expr_isApp(v___x_1058_);
if (v___x_1059_ == 0)
{
lean_dec_ref(v___x_1058_);
lean_dec_ref(v_e_1042_);
goto v___jp_1055_;
}
else
{
lean_object* v_arg_1060_; lean_object* v___x_1061_; uint8_t v___x_1062_; 
v_arg_1060_ = lean_ctor_get(v___x_1058_, 1);
lean_inc_ref(v_arg_1060_);
v___x_1061_ = l_Lean_Expr_appFnCleanup___redArg(v___x_1058_);
v___x_1062_ = l_Lean_Expr_isApp(v___x_1061_);
if (v___x_1062_ == 0)
{
lean_dec_ref(v___x_1061_);
lean_dec_ref(v_arg_1060_);
lean_dec_ref(v_e_1042_);
goto v___jp_1055_;
}
else
{
lean_object* v_arg_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; uint8_t v___x_1066_; 
v_arg_1063_ = lean_ctor_get(v___x_1061_, 1);
lean_inc_ref(v_arg_1063_);
v___x_1064_ = l_Lean_Expr_appFnCleanup___redArg(v___x_1061_);
v___x_1065_ = ((lean_object*)(lp_mathlib_iffComm___closed__1));
v___x_1066_ = l_Lean_Expr_isConstOf(v___x_1064_, v___x_1065_);
lean_dec_ref(v___x_1064_);
if (v___x_1066_ == 0)
{
lean_dec_ref(v_arg_1063_);
lean_dec_ref(v_arg_1060_);
lean_dec_ref(v_e_1042_);
goto v___jp_1055_;
}
else
{
lean_object* v___f_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v_symmExpr_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___f_1075_; lean_object* v___x_1076_; 
lean_inc_ref(v_e_1042_);
v___f_1067_ = lean_alloc_closure((void*)(lp_mathlib_iffComm___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1067_, 0, v_e_1042_);
v___x_1068_ = lean_obj_once(&lp_mathlib_iffComm___closed__2, &lp_mathlib_iffComm___closed__2_once, _init_lp_mathlib_iffComm___closed__2);
lean_inc_ref(v_arg_1060_);
v___x_1069_ = l_Lean_Expr_app___override(v___x_1068_, v_arg_1060_);
lean_inc_ref(v_arg_1063_);
v_symmExpr_1070_ = l_Lean_Expr_app___override(v___x_1069_, v_arg_1063_);
v___x_1071_ = ((lean_object*)(lp_mathlib_iffComm___closed__19));
v___x_1072_ = ((lean_object*)(lp_mathlib_eqComm___closed__44));
v___x_1073_ = ((lean_object*)(lp_mathlib_eqComm___closed__45));
v___x_1074_ = lean_box(v___x_1066_);
lean_inc_ref(v_symmExpr_1070_);
v___f_1075_ = lean_alloc_closure((void*)(lp_mathlib_iffComm___lam__1___boxed), 13, 5);
lean_closure_set(v___f_1075_, 0, v_symmExpr_1070_);
lean_closure_set(v___f_1075_, 1, v___x_1072_);
lean_closure_set(v___f_1075_, 2, v___x_1074_);
lean_closure_set(v___f_1075_, 3, v___x_1073_);
lean_closure_set(v___f_1075_, 4, v___f_1067_);
v___x_1076_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(v___x_1071_, v___f_1075_, v_a_1043_, v_a_1044_, v_a_1045_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_);
if (lean_obj_tag(v___x_1076_) == 0)
{
lean_object* v_a_1077_; lean_object* v___x_1079_; uint8_t v_isShared_1080_; uint8_t v_isSharedCheck_1169_; 
v_a_1077_ = lean_ctor_get(v___x_1076_, 0);
v_isSharedCheck_1169_ = !lean_is_exclusive(v___x_1076_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1079_ = v___x_1076_;
v_isShared_1080_ = v_isSharedCheck_1169_;
goto v_resetjp_1078_;
}
else
{
lean_inc(v_a_1077_);
lean_dec(v___x_1076_);
v___x_1079_ = lean_box(0);
v_isShared_1080_ = v_isSharedCheck_1169_;
goto v_resetjp_1078_;
}
v_resetjp_1078_:
{
lean_object* v_expr_1081_; uint8_t v___y_1083_; uint8_t v___x_1167_; 
v_expr_1081_ = lean_ctor_get(v_a_1077_, 0);
lean_inc_ref(v_expr_1081_);
v___x_1167_ = lean_expr_eqv(v_expr_1081_, v_symmExpr_1070_);
if (v___x_1167_ == 0)
{
uint8_t v___x_1168_; 
v___x_1168_ = lean_expr_eqv(v_expr_1081_, v_e_1042_);
lean_dec_ref(v_e_1042_);
v___y_1083_ = v___x_1168_;
goto v___jp_1082_;
}
else
{
lean_dec_ref(v_e_1042_);
v___y_1083_ = v___x_1066_;
goto v___jp_1082_;
}
v___jp_1082_:
{
if (v___y_1083_ == 0)
{
lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
lean_del_object(v___x_1079_);
v___x_1084_ = ((lean_object*)(lp_mathlib_iffComm___closed__21));
v___x_1085_ = lean_unsigned_to_nat(2u);
v___x_1086_ = lean_mk_empty_array_with_capacity(v___x_1085_);
lean_inc_ref(v___x_1086_);
v___x_1087_ = lean_array_push(v___x_1086_, v_arg_1063_);
v___x_1088_ = lean_array_push(v___x_1087_, v_arg_1060_);
v___x_1089_ = l_Lean_Meta_mkAppM(v___x_1084_, v___x_1088_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_);
if (lean_obj_tag(v___x_1089_) == 0)
{
lean_object* v_a_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; 
v_a_1090_ = lean_ctor_get(v___x_1089_, 0);
lean_inc(v_a_1090_);
lean_dec_ref_known(v___x_1089_, 1);
v___x_1091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1091_, 0, v_a_1090_);
v___x_1092_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1092_, 0, v_symmExpr_1070_);
lean_ctor_set(v___x_1092_, 1, v___x_1091_);
lean_ctor_set_uint8(v___x_1092_, sizeof(void*)*2, v___x_1066_);
v___x_1093_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_1092_, v_a_1077_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_);
if (lean_obj_tag(v___x_1093_) == 0)
{
lean_object* v_a_1094_; lean_object* v___x_1095_; 
v_a_1094_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_a_1094_);
lean_dec_ref_known(v___x_1093_, 1);
v___x_1095_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_expr_1081_, v_a_1047_);
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; lean_object* v___x_1097_; uint8_t v___x_1098_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
lean_inc(v_a_1096_);
lean_dec_ref_known(v___x_1095_, 1);
v___x_1097_ = l_Lean_Expr_cleanupAnnotations(v_a_1096_);
v___x_1098_ = l_Lean_Expr_isApp(v___x_1097_);
if (v___x_1098_ == 0)
{
lean_dec_ref(v___x_1097_);
lean_dec_ref(v___x_1086_);
v___y_1052_ = v_a_1094_;
goto v___jp_1051_;
}
else
{
lean_object* v_arg_1099_; lean_object* v___x_1100_; uint8_t v___x_1101_; 
v_arg_1099_ = lean_ctor_get(v___x_1097_, 1);
lean_inc_ref(v_arg_1099_);
v___x_1100_ = l_Lean_Expr_appFnCleanup___redArg(v___x_1097_);
v___x_1101_ = l_Lean_Expr_isApp(v___x_1100_);
if (v___x_1101_ == 0)
{
lean_dec_ref(v___x_1100_);
lean_dec_ref(v_arg_1099_);
lean_dec_ref(v___x_1086_);
v___y_1052_ = v_a_1094_;
goto v___jp_1051_;
}
else
{
lean_object* v_arg_1102_; lean_object* v___x_1103_; uint8_t v___x_1104_; 
v_arg_1102_ = lean_ctor_get(v___x_1100_, 1);
lean_inc_ref(v_arg_1102_);
v___x_1103_ = l_Lean_Expr_appFnCleanup___redArg(v___x_1100_);
v___x_1104_ = l_Lean_Expr_isConstOf(v___x_1103_, v___x_1065_);
lean_dec_ref(v___x_1103_);
if (v___x_1104_ == 0)
{
lean_dec_ref(v_arg_1102_);
lean_dec_ref(v_arg_1099_);
lean_dec_ref(v___x_1086_);
v___y_1052_ = v_a_1094_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; 
lean_inc_ref(v_arg_1102_);
v___x_1105_ = lean_array_push(v___x_1086_, v_arg_1102_);
lean_inc_ref(v_arg_1099_);
v___x_1106_ = lean_array_push(v___x_1105_, v_arg_1099_);
v___x_1107_ = l_Lean_Meta_mkAppM(v___x_1084_, v___x_1106_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_);
if (lean_obj_tag(v___x_1107_) == 0)
{
lean_object* v_a_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; 
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
lean_inc(v_a_1108_);
lean_dec_ref_known(v___x_1107_, 1);
v___x_1109_ = l_Lean_Expr_app___override(v___x_1068_, v_arg_1099_);
v___x_1110_ = l_Lean_Expr_app___override(v___x_1109_, v_arg_1102_);
v___x_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1111_, 0, v_a_1108_);
v___x_1112_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1112_, 0, v___x_1110_);
lean_ctor_set(v___x_1112_, 1, v___x_1111_);
lean_ctor_set_uint8(v___x_1112_, sizeof(void*)*2, v___x_1066_);
v___x_1113_ = l_Lean_Meta_Simp_Result_mkEqTrans(v_a_1094_, v___x_1112_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_);
if (lean_obj_tag(v___x_1113_) == 0)
{
lean_object* v_a_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1122_; 
v_a_1114_ = lean_ctor_get(v___x_1113_, 0);
v_isSharedCheck_1122_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_1116_ = v___x_1113_;
v_isShared_1117_ = v_isSharedCheck_1122_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_a_1114_);
lean_dec(v___x_1113_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1122_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1118_; lean_object* v___x_1120_; 
v___x_1118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1118_, 0, v_a_1114_);
if (v_isShared_1117_ == 0)
{
lean_ctor_set(v___x_1116_, 0, v___x_1118_);
v___x_1120_ = v___x_1116_;
goto v_reusejp_1119_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v___x_1118_);
v___x_1120_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1119_;
}
v_reusejp_1119_:
{
return v___x_1120_;
}
}
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
v_a_1123_ = lean_ctor_get(v___x_1113_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1113_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1113_);
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
lean_object* v_a_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1138_; 
lean_dec_ref(v_arg_1102_);
lean_dec_ref(v_arg_1099_);
lean_dec(v_a_1094_);
v_a_1131_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1138_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1138_ == 0)
{
v___x_1133_ = v___x_1107_;
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_a_1131_);
lean_dec(v___x_1107_);
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
}
}
else
{
lean_object* v_a_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1146_; 
lean_dec(v_a_1094_);
lean_dec_ref(v___x_1086_);
v_a_1139_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1146_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1146_ == 0)
{
v___x_1141_ = v___x_1095_;
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_a_1139_);
lean_dec(v___x_1095_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1144_; 
if (v_isShared_1142_ == 0)
{
v___x_1144_ = v___x_1141_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v_a_1139_);
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
lean_object* v_a_1147_; lean_object* v___x_1149_; uint8_t v_isShared_1150_; uint8_t v_isSharedCheck_1154_; 
lean_dec_ref(v___x_1086_);
lean_dec_ref(v_expr_1081_);
v_a_1147_ = lean_ctor_get(v___x_1093_, 0);
v_isSharedCheck_1154_ = !lean_is_exclusive(v___x_1093_);
if (v_isSharedCheck_1154_ == 0)
{
v___x_1149_ = v___x_1093_;
v_isShared_1150_ = v_isSharedCheck_1154_;
goto v_resetjp_1148_;
}
else
{
lean_inc(v_a_1147_);
lean_dec(v___x_1093_);
v___x_1149_ = lean_box(0);
v_isShared_1150_ = v_isSharedCheck_1154_;
goto v_resetjp_1148_;
}
v_resetjp_1148_:
{
lean_object* v___x_1152_; 
if (v_isShared_1150_ == 0)
{
v___x_1152_ = v___x_1149_;
goto v_reusejp_1151_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v_a_1147_);
v___x_1152_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1151_;
}
v_reusejp_1151_:
{
return v___x_1152_;
}
}
}
}
else
{
lean_object* v_a_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
lean_dec_ref(v___x_1086_);
lean_dec_ref(v_expr_1081_);
lean_dec(v_a_1077_);
lean_dec_ref(v_symmExpr_1070_);
v_a_1155_ = lean_ctor_get(v___x_1089_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1089_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1157_ = v___x_1089_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_a_1155_);
lean_dec(v___x_1089_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_a_1155_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
else
{
lean_object* v___x_1163_; lean_object* v___x_1165_; 
lean_dec_ref(v_expr_1081_);
lean_dec(v_a_1077_);
lean_dec_ref(v_symmExpr_1070_);
lean_dec_ref(v_arg_1063_);
lean_dec_ref(v_arg_1060_);
v___x_1163_ = ((lean_object*)(lp_mathlib_eqComm___closed__0));
if (v_isShared_1080_ == 0)
{
lean_ctor_set(v___x_1079_, 0, v___x_1163_);
v___x_1165_ = v___x_1079_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
}
}
else
{
lean_object* v_a_1170_; lean_object* v___x_1172_; uint8_t v_isShared_1173_; uint8_t v_isSharedCheck_1177_; 
lean_dec_ref(v_symmExpr_1070_);
lean_dec_ref(v_arg_1063_);
lean_dec_ref(v_arg_1060_);
lean_dec_ref(v_e_1042_);
v_a_1170_ = lean_ctor_get(v___x_1076_, 0);
v_isSharedCheck_1177_ = !lean_is_exclusive(v___x_1076_);
if (v_isSharedCheck_1177_ == 0)
{
v___x_1172_ = v___x_1076_;
v_isShared_1173_ = v_isSharedCheck_1177_;
goto v_resetjp_1171_;
}
else
{
lean_inc(v_a_1170_);
lean_dec(v___x_1076_);
v___x_1172_ = lean_box(0);
v_isShared_1173_ = v_isSharedCheck_1177_;
goto v_resetjp_1171_;
}
v_resetjp_1171_:
{
lean_object* v___x_1175_; 
if (v_isShared_1173_ == 0)
{
v___x_1175_ = v___x_1172_;
goto v_reusejp_1174_;
}
else
{
lean_object* v_reuseFailAlloc_1176_; 
v_reuseFailAlloc_1176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1176_, 0, v_a_1170_);
v___x_1175_ = v_reuseFailAlloc_1176_;
goto v_reusejp_1174_;
}
v_reusejp_1174_:
{
return v___x_1175_;
}
}
}
}
}
}
v___jp_1051_:
{
lean_object* v___x_1053_; lean_object* v___x_1054_; 
v___x_1053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1053_, 0, v___y_1052_);
v___x_1054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1054_, 0, v___x_1053_);
return v___x_1054_;
}
v___jp_1055_:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1056_ = ((lean_object*)(lp_mathlib_eqComm___closed__0));
v___x_1057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1056_);
return v___x_1057_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iffComm___boxed(lean_object* v_e_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_, lean_object* v_a_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_){
_start:
{
lean_object* v_res_1187_; 
v_res_1187_ = lp_mathlib_iffComm(v_e_1178_, v_a_1179_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_, v_a_1184_, v_a_1185_);
lean_dec(v_a_1185_);
lean_dec_ref(v_a_1184_);
lean_dec(v_a_1183_);
lean_dec_ref(v_a_1182_);
lean_dec(v_a_1181_);
lean_dec_ref(v_a_1180_);
lean_dec(v_a_1179_);
return v_res_1187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_hidden___redArg(lean_object* v_a_1188_){
_start:
{
lean_inc(v_a_1188_);
return v_a_1188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_hidden___redArg___boxed(lean_object* v_a_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_hidden___redArg(v_a_1189_);
lean_dec(v_a_1189_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_hidden(lean_object* v_00_u03b1_1191_, lean_object* v_a_1192_){
_start:
{
lean_inc(v_a_1192_);
return v_a_1192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_hidden___boxed(lean_object* v_00_u03b1_1193_, lean_object* v_a_1194_){
_start:
{
lean_object* v_res_1195_; 
v_res_1195_ = lp_mathlib_hidden(v_00_u03b1_1193_, v_a_1194_);
lean_dec(v_a_1194_);
return v_res_1195_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_decidableEq__of__subsingleton(lean_object* v_00_u03b1_1196_, lean_object* v_inst_1197_, lean_object* v_a_1198_, lean_object* v_b_1199_){
_start:
{
uint8_t v___x_1200_; 
v___x_1200_ = 1;
return v___x_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decidableEq__of__subsingleton___boxed(lean_object* v_00_u03b1_1201_, lean_object* v_inst_1202_, lean_object* v_a_1203_, lean_object* v_b_1204_){
_start:
{
uint8_t v_res_1205_; lean_object* v_r_1206_; 
v_res_1205_ = lp_mathlib_decidableEq__of__subsingleton(v_00_u03b1_1201_, v_inst_1202_, v_a_1203_, v_b_1204_);
lean_dec(v_b_1204_);
lean_dec(v_a_1203_);
v_r_1206_ = lean_box(v_res_1205_);
return v_r_1206_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_fact__non_x2dinstances(void){
_start:
{
lean_object* v___x_1207_; 
v___x_1207_ = lean_box(0);
return v___x_1207_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableFact___redArg(uint8_t v_inst_1208_){
_start:
{
return v_inst_1208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableFact___redArg___boxed(lean_object* v_inst_1209_){
_start:
{
uint8_t v_inst_8__boxed_1210_; uint8_t v_res_1211_; lean_object* v_r_1212_; 
v_inst_8__boxed_1210_ = lean_unbox(v_inst_1209_);
v_res_1211_ = lp_mathlib_instDecidableFact___redArg(v_inst_8__boxed_1210_);
v_r_1212_ = lean_box(v_res_1211_);
return v_r_1212_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableFact(lean_object* v_p_1213_, uint8_t v_inst_1214_){
_start:
{
return v_inst_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableFact___boxed(lean_object* v_p_1215_, lean_object* v_inst_1216_){
_start:
{
uint8_t v_inst_11__boxed_1217_; uint8_t v_res_1218_; lean_object* v_r_1219_; 
v_inst_11__boxed_1217_ = lean_unbox(v_inst_1216_);
v_res_1218_ = lp_mathlib_instDecidableFact(v_p_1215_, v_inst_11__boxed_1217_);
v_r_1219_ = lean_box(v_res_1218_);
return v_r_1219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_swap_u2082___redArg(lean_object* v_f_1220_, lean_object* v_i_u2082_1221_, lean_object* v_j_u2082_1222_, lean_object* v_i_u2081_1223_, lean_object* v_j_u2081_1224_){
_start:
{
lean_object* v___x_1225_; 
v___x_1225_ = lean_apply_4(v_f_1220_, v_i_u2081_1223_, v_j_u2081_1224_, v_i_u2082_1221_, v_j_u2082_1222_);
return v___x_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_swap_u2082(lean_object* v_00_u03b9_u2081_1226_, lean_object* v_00_u03b9_u2082_1227_, lean_object* v_00_u03ba_u2081_1228_, lean_object* v_00_u03ba_u2082_1229_, lean_object* v_00_u03c6_1230_, lean_object* v_f_1231_, lean_object* v_i_u2082_1232_, lean_object* v_j_u2082_1233_, lean_object* v_i_u2081_1234_, lean_object* v_j_u2081_1235_){
_start:
{
lean_object* v___x_1236_; 
v___x_1236_ = lean_apply_4(v_f_1231_, v_i_u2081_1234_, v_j_u2081_1235_, v_i_u2082_1232_, v_j_u2082_1233_);
return v___x_1236_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_decidable__namespace(void){
_start:
{
lean_object* v___x_1237_; 
v___x_1237_ = lean_box(0);
return v___x_1237_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_decidable__arguments(void){
_start:
{
lean_object* v___x_1238_; 
v___x_1238_ = lean_box(0);
return v___x_1238_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___aux__1___redArg(uint8_t v_inst_1239_, uint8_t v_inst_1240_){
_start:
{
if (v_inst_1239_ == 0)
{
goto v___jp_1241_;
}
else
{
if (v_inst_1240_ == 0)
{
return v_inst_1239_;
}
else
{
goto v___jp_1241_;
}
}
v___jp_1241_:
{
if (v_inst_1240_ == 0)
{
return v_inst_1240_;
}
else
{
if (v_inst_1239_ == 0)
{
return v_inst_1240_;
}
else
{
uint8_t v___x_1242_; 
v___x_1242_ = 0;
return v___x_1242_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___aux__1___redArg___boxed(lean_object* v_inst_1243_, lean_object* v_inst_1244_){
_start:
{
uint8_t v_inst_172__boxed_1245_; uint8_t v_inst_173__boxed_1246_; uint8_t v_res_1247_; lean_object* v_r_1248_; 
v_inst_172__boxed_1245_ = lean_unbox(v_inst_1243_);
v_inst_173__boxed_1246_ = lean_unbox(v_inst_1244_);
v_res_1247_ = lp_mathlib_instDecidableXor___aux__1___redArg(v_inst_172__boxed_1245_, v_inst_173__boxed_1246_);
v_r_1248_ = lean_box(v_res_1247_);
return v_r_1248_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___aux__1(lean_object* v_a_1249_, lean_object* v_b_1250_, uint8_t v_inst_1251_, uint8_t v_inst_1252_){
_start:
{
uint8_t v___x_1253_; 
v___x_1253_ = lp_mathlib_instDecidableXor___aux__1___redArg(v_inst_1251_, v_inst_1252_);
return v___x_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___aux__1___boxed(lean_object* v_a_1254_, lean_object* v_b_1255_, lean_object* v_inst_1256_, lean_object* v_inst_1257_){
_start:
{
uint8_t v_inst_182__boxed_1258_; uint8_t v_inst_183__boxed_1259_; uint8_t v_res_1260_; lean_object* v_r_1261_; 
v_inst_182__boxed_1258_ = lean_unbox(v_inst_1256_);
v_inst_183__boxed_1259_ = lean_unbox(v_inst_1257_);
v_res_1260_ = lp_mathlib_instDecidableXor___aux__1(v_a_1254_, v_b_1255_, v_inst_182__boxed_1258_, v_inst_183__boxed_1259_);
v_r_1261_ = lean_box(v_res_1260_);
return v_r_1261_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor___redArg(uint8_t v_inst_1262_, uint8_t v_inst_1263_){
_start:
{
uint8_t v___x_1264_; 
v___x_1264_ = lp_mathlib_instDecidableXor___aux__1___redArg(v_inst_1262_, v_inst_1263_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___redArg___boxed(lean_object* v_inst_1265_, lean_object* v_inst_1266_){
_start:
{
uint8_t v_inst_9__boxed_1267_; uint8_t v_inst_10__boxed_1268_; uint8_t v_res_1269_; lean_object* v_r_1270_; 
v_inst_9__boxed_1267_ = lean_unbox(v_inst_1265_);
v_inst_10__boxed_1268_ = lean_unbox(v_inst_1266_);
v_res_1269_ = lp_mathlib_instDecidableXor___redArg(v_inst_9__boxed_1267_, v_inst_10__boxed_1268_);
v_r_1270_ = lean_box(v_res_1269_);
return v_r_1270_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableXor(lean_object* v_a_1271_, lean_object* v_b_1272_, uint8_t v_inst_1273_, uint8_t v_inst_1274_){
_start:
{
uint8_t v___x_1275_; 
v___x_1275_ = lp_mathlib_instDecidableXor___aux__1___redArg(v_inst_1273_, v_inst_1274_);
return v___x_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableXor___boxed(lean_object* v_a_1276_, lean_object* v_b_1277_, lean_object* v_inst_1278_, lean_object* v_inst_1279_){
_start:
{
uint8_t v_inst_17__boxed_1280_; uint8_t v_inst_18__boxed_1281_; uint8_t v_res_1282_; lean_object* v_r_1283_; 
v_inst_17__boxed_1280_ = lean_unbox(v_inst_1278_);
v_inst_18__boxed_1281_ = lean_unbox(v_inst_1279_);
v_res_1282_ = lp_mathlib_instDecidableXor(v_a_1276_, v_b_1277_, v_inst_17__boxed_1280_, v_inst_18__boxed_1281_);
v_r_1283_ = lean_box(v_res_1282_);
return v_r_1283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Classical_choice__of__byContradiction_x27___redArg(lean_object* v_contra_1284_){
_start:
{
lean_object* v___x_1285_; 
v___x_1285_ = lean_apply_1(v_contra_1284_, lean_box(0));
return v___x_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Classical_choice__of__byContradiction_x27(lean_object* v_00_u03b1_1286_, lean_object* v_contra_1287_, lean_object* v_H_1288_){
_start:
{
lean_object* v___x_1289_; 
v___x_1289_ = lean_apply_1(v_contra_1287_, lean_box(0));
return v___x_1289_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_Simp(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_fact__non_x2dinstances = _init_lp_mathlib_LibraryNote_fact__non_x2dinstances();
lean_mark_persistent(lp_mathlib_LibraryNote_fact__non_x2dinstances);
lp_mathlib_LibraryNote_decidable__namespace = _init_lp_mathlib_LibraryNote_decidable__namespace();
lean_mark_persistent(lp_mathlib_LibraryNote_decidable__namespace);
lp_mathlib_LibraryNote_decidable__arguments = _init_lp_mathlib_LibraryNote_decidable__arguments();
lean_mark_persistent(lp_mathlib_LibraryNote_decidable__arguments);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_Simp(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
