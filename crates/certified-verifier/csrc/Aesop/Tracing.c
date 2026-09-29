// Lean compiler output
// Module: Aesop.Tracing
// Imports: public import Init public meta import Init public import Aesop.Util.Basic
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
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_IO_monoNanosNow___boxed(lean_object*);
double lean_float_div(double, double);
lean_object* l_IO_getNumHeartbeats___boxed(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
extern lean_object* l_Lean_crossEmoji;
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Macro_resolveGlobalName(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Option_set___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedOption_default___redArg(lean_object*);
extern lean_object* l_Lean_bombEmoji;
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_checkEmoji;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedTraceOption_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedTraceOption;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_registerTraceOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__0 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__0_value;
static const lean_string_object lp_aesop_Aesop_registerTraceOption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__1 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_registerTraceOption___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_ctor_object lp_aesop_Aesop_registerTraceOption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(126, 87, 90, 160, 119, 158, 62, 117)}};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__2 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__2_value;
static const lean_string_object lp_aesop_Aesop_registerTraceOption___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__3 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value;
static const lean_string_object lp_aesop_Aesop_registerTraceOption___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "registerTraceOption"};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__4 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_registerTraceOption___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_registerTraceOption___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__5_value_aux_0),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__4_value),LEAN_SCALAR_PTR_LITERAL(236, 127, 132, 236, 148, 178, 109, 59)}};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__5 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_registerTraceOption___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_object* lp_aesop_Aesop_registerTraceOption___closed__6 = (const lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_registerTraceOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_registerTraceOption___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "(aesop) Print actions taken by Aesop during the proof search."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_steps;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ruleSet"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(30, 209, 33, 119, 78, 45, 15, 238)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "(aesop) Print the rule set before starting the search."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_ruleSet;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "proof"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 102, 34, 179, 153, 11, 175, 136)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "(aesop) If the search is successful, print the produced proof term."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_proof;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "tree"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 9, 103, 21, 100, 140, 158, 71)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 101, .m_capacity = 101, .m_length = 100, .m_data = "(aesop) Once the search has concluded (successfully or unsuccessfully), print the final search tree."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_tree;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extraction"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 107, 14, 61, 133, 25, 17, 132)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "(aesop) Print a trace of the proof extraction procedure."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_extraction;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "stats"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(68, 13, 178, 114, 105, 210, 78, 86)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "(aesop) If the search is successful, print some statistics."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_stats;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "debug"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(40, 215, 222, 176, 152, 52, 0, 225)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "(aesop) Print various debugging information."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_debug;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "script"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(148, 36, 101, 0, 21, 164, 81, 12)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "(aesop) Print a trace of script generation."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_script;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(71, 99, 59, 33, 27, 114, 107, 49)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "(aesop) Trace forward reasoning."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_forward;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(71, 99, 59, 33, 27, 114, 107, 49)}};
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(132, 224, 126, 128, 5, 129, 183, 67)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 98, .m_capacity = 98, .m_length = 97, .m_data = "(aesop) Trace more information about forward reasoning. Mostly intended for performance analysis."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_forwardDebug;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rpinf"};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(78, 146, 149, 245, 154, 250, 239, 89)}};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "(aesop) Trace RPINF calculations."};
static const lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_rpinf;
LEAN_EXPORT uint8_t lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_resolveTraceOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "TraceOption"};
static const lean_object* lp_aesop_Aesop_resolveTraceOption___closed__0 = (const lean_object*)&lp_aesop_Aesop_resolveTraceOption___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_resolveTraceOption___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_resolveTraceOption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_resolveTraceOption___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_resolveTraceOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 109, 116, 217, 20, 72, 202, 90)}};
static const lean_object* lp_aesop_Aesop_resolveTraceOption___closed__1 = (const lean_object*)&lp_aesop_Aesop_resolveTraceOption___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_resolveTraceOption(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_resolveTraceOption___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "doElemAesop_trace![_]__"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__0 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(171, 249, 143, 116, 55, 41, 199, 165)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__2 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "aesop_trace!["};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__4 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__4_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__5 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__5_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__6 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__7 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__7_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__8 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__5_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__8_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__9 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__9_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__10 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__10_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__11 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__9_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__11_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__12 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__12_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__13 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__14 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__14_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "interpolatedStr"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__15 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(156, 58, 177, 246, 99, 11, 16, 252)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__16 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__16_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__17 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__18 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__19 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__19_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__16_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__19_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__20 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__20_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__14_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__20_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__19_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__21 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__21_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__12_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__21_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__22 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__22_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__22_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__23 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__23_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d____ = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__23_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "doExpr"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(130, 168, 60, 255, 153, 218, 88, 77)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.addTrace"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "addTrace"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(183, 216, 39, 175, 140, 178, 201, 238)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__11 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__12_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__13 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__15 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__17 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__19 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__19_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__20 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__20_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__20_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__21 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__21_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__24 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__24_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__25_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__26 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__26_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__27_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__28 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__28_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__29 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__29_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__29_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__30_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__31 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__31_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__32 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__32_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__32_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__33 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__33_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__34 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__34_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__31_value),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__34_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__35 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__35_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__28_value),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__35_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__36 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__36_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__26_value),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__36_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__37 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__37_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Aesop.TraceOption.traceClass"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__38 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__38_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "traceClass"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__40 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__40_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__41 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__41_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "interpolatedStrKind"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__0 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 118, 32, 248, 73, 51, 110, 198)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__1 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__1_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toMessageData"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 4, 57, 33, 167, 136, 170, 64)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__4 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__4_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ToMessageData"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__5 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(14, 83, 41, 225, 154, 14, 42, 20)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(167, 56, 87, 160, 191, 253, 244, 156)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__7 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__8 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__9 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__9_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__10 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__10_value),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__37_value)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__11 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__11_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "termM!_"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__12 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(241, 254, 249, 246, 41, 222, 210, 184)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "m!"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__14 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__14_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "doElemAesop_trace[_]__"};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__0 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(32, 233, 18, 91, 243, 72, 194, 5)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1_value;
static const lean_string_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "aesop_trace["};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__2 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__2_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__3 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__8_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__4 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__4_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__11_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__5 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__3_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__5_value),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__21_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__6 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__6_value)}};
static const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__7 = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__7_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_doElemAesop__trace_x5b___x5d____ = (const lean_object*)&lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__7_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "do"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__0 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 206, 135, 90, 45, 65, 187, 80)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doNested"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__2 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(220, 154, 41, 109, 103, 76, 110, 63)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "doSeqIndent"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__4 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(93, 115, 138, 230, 225, 195, 43, 46)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "doSeqItem"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__6 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(10, 94, 50, 120, 46, 251, 13, 13)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "doIf"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__8 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(133, 56, 102, 181, 14, 156, 21, 0)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "if"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__10 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__10_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doIfProp"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__11 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(55, 147, 210, 58, 86, 191, 41, 151)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nestedAction"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__14 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_0),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value_aux_2),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(115, 27, 24, 243, 204, 49, 153, 202)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__16 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__16_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Aesop.TraceOption.isEnabled"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__17 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isEnabled"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__19 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__19_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value_aux_0),((lean_object*)&lp_aesop_Aesop_resolveTraceOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 109, 116, 217, 20, 72, 202, 90)}};
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value_aux_1),((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(121, 205, 180, 129, 154, 132, 223, 122)}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__21 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__21_value;
static const lean_ctor_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__22 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__22_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "then"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__23 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__23_value;
static const lean_string_object lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__24 = (const lean_object*)&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__24_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ruleSuccessEmoji;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ruleFailureEmoji;
static const lean_string_object lp_aesop_Aesop_ruleProvedEmoji___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 1, .m_data = "🏁"};
static const lean_object* lp_aesop_Aesop_ruleProvedEmoji___closed__0 = (const lean_object*)&lp_aesop_Aesop_ruleProvedEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ruleProvedEmoji = (const lean_object*)&lp_aesop_Aesop_ruleProvedEmoji___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ruleErrorEmoji;
static const lean_string_object lp_aesop_Aesop_rulePostponedEmoji___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "⏳️"};
static const lean_object* lp_aesop_Aesop_rulePostponedEmoji___closed__0 = (const lean_object*)&lp_aesop_Aesop_rulePostponedEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_rulePostponedEmoji = (const lean_object*)&lp_aesop_Aesop_rulePostponedEmoji___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ruleSkippedEmoji___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "⏩️"};
static const lean_object* lp_aesop_Aesop_ruleSkippedEmoji___closed__0 = (const lean_object*)&lp_aesop_Aesop_ruleSkippedEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ruleSkippedEmoji = (const lean_object*)&lp_aesop_Aesop_ruleSkippedEmoji___closed__0_value;
static const lean_string_object lp_aesop_Aesop_nodeUnknownEmoji___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "❓️"};
static const lean_object* lp_aesop_Aesop_nodeUnknownEmoji___closed__0 = (const lean_object*)&lp_aesop_Aesop_nodeUnknownEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_nodeUnknownEmoji = (const lean_object*)&lp_aesop_Aesop_nodeUnknownEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_nodeProvedEmoji = (const lean_object*)&lp_aesop_Aesop_ruleProvedEmoji___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_nodeUnprovableEmoji;
static const lean_string_object lp_aesop_Aesop_newNodeEmoji___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 1, .m_data = "🆕"};
static const lean_object* lp_aesop_Aesop_newNodeEmoji___closed__0 = (const lean_object*)&lp_aesop_Aesop_newNodeEmoji___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_newNodeEmoji = (const lean_object*)&lp_aesop_Aesop_newNodeEmoji___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__2(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_monoNanosNow___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0 = (const lean_object*)&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_getNumHeartbeats___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1 = (const lean_object*)&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_registerTraceOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0 = (const lean_object*)&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__0 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__1 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__1_value;
static const lean_array_object lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__2 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__0_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_traceSimpTheorems___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_traceSimpTheorems___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__0 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__0_value;
static const lean_array_object lp_aesop_Aesop_traceSimpTheorems___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__1 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__1_value;
static const lean_string_object lp_aesop_Aesop_traceSimpTheorems___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Constants to unfold"};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__2 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_traceSimpTheorems___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__3 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__4;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__5;
static const lean_string_object lp_aesop_Aesop_traceSimpTheorems___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Post lemmas"};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__6 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_traceSimpTheorems___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__7 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__8;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__9;
static const lean_string_object lp_aesop_Aesop_traceSimpTheorems___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Pre lemmas"};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__10 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_traceSimpTheorems___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__11 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__11_value;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__12;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__13;
static const lean_string_object lp_aesop_Aesop_traceSimpTheorems___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 91, .m_capacity = 91, .m_length = 90, .m_data = "(Note: even if these entries appear in the sections below, they will not be used by simp.)"};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__14 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__15;
static const lean_closure_object lp_aesop_Aesop_traceSimpTheorems___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_traceSimpTheorems___lam__4, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__16 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__16_value;
static const lean_array_object lp_aesop_Aesop_traceSimpTheorems___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__17 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__17_value;
static const lean_string_object lp_aesop_Aesop_traceSimpTheorems___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Erased entries"};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__18 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_traceSimpTheorems___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__19 = (const lean_object*)&lp_aesop_Aesop_traceSimpTheorems___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__20;
static lean_once_cell_t lp_aesop_Aesop_traceSimpTheorems___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceSimpTheorems___closed__21;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0(void){
_start:
{
uint8_t v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = 0;
v___x_2_ = lean_box(v___x_1_);
v___x_3_ = l_Lean_instInhabitedOption_default___redArg(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0, &lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedTraceOption_default___closed__0);
v___x_5_ = lean_box(0);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedTraceOption_default(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1, &lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedTraceOption_default___closed__1);
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedTraceOption(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_aesop_Aesop_instInhabitedTraceOption_default;
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0(lean_object* v_name_9_, lean_object* v_decl_10_, lean_object* v_ref_11_){
_start:
{
lean_object* v_defValue_13_; lean_object* v_descr_14_; lean_object* v_deprecation_x3f_15_; lean_object* v___x_16_; uint8_t v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v_defValue_13_ = lean_ctor_get(v_decl_10_, 0);
v_descr_14_ = lean_ctor_get(v_decl_10_, 1);
v_deprecation_x3f_15_ = lean_ctor_get(v_decl_10_, 2);
v___x_16_ = lean_alloc_ctor(1, 0, 1);
v___x_17_ = lean_unbox(v_defValue_13_);
lean_ctor_set_uint8(v___x_16_, 0, v___x_17_);
lean_inc(v_deprecation_x3f_15_);
lean_inc_ref(v_descr_14_);
lean_inc_n(v_name_9_, 2);
v___x_18_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_18_, 0, v_name_9_);
lean_ctor_set(v___x_18_, 1, v_ref_11_);
lean_ctor_set(v___x_18_, 2, v___x_16_);
lean_ctor_set(v___x_18_, 3, v_descr_14_);
lean_ctor_set(v___x_18_, 4, v_deprecation_x3f_15_);
v___x_19_ = lean_register_option(v_name_9_, v___x_18_);
if (lean_obj_tag(v___x_19_) == 0)
{
lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_27_; 
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_27_ == 0)
{
lean_object* v_unused_28_; 
v_unused_28_ = lean_ctor_get(v___x_19_, 0);
lean_dec(v_unused_28_);
v___x_21_ = v___x_19_;
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
else
{
lean_dec(v___x_19_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
lean_object* v___x_23_; lean_object* v___x_25_; 
lean_inc(v_defValue_13_);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v_name_9_);
lean_ctor_set(v___x_23_, 1, v_defValue_13_);
if (v_isShared_22_ == 0)
{
lean_ctor_set(v___x_21_, 0, v___x_23_);
v___x_25_ = v___x_21_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_23_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
else
{
lean_object* v_a_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_36_; 
lean_dec(v_name_9_);
v_a_29_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_36_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_36_ == 0)
{
v___x_31_ = v___x_19_;
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_a_29_);
lean_dec(v___x_19_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___x_34_; 
if (v_isShared_32_ == 0)
{
v___x_34_ = v___x_31_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v_a_29_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0___boxed(lean_object* v_name_37_, lean_object* v_decl_38_, lean_object* v_ref_39_, lean_object* v_a_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0(v_name_37_, v_decl_38_, v_ref_39_);
lean_dec_ref(v_decl_38_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_registerTraceOption(lean_object* v_traceName_54_, lean_object* v_descr_55_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; uint8_t v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_57_ = ((lean_object*)(lp_aesop_Aesop_registerTraceOption___closed__2));
lean_inc(v_traceName_54_);
v___x_58_ = l_Lean_Name_append(v___x_57_, v_traceName_54_);
v___x_59_ = 0;
v___x_60_ = lean_box(0);
v___x_61_ = lean_box(v___x_59_);
v___x_62_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v_descr_55_);
lean_ctor_set(v___x_62_, 2, v___x_60_);
v___x_63_ = ((lean_object*)(lp_aesop_Aesop_registerTraceOption___closed__5));
v___x_64_ = lp_aesop_Lean_Option_register___at___00Aesop_registerTraceOption_spec__0(v___x_58_, v___x_62_, v___x_63_);
lean_dec_ref_known(v___x_62_, 3);
if (lean_obj_tag(v___x_64_) == 0)
{
lean_object* v_a_65_; lean_object* v___x_67_; uint8_t v_isShared_68_; uint8_t v_isSharedCheck_75_; 
v_a_65_ = lean_ctor_get(v___x_64_, 0);
v_isSharedCheck_75_ = !lean_is_exclusive(v___x_64_);
if (v_isSharedCheck_75_ == 0)
{
v___x_67_ = v___x_64_;
v_isShared_68_ = v_isSharedCheck_75_;
goto v_resetjp_66_;
}
else
{
lean_inc(v_a_65_);
lean_dec(v___x_64_);
v___x_67_ = lean_box(0);
v_isShared_68_ = v_isSharedCheck_75_;
goto v_resetjp_66_;
}
v_resetjp_66_:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_73_; 
v___x_69_ = ((lean_object*)(lp_aesop_Aesop_registerTraceOption___closed__6));
v___x_70_ = l_Lean_Name_append(v___x_69_, v_traceName_54_);
v___x_71_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v_a_65_);
if (v_isShared_68_ == 0)
{
lean_ctor_set(v___x_67_, 0, v___x_71_);
v___x_73_ = v___x_67_;
goto v_reusejp_72_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v___x_71_);
v___x_73_ = v_reuseFailAlloc_74_;
goto v_reusejp_72_;
}
v_reusejp_72_:
{
return v___x_73_;
}
}
}
else
{
lean_object* v_a_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_83_; 
lean_dec(v_traceName_54_);
v_a_76_ = lean_ctor_get(v___x_64_, 0);
v_isSharedCheck_83_ = !lean_is_exclusive(v___x_64_);
if (v_isSharedCheck_83_ == 0)
{
v___x_78_ = v___x_64_;
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_a_76_);
lean_dec(v___x_64_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_81_; 
if (v_isShared_79_ == 0)
{
v___x_81_ = v___x_78_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v_a_76_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_registerTraceOption___boxed(lean_object* v_traceName_84_, lean_object* v_descr_85_, lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop_Aesop_registerTraceOption(v_traceName_84_, v_descr_85_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0(lean_object* v_opt_88_, lean_object* v___x_89_, lean_object* v_toPure_90_, lean_object* v_____do__lift_91_){
_start:
{
lean_object* v_option_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v_option_92_ = lean_ctor_get(v_opt_88_, 1);
v___x_93_ = l_Lean_Option_get___redArg(v___x_89_, v_____do__lift_91_, v_option_92_);
v___x_94_ = lean_apply_2(v_toPure_90_, lean_box(0), v___x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0___boxed(lean_object* v_opt_95_, lean_object* v___x_96_, lean_object* v_toPure_97_, lean_object* v_____do__lift_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0(v_opt_95_, v___x_96_, v_toPure_97_, v_____do__lift_98_);
lean_dec_ref(v_____do__lift_98_);
lean_dec_ref(v_opt_95_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_opt_102_){
_start:
{
lean_object* v___x_103_; lean_object* v_toApplicative_104_; lean_object* v_toBind_105_; lean_object* v_toPure_106_; lean_object* v___f_107_; lean_object* v___x_108_; 
v___x_103_ = l_Lean_KVMap_instValueBool;
v_toApplicative_104_ = lean_ctor_get(v_inst_100_, 0);
lean_inc_ref(v_toApplicative_104_);
v_toBind_105_ = lean_ctor_get(v_inst_100_, 1);
lean_inc(v_toBind_105_);
lean_dec_ref(v_inst_100_);
v_toPure_106_ = lean_ctor_get(v_toApplicative_104_, 1);
lean_inc(v_toPure_106_);
lean_dec_ref(v_toApplicative_104_);
v___f_107_ = lean_alloc_closure((void*)(lp_aesop_Aesop_TraceOption_isEnabled___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_107_, 0, v_opt_102_);
lean_closure_set(v___f_107_, 1, v___x_103_);
lean_closure_set(v___f_107_, 2, v_toPure_106_);
v___x_108_ = lean_apply_4(v_toBind_105_, lean_box(0), lean_box(0), v_inst_101_, v___f_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled(lean_object* v_m_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_opt_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v_inst_110_, v_inst_111_, v_opt_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled___redArg___lam__0(lean_object* v_opt_114_, lean_object* v___x_115_, lean_object* v_opts_116_){
_start:
{
lean_object* v_option_117_; uint8_t v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_option_117_ = lean_ctor_get(v_opt_114_, 1);
lean_inc_ref(v_option_117_);
lean_dec_ref(v_opt_114_);
v___x_118_ = 1;
v___x_119_ = lean_box(v___x_118_);
v___x_120_ = l_Lean_Option_set___redArg(v___x_115_, v_opts_116_, v_option_117_, v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled___redArg(lean_object* v_inst_121_, lean_object* v_opt_122_, lean_object* v_k_123_){
_start:
{
lean_object* v___x_124_; lean_object* v___f_125_; lean_object* v___x_126_; 
v___x_124_ = l_Lean_KVMap_instValueBool;
v___f_125_ = lean_alloc_closure((void*)(lp_aesop_Aesop_TraceOption_withEnabled___redArg___lam__0), 3, 2);
lean_closure_set(v___f_125_, 0, v_opt_122_);
lean_closure_set(v___f_125_, 1, v___x_124_);
v___x_126_ = lean_apply_3(v_inst_121_, lean_box(0), v___f_125_, v_k_123_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_withEnabled(lean_object* v_m_127_, lean_object* v_00_u03b1_128_, lean_object* v_inst_129_, lean_object* v_opt_130_, lean_object* v_k_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_aesop_Aesop_TraceOption_withEnabled___redArg(v_inst_129_, v_opt_130_, v_k_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_135_ = lean_box(0);
v___x_136_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_));
v___x_137_ = lp_aesop_Aesop_registerTraceOption(v___x_135_, v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2____boxed(lean_object* v_a_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_();
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_145_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_));
v___x_146_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_));
v___x_147_ = lp_aesop_Aesop_registerTraceOption(v___x_145_, v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2____boxed(lean_object* v_a_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_();
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_155_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_));
v___x_156_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_));
v___x_157_ = lp_aesop_Aesop_registerTraceOption(v___x_155_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2____boxed(lean_object* v_a_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_();
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_165_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_));
v___x_166_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_));
v___x_167_ = lp_aesop_Aesop_registerTraceOption(v___x_165_, v___x_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2____boxed(lean_object* v_a_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_();
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_175_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_));
v___x_176_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_));
v___x_177_ = lp_aesop_Aesop_registerTraceOption(v___x_175_, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2____boxed(lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_();
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_185_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_));
v___x_186_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_));
v___x_187_ = lp_aesop_Aesop_registerTraceOption(v___x_185_, v___x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2____boxed(lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_();
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_195_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_));
v___x_196_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_));
v___x_197_ = lp_aesop_Aesop_registerTraceOption(v___x_195_, v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2____boxed(lean_object* v_a_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_();
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_205_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_));
v___x_206_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_));
v___x_207_ = lp_aesop_Aesop_registerTraceOption(v___x_205_, v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2____boxed(lean_object* v_a_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_();
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_215_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_));
v___x_216_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_));
v___x_217_ = lp_aesop_Aesop_registerTraceOption(v___x_215_, v___x_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2____boxed(lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_();
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_225_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__0_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_));
v___x_226_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_));
v___x_227_ = lp_aesop_Aesop_registerTraceOption(v___x_225_, v___x_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2____boxed(lean_object* v_a_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_();
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_235_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__1_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_));
v___x_236_ = ((lean_object*)(lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn___closed__2_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_));
v___x_237_ = lp_aesop_Aesop_registerTraceOption(v___x_235_, v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2____boxed(lean_object* v_a_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_();
return v_res_239_;
}
}
LEAN_EXPORT uint8_t lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0(lean_object* v_n_240_, lean_object* v_x_241_){
_start:
{
if (lean_obj_tag(v_x_241_) == 0)
{
uint8_t v___x_242_; 
v___x_242_ = 0;
return v___x_242_;
}
else
{
lean_object* v_head_243_; lean_object* v_tail_244_; lean_object* v_fst_245_; uint8_t v___x_246_; 
v_head_243_ = lean_ctor_get(v_x_241_, 0);
v_tail_244_ = lean_ctor_get(v_x_241_, 1);
v_fst_245_ = lean_ctor_get(v_head_243_, 0);
v___x_246_ = lean_name_eq(v_fst_245_, v_n_240_);
if (v___x_246_ == 0)
{
v_x_241_ = v_tail_244_;
goto _start;
}
else
{
return v___x_246_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0___boxed(lean_object* v_n_248_, lean_object* v_x_249_){
_start:
{
uint8_t v_res_250_; lean_object* v_r_251_; 
v_res_250_ = lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0(v_n_248_, v_x_249_);
lean_dec(v_x_249_);
lean_dec(v_n_248_);
v_r_251_ = lean_box(v_res_250_);
return v_r_251_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName(lean_object* v_n_252_, lean_object* v_a_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_255_; 
lean_inc(v_n_252_);
v___x_255_ = l_Lean_Macro_resolveGlobalName(v_n_252_, v_a_253_, v_a_254_);
if (lean_obj_tag(v___x_255_) == 0)
{
lean_object* v_a_256_; lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_266_; 
v_a_256_ = lean_ctor_get(v___x_255_, 0);
v_a_257_ = lean_ctor_get(v___x_255_, 1);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_266_ == 0)
{
v___x_259_ = v___x_255_;
v_isShared_260_ = v_isSharedCheck_266_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_inc(v_a_256_);
lean_dec(v___x_255_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_266_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
uint8_t v___x_261_; lean_object* v___x_262_; lean_object* v___x_264_; 
v___x_261_ = lp_aesop_List_any___at___00__private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName_spec__0(v_n_252_, v_a_256_);
lean_dec(v_a_256_);
lean_dec(v_n_252_);
v___x_262_ = lean_box(v___x_261_);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 0, v___x_262_);
v___x_264_ = v___x_259_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v___x_262_);
lean_ctor_set(v_reuseFailAlloc_265_, 1, v_a_257_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
else
{
lean_object* v_a_267_; lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_dec(v_n_252_);
v_a_267_ = lean_ctor_get(v___x_255_, 0);
v_a_268_ = lean_ctor_get(v___x_255_, 1);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_255_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_inc(v_a_267_);
lean_dec(v___x_255_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_267_);
lean_ctor_set(v_reuseFailAlloc_274_, 1, v_a_268_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName___boxed(lean_object* v_n_276_, lean_object* v_a_277_, lean_object* v_a_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName(v_n_276_, v_a_277_, v_a_278_);
lean_dec_ref(v_a_277_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_resolveTraceOption(lean_object* v_stx_284_, lean_object* v_a_285_, lean_object* v_a_286_){
_start:
{
lean_object* v_methods_287_; lean_object* v_quotContext_288_; lean_object* v_currMacroScope_289_; lean_object* v_currRecDepth_290_; lean_object* v_maxRecDepth_291_; lean_object* v_ref_292_; lean_object* v_n_293_; lean_object* v___x_294_; lean_object* v_fqn_295_; lean_object* v_ref_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_methods_287_ = lean_ctor_get(v_a_285_, 0);
v_quotContext_288_ = lean_ctor_get(v_a_285_, 1);
v_currMacroScope_289_ = lean_ctor_get(v_a_285_, 2);
v_currRecDepth_290_ = lean_ctor_get(v_a_285_, 3);
v_maxRecDepth_291_ = lean_ctor_get(v_a_285_, 4);
v_ref_292_ = lean_ctor_get(v_a_285_, 5);
v_n_293_ = l_Lean_TSyntax_getId(v_stx_284_);
v___x_294_ = ((lean_object*)(lp_aesop_Aesop_resolveTraceOption___closed__1));
lean_inc(v_n_293_);
v_fqn_295_ = l_Lean_Name_append(v___x_294_, v_n_293_);
v_ref_296_ = l_Lean_replaceRef(v_stx_284_, v_ref_292_);
lean_inc(v_maxRecDepth_291_);
lean_inc(v_currRecDepth_290_);
lean_inc(v_currMacroScope_289_);
lean_inc(v_quotContext_288_);
lean_inc(v_methods_287_);
v___x_297_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_297_, 0, v_methods_287_);
lean_ctor_set(v___x_297_, 1, v_quotContext_288_);
lean_ctor_set(v___x_297_, 2, v_currMacroScope_289_);
lean_ctor_set(v___x_297_, 3, v_currRecDepth_290_);
lean_ctor_set(v___x_297_, 4, v_maxRecDepth_291_);
lean_ctor_set(v___x_297_, 5, v_ref_296_);
lean_inc(v_fqn_295_);
v___x_298_ = lp_aesop___private_Aesop_Tracing_0__Aesop_isFullyQualifiedGlobalName(v_fqn_295_, v___x_297_, v_a_286_);
lean_dec_ref_known(v___x_297_, 6);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v_a_299_; uint8_t v___x_300_; 
v_a_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_a_299_);
v___x_300_ = lean_unbox(v_a_299_);
lean_dec(v_a_299_);
if (v___x_300_ == 0)
{
lean_object* v_a_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_308_; 
lean_dec(v_fqn_295_);
v_a_301_ = lean_ctor_get(v___x_298_, 1);
v_isSharedCheck_308_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_308_ == 0)
{
lean_object* v_unused_309_; 
v_unused_309_ = lean_ctor_get(v___x_298_, 0);
lean_dec(v_unused_309_);
v___x_303_ = v___x_298_;
v_isShared_304_ = v_isSharedCheck_308_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_a_301_);
lean_dec(v___x_298_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_308_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v___x_306_; 
if (v_isShared_304_ == 0)
{
lean_ctor_set(v___x_303_, 0, v_n_293_);
v___x_306_ = v___x_303_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_307_; 
v_reuseFailAlloc_307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_307_, 0, v_n_293_);
lean_ctor_set(v_reuseFailAlloc_307_, 1, v_a_301_);
v___x_306_ = v_reuseFailAlloc_307_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
return v___x_306_;
}
}
}
else
{
lean_object* v_a_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_317_; 
lean_dec(v_n_293_);
v_a_310_ = lean_ctor_get(v___x_298_, 1);
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_317_ == 0)
{
lean_object* v_unused_318_; 
v_unused_318_ = lean_ctor_get(v___x_298_, 0);
lean_dec(v_unused_318_);
v___x_312_ = v___x_298_;
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_a_310_);
lean_dec(v___x_298_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_315_; 
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v_fqn_295_);
v___x_315_ = v___x_312_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v_fqn_295_);
lean_ctor_set(v_reuseFailAlloc_316_, 1, v_a_310_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
}
else
{
lean_object* v_a_319_; lean_object* v_a_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_327_; 
lean_dec(v_fqn_295_);
lean_dec(v_n_293_);
v_a_319_ = lean_ctor_get(v___x_298_, 0);
v_a_320_ = lean_ctor_get(v___x_298_, 1);
v_isSharedCheck_327_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_327_ == 0)
{
v___x_322_ = v___x_298_;
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_a_320_);
lean_inc(v_a_319_);
lean_dec(v___x_298_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_325_; 
if (v_isShared_323_ == 0)
{
v___x_325_ = v___x_322_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_a_319_);
lean_ctor_set(v_reuseFailAlloc_326_, 1, v_a_320_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_resolveTraceOption___boxed(lean_object* v_stx_328_, lean_object* v_a_329_, lean_object* v_a_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_aesop_Aesop_resolveTraceOption(v_stx_328_, v_a_329_, v_a_330_);
lean_dec_ref(v_a_329_);
lean_dec(v_stx_328_);
return v_res_331_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8(void){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_402_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__7));
v___x_403_ = l_String_toRawSubstring_x27(v___x_402_);
return v___x_403_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23(void){
_start:
{
lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_434_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_435_ = l_String_toRawSubstring_x27(v___x_434_);
return v___x_435_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39(void){
_start:
{
lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_471_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__38));
v___x_472_ = l_String_toRawSubstring_x27(v___x_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0(lean_object* v___x_475_, lean_object* v___x_476_, lean_object* v___x_477_, lean_object* v_msg_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v_quotContext_481_; lean_object* v_currMacroScope_482_; lean_object* v_ref_483_; uint8_t v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v_quotContext_481_ = lean_ctor_get(v___y_479_, 1);
v_currMacroScope_482_ = lean_ctor_get(v___y_479_, 2);
v_ref_483_ = lean_ctor_get(v___y_479_, 5);
v___x_484_ = 0;
v___x_485_ = l_Lean_SourceInfo_fromRef(v_ref_483_, v___x_484_);
v___x_486_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4));
v___x_487_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6));
v___x_488_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__8);
v___x_489_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__10));
lean_inc_n(v_currMacroScope_482_, 3);
lean_inc_n(v_quotContext_481_, 3);
v___x_490_ = l_Lean_addMacroScope(v_quotContext_481_, v___x_489_, v_currMacroScope_482_);
v___x_491_ = lean_box(0);
v___x_492_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__12));
lean_inc_n(v___x_485_, 12);
v___x_493_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_493_, 0, v___x_485_);
lean_ctor_set(v___x_493_, 1, v___x_488_);
lean_ctor_set(v___x_493_, 2, v___x_490_);
lean_ctor_set(v___x_493_, 3, v___x_492_);
v___x_494_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14));
v___x_495_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16));
v___x_496_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18));
v___x_497_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__19));
v___x_498_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_498_, 0, v___x_485_);
lean_ctor_set(v___x_498_, 1, v___x_497_);
v___x_499_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__21));
v___x_500_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23);
v___x_501_ = l_Lean_addMacroScope(v_quotContext_481_, v___x_475_, v_currMacroScope_482_);
lean_inc_ref(v___x_476_);
v___x_502_ = l_Lean_Name_mkStr1(v___x_476_);
v___x_503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
v___x_504_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__37));
v___x_505_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_503_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_506_, 0, v___x_485_);
lean_ctor_set(v___x_506_, 1, v___x_500_);
lean_ctor_set(v___x_506_, 2, v___x_501_);
lean_ctor_set(v___x_506_, 3, v___x_505_);
v___x_507_ = l_Lean_Syntax_node1(v___x_485_, v___x_499_, v___x_506_);
v___x_508_ = l_Lean_Syntax_node2(v___x_485_, v___x_496_, v___x_498_, v___x_507_);
v___x_509_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__39);
v___x_510_ = ((lean_object*)(lp_aesop_Aesop_resolveTraceOption___closed__0));
v___x_511_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__40));
v___x_512_ = l_Lean_Name_mkStr3(v___x_476_, v___x_510_, v___x_511_);
lean_inc(v___x_512_);
v___x_513_ = l_Lean_addMacroScope(v_quotContext_481_, v___x_512_, v_currMacroScope_482_);
v___x_514_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_512_);
lean_ctor_set(v___x_514_, 1, v___x_491_);
v___x_515_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_515_, 0, v___x_514_);
lean_ctor_set(v___x_515_, 1, v___x_491_);
v___x_516_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_516_, 0, v___x_485_);
lean_ctor_set(v___x_516_, 1, v___x_509_);
lean_ctor_set(v___x_516_, 2, v___x_513_);
lean_ctor_set(v___x_516_, 3, v___x_515_);
v___x_517_ = l_Lean_Syntax_node1(v___x_485_, v___x_494_, v___x_477_);
v___x_518_ = l_Lean_Syntax_node2(v___x_485_, v___x_487_, v___x_516_, v___x_517_);
v___x_519_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__41));
v___x_520_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_485_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
v___x_521_ = l_Lean_Syntax_node3(v___x_485_, v___x_495_, v___x_508_, v___x_518_, v___x_520_);
v___x_522_ = l_Lean_Syntax_node2(v___x_485_, v___x_494_, v___x_521_, v_msg_478_);
v___x_523_ = l_Lean_Syntax_node2(v___x_485_, v___x_487_, v___x_493_, v___x_522_);
v___x_524_ = l_Lean_Syntax_node1(v___x_485_, v___x_486_, v___x_523_);
v___x_525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
lean_ctor_set(v___x_525_, 1, v___y_480_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___boxed(lean_object* v___x_526_, lean_object* v___x_527_, lean_object* v___x_528_, lean_object* v_msg_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0(v___x_526_, v___x_527_, v___x_528_, v_msg_529_, v___y_530_, v___y_531_);
lean_dec_ref(v___y_530_);
return v_res_532_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; 
v___x_537_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__2));
v___x_538_ = l_String_toRawSubstring_x27(v___x_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1(lean_object* v_x_564_, lean_object* v_a_565_, lean_object* v_a_566_){
_start:
{
lean_object* v___y_568_; lean_object* v___x_578_; lean_object* v___x_579_; uint8_t v___x_580_; 
v___x_578_ = ((lean_object*)(lp_aesop_Aesop_registerTraceOption___closed__3));
v___x_579_ = ((lean_object*)(lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1));
lean_inc(v_x_564_);
v___x_580_ = l_Lean_Syntax_isOfKind(v_x_564_, v___x_579_);
if (v___x_580_ == 0)
{
lean_object* v___x_581_; lean_object* v___x_582_; 
lean_dec(v_x_564_);
v___x_581_ = lean_box(1);
v___x_582_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v_a_566_);
return v___x_582_;
}
else
{
lean_object* v___x_583_; lean_object* v_opt_584_; lean_object* v___x_585_; 
v___x_583_ = lean_unsigned_to_nat(1u);
v_opt_584_ = l_Lean_Syntax_getArg(v_x_564_, v___x_583_);
v___x_585_ = lp_aesop_Aesop_resolveTraceOption(v_opt_584_, v_a_565_, v_a_566_);
lean_dec(v_opt_584_);
if (lean_obj_tag(v___x_585_) == 0)
{
lean_object* v_a_586_; lean_object* v_a_587_; lean_object* v___x_588_; lean_object* v_msg_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; uint8_t v___x_594_; 
v_a_586_ = lean_ctor_get(v___x_585_, 0);
lean_inc(v_a_586_);
v_a_587_ = lean_ctor_get(v___x_585_, 1);
lean_inc(v_a_587_);
lean_dec_ref_known(v___x_585_, 2);
v___x_588_ = lean_unsigned_to_nat(3u);
v_msg_589_ = l_Lean_Syntax_getArg(v_x_564_, v___x_588_);
lean_dec(v_x_564_);
v___x_590_ = lean_box(0);
v___x_591_ = l_Lean_mkIdent(v_a_586_);
lean_inc(v_msg_589_);
v___x_592_ = l_Lean_Syntax_getKind(v_msg_589_);
v___x_593_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__1));
v___x_594_ = lean_name_eq(v___x_592_, v___x_593_);
lean_dec(v___x_592_);
if (v___x_594_ == 0)
{
lean_object* v_quotContext_595_; lean_object* v_currMacroScope_596_; lean_object* v_ref_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v_quotContext_595_ = lean_ctor_get(v_a_565_, 1);
v_currMacroScope_596_ = lean_ctor_get(v_a_565_, 2);
v_ref_597_ = lean_ctor_get(v_a_565_, 5);
v___x_598_ = l_Lean_SourceInfo_fromRef(v_ref_597_, v___x_594_);
v___x_599_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6));
v___x_600_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__3);
v___x_601_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__4));
lean_inc_n(v_currMacroScope_596_, 2);
lean_inc_n(v_quotContext_595_, 2);
v___x_602_ = l_Lean_addMacroScope(v_quotContext_595_, v___x_601_, v_currMacroScope_596_);
v___x_603_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__8));
lean_inc_n(v___x_598_, 8);
v___x_604_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_604_, 0, v___x_598_);
lean_ctor_set(v___x_604_, 1, v___x_600_);
lean_ctor_set(v___x_604_, 2, v___x_602_);
lean_ctor_set(v___x_604_, 3, v___x_603_);
v___x_605_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14));
v___x_606_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__16));
v___x_607_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__18));
v___x_608_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__19));
v___x_609_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_609_, 0, v___x_598_);
lean_ctor_set(v___x_609_, 1, v___x_608_);
v___x_610_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__21));
v___x_611_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__23);
v___x_612_ = l_Lean_addMacroScope(v_quotContext_595_, v___x_590_, v_currMacroScope_596_);
v___x_613_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__11));
v___x_614_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_614_, 0, v___x_598_);
lean_ctor_set(v___x_614_, 1, v___x_611_);
lean_ctor_set(v___x_614_, 2, v___x_612_);
lean_ctor_set(v___x_614_, 3, v___x_613_);
v___x_615_ = l_Lean_Syntax_node1(v___x_598_, v___x_610_, v___x_614_);
v___x_616_ = l_Lean_Syntax_node2(v___x_598_, v___x_607_, v___x_609_, v___x_615_);
v___x_617_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__41));
v___x_618_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_618_, 0, v___x_598_);
lean_ctor_set(v___x_618_, 1, v___x_617_);
v___x_619_ = l_Lean_Syntax_node3(v___x_598_, v___x_606_, v___x_616_, v_msg_589_, v___x_618_);
v___x_620_ = l_Lean_Syntax_node1(v___x_598_, v___x_605_, v___x_619_);
v___x_621_ = l_Lean_Syntax_node2(v___x_598_, v___x_599_, v___x_604_, v___x_620_);
v___x_622_ = lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0(v___x_590_, v___x_578_, v___x_591_, v___x_621_, v_a_565_, v_a_587_);
v___y_568_ = v___x_622_;
goto v___jp_567_;
}
else
{
lean_object* v_ref_623_; uint8_t v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v_ref_623_ = lean_ctor_get(v_a_565_, 5);
v___x_624_ = 0;
v___x_625_ = l_Lean_SourceInfo_fromRef(v_ref_623_, v___x_624_);
v___x_626_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__13));
v___x_627_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___closed__14));
lean_inc(v___x_625_);
v___x_628_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_625_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
v___x_629_ = l_Lean_Syntax_node2(v___x_625_, v___x_626_, v___x_628_, v_msg_589_);
v___x_630_ = lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0(v___x_590_, v___x_578_, v___x_591_, v___x_629_, v_a_565_, v_a_587_);
v___y_568_ = v___x_630_;
goto v___jp_567_;
}
}
else
{
lean_object* v_a_631_; lean_object* v_a_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
lean_dec(v_x_564_);
v_a_631_ = lean_ctor_get(v___x_585_, 0);
v_a_632_ = lean_ctor_get(v___x_585_, 1);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_585_);
if (v_isSharedCheck_639_ == 0)
{
v___x_634_ = v___x_585_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_a_632_);
lean_inc(v_a_631_);
lean_dec(v___x_585_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_631_);
lean_ctor_set(v_reuseFailAlloc_638_, 1, v_a_632_);
v___x_637_ = v_reuseFailAlloc_638_;
goto v_reusejp_636_;
}
v_reusejp_636_:
{
return v___x_637_;
}
}
}
}
v___jp_567_:
{
lean_object* v_a_569_; lean_object* v_a_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_577_; 
v_a_569_ = lean_ctor_get(v___y_568_, 0);
v_a_570_ = lean_ctor_get(v___y_568_, 1);
v_isSharedCheck_577_ = !lean_is_exclusive(v___y_568_);
if (v_isSharedCheck_577_ == 0)
{
v___x_572_ = v___y_568_;
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_a_570_);
lean_inc(v_a_569_);
lean_dec(v___y_568_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_575_; 
if (v_isShared_573_ == 0)
{
v___x_575_ = v___x_572_;
goto v_reusejp_574_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v_a_569_);
lean_ctor_set(v_reuseFailAlloc_576_, 1, v_a_570_);
v___x_575_ = v_reuseFailAlloc_576_;
goto v_reusejp_574_;
}
v_reusejp_574_:
{
return v___x_575_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___boxed(lean_object* v_x_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1(v_x_640_, v_a_641_, v_a_642_);
lean_dec_ref(v_a_641_);
return v_res_643_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13(void){
_start:
{
lean_object* v___x_705_; 
v___x_705_ = l_Array_mkArray0(lean_box(0));
return v___x_705_;
}
}
static lean_object* _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18(void){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_714_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__17));
v___x_715_ = l_String_toRawSubstring_x27(v___x_714_);
return v___x_715_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1(lean_object* v_x_729_, lean_object* v_a_730_, lean_object* v_a_731_){
_start:
{
lean_object* v___x_732_; uint8_t v___x_733_; 
v___x_732_ = ((lean_object*)(lp_aesop_Aesop_doElemAesop__trace_x5b___x5d_____00__closed__1));
lean_inc(v_x_729_);
v___x_733_ = l_Lean_Syntax_isOfKind(v_x_729_, v___x_732_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; lean_object* v___x_735_; 
lean_dec(v_x_729_);
v___x_734_ = lean_box(1);
v___x_735_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_734_);
lean_ctor_set(v___x_735_, 1, v_a_731_);
return v___x_735_;
}
else
{
lean_object* v___x_736_; lean_object* v_opt_737_; lean_object* v___x_738_; 
v___x_736_ = lean_unsigned_to_nat(1u);
v_opt_737_ = l_Lean_Syntax_getArg(v_x_729_, v___x_736_);
v___x_738_ = lp_aesop_Aesop_resolveTraceOption(v_opt_737_, v_a_730_, v_a_731_);
lean_dec(v_opt_737_);
if (lean_obj_tag(v___x_738_) == 0)
{
lean_object* v_a_739_; lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_841_; 
v_a_739_ = lean_ctor_get(v___x_738_, 0);
v_a_740_ = lean_ctor_get(v___x_738_, 1);
v_isSharedCheck_841_ = !lean_is_exclusive(v___x_738_);
if (v_isSharedCheck_841_ == 0)
{
v___x_742_ = v___x_738_;
v_isShared_743_ = v_isSharedCheck_841_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_inc(v_a_739_);
lean_dec(v___x_738_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_841_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; lean_object* v_msg_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; uint8_t v___x_749_; 
v___x_744_ = lean_unsigned_to_nat(3u);
v_msg_745_ = l_Lean_Syntax_getArg(v_x_729_, v___x_744_);
lean_dec(v_x_729_);
v___x_746_ = l_Lean_mkIdent(v_a_739_);
v___x_747_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__0));
v___x_748_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__1));
lean_inc(v_msg_745_);
v___x_749_ = l_Lean_Syntax_isOfKind(v_msg_745_, v___x_748_);
if (v___x_749_ == 0)
{
lean_object* v_quotContext_750_; lean_object* v_currMacroScope_751_; lean_object* v_ref_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_797_; 
v_quotContext_750_ = lean_ctor_get(v_a_730_, 1);
v_currMacroScope_751_ = lean_ctor_get(v_a_730_, 2);
v_ref_752_ = lean_ctor_get(v_a_730_, 5);
v___x_753_ = l_Lean_SourceInfo_fromRef(v_ref_752_, v___x_749_);
v___x_754_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3));
lean_inc_n(v___x_753_, 21);
v___x_755_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_755_, 0, v___x_753_);
lean_ctor_set(v___x_755_, 1, v___x_747_);
v___x_756_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5));
v___x_757_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14));
v___x_758_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7));
v___x_759_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9));
v___x_760_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__10));
v___x_761_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_761_, 0, v___x_753_);
lean_ctor_set(v___x_761_, 1, v___x_760_);
v___x_762_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12));
v___x_763_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13);
v___x_764_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_764_, 0, v___x_753_);
lean_ctor_set(v___x_764_, 1, v___x_757_);
lean_ctor_set(v___x_764_, 2, v___x_763_);
v___x_765_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15));
v___x_766_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__16));
v___x_767_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_767_, 0, v___x_753_);
lean_ctor_set(v___x_767_, 1, v___x_766_);
v___x_768_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4));
v___x_769_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6));
v___x_770_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18);
v___x_771_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20));
lean_inc(v_currMacroScope_751_);
lean_inc(v_quotContext_750_);
v___x_772_ = l_Lean_addMacroScope(v_quotContext_750_, v___x_771_, v_currMacroScope_751_);
v___x_773_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__22));
v___x_774_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_774_, 0, v___x_753_);
lean_ctor_set(v___x_774_, 1, v___x_770_);
lean_ctor_set(v___x_774_, 2, v___x_772_);
lean_ctor_set(v___x_774_, 3, v___x_773_);
lean_inc(v___x_746_);
v___x_775_ = l_Lean_Syntax_node1(v___x_753_, v___x_757_, v___x_746_);
v___x_776_ = l_Lean_Syntax_node2(v___x_753_, v___x_769_, v___x_774_, v___x_775_);
v___x_777_ = l_Lean_Syntax_node1(v___x_753_, v___x_768_, v___x_776_);
v___x_778_ = l_Lean_Syntax_node2(v___x_753_, v___x_765_, v___x_767_, v___x_777_);
lean_inc_ref_n(v___x_764_, 4);
v___x_779_ = l_Lean_Syntax_node2(v___x_753_, v___x_762_, v___x_764_, v___x_778_);
v___x_780_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__23));
v___x_781_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_781_, 0, v___x_753_);
lean_ctor_set(v___x_781_, 1, v___x_780_);
v___x_782_ = ((lean_object*)(lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__1));
v___x_783_ = ((lean_object*)(lp_aesop_Aesop_doElemAesop__trace_x21_x5b___x5d_____00__closed__4));
v___x_784_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_784_, 0, v___x_753_);
lean_ctor_set(v___x_784_, 1, v___x_783_);
v___x_785_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__24));
v___x_786_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_786_, 0, v___x_753_);
lean_ctor_set(v___x_786_, 1, v___x_785_);
v___x_787_ = l_Lean_Syntax_node4(v___x_753_, v___x_782_, v___x_784_, v___x_746_, v___x_786_, v_msg_745_);
v___x_788_ = l_Lean_Syntax_node2(v___x_753_, v___x_758_, v___x_787_, v___x_764_);
v___x_789_ = l_Lean_Syntax_node1(v___x_753_, v___x_757_, v___x_788_);
v___x_790_ = l_Lean_Syntax_node1(v___x_753_, v___x_756_, v___x_789_);
v___x_791_ = l_Lean_Syntax_node6(v___x_753_, v___x_759_, v___x_761_, v___x_779_, v___x_781_, v___x_790_, v___x_764_, v___x_764_);
v___x_792_ = l_Lean_Syntax_node2(v___x_753_, v___x_758_, v___x_791_, v___x_764_);
v___x_793_ = l_Lean_Syntax_node1(v___x_753_, v___x_757_, v___x_792_);
v___x_794_ = l_Lean_Syntax_node1(v___x_753_, v___x_756_, v___x_793_);
v___x_795_ = l_Lean_Syntax_node2(v___x_753_, v___x_754_, v___x_755_, v___x_794_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 0, v___x_795_);
v___x_797_ = v___x_742_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v___x_795_);
lean_ctor_set(v_reuseFailAlloc_798_, 1, v_a_740_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
else
{
lean_object* v_quotContext_799_; lean_object* v_currMacroScope_800_; lean_object* v_ref_801_; lean_object* v___x_802_; uint8_t v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_839_; 
v_quotContext_799_ = lean_ctor_get(v_a_730_, 1);
v_currMacroScope_800_ = lean_ctor_get(v_a_730_, 2);
v_ref_801_ = lean_ctor_get(v_a_730_, 5);
v___x_802_ = l_Lean_Syntax_getArg(v_msg_745_, v___x_736_);
lean_dec(v_msg_745_);
v___x_803_ = 0;
v___x_804_ = l_Lean_SourceInfo_fromRef(v_ref_801_, v___x_803_);
v___x_805_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__3));
lean_inc_n(v___x_804_, 15);
v___x_806_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_806_, 0, v___x_804_);
lean_ctor_set(v___x_806_, 1, v___x_747_);
v___x_807_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__5));
v___x_808_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__14));
v___x_809_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__7));
v___x_810_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__9));
v___x_811_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__10));
v___x_812_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_812_, 0, v___x_804_);
lean_ctor_set(v___x_812_, 1, v___x_811_);
v___x_813_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__12));
v___x_814_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__13);
v___x_815_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_815_, 0, v___x_804_);
lean_ctor_set(v___x_815_, 1, v___x_808_);
lean_ctor_set(v___x_815_, 2, v___x_814_);
v___x_816_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__15));
v___x_817_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__16));
v___x_818_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_818_, 0, v___x_804_);
lean_ctor_set(v___x_818_, 1, v___x_817_);
v___x_819_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__4));
v___x_820_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__6));
v___x_821_ = lean_obj_once(&lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18, &lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18_once, _init_lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__18);
v___x_822_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__20));
lean_inc(v_currMacroScope_800_);
lean_inc(v_quotContext_799_);
v___x_823_ = l_Lean_addMacroScope(v_quotContext_799_, v___x_822_, v_currMacroScope_800_);
v___x_824_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__22));
v___x_825_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_825_, 0, v___x_804_);
lean_ctor_set(v___x_825_, 1, v___x_821_);
lean_ctor_set(v___x_825_, 2, v___x_823_);
lean_ctor_set(v___x_825_, 3, v___x_824_);
v___x_826_ = l_Lean_Syntax_node1(v___x_804_, v___x_808_, v___x_746_);
v___x_827_ = l_Lean_Syntax_node2(v___x_804_, v___x_820_, v___x_825_, v___x_826_);
v___x_828_ = l_Lean_Syntax_node1(v___x_804_, v___x_819_, v___x_827_);
v___x_829_ = l_Lean_Syntax_node2(v___x_804_, v___x_816_, v___x_818_, v___x_828_);
lean_inc_ref_n(v___x_815_, 3);
v___x_830_ = l_Lean_Syntax_node2(v___x_804_, v___x_813_, v___x_815_, v___x_829_);
v___x_831_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___closed__23));
v___x_832_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_832_, 0, v___x_804_);
lean_ctor_set(v___x_832_, 1, v___x_831_);
v___x_833_ = l_Lean_Syntax_node6(v___x_804_, v___x_810_, v___x_812_, v___x_830_, v___x_832_, v___x_802_, v___x_815_, v___x_815_);
v___x_834_ = l_Lean_Syntax_node2(v___x_804_, v___x_809_, v___x_833_, v___x_815_);
v___x_835_ = l_Lean_Syntax_node1(v___x_804_, v___x_808_, v___x_834_);
v___x_836_ = l_Lean_Syntax_node1(v___x_804_, v___x_807_, v___x_835_);
v___x_837_ = l_Lean_Syntax_node2(v___x_804_, v___x_805_, v___x_806_, v___x_836_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 0, v___x_837_);
v___x_839_ = v___x_742_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_837_);
lean_ctor_set(v_reuseFailAlloc_840_, 1, v_a_740_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
else
{
lean_object* v_a_842_; lean_object* v_a_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_850_; 
lean_dec(v_x_729_);
v_a_842_ = lean_ctor_get(v___x_738_, 0);
v_a_843_ = lean_ctor_get(v___x_738_, 1);
v_isSharedCheck_850_ = !lean_is_exclusive(v___x_738_);
if (v_isSharedCheck_850_ == 0)
{
v___x_845_ = v___x_738_;
v_isShared_846_ = v_isSharedCheck_850_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_a_843_);
lean_inc(v_a_842_);
lean_dec(v___x_738_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_850_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_848_; 
if (v_isShared_846_ == 0)
{
v___x_848_ = v___x_845_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_a_842_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v_a_843_);
v___x_848_ = v_reuseFailAlloc_849_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
return v___x_848_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1___boxed(lean_object* v_x_851_, lean_object* v_a_852_, lean_object* v_a_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x5b___x5d______1(v_x_851_, v_a_852_, v_a_853_);
lean_dec_ref(v_a_852_);
return v_res_854_;
}
}
static lean_object* _init_lp_aesop_Aesop_ruleSuccessEmoji(void){
_start:
{
lean_object* v___x_855_; 
v___x_855_ = l_Lean_checkEmoji;
return v___x_855_;
}
}
static lean_object* _init_lp_aesop_Aesop_ruleFailureEmoji(void){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = l_Lean_crossEmoji;
return v___x_856_;
}
}
static lean_object* _init_lp_aesop_Aesop_ruleErrorEmoji(void){
_start:
{
lean_object* v___x_859_; 
v___x_859_ = l_Lean_bombEmoji;
return v___x_859_;
}
}
static lean_object* _init_lp_aesop_Aesop_nodeUnprovableEmoji(void){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = l_Lean_crossEmoji;
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object* v_toEmoji_870_, lean_object* v_x_871_){
_start:
{
if (lean_obj_tag(v_x_871_) == 0)
{
lean_object* v___x_872_; 
lean_dec_ref_known(v_x_871_, 1);
lean_dec_ref(v_toEmoji_870_);
v___x_872_ = l_Lean_crossEmoji;
return v___x_872_;
}
else
{
lean_object* v_a_873_; lean_object* v___x_874_; 
v_a_873_ = lean_ctor_get(v_x_871_, 0);
lean_inc(v_a_873_);
lean_dec_ref_known(v_x_871_, 1);
v___x_874_ = lean_apply_1(v_toEmoji_870_, v_a_873_);
return v___x_874_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji(lean_object* v_00_u03b1_875_, lean_object* v_00_u03b5_876_, lean_object* v_toEmoji_877_, lean_object* v_x_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(v_toEmoji_877_, v_x_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0(lean_object* v_inst_880_, lean_object* v_inst_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_traceClass_886_, uint8_t v_collapsed_887_, lean_object* v___x_888_, lean_object* v_opts_889_, uint8_t v_clsEnabled_890_, lean_object* v_oldTraces_891_, lean_object* v_msg_892_, lean_object* v_resStartStop_893_){
_start:
{
lean_object* v___x_894_; 
v___x_894_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v_inst_880_, v_inst_881_, v_inst_882_, v_inst_883_, lean_box(0), v_inst_884_, v_inst_885_, v_traceClass_886_, v_collapsed_887_, v___x_888_, v_opts_889_, v_clsEnabled_890_, v_oldTraces_891_, v_msg_892_, v_resStartStop_893_);
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0___boxed(lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_inst_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_traceClass_901_, lean_object* v_collapsed_902_, lean_object* v___x_903_, lean_object* v_opts_904_, lean_object* v_clsEnabled_905_, lean_object* v_oldTraces_906_, lean_object* v_msg_907_, lean_object* v_resStartStop_908_){
_start:
{
uint8_t v_collapsed_boxed_909_; uint8_t v_clsEnabled_boxed_910_; lean_object* v_res_911_; 
v_collapsed_boxed_909_ = lean_unbox(v_collapsed_902_);
v_clsEnabled_boxed_910_ = lean_unbox(v_clsEnabled_905_);
v_res_911_ = lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0(v_inst_895_, v_inst_896_, v_inst_897_, v_inst_898_, v_inst_899_, v_inst_900_, v_traceClass_901_, v_collapsed_boxed_909_, v___x_903_, v_opts_904_, v_clsEnabled_boxed_910_, v_oldTraces_906_, v_msg_907_, v_resStartStop_908_);
lean_dec_ref(v_opts_904_);
return v_res_911_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__1(lean_object* v_toPure_912_, lean_object* v_a_913_){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; 
v___x_914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_914_, 0, v_a_913_);
v___x_915_ = lean_apply_2(v_toPure_912_, lean_box(0), v___x_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__2(lean_object* v_toPure_916_, lean_object* v_ex_917_){
_start:
{
lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_918_, 0, v_ex_917_);
v___x_919_ = lean_apply_2(v_toPure_916_, lean_box(0), v___x_918_);
return v___x_919_;
}
}
static double _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0(void){
_start:
{
lean_object* v___x_920_; double v___x_921_; 
v___x_920_ = lean_unsigned_to_nat(1000000000u);
v___x_921_ = lean_float_of_nat(v___x_920_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3(lean_object* v_start_922_, lean_object* v_a_923_, lean_object* v_toPure_924_, lean_object* v_stop_925_){
_start:
{
double v___x_926_; double v___x_927_; double v___x_928_; double v___x_929_; double v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v___x_926_ = lean_float_of_nat(v_start_922_);
v___x_927_ = lean_float_once(&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0, &lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once, _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0);
v___x_928_ = lean_float_div(v___x_926_, v___x_927_);
v___x_929_ = lean_float_of_nat(v_stop_925_);
v___x_930_ = lean_float_div(v___x_929_, v___x_927_);
v___x_931_ = lean_box_float(v___x_928_);
v___x_932_ = lean_box_float(v___x_930_);
v___x_933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_931_);
lean_ctor_set(v___x_933_, 1, v___x_932_);
v___x_934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_934_, 0, v_a_923_);
lean_ctor_set(v___x_934_, 1, v___x_933_);
v___x_935_ = lean_apply_2(v_toPure_924_, lean_box(0), v___x_934_);
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__4(lean_object* v_start_936_, lean_object* v_toPure_937_, lean_object* v_toBind_938_, lean_object* v___x_939_, lean_object* v_a_940_){
_start:
{
lean_object* v___f_941_; lean_object* v___x_942_; 
v___f_941_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3), 4, 3);
lean_closure_set(v___f_941_, 0, v_start_936_);
lean_closure_set(v___f_941_, 1, v_a_940_);
lean_closure_set(v___f_941_, 2, v_toPure_937_);
v___x_942_ = lean_apply_4(v_toBind_938_, lean_box(0), lean_box(0), v___x_939_, v___f_941_);
return v___x_942_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__5(lean_object* v_toPure_943_, lean_object* v_toBind_944_, lean_object* v___x_945_, lean_object* v___x_946_, lean_object* v_start_947_){
_start:
{
lean_object* v___f_948_; lean_object* v___x_949_; 
lean_inc(v_toBind_944_);
v___f_948_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__4), 5, 4);
lean_closure_set(v___f_948_, 0, v_start_947_);
lean_closure_set(v___f_948_, 1, v_toPure_943_);
lean_closure_set(v___f_948_, 2, v_toBind_944_);
lean_closure_set(v___f_948_, 3, v___x_945_);
v___x_949_ = lean_apply_4(v_toBind_944_, lean_box(0), lean_box(0), v___x_946_, v___f_948_);
return v___x_949_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__6(lean_object* v_start_950_, lean_object* v_a_951_, lean_object* v_toPure_952_, lean_object* v_stop_953_){
_start:
{
double v___x_954_; double v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; 
v___x_954_ = lean_float_of_nat(v_start_950_);
v___x_955_ = lean_float_of_nat(v_stop_953_);
v___x_956_ = lean_box_float(v___x_954_);
v___x_957_ = lean_box_float(v___x_955_);
v___x_958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_958_, 0, v___x_956_);
lean_ctor_set(v___x_958_, 1, v___x_957_);
v___x_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_959_, 0, v_a_951_);
lean_ctor_set(v___x_959_, 1, v___x_958_);
v___x_960_ = lean_apply_2(v_toPure_952_, lean_box(0), v___x_959_);
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__7(lean_object* v_start_961_, lean_object* v_toPure_962_, lean_object* v_toBind_963_, lean_object* v___x_964_, lean_object* v_a_965_){
_start:
{
lean_object* v___f_966_; lean_object* v___x_967_; 
v___f_966_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__6), 4, 3);
lean_closure_set(v___f_966_, 0, v_start_961_);
lean_closure_set(v___f_966_, 1, v_a_965_);
lean_closure_set(v___f_966_, 2, v_toPure_962_);
v___x_967_ = lean_apply_4(v_toBind_963_, lean_box(0), lean_box(0), v___x_964_, v___f_966_);
return v___x_967_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__8(lean_object* v_toPure_968_, lean_object* v_toBind_969_, lean_object* v___x_970_, lean_object* v___x_971_, lean_object* v_start_972_){
_start:
{
lean_object* v___f_973_; lean_object* v___x_974_; 
lean_inc(v_toBind_969_);
v___f_973_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__7), 5, 4);
lean_closure_set(v___f_973_, 0, v_start_972_);
lean_closure_set(v___f_973_, 1, v_toPure_968_);
lean_closure_set(v___f_973_, 2, v_toBind_969_);
lean_closure_set(v___f_973_, 3, v___x_970_);
v___x_974_ = lean_apply_4(v_toBind_969_, lean_box(0), lean_box(0), v___x_971_, v___f_973_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9(lean_object* v_inst_977_, lean_object* v_inst_978_, lean_object* v_inst_979_, lean_object* v_inst_980_, lean_object* v_inst_981_, lean_object* v_inst_982_, lean_object* v_traceClass_983_, uint8_t v_collapsed_984_, lean_object* v___x_985_, lean_object* v_opts_986_, uint8_t v_clsEnabled_987_, lean_object* v_msg_988_, lean_object* v_toPure_989_, lean_object* v_toBind_990_, lean_object* v_k_991_, lean_object* v_inst_992_, lean_object* v_oldTraces_993_){
_start:
{
lean_object* v_tryCatch_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___f_997_; lean_object* v___f_998_; lean_object* v___f_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; uint8_t v___x_1005_; 
v_tryCatch_994_ = lean_ctor_get(v_inst_977_, 1);
lean_inc(v_tryCatch_994_);
v___x_995_ = lean_box(v_collapsed_984_);
v___x_996_ = lean_box(v_clsEnabled_987_);
lean_inc_ref(v_opts_986_);
v___f_997_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__0___boxed), 14, 13);
lean_closure_set(v___f_997_, 0, v_inst_978_);
lean_closure_set(v___f_997_, 1, v_inst_979_);
lean_closure_set(v___f_997_, 2, v_inst_980_);
lean_closure_set(v___f_997_, 3, v_inst_981_);
lean_closure_set(v___f_997_, 4, v_inst_977_);
lean_closure_set(v___f_997_, 5, v_inst_982_);
lean_closure_set(v___f_997_, 6, v_traceClass_983_);
lean_closure_set(v___f_997_, 7, v___x_995_);
lean_closure_set(v___f_997_, 8, v___x_985_);
lean_closure_set(v___f_997_, 9, v_opts_986_);
lean_closure_set(v___f_997_, 10, v___x_996_);
lean_closure_set(v___f_997_, 11, v_oldTraces_993_);
lean_closure_set(v___f_997_, 12, v_msg_988_);
lean_inc_n(v_toPure_989_, 2);
v___f_998_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__1), 2, 1);
lean_closure_set(v___f_998_, 0, v_toPure_989_);
v___f_999_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__2), 2, 1);
lean_closure_set(v___f_999_, 0, v_toPure_989_);
lean_inc(v_toBind_990_);
v___x_1000_ = lean_apply_4(v_toBind_990_, lean_box(0), lean_box(0), v_k_991_, v___f_998_);
v___x_1001_ = lean_apply_3(v_tryCatch_994_, lean_box(0), v___x_1000_, v___f_999_);
v___x_1002_ = l_Lean_KVMap_instValueBool;
v___x_1003_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1004_ = l_Lean_Option_get___redArg(v___x_1002_, v_opts_986_, v___x_1003_);
lean_dec_ref(v_opts_986_);
v___x_1005_ = lean_unbox(v___x_1004_);
lean_dec(v___x_1004_);
if (v___x_1005_ == 0)
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___f_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1006_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0));
v___x_1007_ = lean_apply_2(v_inst_992_, lean_box(0), v___x_1006_);
lean_inc(v___x_1007_);
lean_inc_n(v_toBind_990_, 2);
v___f_1008_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__5), 5, 4);
lean_closure_set(v___f_1008_, 0, v_toPure_989_);
lean_closure_set(v___f_1008_, 1, v_toBind_990_);
lean_closure_set(v___f_1008_, 2, v___x_1007_);
lean_closure_set(v___f_1008_, 3, v___x_1001_);
v___x_1009_ = lean_apply_4(v_toBind_990_, lean_box(0), lean_box(0), v___x_1007_, v___f_1008_);
v___x_1010_ = lean_apply_4(v_toBind_990_, lean_box(0), lean_box(0), v___x_1009_, v___f_997_);
return v___x_1010_;
}
else
{
lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___f_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1011_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1));
v___x_1012_ = lean_apply_2(v_inst_992_, lean_box(0), v___x_1011_);
lean_inc(v___x_1012_);
lean_inc_n(v_toBind_990_, 2);
v___f_1013_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__8), 5, 4);
lean_closure_set(v___f_1013_, 0, v_toPure_989_);
lean_closure_set(v___f_1013_, 1, v_toBind_990_);
lean_closure_set(v___f_1013_, 2, v___x_1012_);
lean_closure_set(v___f_1013_, 3, v___x_1001_);
v___x_1014_ = lean_apply_4(v_toBind_990_, lean_box(0), lean_box(0), v___x_1012_, v___f_1013_);
v___x_1015_ = lean_apply_4(v_toBind_990_, lean_box(0), lean_box(0), v___x_1014_, v___f_997_);
return v___x_1015_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___boxed(lean_object** _args){
lean_object* v_inst_1016_ = _args[0];
lean_object* v_inst_1017_ = _args[1];
lean_object* v_inst_1018_ = _args[2];
lean_object* v_inst_1019_ = _args[3];
lean_object* v_inst_1020_ = _args[4];
lean_object* v_inst_1021_ = _args[5];
lean_object* v_traceClass_1022_ = _args[6];
lean_object* v_collapsed_1023_ = _args[7];
lean_object* v___x_1024_ = _args[8];
lean_object* v_opts_1025_ = _args[9];
lean_object* v_clsEnabled_1026_ = _args[10];
lean_object* v_msg_1027_ = _args[11];
lean_object* v_toPure_1028_ = _args[12];
lean_object* v_toBind_1029_ = _args[13];
lean_object* v_k_1030_ = _args[14];
lean_object* v_inst_1031_ = _args[15];
lean_object* v_oldTraces_1032_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_1033_; uint8_t v_clsEnabled_boxed_1034_; lean_object* v_res_1035_; 
v_collapsed_boxed_1033_ = lean_unbox(v_collapsed_1023_);
v_clsEnabled_boxed_1034_ = lean_unbox(v_clsEnabled_1026_);
v_res_1035_ = lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9(v_inst_1016_, v_inst_1017_, v_inst_1018_, v_inst_1019_, v_inst_1020_, v_inst_1021_, v_traceClass_1022_, v_collapsed_boxed_1033_, v___x_1024_, v_opts_1025_, v_clsEnabled_boxed_1034_, v_msg_1027_, v_toPure_1028_, v_toBind_1029_, v_k_1030_, v_inst_1031_, v_oldTraces_1032_);
return v_res_1035_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10(lean_object* v_inst_1036_, lean_object* v_inst_1037_, lean_object* v_inst_1038_, lean_object* v_inst_1039_, lean_object* v_inst_1040_, lean_object* v_inst_1041_, lean_object* v_traceClass_1042_, uint8_t v_collapsed_1043_, lean_object* v___x_1044_, lean_object* v_opts_1045_, lean_object* v_msg_1046_, lean_object* v_toPure_1047_, lean_object* v_toBind_1048_, lean_object* v_k_1049_, lean_object* v_inst_1050_, uint8_t v_clsEnabled_1051_){
_start:
{
lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___f_1054_; 
v___x_1052_ = lean_box(v_collapsed_1043_);
v___x_1053_ = lean_box(v_clsEnabled_1051_);
lean_inc(v_k_1049_);
lean_inc(v_toBind_1048_);
lean_inc_ref(v_opts_1045_);
lean_inc_ref(v_inst_1038_);
lean_inc_ref(v_inst_1037_);
v___f_1054_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___boxed), 17, 16);
lean_closure_set(v___f_1054_, 0, v_inst_1036_);
lean_closure_set(v___f_1054_, 1, v_inst_1037_);
lean_closure_set(v___f_1054_, 2, v_inst_1038_);
lean_closure_set(v___f_1054_, 3, v_inst_1039_);
lean_closure_set(v___f_1054_, 4, v_inst_1040_);
lean_closure_set(v___f_1054_, 5, v_inst_1041_);
lean_closure_set(v___f_1054_, 6, v_traceClass_1042_);
lean_closure_set(v___f_1054_, 7, v___x_1052_);
lean_closure_set(v___f_1054_, 8, v___x_1044_);
lean_closure_set(v___f_1054_, 9, v_opts_1045_);
lean_closure_set(v___f_1054_, 10, v___x_1053_);
lean_closure_set(v___f_1054_, 11, v_msg_1046_);
lean_closure_set(v___f_1054_, 12, v_toPure_1047_);
lean_closure_set(v___f_1054_, 13, v_toBind_1048_);
lean_closure_set(v___f_1054_, 14, v_k_1049_);
lean_closure_set(v___f_1054_, 15, v_inst_1050_);
if (v_clsEnabled_1051_ == 0)
{
lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; uint8_t v___x_1061_; 
v___x_1058_ = l_Lean_KVMap_instValueBool;
v___x_1059_ = l_Lean_trace_profiler;
v___x_1060_ = l_Lean_Option_get___redArg(v___x_1058_, v_opts_1045_, v___x_1059_);
lean_dec_ref(v_opts_1045_);
v___x_1061_ = lean_unbox(v___x_1060_);
lean_dec(v___x_1060_);
if (v___x_1061_ == 0)
{
lean_dec_ref(v___f_1054_);
lean_dec(v_toBind_1048_);
lean_dec_ref(v_inst_1038_);
lean_dec_ref(v_inst_1037_);
return v_k_1049_;
}
else
{
lean_dec(v_k_1049_);
goto v___jp_1055_;
}
}
else
{
lean_dec(v_k_1049_);
lean_dec_ref(v_opts_1045_);
goto v___jp_1055_;
}
v___jp_1055_:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1056_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v_inst_1037_, v_inst_1038_);
v___x_1057_ = lean_apply_4(v_toBind_1048_, lean_box(0), lean_box(0), v___x_1056_, v___f_1054_);
return v___x_1057_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10___boxed(lean_object* v_inst_1062_, lean_object* v_inst_1063_, lean_object* v_inst_1064_, lean_object* v_inst_1065_, lean_object* v_inst_1066_, lean_object* v_inst_1067_, lean_object* v_traceClass_1068_, lean_object* v_collapsed_1069_, lean_object* v___x_1070_, lean_object* v_opts_1071_, lean_object* v_msg_1072_, lean_object* v_toPure_1073_, lean_object* v_toBind_1074_, lean_object* v_k_1075_, lean_object* v_inst_1076_, lean_object* v_clsEnabled_1077_){
_start:
{
uint8_t v_collapsed_boxed_1078_; uint8_t v_clsEnabled_boxed_1079_; lean_object* v_res_1080_; 
v_collapsed_boxed_1078_ = lean_unbox(v_collapsed_1069_);
v_clsEnabled_boxed_1079_ = lean_unbox(v_clsEnabled_1077_);
v_res_1080_ = lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10(v_inst_1062_, v_inst_1063_, v_inst_1064_, v_inst_1065_, v_inst_1066_, v_inst_1067_, v_traceClass_1068_, v_collapsed_boxed_1078_, v___x_1070_, v_opts_1071_, v_msg_1072_, v_toPure_1073_, v_toBind_1074_, v_k_1075_, v_inst_1076_, v_clsEnabled_boxed_1079_);
return v_res_1080_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11(lean_object* v_toPure_1083_, lean_object* v_traceClass_1084_, lean_object* v_____do__lift_1085_, lean_object* v_____do__lift_1086_){
_start:
{
uint8_t v_hasTrace_1087_; 
v_hasTrace_1087_ = lean_ctor_get_uint8(v_____do__lift_1086_, sizeof(void*)*1);
if (v_hasTrace_1087_ == 0)
{
lean_object* v___x_1088_; lean_object* v___x_1089_; 
lean_dec(v_traceClass_1084_);
v___x_1088_ = lean_box(v_hasTrace_1087_);
v___x_1089_ = lean_apply_2(v_toPure_1083_, lean_box(0), v___x_1088_);
return v___x_1089_;
}
else
{
lean_object* v___x_1090_; lean_object* v___x_1091_; uint8_t v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1090_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0));
v___x_1091_ = l_Lean_Name_append(v___x_1090_, v_traceClass_1084_);
v___x_1092_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_____do__lift_1085_, v_____do__lift_1086_, v___x_1091_);
lean_dec(v___x_1091_);
v___x_1093_ = lean_box(v___x_1092_);
v___x_1094_ = lean_apply_2(v_toPure_1083_, lean_box(0), v___x_1093_);
return v___x_1094_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___boxed(lean_object* v_toPure_1095_, lean_object* v_traceClass_1096_, lean_object* v_____do__lift_1097_, lean_object* v_____do__lift_1098_){
_start:
{
lean_object* v_res_1099_; 
v_res_1099_ = lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11(v_toPure_1095_, v_traceClass_1096_, v_____do__lift_1097_, v_____do__lift_1098_);
lean_dec_ref(v_____do__lift_1098_);
lean_dec_ref(v_____do__lift_1097_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__12(lean_object* v_toPure_1100_, lean_object* v_traceClass_1101_, lean_object* v_toBind_1102_, lean_object* v_inst_1103_, lean_object* v_____do__lift_1104_){
_start:
{
lean_object* v___f_1105_; lean_object* v___x_1106_; 
v___f_1105_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___boxed), 4, 3);
lean_closure_set(v___f_1105_, 0, v_toPure_1100_);
lean_closure_set(v___f_1105_, 1, v_traceClass_1101_);
lean_closure_set(v___f_1105_, 2, v_____do__lift_1104_);
v___x_1106_ = lean_apply_4(v_toBind_1102_, lean_box(0), lean_box(0), v_inst_1103_, v___f_1105_);
return v___x_1106_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13(lean_object* v_k_1107_, lean_object* v_inst_1108_, lean_object* v_toApplicative_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_inst_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_, lean_object* v_traceClass_1115_, uint8_t v_collapsed_1116_, lean_object* v___x_1117_, lean_object* v_msg_1118_, lean_object* v_toBind_1119_, lean_object* v_inst_1120_, lean_object* v_inst_1121_, lean_object* v_opts_1122_){
_start:
{
uint8_t v_hasTrace_1123_; 
v_hasTrace_1123_ = lean_ctor_get_uint8(v_opts_1122_, sizeof(void*)*1);
if (v_hasTrace_1123_ == 0)
{
lean_dec_ref(v_opts_1122_);
lean_dec(v_inst_1121_);
lean_dec(v_inst_1120_);
lean_dec(v_toBind_1119_);
lean_dec(v_msg_1118_);
lean_dec_ref(v___x_1117_);
lean_dec(v_traceClass_1115_);
lean_dec_ref(v_inst_1114_);
lean_dec(v_inst_1113_);
lean_dec_ref(v_inst_1112_);
lean_dec_ref(v_inst_1111_);
lean_dec_ref(v_inst_1110_);
lean_dec_ref(v_toApplicative_1109_);
lean_dec_ref(v_inst_1108_);
return v_k_1107_;
}
else
{
lean_object* v_getInheritedTraceOptions_1124_; lean_object* v_toPure_1125_; lean_object* v___x_1126_; lean_object* v___f_1127_; lean_object* v___f_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; 
v_getInheritedTraceOptions_1124_ = lean_ctor_get(v_inst_1108_, 2);
lean_inc(v_getInheritedTraceOptions_1124_);
v_toPure_1125_ = lean_ctor_get(v_toApplicative_1109_, 1);
lean_inc_n(v_toPure_1125_, 2);
lean_dec_ref(v_toApplicative_1109_);
v___x_1126_ = lean_box(v_collapsed_1116_);
lean_inc_n(v_toBind_1119_, 3);
lean_inc(v_traceClass_1115_);
v___f_1127_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__10___boxed), 16, 15);
lean_closure_set(v___f_1127_, 0, v_inst_1110_);
lean_closure_set(v___f_1127_, 1, v_inst_1111_);
lean_closure_set(v___f_1127_, 2, v_inst_1108_);
lean_closure_set(v___f_1127_, 3, v_inst_1112_);
lean_closure_set(v___f_1127_, 4, v_inst_1113_);
lean_closure_set(v___f_1127_, 5, v_inst_1114_);
lean_closure_set(v___f_1127_, 6, v_traceClass_1115_);
lean_closure_set(v___f_1127_, 7, v___x_1126_);
lean_closure_set(v___f_1127_, 8, v___x_1117_);
lean_closure_set(v___f_1127_, 9, v_opts_1122_);
lean_closure_set(v___f_1127_, 10, v_msg_1118_);
lean_closure_set(v___f_1127_, 11, v_toPure_1125_);
lean_closure_set(v___f_1127_, 12, v_toBind_1119_);
lean_closure_set(v___f_1127_, 13, v_k_1107_);
lean_closure_set(v___f_1127_, 14, v_inst_1120_);
v___f_1128_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__12), 5, 4);
lean_closure_set(v___f_1128_, 0, v_toPure_1125_);
lean_closure_set(v___f_1128_, 1, v_traceClass_1115_);
lean_closure_set(v___f_1128_, 2, v_toBind_1119_);
lean_closure_set(v___f_1128_, 3, v_inst_1121_);
v___x_1129_ = lean_apply_4(v_toBind_1119_, lean_box(0), lean_box(0), v_getInheritedTraceOptions_1124_, v___f_1128_);
v___x_1130_ = lean_apply_4(v_toBind_1119_, lean_box(0), lean_box(0), v___x_1129_, v___f_1127_);
return v___x_1130_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13___boxed(lean_object* v_k_1131_, lean_object* v_inst_1132_, lean_object* v_toApplicative_1133_, lean_object* v_inst_1134_, lean_object* v_inst_1135_, lean_object* v_inst_1136_, lean_object* v_inst_1137_, lean_object* v_inst_1138_, lean_object* v_traceClass_1139_, lean_object* v_collapsed_1140_, lean_object* v___x_1141_, lean_object* v_msg_1142_, lean_object* v_toBind_1143_, lean_object* v_inst_1144_, lean_object* v_inst_1145_, lean_object* v_opts_1146_){
_start:
{
uint8_t v_collapsed_boxed_1147_; lean_object* v_res_1148_; 
v_collapsed_boxed_1147_ = lean_unbox(v_collapsed_1140_);
v_res_1148_ = lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13(v_k_1131_, v_inst_1132_, v_toApplicative_1133_, v_inst_1134_, v_inst_1135_, v_inst_1136_, v_inst_1137_, v_inst_1138_, v_traceClass_1139_, v_collapsed_boxed_1147_, v___x_1141_, v_msg_1142_, v_toBind_1143_, v_inst_1144_, v_inst_1145_, v_opts_1146_);
return v_res_1148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg(lean_object* v_inst_1149_, lean_object* v_inst_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_, lean_object* v_inst_1153_, lean_object* v_inst_1154_, lean_object* v_inst_1155_, lean_object* v_inst_1156_, lean_object* v_opt_1157_, lean_object* v_msg_1158_, lean_object* v_k_1159_, uint8_t v_collapsed_1160_){
_start:
{
lean_object* v_traceClass_1161_; lean_object* v_toApplicative_1162_; lean_object* v_toBind_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___f_1166_; lean_object* v___x_1167_; 
v_traceClass_1161_ = lean_ctor_get(v_opt_1157_, 0);
lean_inc(v_traceClass_1161_);
lean_dec_ref(v_opt_1157_);
v_toApplicative_1162_ = lean_ctor_get(v_inst_1149_, 0);
lean_inc_ref(v_toApplicative_1162_);
v_toBind_1163_ = lean_ctor_get(v_inst_1149_, 1);
lean_inc_n(v_toBind_1163_, 2);
v___x_1164_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1165_ = lean_box(v_collapsed_1160_);
lean_inc(v_inst_1154_);
v___f_1166_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13___boxed), 16, 15);
lean_closure_set(v___f_1166_, 0, v_k_1159_);
lean_closure_set(v___f_1166_, 1, v_inst_1150_);
lean_closure_set(v___f_1166_, 2, v_toApplicative_1162_);
lean_closure_set(v___f_1166_, 3, v_inst_1155_);
lean_closure_set(v___f_1166_, 4, v_inst_1149_);
lean_closure_set(v___f_1166_, 5, v_inst_1152_);
lean_closure_set(v___f_1166_, 6, v_inst_1153_);
lean_closure_set(v___f_1166_, 7, v_inst_1156_);
lean_closure_set(v___f_1166_, 8, v_traceClass_1161_);
lean_closure_set(v___f_1166_, 9, v___x_1165_);
lean_closure_set(v___f_1166_, 10, v___x_1164_);
lean_closure_set(v___f_1166_, 11, v_msg_1158_);
lean_closure_set(v___f_1166_, 12, v_toBind_1163_);
lean_closure_set(v___f_1166_, 13, v_inst_1151_);
lean_closure_set(v___f_1166_, 14, v_inst_1154_);
v___x_1167_ = lean_apply_4(v_toBind_1163_, lean_box(0), lean_box(0), v_inst_1154_, v___f_1166_);
return v___x_1167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___redArg___boxed(lean_object* v_inst_1168_, lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_inst_1171_, lean_object* v_inst_1172_, lean_object* v_inst_1173_, lean_object* v_inst_1174_, lean_object* v_inst_1175_, lean_object* v_opt_1176_, lean_object* v_msg_1177_, lean_object* v_k_1178_, lean_object* v_collapsed_1179_){
_start:
{
uint8_t v_collapsed_boxed_1180_; lean_object* v_res_1181_; 
v_collapsed_boxed_1180_ = lean_unbox(v_collapsed_1179_);
v_res_1181_ = lp_aesop_Aesop_withAesopTraceNode___redArg(v_inst_1168_, v_inst_1169_, v_inst_1170_, v_inst_1171_, v_inst_1172_, v_inst_1173_, v_inst_1174_, v_inst_1175_, v_opt_1176_, v_msg_1177_, v_k_1178_, v_collapsed_boxed_1180_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode(lean_object* v_m_1182_, lean_object* v_00_u03b5_1183_, lean_object* v_inst_1184_, lean_object* v_inst_1185_, lean_object* v_inst_1186_, lean_object* v_inst_1187_, lean_object* v_inst_1188_, lean_object* v_inst_1189_, lean_object* v_inst_1190_, lean_object* v_00_u03b1_1191_, lean_object* v_inst_1192_, lean_object* v_opt_1193_, lean_object* v_msg_1194_, lean_object* v_k_1195_, uint8_t v_collapsed_1196_){
_start:
{
lean_object* v_traceClass_1197_; lean_object* v_toApplicative_1198_; lean_object* v_toBind_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___f_1202_; lean_object* v___x_1203_; 
v_traceClass_1197_ = lean_ctor_get(v_opt_1193_, 0);
lean_inc(v_traceClass_1197_);
lean_dec_ref(v_opt_1193_);
v_toApplicative_1198_ = lean_ctor_get(v_inst_1184_, 0);
lean_inc_ref(v_toApplicative_1198_);
v_toBind_1199_ = lean_ctor_get(v_inst_1184_, 1);
lean_inc_n(v_toBind_1199_, 2);
v___x_1200_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1201_ = lean_box(v_collapsed_1196_);
lean_inc(v_inst_1189_);
v___f_1202_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__13___boxed), 16, 15);
lean_closure_set(v___f_1202_, 0, v_k_1195_);
lean_closure_set(v___f_1202_, 1, v_inst_1185_);
lean_closure_set(v___f_1202_, 2, v_toApplicative_1198_);
lean_closure_set(v___f_1202_, 3, v_inst_1190_);
lean_closure_set(v___f_1202_, 4, v_inst_1184_);
lean_closure_set(v___f_1202_, 5, v_inst_1187_);
lean_closure_set(v___f_1202_, 6, v_inst_1188_);
lean_closure_set(v___f_1202_, 7, v_inst_1192_);
lean_closure_set(v___f_1202_, 8, v_traceClass_1197_);
lean_closure_set(v___f_1202_, 9, v___x_1201_);
lean_closure_set(v___f_1202_, 10, v___x_1200_);
lean_closure_set(v___f_1202_, 11, v_msg_1194_);
lean_closure_set(v___f_1202_, 12, v_toBind_1199_);
lean_closure_set(v___f_1202_, 13, v_inst_1186_);
lean_closure_set(v___f_1202_, 14, v_inst_1189_);
v___x_1203_ = lean_apply_4(v_toBind_1199_, lean_box(0), lean_box(0), v_inst_1189_, v___f_1202_);
return v___x_1203_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNode___boxed(lean_object* v_m_1204_, lean_object* v_00_u03b5_1205_, lean_object* v_inst_1206_, lean_object* v_inst_1207_, lean_object* v_inst_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_inst_1211_, lean_object* v_inst_1212_, lean_object* v_00_u03b1_1213_, lean_object* v_inst_1214_, lean_object* v_opt_1215_, lean_object* v_msg_1216_, lean_object* v_k_1217_, lean_object* v_collapsed_1218_){
_start:
{
uint8_t v_collapsed_boxed_1219_; lean_object* v_res_1220_; 
v_collapsed_boxed_1219_ = lean_unbox(v_collapsed_1218_);
v_res_1220_ = lp_aesop_Aesop_withAesopTraceNode(v_m_1204_, v_00_u03b5_1205_, v_inst_1206_, v_inst_1207_, v_inst_1208_, v_inst_1209_, v_inst_1210_, v_inst_1211_, v_inst_1212_, v_00_u03b1_1213_, v_inst_1214_, v_opt_1215_, v_msg_1216_, v_k_1217_, v_collapsed_boxed_1219_);
return v_res_1220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__0(lean_object* v_inst_1221_, lean_object* v_____do__lift_1222_){
_start:
{
lean_object* v___x_1223_; 
v___x_1223_ = lean_apply_1(v_inst_1221_, v_____do__lift_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1(lean_object* v_inst_1224_, lean_object* v_inst_1225_, lean_object* v_inst_1226_, lean_object* v_inst_1227_, lean_object* v_inst_1228_, lean_object* v_inst_1229_, lean_object* v_traceClass_1230_, uint8_t v_collapsed_1231_, lean_object* v___x_1232_, lean_object* v_opts_1233_, uint8_t v_clsEnabled_1234_, lean_object* v_oldTraces_1235_, lean_object* v_ref_1236_, lean_object* v_msg_1237_, lean_object* v_resStartStop_1238_){
_start:
{
lean_object* v___x_1239_; 
v___x_1239_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v_inst_1224_, v_inst_1225_, lean_box(0), v_inst_1226_, v_inst_1227_, v_inst_1228_, v_inst_1229_, v_traceClass_1230_, v_collapsed_1231_, v___x_1232_, v_opts_1233_, v_clsEnabled_1234_, v_oldTraces_1235_, v_ref_1236_, v_msg_1237_, v_resStartStop_1238_);
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1___boxed(lean_object* v_inst_1240_, lean_object* v_inst_1241_, lean_object* v_inst_1242_, lean_object* v_inst_1243_, lean_object* v_inst_1244_, lean_object* v_inst_1245_, lean_object* v_traceClass_1246_, lean_object* v_collapsed_1247_, lean_object* v___x_1248_, lean_object* v_opts_1249_, lean_object* v_clsEnabled_1250_, lean_object* v_oldTraces_1251_, lean_object* v_ref_1252_, lean_object* v_msg_1253_, lean_object* v_resStartStop_1254_){
_start:
{
uint8_t v_collapsed_boxed_1255_; uint8_t v_clsEnabled_boxed_1256_; lean_object* v_res_1257_; 
v_collapsed_boxed_1255_ = lean_unbox(v_collapsed_1247_);
v_clsEnabled_boxed_1256_ = lean_unbox(v_clsEnabled_1250_);
v_res_1257_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1(v_inst_1240_, v_inst_1241_, v_inst_1242_, v_inst_1243_, v_inst_1244_, v_inst_1245_, v_traceClass_1246_, v_collapsed_boxed_1255_, v___x_1248_, v_opts_1249_, v_clsEnabled_boxed_1256_, v_oldTraces_1251_, v_ref_1252_, v_msg_1253_, v_resStartStop_1254_);
lean_dec_ref(v_opts_1249_);
return v_res_1257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10(lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_inst_1260_, lean_object* v_inst_1261_, lean_object* v_inst_1262_, lean_object* v_inst_1263_, lean_object* v_traceClass_1264_, uint8_t v_collapsed_1265_, lean_object* v___x_1266_, lean_object* v_opts_1267_, uint8_t v_clsEnabled_1268_, lean_object* v_oldTraces_1269_, lean_object* v_ref_1270_, lean_object* v_toPure_1271_, lean_object* v_toBind_1272_, lean_object* v_k_1273_, lean_object* v_inst_1274_, lean_object* v_msg_1275_){
_start:
{
lean_object* v_tryCatch_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___f_1279_; lean_object* v___f_1280_; lean_object* v___f_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; uint8_t v___x_1287_; 
v_tryCatch_1276_ = lean_ctor_get(v_inst_1258_, 1);
lean_inc(v_tryCatch_1276_);
v___x_1277_ = lean_box(v_collapsed_1265_);
v___x_1278_ = lean_box(v_clsEnabled_1268_);
lean_inc_ref(v_opts_1267_);
v___f_1279_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__1___boxed), 15, 14);
lean_closure_set(v___f_1279_, 0, v_inst_1259_);
lean_closure_set(v___f_1279_, 1, v_inst_1260_);
lean_closure_set(v___f_1279_, 2, v_inst_1261_);
lean_closure_set(v___f_1279_, 3, v_inst_1262_);
lean_closure_set(v___f_1279_, 4, v_inst_1258_);
lean_closure_set(v___f_1279_, 5, v_inst_1263_);
lean_closure_set(v___f_1279_, 6, v_traceClass_1264_);
lean_closure_set(v___f_1279_, 7, v___x_1277_);
lean_closure_set(v___f_1279_, 8, v___x_1266_);
lean_closure_set(v___f_1279_, 9, v_opts_1267_);
lean_closure_set(v___f_1279_, 10, v___x_1278_);
lean_closure_set(v___f_1279_, 11, v_oldTraces_1269_);
lean_closure_set(v___f_1279_, 12, v_ref_1270_);
lean_closure_set(v___f_1279_, 13, v_msg_1275_);
lean_inc_n(v_toPure_1271_, 2);
v___f_1280_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1280_, 0, v_toPure_1271_);
v___f_1281_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1281_, 0, v_toPure_1271_);
lean_inc(v_toBind_1272_);
v___x_1282_ = lean_apply_4(v_toBind_1272_, lean_box(0), lean_box(0), v_k_1273_, v___f_1280_);
v___x_1283_ = lean_apply_3(v_tryCatch_1276_, lean_box(0), v___x_1282_, v___f_1281_);
v___x_1284_ = l_Lean_KVMap_instValueBool;
v___x_1285_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1286_ = l_Lean_Option_get___redArg(v___x_1284_, v_opts_1267_, v___x_1285_);
lean_dec_ref(v_opts_1267_);
v___x_1287_ = lean_unbox(v___x_1286_);
lean_dec(v___x_1286_);
if (v___x_1287_ == 0)
{
lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___f_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; 
v___x_1288_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0));
v___x_1289_ = lean_apply_2(v_inst_1274_, lean_box(0), v___x_1288_);
lean_inc(v___x_1289_);
lean_inc_n(v_toBind_1272_, 2);
v___f_1290_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__5), 5, 4);
lean_closure_set(v___f_1290_, 0, v_toPure_1271_);
lean_closure_set(v___f_1290_, 1, v_toBind_1272_);
lean_closure_set(v___f_1290_, 2, v___x_1289_);
lean_closure_set(v___f_1290_, 3, v___x_1283_);
v___x_1291_ = lean_apply_4(v_toBind_1272_, lean_box(0), lean_box(0), v___x_1289_, v___f_1290_);
v___x_1292_ = lean_apply_4(v_toBind_1272_, lean_box(0), lean_box(0), v___x_1291_, v___f_1279_);
return v___x_1292_;
}
else
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___f_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; 
v___x_1293_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1));
v___x_1294_ = lean_apply_2(v_inst_1274_, lean_box(0), v___x_1293_);
lean_inc(v___x_1294_);
lean_inc_n(v_toBind_1272_, 2);
v___f_1295_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__8), 5, 4);
lean_closure_set(v___f_1295_, 0, v_toPure_1271_);
lean_closure_set(v___f_1295_, 1, v_toBind_1272_);
lean_closure_set(v___f_1295_, 2, v___x_1294_);
lean_closure_set(v___f_1295_, 3, v___x_1283_);
v___x_1296_ = lean_apply_4(v_toBind_1272_, lean_box(0), lean_box(0), v___x_1294_, v___f_1295_);
v___x_1297_ = lean_apply_4(v_toBind_1272_, lean_box(0), lean_box(0), v___x_1296_, v___f_1279_);
return v___x_1297_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10___boxed(lean_object** _args){
lean_object* v_inst_1298_ = _args[0];
lean_object* v_inst_1299_ = _args[1];
lean_object* v_inst_1300_ = _args[2];
lean_object* v_inst_1301_ = _args[3];
lean_object* v_inst_1302_ = _args[4];
lean_object* v_inst_1303_ = _args[5];
lean_object* v_traceClass_1304_ = _args[6];
lean_object* v_collapsed_1305_ = _args[7];
lean_object* v___x_1306_ = _args[8];
lean_object* v_opts_1307_ = _args[9];
lean_object* v_clsEnabled_1308_ = _args[10];
lean_object* v_oldTraces_1309_ = _args[11];
lean_object* v_ref_1310_ = _args[12];
lean_object* v_toPure_1311_ = _args[13];
lean_object* v_toBind_1312_ = _args[14];
lean_object* v_k_1313_ = _args[15];
lean_object* v_inst_1314_ = _args[16];
lean_object* v_msg_1315_ = _args[17];
_start:
{
uint8_t v_collapsed_boxed_1316_; uint8_t v_clsEnabled_boxed_1317_; lean_object* v_res_1318_; 
v_collapsed_boxed_1316_ = lean_unbox(v_collapsed_1305_);
v_clsEnabled_boxed_1317_ = lean_unbox(v_clsEnabled_1308_);
v_res_1318_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10(v_inst_1298_, v_inst_1299_, v_inst_1300_, v_inst_1301_, v_inst_1302_, v_inst_1303_, v_traceClass_1304_, v_collapsed_boxed_1316_, v___x_1306_, v_opts_1307_, v_clsEnabled_boxed_1317_, v_oldTraces_1309_, v_ref_1310_, v_toPure_1311_, v_toBind_1312_, v_k_1313_, v_inst_1314_, v_msg_1315_);
return v_res_1318_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2(lean_object* v_ref_1319_, lean_object* v_withRef_1320_, lean_object* v___x_1321_, lean_object* v_oldRef_1322_){
_start:
{
lean_object* v_ref_1323_; lean_object* v___x_1324_; 
v_ref_1323_ = l_Lean_replaceRef(v_ref_1319_, v_oldRef_1322_);
v___x_1324_ = lean_apply_3(v_withRef_1320_, lean_box(0), v_ref_1323_, v___x_1321_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2___boxed(lean_object* v_ref_1325_, lean_object* v_withRef_1326_, lean_object* v___x_1327_, lean_object* v_oldRef_1328_){
_start:
{
lean_object* v_res_1329_; 
v_res_1329_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2(v_ref_1325_, v_withRef_1326_, v___x_1327_, v_oldRef_1328_);
lean_dec(v_oldRef_1328_);
lean_dec(v_ref_1325_);
return v_res_1329_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3(lean_object* v_inst_1330_, lean_object* v_inst_1331_, lean_object* v_inst_1332_, lean_object* v_inst_1333_, lean_object* v_inst_1334_, lean_object* v_inst_1335_, lean_object* v_traceClass_1336_, uint8_t v_collapsed_1337_, lean_object* v___x_1338_, lean_object* v_opts_1339_, uint8_t v_clsEnabled_1340_, lean_object* v_oldTraces_1341_, lean_object* v_toPure_1342_, lean_object* v_toBind_1343_, lean_object* v_k_1344_, lean_object* v_inst_1345_, lean_object* v_msg_1346_, lean_object* v___f_1347_, lean_object* v_withRef_1348_, lean_object* v_getRef_1349_, lean_object* v_ref_1350_){
_start:
{
lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___f_1353_; lean_object* v___x_1354_; lean_object* v___f_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; 
v___x_1351_ = lean_box(v_collapsed_1337_);
v___x_1352_ = lean_box(v_clsEnabled_1340_);
lean_inc_n(v_toBind_1343_, 3);
lean_inc(v_ref_1350_);
v___f_1353_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__10___boxed), 18, 17);
lean_closure_set(v___f_1353_, 0, v_inst_1330_);
lean_closure_set(v___f_1353_, 1, v_inst_1331_);
lean_closure_set(v___f_1353_, 2, v_inst_1332_);
lean_closure_set(v___f_1353_, 3, v_inst_1333_);
lean_closure_set(v___f_1353_, 4, v_inst_1334_);
lean_closure_set(v___f_1353_, 5, v_inst_1335_);
lean_closure_set(v___f_1353_, 6, v_traceClass_1336_);
lean_closure_set(v___f_1353_, 7, v___x_1351_);
lean_closure_set(v___f_1353_, 8, v___x_1338_);
lean_closure_set(v___f_1353_, 9, v_opts_1339_);
lean_closure_set(v___f_1353_, 10, v___x_1352_);
lean_closure_set(v___f_1353_, 11, v_oldTraces_1341_);
lean_closure_set(v___f_1353_, 12, v_ref_1350_);
lean_closure_set(v___f_1353_, 13, v_toPure_1342_);
lean_closure_set(v___f_1353_, 14, v_toBind_1343_);
lean_closure_set(v___f_1353_, 15, v_k_1344_);
lean_closure_set(v___f_1353_, 16, v_inst_1345_);
v___x_1354_ = lean_apply_4(v_toBind_1343_, lean_box(0), lean_box(0), v_msg_1346_, v___f_1347_);
v___f_1355_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_1355_, 0, v_ref_1350_);
lean_closure_set(v___f_1355_, 1, v_withRef_1348_);
lean_closure_set(v___f_1355_, 2, v___x_1354_);
v___x_1356_ = lean_apply_4(v_toBind_1343_, lean_box(0), lean_box(0), v_getRef_1349_, v___f_1355_);
v___x_1357_ = lean_apply_4(v_toBind_1343_, lean_box(0), lean_box(0), v___x_1356_, v___f_1353_);
return v___x_1357_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3___boxed(lean_object** _args){
lean_object* v_inst_1358_ = _args[0];
lean_object* v_inst_1359_ = _args[1];
lean_object* v_inst_1360_ = _args[2];
lean_object* v_inst_1361_ = _args[3];
lean_object* v_inst_1362_ = _args[4];
lean_object* v_inst_1363_ = _args[5];
lean_object* v_traceClass_1364_ = _args[6];
lean_object* v_collapsed_1365_ = _args[7];
lean_object* v___x_1366_ = _args[8];
lean_object* v_opts_1367_ = _args[9];
lean_object* v_clsEnabled_1368_ = _args[10];
lean_object* v_oldTraces_1369_ = _args[11];
lean_object* v_toPure_1370_ = _args[12];
lean_object* v_toBind_1371_ = _args[13];
lean_object* v_k_1372_ = _args[14];
lean_object* v_inst_1373_ = _args[15];
lean_object* v_msg_1374_ = _args[16];
lean_object* v___f_1375_ = _args[17];
lean_object* v_withRef_1376_ = _args[18];
lean_object* v_getRef_1377_ = _args[19];
lean_object* v_ref_1378_ = _args[20];
_start:
{
uint8_t v_collapsed_boxed_1379_; uint8_t v_clsEnabled_boxed_1380_; lean_object* v_res_1381_; 
v_collapsed_boxed_1379_ = lean_unbox(v_collapsed_1365_);
v_clsEnabled_boxed_1380_ = lean_unbox(v_clsEnabled_1368_);
v_res_1381_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3(v_inst_1358_, v_inst_1359_, v_inst_1360_, v_inst_1361_, v_inst_1362_, v_inst_1363_, v_traceClass_1364_, v_collapsed_boxed_1379_, v___x_1366_, v_opts_1367_, v_clsEnabled_boxed_1380_, v_oldTraces_1369_, v_toPure_1370_, v_toBind_1371_, v_k_1372_, v_inst_1373_, v_msg_1374_, v___f_1375_, v_withRef_1376_, v_getRef_1377_, v_ref_1378_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4(lean_object* v_inst_1382_, lean_object* v_inst_1383_, lean_object* v_inst_1384_, lean_object* v_inst_1385_, lean_object* v_inst_1386_, lean_object* v_inst_1387_, lean_object* v_traceClass_1388_, uint8_t v_collapsed_1389_, lean_object* v___x_1390_, lean_object* v_opts_1391_, uint8_t v_clsEnabled_1392_, lean_object* v_toPure_1393_, lean_object* v_toBind_1394_, lean_object* v_k_1395_, lean_object* v_inst_1396_, lean_object* v_msg_1397_, lean_object* v___f_1398_, lean_object* v_oldTraces_1399_){
_start:
{
lean_object* v_getRef_1400_; lean_object* v_withRef_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___f_1404_; lean_object* v___x_1405_; 
v_getRef_1400_ = lean_ctor_get(v_inst_1382_, 0);
lean_inc_n(v_getRef_1400_, 2);
v_withRef_1401_ = lean_ctor_get(v_inst_1382_, 1);
lean_inc(v_withRef_1401_);
v___x_1402_ = lean_box(v_collapsed_1389_);
v___x_1403_ = lean_box(v_clsEnabled_1392_);
lean_inc(v_toBind_1394_);
v___f_1404_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__3___boxed), 21, 20);
lean_closure_set(v___f_1404_, 0, v_inst_1383_);
lean_closure_set(v___f_1404_, 1, v_inst_1384_);
lean_closure_set(v___f_1404_, 2, v_inst_1385_);
lean_closure_set(v___f_1404_, 3, v_inst_1382_);
lean_closure_set(v___f_1404_, 4, v_inst_1386_);
lean_closure_set(v___f_1404_, 5, v_inst_1387_);
lean_closure_set(v___f_1404_, 6, v_traceClass_1388_);
lean_closure_set(v___f_1404_, 7, v___x_1402_);
lean_closure_set(v___f_1404_, 8, v___x_1390_);
lean_closure_set(v___f_1404_, 9, v_opts_1391_);
lean_closure_set(v___f_1404_, 10, v___x_1403_);
lean_closure_set(v___f_1404_, 11, v_oldTraces_1399_);
lean_closure_set(v___f_1404_, 12, v_toPure_1393_);
lean_closure_set(v___f_1404_, 13, v_toBind_1394_);
lean_closure_set(v___f_1404_, 14, v_k_1395_);
lean_closure_set(v___f_1404_, 15, v_inst_1396_);
lean_closure_set(v___f_1404_, 16, v_msg_1397_);
lean_closure_set(v___f_1404_, 17, v___f_1398_);
lean_closure_set(v___f_1404_, 18, v_withRef_1401_);
lean_closure_set(v___f_1404_, 19, v_getRef_1400_);
v___x_1405_ = lean_apply_4(v_toBind_1394_, lean_box(0), lean_box(0), v_getRef_1400_, v___f_1404_);
return v___x_1405_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4___boxed(lean_object** _args){
lean_object* v_inst_1406_ = _args[0];
lean_object* v_inst_1407_ = _args[1];
lean_object* v_inst_1408_ = _args[2];
lean_object* v_inst_1409_ = _args[3];
lean_object* v_inst_1410_ = _args[4];
lean_object* v_inst_1411_ = _args[5];
lean_object* v_traceClass_1412_ = _args[6];
lean_object* v_collapsed_1413_ = _args[7];
lean_object* v___x_1414_ = _args[8];
lean_object* v_opts_1415_ = _args[9];
lean_object* v_clsEnabled_1416_ = _args[10];
lean_object* v_toPure_1417_ = _args[11];
lean_object* v_toBind_1418_ = _args[12];
lean_object* v_k_1419_ = _args[13];
lean_object* v_inst_1420_ = _args[14];
lean_object* v_msg_1421_ = _args[15];
lean_object* v___f_1422_ = _args[16];
lean_object* v_oldTraces_1423_ = _args[17];
_start:
{
uint8_t v_collapsed_boxed_1424_; uint8_t v_clsEnabled_boxed_1425_; lean_object* v_res_1426_; 
v_collapsed_boxed_1424_ = lean_unbox(v_collapsed_1413_);
v_clsEnabled_boxed_1425_ = lean_unbox(v_clsEnabled_1416_);
v_res_1426_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4(v_inst_1406_, v_inst_1407_, v_inst_1408_, v_inst_1409_, v_inst_1410_, v_inst_1411_, v_traceClass_1412_, v_collapsed_boxed_1424_, v___x_1414_, v_opts_1415_, v_clsEnabled_boxed_1425_, v_toPure_1417_, v_toBind_1418_, v_k_1419_, v_inst_1420_, v_msg_1421_, v___f_1422_, v_oldTraces_1423_);
return v_res_1426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5(lean_object* v_inst_1427_, lean_object* v_inst_1428_, lean_object* v_inst_1429_, lean_object* v_inst_1430_, lean_object* v_inst_1431_, lean_object* v_inst_1432_, lean_object* v_traceClass_1433_, uint8_t v_collapsed_1434_, lean_object* v___x_1435_, lean_object* v_opts_1436_, lean_object* v_toPure_1437_, lean_object* v_toBind_1438_, lean_object* v_k_1439_, lean_object* v_inst_1440_, lean_object* v_msg_1441_, lean_object* v___f_1442_, uint8_t v_clsEnabled_1443_){
_start:
{
lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___f_1446_; 
v___x_1444_ = lean_box(v_collapsed_1434_);
v___x_1445_ = lean_box(v_clsEnabled_1443_);
lean_inc(v_k_1439_);
lean_inc(v_toBind_1438_);
lean_inc_ref(v_opts_1436_);
lean_inc_ref(v_inst_1430_);
lean_inc_ref(v_inst_1429_);
v___f_1446_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__4___boxed), 18, 17);
lean_closure_set(v___f_1446_, 0, v_inst_1427_);
lean_closure_set(v___f_1446_, 1, v_inst_1428_);
lean_closure_set(v___f_1446_, 2, v_inst_1429_);
lean_closure_set(v___f_1446_, 3, v_inst_1430_);
lean_closure_set(v___f_1446_, 4, v_inst_1431_);
lean_closure_set(v___f_1446_, 5, v_inst_1432_);
lean_closure_set(v___f_1446_, 6, v_traceClass_1433_);
lean_closure_set(v___f_1446_, 7, v___x_1444_);
lean_closure_set(v___f_1446_, 8, v___x_1435_);
lean_closure_set(v___f_1446_, 9, v_opts_1436_);
lean_closure_set(v___f_1446_, 10, v___x_1445_);
lean_closure_set(v___f_1446_, 11, v_toPure_1437_);
lean_closure_set(v___f_1446_, 12, v_toBind_1438_);
lean_closure_set(v___f_1446_, 13, v_k_1439_);
lean_closure_set(v___f_1446_, 14, v_inst_1440_);
lean_closure_set(v___f_1446_, 15, v_msg_1441_);
lean_closure_set(v___f_1446_, 16, v___f_1442_);
if (v_clsEnabled_1443_ == 0)
{
lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; uint8_t v___x_1453_; 
v___x_1450_ = l_Lean_KVMap_instValueBool;
v___x_1451_ = l_Lean_trace_profiler;
v___x_1452_ = l_Lean_Option_get___redArg(v___x_1450_, v_opts_1436_, v___x_1451_);
lean_dec_ref(v_opts_1436_);
v___x_1453_ = lean_unbox(v___x_1452_);
lean_dec(v___x_1452_);
if (v___x_1453_ == 0)
{
lean_dec_ref(v___f_1446_);
lean_dec(v_toBind_1438_);
lean_dec_ref(v_inst_1430_);
lean_dec_ref(v_inst_1429_);
return v_k_1439_;
}
else
{
lean_dec(v_k_1439_);
goto v___jp_1447_;
}
}
else
{
lean_dec(v_k_1439_);
lean_dec_ref(v_opts_1436_);
goto v___jp_1447_;
}
v___jp_1447_:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; 
v___x_1448_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v_inst_1429_, v_inst_1430_);
v___x_1449_ = lean_apply_4(v_toBind_1438_, lean_box(0), lean_box(0), v___x_1448_, v___f_1446_);
return v___x_1449_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5___boxed(lean_object** _args){
lean_object* v_inst_1454_ = _args[0];
lean_object* v_inst_1455_ = _args[1];
lean_object* v_inst_1456_ = _args[2];
lean_object* v_inst_1457_ = _args[3];
lean_object* v_inst_1458_ = _args[4];
lean_object* v_inst_1459_ = _args[5];
lean_object* v_traceClass_1460_ = _args[6];
lean_object* v_collapsed_1461_ = _args[7];
lean_object* v___x_1462_ = _args[8];
lean_object* v_opts_1463_ = _args[9];
lean_object* v_toPure_1464_ = _args[10];
lean_object* v_toBind_1465_ = _args[11];
lean_object* v_k_1466_ = _args[12];
lean_object* v_inst_1467_ = _args[13];
lean_object* v_msg_1468_ = _args[14];
lean_object* v___f_1469_ = _args[15];
lean_object* v_clsEnabled_1470_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_1471_; uint8_t v_clsEnabled_boxed_1472_; lean_object* v_res_1473_; 
v_collapsed_boxed_1471_ = lean_unbox(v_collapsed_1461_);
v_clsEnabled_boxed_1472_ = lean_unbox(v_clsEnabled_1470_);
v_res_1473_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5(v_inst_1454_, v_inst_1455_, v_inst_1456_, v_inst_1457_, v_inst_1458_, v_inst_1459_, v_traceClass_1460_, v_collapsed_boxed_1471_, v___x_1462_, v_opts_1463_, v_toPure_1464_, v_toBind_1465_, v_k_1466_, v_inst_1467_, v_msg_1468_, v___f_1469_, v_clsEnabled_boxed_1472_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8(lean_object* v_k_1474_, lean_object* v_inst_1475_, lean_object* v_toApplicative_1476_, lean_object* v_inst_1477_, lean_object* v_inst_1478_, lean_object* v_inst_1479_, lean_object* v_inst_1480_, lean_object* v_inst_1481_, lean_object* v_traceClass_1482_, uint8_t v_collapsed_1483_, lean_object* v___x_1484_, lean_object* v_toBind_1485_, lean_object* v_inst_1486_, lean_object* v_msg_1487_, lean_object* v___f_1488_, lean_object* v_inst_1489_, lean_object* v_opts_1490_){
_start:
{
uint8_t v_hasTrace_1491_; 
v_hasTrace_1491_ = lean_ctor_get_uint8(v_opts_1490_, sizeof(void*)*1);
if (v_hasTrace_1491_ == 0)
{
lean_dec_ref(v_opts_1490_);
lean_dec(v_inst_1489_);
lean_dec(v___f_1488_);
lean_dec(v_msg_1487_);
lean_dec(v_inst_1486_);
lean_dec(v_toBind_1485_);
lean_dec_ref(v___x_1484_);
lean_dec(v_traceClass_1482_);
lean_dec_ref(v_inst_1481_);
lean_dec(v_inst_1480_);
lean_dec_ref(v_inst_1479_);
lean_dec_ref(v_inst_1478_);
lean_dec_ref(v_inst_1477_);
lean_dec_ref(v_toApplicative_1476_);
lean_dec_ref(v_inst_1475_);
return v_k_1474_;
}
else
{
lean_object* v_getInheritedTraceOptions_1492_; lean_object* v_toPure_1493_; lean_object* v___x_1494_; lean_object* v___f_1495_; lean_object* v___f_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; 
v_getInheritedTraceOptions_1492_ = lean_ctor_get(v_inst_1475_, 2);
lean_inc(v_getInheritedTraceOptions_1492_);
v_toPure_1493_ = lean_ctor_get(v_toApplicative_1476_, 1);
lean_inc_n(v_toPure_1493_, 2);
lean_dec_ref(v_toApplicative_1476_);
v___x_1494_ = lean_box(v_collapsed_1483_);
lean_inc_n(v_toBind_1485_, 3);
lean_inc(v_traceClass_1482_);
v___f_1495_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__5___boxed), 17, 16);
lean_closure_set(v___f_1495_, 0, v_inst_1477_);
lean_closure_set(v___f_1495_, 1, v_inst_1478_);
lean_closure_set(v___f_1495_, 2, v_inst_1479_);
lean_closure_set(v___f_1495_, 3, v_inst_1475_);
lean_closure_set(v___f_1495_, 4, v_inst_1480_);
lean_closure_set(v___f_1495_, 5, v_inst_1481_);
lean_closure_set(v___f_1495_, 6, v_traceClass_1482_);
lean_closure_set(v___f_1495_, 7, v___x_1494_);
lean_closure_set(v___f_1495_, 8, v___x_1484_);
lean_closure_set(v___f_1495_, 9, v_opts_1490_);
lean_closure_set(v___f_1495_, 10, v_toPure_1493_);
lean_closure_set(v___f_1495_, 11, v_toBind_1485_);
lean_closure_set(v___f_1495_, 12, v_k_1474_);
lean_closure_set(v___f_1495_, 13, v_inst_1486_);
lean_closure_set(v___f_1495_, 14, v_msg_1487_);
lean_closure_set(v___f_1495_, 15, v___f_1488_);
v___f_1496_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__12), 5, 4);
lean_closure_set(v___f_1496_, 0, v_toPure_1493_);
lean_closure_set(v___f_1496_, 1, v_traceClass_1482_);
lean_closure_set(v___f_1496_, 2, v_toBind_1485_);
lean_closure_set(v___f_1496_, 3, v_inst_1489_);
v___x_1497_ = lean_apply_4(v_toBind_1485_, lean_box(0), lean_box(0), v_getInheritedTraceOptions_1492_, v___f_1496_);
v___x_1498_ = lean_apply_4(v_toBind_1485_, lean_box(0), lean_box(0), v___x_1497_, v___f_1495_);
return v___x_1498_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8___boxed(lean_object** _args){
lean_object* v_k_1499_ = _args[0];
lean_object* v_inst_1500_ = _args[1];
lean_object* v_toApplicative_1501_ = _args[2];
lean_object* v_inst_1502_ = _args[3];
lean_object* v_inst_1503_ = _args[4];
lean_object* v_inst_1504_ = _args[5];
lean_object* v_inst_1505_ = _args[6];
lean_object* v_inst_1506_ = _args[7];
lean_object* v_traceClass_1507_ = _args[8];
lean_object* v_collapsed_1508_ = _args[9];
lean_object* v___x_1509_ = _args[10];
lean_object* v_toBind_1510_ = _args[11];
lean_object* v_inst_1511_ = _args[12];
lean_object* v_msg_1512_ = _args[13];
lean_object* v___f_1513_ = _args[14];
lean_object* v_inst_1514_ = _args[15];
lean_object* v_opts_1515_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_1516_; lean_object* v_res_1517_; 
v_collapsed_boxed_1516_ = lean_unbox(v_collapsed_1508_);
v_res_1517_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8(v_k_1499_, v_inst_1500_, v_toApplicative_1501_, v_inst_1502_, v_inst_1503_, v_inst_1504_, v_inst_1505_, v_inst_1506_, v_traceClass_1507_, v_collapsed_boxed_1516_, v___x_1509_, v_toBind_1510_, v_inst_1511_, v_msg_1512_, v___f_1513_, v_inst_1514_, v_opts_1515_);
return v_res_1517_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg(lean_object* v_inst_1518_, lean_object* v_inst_1519_, lean_object* v_inst_1520_, lean_object* v_inst_1521_, lean_object* v_inst_1522_, lean_object* v_inst_1523_, lean_object* v_inst_1524_, lean_object* v_inst_1525_, lean_object* v_opt_1526_, lean_object* v_msg_1527_, lean_object* v_k_1528_, uint8_t v_collapsed_1529_){
_start:
{
lean_object* v_traceClass_1530_; lean_object* v_toApplicative_1531_; lean_object* v_toBind_1532_; lean_object* v___f_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___f_1536_; lean_object* v___x_1537_; 
v_traceClass_1530_ = lean_ctor_get(v_opt_1526_, 0);
lean_inc(v_traceClass_1530_);
lean_dec_ref(v_opt_1526_);
v_toApplicative_1531_ = lean_ctor_get(v_inst_1518_, 0);
lean_inc_ref(v_toApplicative_1531_);
v_toBind_1532_ = lean_ctor_get(v_inst_1518_, 1);
lean_inc_n(v_toBind_1532_, 2);
lean_inc(v_inst_1522_);
v___f_1533_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1533_, 0, v_inst_1522_);
v___x_1534_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1535_ = lean_box(v_collapsed_1529_);
lean_inc(v_inst_1523_);
v___f_1536_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8___boxed), 17, 16);
lean_closure_set(v___f_1536_, 0, v_k_1528_);
lean_closure_set(v___f_1536_, 1, v_inst_1519_);
lean_closure_set(v___f_1536_, 2, v_toApplicative_1531_);
lean_closure_set(v___f_1536_, 3, v_inst_1521_);
lean_closure_set(v___f_1536_, 4, v_inst_1524_);
lean_closure_set(v___f_1536_, 5, v_inst_1518_);
lean_closure_set(v___f_1536_, 6, v_inst_1522_);
lean_closure_set(v___f_1536_, 7, v_inst_1525_);
lean_closure_set(v___f_1536_, 8, v_traceClass_1530_);
lean_closure_set(v___f_1536_, 9, v___x_1535_);
lean_closure_set(v___f_1536_, 10, v___x_1534_);
lean_closure_set(v___f_1536_, 11, v_toBind_1532_);
lean_closure_set(v___f_1536_, 12, v_inst_1520_);
lean_closure_set(v___f_1536_, 13, v_msg_1527_);
lean_closure_set(v___f_1536_, 14, v___f_1533_);
lean_closure_set(v___f_1536_, 15, v_inst_1523_);
v___x_1537_ = lean_apply_4(v_toBind_1532_, lean_box(0), lean_box(0), v_inst_1523_, v___f_1536_);
return v___x_1537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___boxed(lean_object* v_inst_1538_, lean_object* v_inst_1539_, lean_object* v_inst_1540_, lean_object* v_inst_1541_, lean_object* v_inst_1542_, lean_object* v_inst_1543_, lean_object* v_inst_1544_, lean_object* v_inst_1545_, lean_object* v_opt_1546_, lean_object* v_msg_1547_, lean_object* v_k_1548_, lean_object* v_collapsed_1549_){
_start:
{
uint8_t v_collapsed_boxed_1550_; lean_object* v_res_1551_; 
v_collapsed_boxed_1550_ = lean_unbox(v_collapsed_1549_);
v_res_1551_ = lp_aesop_Aesop_withAesopTraceNodeBefore___redArg(v_inst_1538_, v_inst_1539_, v_inst_1540_, v_inst_1541_, v_inst_1542_, v_inst_1543_, v_inst_1544_, v_inst_1545_, v_opt_1546_, v_msg_1547_, v_k_1548_, v_collapsed_boxed_1550_);
return v_res_1551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore(lean_object* v_m_1552_, lean_object* v_00_u03b5_1553_, lean_object* v_inst_1554_, lean_object* v_inst_1555_, lean_object* v_inst_1556_, lean_object* v_inst_1557_, lean_object* v_inst_1558_, lean_object* v_inst_1559_, lean_object* v_inst_1560_, lean_object* v_00_u03b1_1561_, lean_object* v_inst_1562_, lean_object* v_opt_1563_, lean_object* v_msg_1564_, lean_object* v_k_1565_, uint8_t v_collapsed_1566_){
_start:
{
lean_object* v_traceClass_1567_; lean_object* v_toApplicative_1568_; lean_object* v_toBind_1569_; lean_object* v___f_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___f_1573_; lean_object* v___x_1574_; 
v_traceClass_1567_ = lean_ctor_get(v_opt_1563_, 0);
lean_inc(v_traceClass_1567_);
lean_dec_ref(v_opt_1563_);
v_toApplicative_1568_ = lean_ctor_get(v_inst_1554_, 0);
lean_inc_ref(v_toApplicative_1568_);
v_toBind_1569_ = lean_ctor_get(v_inst_1554_, 1);
lean_inc_n(v_toBind_1569_, 2);
lean_inc(v_inst_1558_);
v___f_1570_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1570_, 0, v_inst_1558_);
v___x_1571_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1572_ = lean_box(v_collapsed_1566_);
lean_inc(v_inst_1559_);
v___f_1573_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNodeBefore___redArg___lam__8___boxed), 17, 16);
lean_closure_set(v___f_1573_, 0, v_k_1565_);
lean_closure_set(v___f_1573_, 1, v_inst_1555_);
lean_closure_set(v___f_1573_, 2, v_toApplicative_1568_);
lean_closure_set(v___f_1573_, 3, v_inst_1557_);
lean_closure_set(v___f_1573_, 4, v_inst_1560_);
lean_closure_set(v___f_1573_, 5, v_inst_1554_);
lean_closure_set(v___f_1573_, 6, v_inst_1558_);
lean_closure_set(v___f_1573_, 7, v_inst_1562_);
lean_closure_set(v___f_1573_, 8, v_traceClass_1567_);
lean_closure_set(v___f_1573_, 9, v___x_1572_);
lean_closure_set(v___f_1573_, 10, v___x_1571_);
lean_closure_set(v___f_1573_, 11, v_toBind_1569_);
lean_closure_set(v___f_1573_, 12, v_inst_1556_);
lean_closure_set(v___f_1573_, 13, v_msg_1564_);
lean_closure_set(v___f_1573_, 14, v___f_1570_);
lean_closure_set(v___f_1573_, 15, v_inst_1559_);
v___x_1574_ = lean_apply_4(v_toBind_1569_, lean_box(0), lean_box(0), v_inst_1559_, v___f_1573_);
return v___x_1574_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withAesopTraceNodeBefore___boxed(lean_object* v_m_1575_, lean_object* v_00_u03b5_1576_, lean_object* v_inst_1577_, lean_object* v_inst_1578_, lean_object* v_inst_1579_, lean_object* v_inst_1580_, lean_object* v_inst_1581_, lean_object* v_inst_1582_, lean_object* v_inst_1583_, lean_object* v_00_u03b1_1584_, lean_object* v_inst_1585_, lean_object* v_opt_1586_, lean_object* v_msg_1587_, lean_object* v_k_1588_, lean_object* v_collapsed_1589_){
_start:
{
uint8_t v_collapsed_boxed_1590_; lean_object* v_res_1591_; 
v_collapsed_boxed_1590_ = lean_unbox(v_collapsed_1589_);
v_res_1591_ = lp_aesop_Aesop_withAesopTraceNodeBefore(v_m_1575_, v_00_u03b5_1576_, v_inst_1577_, v_inst_1578_, v_inst_1579_, v_inst_1580_, v_inst_1581_, v_inst_1582_, v_inst_1583_, v_00_u03b1_1584_, v_inst_1585_, v_opt_1586_, v_msg_1587_, v_k_1588_, v_collapsed_boxed_1590_);
return v_res_1591_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0(lean_object* v_msg_1592_, lean_object* v_x_1593_){
_start:
{
lean_inc(v_msg_1592_);
return v_msg_1592_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0___boxed(lean_object* v_msg_1594_, lean_object* v_x_1595_){
_start:
{
lean_object* v_res_1596_; 
v_res_1596_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0(v_msg_1594_, v_x_1595_);
lean_dec_ref(v_x_1595_);
lean_dec(v_msg_1594_);
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1(lean_object* v_inst_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_, lean_object* v_inst_1602_, lean_object* v_traceClass_1603_, uint8_t v_collapsed_1604_, lean_object* v___x_1605_, lean_object* v_opts_1606_, uint8_t v_clsEnabled_1607_, lean_object* v_oldTraces_1608_, lean_object* v___f_1609_, lean_object* v_resStartStop_1610_){
_start:
{
lean_object* v___x_1611_; 
v___x_1611_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v_inst_1597_, v_inst_1598_, v_inst_1599_, v_inst_1600_, lean_box(0), v_inst_1601_, v_inst_1602_, v_traceClass_1603_, v_collapsed_1604_, v___x_1605_, v_opts_1606_, v_clsEnabled_1607_, v_oldTraces_1608_, v___f_1609_, v_resStartStop_1610_);
return v___x_1611_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1___boxed(lean_object* v_inst_1612_, lean_object* v_inst_1613_, lean_object* v_inst_1614_, lean_object* v_inst_1615_, lean_object* v_inst_1616_, lean_object* v_inst_1617_, lean_object* v_traceClass_1618_, lean_object* v_collapsed_1619_, lean_object* v___x_1620_, lean_object* v_opts_1621_, lean_object* v_clsEnabled_1622_, lean_object* v_oldTraces_1623_, lean_object* v___f_1624_, lean_object* v_resStartStop_1625_){
_start:
{
uint8_t v_collapsed_boxed_1626_; uint8_t v_clsEnabled_boxed_1627_; lean_object* v_res_1628_; 
v_collapsed_boxed_1626_ = lean_unbox(v_collapsed_1619_);
v_clsEnabled_boxed_1627_ = lean_unbox(v_clsEnabled_1622_);
v_res_1628_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1(v_inst_1612_, v_inst_1613_, v_inst_1614_, v_inst_1615_, v_inst_1616_, v_inst_1617_, v_traceClass_1618_, v_collapsed_boxed_1626_, v___x_1620_, v_opts_1621_, v_clsEnabled_boxed_1627_, v_oldTraces_1623_, v___f_1624_, v_resStartStop_1625_);
lean_dec_ref(v_opts_1621_);
return v_res_1628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10(lean_object* v_inst_1629_, lean_object* v_inst_1630_, lean_object* v_inst_1631_, lean_object* v_inst_1632_, lean_object* v_inst_1633_, lean_object* v_inst_1634_, lean_object* v_traceClass_1635_, uint8_t v_collapsed_1636_, lean_object* v___x_1637_, lean_object* v_opts_1638_, uint8_t v_clsEnabled_1639_, lean_object* v___f_1640_, lean_object* v_toPure_1641_, lean_object* v_toBind_1642_, lean_object* v_k_1643_, lean_object* v_inst_1644_, lean_object* v_oldTraces_1645_){
_start:
{
lean_object* v_tryCatch_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___f_1649_; lean_object* v___f_1650_; lean_object* v___f_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; uint8_t v___x_1657_; 
v_tryCatch_1646_ = lean_ctor_get(v_inst_1629_, 1);
lean_inc(v_tryCatch_1646_);
v___x_1647_ = lean_box(v_collapsed_1636_);
v___x_1648_ = lean_box(v_clsEnabled_1639_);
lean_inc_ref(v_opts_1638_);
v___f_1649_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__1___boxed), 14, 13);
lean_closure_set(v___f_1649_, 0, v_inst_1630_);
lean_closure_set(v___f_1649_, 1, v_inst_1631_);
lean_closure_set(v___f_1649_, 2, v_inst_1632_);
lean_closure_set(v___f_1649_, 3, v_inst_1633_);
lean_closure_set(v___f_1649_, 4, v_inst_1629_);
lean_closure_set(v___f_1649_, 5, v_inst_1634_);
lean_closure_set(v___f_1649_, 6, v_traceClass_1635_);
lean_closure_set(v___f_1649_, 7, v___x_1647_);
lean_closure_set(v___f_1649_, 8, v___x_1637_);
lean_closure_set(v___f_1649_, 9, v_opts_1638_);
lean_closure_set(v___f_1649_, 10, v___x_1648_);
lean_closure_set(v___f_1649_, 11, v_oldTraces_1645_);
lean_closure_set(v___f_1649_, 12, v___f_1640_);
lean_inc_n(v_toPure_1641_, 2);
v___f_1650_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1650_, 0, v_toPure_1641_);
v___f_1651_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1651_, 0, v_toPure_1641_);
lean_inc(v_toBind_1642_);
v___x_1652_ = lean_apply_4(v_toBind_1642_, lean_box(0), lean_box(0), v_k_1643_, v___f_1651_);
v___x_1653_ = lean_apply_3(v_tryCatch_1646_, lean_box(0), v___x_1652_, v___f_1650_);
v___x_1654_ = l_Lean_KVMap_instValueBool;
v___x_1655_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1656_ = l_Lean_Option_get___redArg(v___x_1654_, v_opts_1638_, v___x_1655_);
lean_dec_ref(v_opts_1638_);
v___x_1657_ = lean_unbox(v___x_1656_);
lean_dec(v___x_1656_);
if (v___x_1657_ == 0)
{
lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___f_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; 
v___x_1658_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__0));
v___x_1659_ = lean_apply_2(v_inst_1644_, lean_box(0), v___x_1658_);
lean_inc(v___x_1659_);
lean_inc_n(v_toBind_1642_, 2);
v___f_1660_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__5), 5, 4);
lean_closure_set(v___f_1660_, 0, v_toPure_1641_);
lean_closure_set(v___f_1660_, 1, v_toBind_1642_);
lean_closure_set(v___f_1660_, 2, v___x_1659_);
lean_closure_set(v___f_1660_, 3, v___x_1653_);
v___x_1661_ = lean_apply_4(v_toBind_1642_, lean_box(0), lean_box(0), v___x_1659_, v___f_1660_);
v___x_1662_ = lean_apply_4(v_toBind_1642_, lean_box(0), lean_box(0), v___x_1661_, v___f_1649_);
return v___x_1662_;
}
else
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___f_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; 
v___x_1663_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__9___closed__1));
v___x_1664_ = lean_apply_2(v_inst_1644_, lean_box(0), v___x_1663_);
lean_inc(v___x_1664_);
lean_inc_n(v_toBind_1642_, 2);
v___f_1665_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__8), 5, 4);
lean_closure_set(v___f_1665_, 0, v_toPure_1641_);
lean_closure_set(v___f_1665_, 1, v_toBind_1642_);
lean_closure_set(v___f_1665_, 2, v___x_1664_);
lean_closure_set(v___f_1665_, 3, v___x_1653_);
v___x_1666_ = lean_apply_4(v_toBind_1642_, lean_box(0), lean_box(0), v___x_1664_, v___f_1665_);
v___x_1667_ = lean_apply_4(v_toBind_1642_, lean_box(0), lean_box(0), v___x_1666_, v___f_1649_);
return v___x_1667_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10___boxed(lean_object** _args){
lean_object* v_inst_1668_ = _args[0];
lean_object* v_inst_1669_ = _args[1];
lean_object* v_inst_1670_ = _args[2];
lean_object* v_inst_1671_ = _args[3];
lean_object* v_inst_1672_ = _args[4];
lean_object* v_inst_1673_ = _args[5];
lean_object* v_traceClass_1674_ = _args[6];
lean_object* v_collapsed_1675_ = _args[7];
lean_object* v___x_1676_ = _args[8];
lean_object* v_opts_1677_ = _args[9];
lean_object* v_clsEnabled_1678_ = _args[10];
lean_object* v___f_1679_ = _args[11];
lean_object* v_toPure_1680_ = _args[12];
lean_object* v_toBind_1681_ = _args[13];
lean_object* v_k_1682_ = _args[14];
lean_object* v_inst_1683_ = _args[15];
lean_object* v_oldTraces_1684_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_1685_; uint8_t v_clsEnabled_boxed_1686_; lean_object* v_res_1687_; 
v_collapsed_boxed_1685_ = lean_unbox(v_collapsed_1675_);
v_clsEnabled_boxed_1686_ = lean_unbox(v_clsEnabled_1678_);
v_res_1687_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10(v_inst_1668_, v_inst_1669_, v_inst_1670_, v_inst_1671_, v_inst_1672_, v_inst_1673_, v_traceClass_1674_, v_collapsed_boxed_1685_, v___x_1676_, v_opts_1677_, v_clsEnabled_boxed_1686_, v___f_1679_, v_toPure_1680_, v_toBind_1681_, v_k_1682_, v_inst_1683_, v_oldTraces_1684_);
return v_res_1687_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2(lean_object* v_inst_1688_, lean_object* v_inst_1689_, lean_object* v_inst_1690_, lean_object* v_inst_1691_, lean_object* v_inst_1692_, lean_object* v_inst_1693_, lean_object* v_traceClass_1694_, uint8_t v_collapsed_1695_, lean_object* v___x_1696_, lean_object* v_opts_1697_, lean_object* v___f_1698_, lean_object* v_toPure_1699_, lean_object* v_toBind_1700_, lean_object* v_k_1701_, lean_object* v_inst_1702_, uint8_t v_clsEnabled_1703_){
_start:
{
lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___f_1706_; 
v___x_1704_ = lean_box(v_collapsed_1695_);
v___x_1705_ = lean_box(v_clsEnabled_1703_);
lean_inc(v_k_1701_);
lean_inc(v_toBind_1700_);
lean_inc_ref(v_opts_1697_);
lean_inc_ref(v_inst_1690_);
lean_inc_ref(v_inst_1689_);
v___f_1706_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__10___boxed), 17, 16);
lean_closure_set(v___f_1706_, 0, v_inst_1688_);
lean_closure_set(v___f_1706_, 1, v_inst_1689_);
lean_closure_set(v___f_1706_, 2, v_inst_1690_);
lean_closure_set(v___f_1706_, 3, v_inst_1691_);
lean_closure_set(v___f_1706_, 4, v_inst_1692_);
lean_closure_set(v___f_1706_, 5, v_inst_1693_);
lean_closure_set(v___f_1706_, 6, v_traceClass_1694_);
lean_closure_set(v___f_1706_, 7, v___x_1704_);
lean_closure_set(v___f_1706_, 8, v___x_1696_);
lean_closure_set(v___f_1706_, 9, v_opts_1697_);
lean_closure_set(v___f_1706_, 10, v___x_1705_);
lean_closure_set(v___f_1706_, 11, v___f_1698_);
lean_closure_set(v___f_1706_, 12, v_toPure_1699_);
lean_closure_set(v___f_1706_, 13, v_toBind_1700_);
lean_closure_set(v___f_1706_, 14, v_k_1701_);
lean_closure_set(v___f_1706_, 15, v_inst_1702_);
if (v_clsEnabled_1703_ == 0)
{
lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; uint8_t v___x_1713_; 
v___x_1710_ = l_Lean_KVMap_instValueBool;
v___x_1711_ = l_Lean_trace_profiler;
v___x_1712_ = l_Lean_Option_get___redArg(v___x_1710_, v_opts_1697_, v___x_1711_);
lean_dec_ref(v_opts_1697_);
v___x_1713_ = lean_unbox(v___x_1712_);
lean_dec(v___x_1712_);
if (v___x_1713_ == 0)
{
lean_dec_ref(v___f_1706_);
lean_dec(v_toBind_1700_);
lean_dec_ref(v_inst_1690_);
lean_dec_ref(v_inst_1689_);
return v_k_1701_;
}
else
{
lean_dec(v_k_1701_);
goto v___jp_1707_;
}
}
else
{
lean_dec(v_k_1701_);
lean_dec_ref(v_opts_1697_);
goto v___jp_1707_;
}
v___jp_1707_:
{
lean_object* v___x_1708_; lean_object* v___x_1709_; 
v___x_1708_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v_inst_1689_, v_inst_1690_);
v___x_1709_ = lean_apply_4(v_toBind_1700_, lean_box(0), lean_box(0), v___x_1708_, v___f_1706_);
return v___x_1709_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2___boxed(lean_object* v_inst_1714_, lean_object* v_inst_1715_, lean_object* v_inst_1716_, lean_object* v_inst_1717_, lean_object* v_inst_1718_, lean_object* v_inst_1719_, lean_object* v_traceClass_1720_, lean_object* v_collapsed_1721_, lean_object* v___x_1722_, lean_object* v_opts_1723_, lean_object* v___f_1724_, lean_object* v_toPure_1725_, lean_object* v_toBind_1726_, lean_object* v_k_1727_, lean_object* v_inst_1728_, lean_object* v_clsEnabled_1729_){
_start:
{
uint8_t v_collapsed_boxed_1730_; uint8_t v_clsEnabled_boxed_1731_; lean_object* v_res_1732_; 
v_collapsed_boxed_1730_ = lean_unbox(v_collapsed_1721_);
v_clsEnabled_boxed_1731_ = lean_unbox(v_clsEnabled_1729_);
v_res_1732_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2(v_inst_1714_, v_inst_1715_, v_inst_1716_, v_inst_1717_, v_inst_1718_, v_inst_1719_, v_traceClass_1720_, v_collapsed_boxed_1730_, v___x_1722_, v_opts_1723_, v___f_1724_, v_toPure_1725_, v_toBind_1726_, v_k_1727_, v_inst_1728_, v_clsEnabled_boxed_1731_);
return v_res_1732_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5(lean_object* v_k_1733_, lean_object* v_inst_1734_, lean_object* v_toApplicative_1735_, lean_object* v_inst_1736_, lean_object* v_inst_1737_, lean_object* v_inst_1738_, lean_object* v_inst_1739_, lean_object* v_inst_1740_, lean_object* v_traceClass_1741_, uint8_t v_collapsed_1742_, lean_object* v___x_1743_, lean_object* v___f_1744_, lean_object* v_toBind_1745_, lean_object* v_inst_1746_, lean_object* v_inst_1747_, lean_object* v_opts_1748_){
_start:
{
uint8_t v_hasTrace_1749_; 
v_hasTrace_1749_ = lean_ctor_get_uint8(v_opts_1748_, sizeof(void*)*1);
if (v_hasTrace_1749_ == 0)
{
lean_dec_ref(v_opts_1748_);
lean_dec(v_inst_1747_);
lean_dec(v_inst_1746_);
lean_dec(v_toBind_1745_);
lean_dec(v___f_1744_);
lean_dec_ref(v___x_1743_);
lean_dec(v_traceClass_1741_);
lean_dec_ref(v_inst_1740_);
lean_dec(v_inst_1739_);
lean_dec_ref(v_inst_1738_);
lean_dec_ref(v_inst_1737_);
lean_dec_ref(v_inst_1736_);
lean_dec_ref(v_toApplicative_1735_);
lean_dec_ref(v_inst_1734_);
return v_k_1733_;
}
else
{
lean_object* v_getInheritedTraceOptions_1750_; lean_object* v_toPure_1751_; lean_object* v___x_1752_; lean_object* v___f_1753_; lean_object* v___f_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; 
v_getInheritedTraceOptions_1750_ = lean_ctor_get(v_inst_1734_, 2);
lean_inc(v_getInheritedTraceOptions_1750_);
v_toPure_1751_ = lean_ctor_get(v_toApplicative_1735_, 1);
lean_inc_n(v_toPure_1751_, 2);
lean_dec_ref(v_toApplicative_1735_);
v___x_1752_ = lean_box(v_collapsed_1742_);
lean_inc_n(v_toBind_1745_, 3);
lean_inc(v_traceClass_1741_);
v___f_1753_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__2___boxed), 16, 15);
lean_closure_set(v___f_1753_, 0, v_inst_1736_);
lean_closure_set(v___f_1753_, 1, v_inst_1737_);
lean_closure_set(v___f_1753_, 2, v_inst_1734_);
lean_closure_set(v___f_1753_, 3, v_inst_1738_);
lean_closure_set(v___f_1753_, 4, v_inst_1739_);
lean_closure_set(v___f_1753_, 5, v_inst_1740_);
lean_closure_set(v___f_1753_, 6, v_traceClass_1741_);
lean_closure_set(v___f_1753_, 7, v___x_1752_);
lean_closure_set(v___f_1753_, 8, v___x_1743_);
lean_closure_set(v___f_1753_, 9, v_opts_1748_);
lean_closure_set(v___f_1753_, 10, v___f_1744_);
lean_closure_set(v___f_1753_, 11, v_toPure_1751_);
lean_closure_set(v___f_1753_, 12, v_toBind_1745_);
lean_closure_set(v___f_1753_, 13, v_k_1733_);
lean_closure_set(v___f_1753_, 14, v_inst_1746_);
v___f_1754_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__12), 5, 4);
lean_closure_set(v___f_1754_, 0, v_toPure_1751_);
lean_closure_set(v___f_1754_, 1, v_traceClass_1741_);
lean_closure_set(v___f_1754_, 2, v_toBind_1745_);
lean_closure_set(v___f_1754_, 3, v_inst_1747_);
v___x_1755_ = lean_apply_4(v_toBind_1745_, lean_box(0), lean_box(0), v_getInheritedTraceOptions_1750_, v___f_1754_);
v___x_1756_ = lean_apply_4(v_toBind_1745_, lean_box(0), lean_box(0), v___x_1755_, v___f_1753_);
return v___x_1756_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5___boxed(lean_object* v_k_1757_, lean_object* v_inst_1758_, lean_object* v_toApplicative_1759_, lean_object* v_inst_1760_, lean_object* v_inst_1761_, lean_object* v_inst_1762_, lean_object* v_inst_1763_, lean_object* v_inst_1764_, lean_object* v_traceClass_1765_, lean_object* v_collapsed_1766_, lean_object* v___x_1767_, lean_object* v___f_1768_, lean_object* v_toBind_1769_, lean_object* v_inst_1770_, lean_object* v_inst_1771_, lean_object* v_opts_1772_){
_start:
{
uint8_t v_collapsed_boxed_1773_; lean_object* v_res_1774_; 
v_collapsed_boxed_1773_ = lean_unbox(v_collapsed_1766_);
v_res_1774_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5(v_k_1757_, v_inst_1758_, v_toApplicative_1759_, v_inst_1760_, v_inst_1761_, v_inst_1762_, v_inst_1763_, v_inst_1764_, v_traceClass_1765_, v_collapsed_boxed_1773_, v___x_1767_, v___f_1768_, v_toBind_1769_, v_inst_1770_, v_inst_1771_, v_opts_1772_);
return v_res_1774_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg(lean_object* v_inst_1775_, lean_object* v_inst_1776_, lean_object* v_inst_1777_, lean_object* v_inst_1778_, lean_object* v_inst_1779_, lean_object* v_inst_1780_, lean_object* v_inst_1781_, lean_object* v_inst_1782_, lean_object* v_opt_1783_, lean_object* v_msg_1784_, lean_object* v_k_1785_, uint8_t v_collapsed_1786_){
_start:
{
lean_object* v_traceClass_1787_; lean_object* v_toApplicative_1788_; lean_object* v_toBind_1789_; lean_object* v___f_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___f_1793_; lean_object* v___x_1794_; 
v_traceClass_1787_ = lean_ctor_get(v_opt_1783_, 0);
lean_inc(v_traceClass_1787_);
lean_dec_ref(v_opt_1783_);
v_toApplicative_1788_ = lean_ctor_get(v_inst_1775_, 0);
lean_inc_ref(v_toApplicative_1788_);
v_toBind_1789_ = lean_ctor_get(v_inst_1775_, 1);
lean_inc_n(v_toBind_1789_, 2);
v___f_1790_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1790_, 0, v_msg_1784_);
v___x_1791_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1792_ = lean_box(v_collapsed_1786_);
lean_inc(v_inst_1780_);
v___f_1793_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5___boxed), 16, 15);
lean_closure_set(v___f_1793_, 0, v_k_1785_);
lean_closure_set(v___f_1793_, 1, v_inst_1776_);
lean_closure_set(v___f_1793_, 2, v_toApplicative_1788_);
lean_closure_set(v___f_1793_, 3, v_inst_1781_);
lean_closure_set(v___f_1793_, 4, v_inst_1775_);
lean_closure_set(v___f_1793_, 5, v_inst_1778_);
lean_closure_set(v___f_1793_, 6, v_inst_1779_);
lean_closure_set(v___f_1793_, 7, v_inst_1782_);
lean_closure_set(v___f_1793_, 8, v_traceClass_1787_);
lean_closure_set(v___f_1793_, 9, v___x_1792_);
lean_closure_set(v___f_1793_, 10, v___x_1791_);
lean_closure_set(v___f_1793_, 11, v___f_1790_);
lean_closure_set(v___f_1793_, 12, v_toBind_1789_);
lean_closure_set(v___f_1793_, 13, v_inst_1777_);
lean_closure_set(v___f_1793_, 14, v_inst_1780_);
v___x_1794_ = lean_apply_4(v_toBind_1789_, lean_box(0), lean_box(0), v_inst_1780_, v___f_1793_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___redArg___boxed(lean_object* v_inst_1795_, lean_object* v_inst_1796_, lean_object* v_inst_1797_, lean_object* v_inst_1798_, lean_object* v_inst_1799_, lean_object* v_inst_1800_, lean_object* v_inst_1801_, lean_object* v_inst_1802_, lean_object* v_opt_1803_, lean_object* v_msg_1804_, lean_object* v_k_1805_, lean_object* v_collapsed_1806_){
_start:
{
uint8_t v_collapsed_boxed_1807_; lean_object* v_res_1808_; 
v_collapsed_boxed_1807_ = lean_unbox(v_collapsed_1806_);
v_res_1808_ = lp_aesop_Aesop_withConstAesopTraceNode___redArg(v_inst_1795_, v_inst_1796_, v_inst_1797_, v_inst_1798_, v_inst_1799_, v_inst_1800_, v_inst_1801_, v_inst_1802_, v_opt_1803_, v_msg_1804_, v_k_1805_, v_collapsed_boxed_1807_);
return v_res_1808_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode(lean_object* v_m_1809_, lean_object* v_00_u03b5_1810_, lean_object* v_inst_1811_, lean_object* v_inst_1812_, lean_object* v_inst_1813_, lean_object* v_inst_1814_, lean_object* v_inst_1815_, lean_object* v_inst_1816_, lean_object* v_inst_1817_, lean_object* v_00_u03b1_1818_, lean_object* v_inst_1819_, lean_object* v_opt_1820_, lean_object* v_msg_1821_, lean_object* v_k_1822_, uint8_t v_collapsed_1823_){
_start:
{
lean_object* v_traceClass_1824_; lean_object* v_toApplicative_1825_; lean_object* v_toBind_1826_; lean_object* v___f_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___f_1830_; lean_object* v___x_1831_; 
v_traceClass_1824_ = lean_ctor_get(v_opt_1820_, 0);
lean_inc(v_traceClass_1824_);
lean_dec_ref(v_opt_1820_);
v_toApplicative_1825_ = lean_ctor_get(v_inst_1811_, 0);
lean_inc_ref(v_toApplicative_1825_);
v_toBind_1826_ = lean_ctor_get(v_inst_1811_, 1);
lean_inc_n(v_toBind_1826_, 2);
v___f_1827_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1827_, 0, v_msg_1821_);
v___x_1828_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_1829_ = lean_box(v_collapsed_1823_);
lean_inc(v_inst_1816_);
v___f_1830_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withConstAesopTraceNode___redArg___lam__5___boxed), 16, 15);
lean_closure_set(v___f_1830_, 0, v_k_1822_);
lean_closure_set(v___f_1830_, 1, v_inst_1812_);
lean_closure_set(v___f_1830_, 2, v_toApplicative_1825_);
lean_closure_set(v___f_1830_, 3, v_inst_1817_);
lean_closure_set(v___f_1830_, 4, v_inst_1811_);
lean_closure_set(v___f_1830_, 5, v_inst_1814_);
lean_closure_set(v___f_1830_, 6, v_inst_1815_);
lean_closure_set(v___f_1830_, 7, v_inst_1819_);
lean_closure_set(v___f_1830_, 8, v_traceClass_1824_);
lean_closure_set(v___f_1830_, 9, v___x_1829_);
lean_closure_set(v___f_1830_, 10, v___x_1828_);
lean_closure_set(v___f_1830_, 11, v___f_1827_);
lean_closure_set(v___f_1830_, 12, v_toBind_1826_);
lean_closure_set(v___f_1830_, 13, v_inst_1813_);
lean_closure_set(v___f_1830_, 14, v_inst_1816_);
v___x_1831_ = lean_apply_4(v_toBind_1826_, lean_box(0), lean_box(0), v_inst_1816_, v___f_1830_);
return v___x_1831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withConstAesopTraceNode___boxed(lean_object* v_m_1832_, lean_object* v_00_u03b5_1833_, lean_object* v_inst_1834_, lean_object* v_inst_1835_, lean_object* v_inst_1836_, lean_object* v_inst_1837_, lean_object* v_inst_1838_, lean_object* v_inst_1839_, lean_object* v_inst_1840_, lean_object* v_00_u03b1_1841_, lean_object* v_inst_1842_, lean_object* v_opt_1843_, lean_object* v_msg_1844_, lean_object* v_k_1845_, lean_object* v_collapsed_1846_){
_start:
{
uint8_t v_collapsed_boxed_1847_; lean_object* v_res_1848_; 
v_collapsed_boxed_1847_ = lean_unbox(v_collapsed_1846_);
v_res_1848_ = lp_aesop_Aesop_withConstAesopTraceNode(v_m_1832_, v_00_u03b5_1833_, v_inst_1834_, v_inst_1835_, v_inst_1836_, v_inst_1837_, v_inst_1838_, v_inst_1839_, v_inst_1840_, v_00_u03b1_1841_, v_inst_1842_, v_opt_1843_, v_msg_1844_, v_k_1845_, v_collapsed_boxed_1847_);
return v_res_1848_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__0(lean_object* v_x1_1849_, lean_object* v_x2_1850_){
_start:
{
lean_object* v___x_1851_; 
v___x_1851_ = lean_array_push(v_x1_1849_, v_x2_1850_);
return v___x_1851_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(lean_object* v_f_1852_, lean_object* v_as_1853_, size_t v_i_1854_, size_t v_stop_1855_, lean_object* v_b_1856_){
_start:
{
uint8_t v___x_1857_; 
v___x_1857_ = lean_usize_dec_eq(v_i_1854_, v_stop_1855_);
if (v___x_1857_ == 0)
{
lean_object* v___x_1858_; lean_object* v___x_1859_; size_t v___x_1860_; size_t v___x_1861_; 
v___x_1858_ = lean_array_uget_borrowed(v_as_1853_, v_i_1854_);
lean_inc(v_f_1852_);
lean_inc(v___x_1858_);
v___x_1859_ = lean_apply_2(v_f_1852_, v_b_1856_, v___x_1858_);
v___x_1860_ = ((size_t)1ULL);
v___x_1861_ = lean_usize_add(v_i_1854_, v___x_1860_);
v_i_1854_ = v___x_1861_;
v_b_1856_ = v___x_1859_;
goto _start;
}
else
{
lean_dec(v_f_1852_);
return v_b_1856_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg___boxed(lean_object* v_f_1863_, lean_object* v_as_1864_, lean_object* v_i_1865_, lean_object* v_stop_1866_, lean_object* v_b_1867_){
_start:
{
size_t v_i_boxed_1868_; size_t v_stop_boxed_1869_; lean_object* v_res_1870_; 
v_i_boxed_1868_ = lean_unbox_usize(v_i_1865_);
lean_dec(v_i_1865_);
v_stop_boxed_1869_ = lean_unbox_usize(v_stop_1866_);
lean_dec(v_stop_1866_);
v_res_1870_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(v_f_1863_, v_as_1864_, v_i_boxed_1868_, v_stop_boxed_1869_, v_b_1867_);
lean_dec_ref(v_as_1864_);
return v_res_1870_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(lean_object* v_f_1871_, lean_object* v_x_1872_, lean_object* v_x_1873_){
_start:
{
lean_object* v_vs_1874_; lean_object* v_children_1875_; lean_object* v___x_1876_; lean_object* v_s_1878_; lean_object* v___x_1888_; uint8_t v___x_1889_; 
v_vs_1874_ = lean_ctor_get(v_x_1873_, 0);
v_children_1875_ = lean_ctor_get(v_x_1873_, 1);
v___x_1876_ = lean_unsigned_to_nat(0u);
v___x_1888_ = lean_array_get_size(v_vs_1874_);
v___x_1889_ = lean_nat_dec_lt(v___x_1876_, v___x_1888_);
if (v___x_1889_ == 0)
{
lean_object* v___x_1890_; uint8_t v___x_1891_; 
v___x_1890_ = lean_array_get_size(v_children_1875_);
v___x_1891_ = lean_nat_dec_lt(v___x_1876_, v___x_1890_);
if (v___x_1891_ == 0)
{
lean_dec(v_f_1871_);
return v_x_1872_;
}
else
{
uint8_t v___x_1892_; 
v___x_1892_ = lean_nat_dec_le(v___x_1890_, v___x_1890_);
if (v___x_1892_ == 0)
{
if (v___x_1891_ == 0)
{
lean_dec(v_f_1871_);
return v_x_1872_;
}
else
{
size_t v___x_1893_; size_t v___x_1894_; lean_object* v___x_1895_; 
v___x_1893_ = ((size_t)0ULL);
v___x_1894_ = lean_usize_of_nat(v___x_1890_);
v___x_1895_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_1871_, v_children_1875_, v___x_1893_, v___x_1894_, v_x_1872_);
return v___x_1895_;
}
}
else
{
size_t v___x_1896_; size_t v___x_1897_; lean_object* v___x_1898_; 
v___x_1896_ = ((size_t)0ULL);
v___x_1897_ = lean_usize_of_nat(v___x_1890_);
v___x_1898_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_1871_, v_children_1875_, v___x_1896_, v___x_1897_, v_x_1872_);
return v___x_1898_;
}
}
}
else
{
uint8_t v___x_1899_; 
v___x_1899_ = lean_nat_dec_le(v___x_1888_, v___x_1888_);
if (v___x_1899_ == 0)
{
if (v___x_1889_ == 0)
{
v_s_1878_ = v_x_1872_;
goto v___jp_1877_;
}
else
{
size_t v___x_1900_; size_t v___x_1901_; lean_object* v___x_1902_; 
v___x_1900_ = ((size_t)0ULL);
v___x_1901_ = lean_usize_of_nat(v___x_1888_);
lean_inc(v_f_1871_);
v___x_1902_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(v_f_1871_, v_vs_1874_, v___x_1900_, v___x_1901_, v_x_1872_);
v_s_1878_ = v___x_1902_;
goto v___jp_1877_;
}
}
else
{
size_t v___x_1903_; size_t v___x_1904_; lean_object* v___x_1905_; 
v___x_1903_ = ((size_t)0ULL);
v___x_1904_ = lean_usize_of_nat(v___x_1888_);
lean_inc(v_f_1871_);
v___x_1905_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(v_f_1871_, v_vs_1874_, v___x_1903_, v___x_1904_, v_x_1872_);
v_s_1878_ = v___x_1905_;
goto v___jp_1877_;
}
}
v___jp_1877_:
{
lean_object* v___x_1879_; uint8_t v___x_1880_; 
v___x_1879_ = lean_array_get_size(v_children_1875_);
v___x_1880_ = lean_nat_dec_lt(v___x_1876_, v___x_1879_);
if (v___x_1880_ == 0)
{
lean_dec(v_f_1871_);
return v_s_1878_;
}
else
{
uint8_t v___x_1881_; 
v___x_1881_ = lean_nat_dec_le(v___x_1879_, v___x_1879_);
if (v___x_1881_ == 0)
{
if (v___x_1880_ == 0)
{
lean_dec(v_f_1871_);
return v_s_1878_;
}
else
{
size_t v___x_1882_; size_t v___x_1883_; lean_object* v___x_1884_; 
v___x_1882_ = ((size_t)0ULL);
v___x_1883_ = lean_usize_of_nat(v___x_1879_);
v___x_1884_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_1871_, v_children_1875_, v___x_1882_, v___x_1883_, v_s_1878_);
return v___x_1884_;
}
}
else
{
size_t v___x_1885_; size_t v___x_1886_; lean_object* v___x_1887_; 
v___x_1885_ = ((size_t)0ULL);
v___x_1886_ = lean_usize_of_nat(v___x_1879_);
v___x_1887_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_1871_, v_children_1875_, v___x_1885_, v___x_1886_, v_s_1878_);
return v___x_1887_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(lean_object* v_f_1906_, lean_object* v_as_1907_, size_t v_i_1908_, size_t v_stop_1909_, lean_object* v_b_1910_){
_start:
{
uint8_t v___x_1911_; 
v___x_1911_ = lean_usize_dec_eq(v_i_1908_, v_stop_1909_);
if (v___x_1911_ == 0)
{
lean_object* v___x_1912_; lean_object* v_snd_1913_; lean_object* v___x_1914_; size_t v___x_1915_; size_t v___x_1916_; 
v___x_1912_ = lean_array_uget_borrowed(v_as_1907_, v_i_1908_);
v_snd_1913_ = lean_ctor_get(v___x_1912_, 1);
lean_inc(v_f_1906_);
v___x_1914_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(v_f_1906_, v_b_1910_, v_snd_1913_);
v___x_1915_ = ((size_t)1ULL);
v___x_1916_ = lean_usize_add(v_i_1908_, v___x_1915_);
v_i_1908_ = v___x_1916_;
v_b_1910_ = v___x_1914_;
goto _start;
}
else
{
lean_dec(v_f_1906_);
return v_b_1910_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg___boxed(lean_object* v_f_1918_, lean_object* v_as_1919_, lean_object* v_i_1920_, lean_object* v_stop_1921_, lean_object* v_b_1922_){
_start:
{
size_t v_i_boxed_1923_; size_t v_stop_boxed_1924_; lean_object* v_res_1925_; 
v_i_boxed_1923_ = lean_unbox_usize(v_i_1920_);
lean_dec(v_i_1920_);
v_stop_boxed_1924_ = lean_unbox_usize(v_stop_1921_);
lean_dec(v_stop_1921_);
v_res_1925_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_1918_, v_as_1919_, v_i_boxed_1923_, v_stop_boxed_1924_, v_b_1922_);
lean_dec_ref(v_as_1919_);
return v_res_1925_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg___boxed(lean_object* v_f_1926_, lean_object* v_x_1927_, lean_object* v_x_1928_){
_start:
{
lean_object* v_res_1929_; 
v_res_1929_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(v_f_1926_, v_x_1927_, v_x_1928_);
lean_dec_ref(v_x_1928_);
return v_res_1929_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1(lean_object* v___f_1930_, lean_object* v_s_1931_, lean_object* v_x_1932_, lean_object* v_t_1933_){
_start:
{
lean_object* v___x_1934_; 
v___x_1934_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(v___f_1930_, v_s_1931_, v_t_1933_);
return v___x_1934_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1___boxed(lean_object* v___f_1935_, lean_object* v_s_1936_, lean_object* v_x_1937_, lean_object* v_t_1938_){
_start:
{
lean_object* v_res_1939_; 
v_res_1939_ = lp_aesop_Aesop_traceSimpTheoremTreeContents___lam__1(v___f_1935_, v_s_1936_, v_x_1937_, v_t_1938_);
lean_dec_ref(v_t_1938_);
lean_dec(v_x_1937_);
return v_res_1939_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(lean_object* v_opts_1940_, lean_object* v_opt_1941_){
_start:
{
lean_object* v_name_1942_; lean_object* v_defValue_1943_; lean_object* v_map_1944_; lean_object* v___x_1945_; 
v_name_1942_ = lean_ctor_get(v_opt_1941_, 0);
v_defValue_1943_ = lean_ctor_get(v_opt_1941_, 1);
v_map_1944_ = lean_ctor_get(v_opts_1940_, 0);
v___x_1945_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1944_, v_name_1942_);
if (lean_obj_tag(v___x_1945_) == 0)
{
uint8_t v___x_1946_; 
v___x_1946_ = lean_unbox(v_defValue_1943_);
return v___x_1946_;
}
else
{
lean_object* v_val_1947_; 
v_val_1947_ = lean_ctor_get(v___x_1945_, 0);
lean_inc(v_val_1947_);
lean_dec_ref_known(v___x_1945_, 1);
if (lean_obj_tag(v_val_1947_) == 1)
{
uint8_t v_v_1948_; 
v_v_1948_ = lean_ctor_get_uint8(v_val_1947_, 0);
lean_dec_ref_known(v_val_1947_, 0);
return v_v_1948_;
}
else
{
uint8_t v___x_1949_; 
lean_dec(v_val_1947_);
v___x_1949_ = lean_unbox(v_defValue_1943_);
return v___x_1949_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0___boxed(lean_object* v_opts_1950_, lean_object* v_opt_1951_){
_start:
{
uint8_t v_res_1952_; lean_object* v_r_1953_; 
v_res_1952_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_opts_1950_, v_opt_1951_);
lean_dec_ref(v_opt_1951_);
lean_dec_ref(v_opts_1950_);
v_r_1953_ = lean_box(v_res_1952_);
return v_r_1953_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(lean_object* v_opt_1954_, lean_object* v___y_1955_){
_start:
{
lean_object* v_options_1957_; lean_object* v_option_1958_; uint8_t v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; 
v_options_1957_ = lean_ctor_get(v___y_1955_, 2);
v_option_1958_ = lean_ctor_get(v_opt_1954_, 1);
v___x_1959_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_1957_, v_option_1958_);
v___x_1960_ = lean_box(v___x_1959_);
v___x_1961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1961_, 0, v___x_1960_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg___boxed(lean_object* v_opt_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_){
_start:
{
lean_object* v_res_1965_; 
v_res_1965_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(v_opt_1962_, v___y_1963_);
lean_dec_ref(v___y_1963_);
lean_dec_ref(v_opt_1962_);
return v_res_1965_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg(lean_object* v_hi_1966_, lean_object* v_pivot_1967_, lean_object* v_as_1968_, lean_object* v_i_1969_, lean_object* v_k_1970_){
_start:
{
uint8_t v___x_1971_; 
v___x_1971_ = lean_nat_dec_lt(v_k_1970_, v_hi_1966_);
if (v___x_1971_ == 0)
{
lean_object* v___x_1972_; lean_object* v___x_1973_; 
lean_dec(v_k_1970_);
v___x_1972_ = lean_array_fswap(v_as_1968_, v_i_1969_, v_hi_1966_);
v___x_1973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1973_, 0, v_i_1969_);
lean_ctor_set(v___x_1973_, 1, v___x_1972_);
return v___x_1973_;
}
else
{
lean_object* v___x_1974_; uint8_t v___x_1975_; 
v___x_1974_ = lean_array_fget_borrowed(v_as_1968_, v_k_1970_);
v___x_1975_ = lean_string_compare(v___x_1974_, v_pivot_1967_);
if (v___x_1975_ == 0)
{
lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; 
v___x_1976_ = lean_array_fswap(v_as_1968_, v_i_1969_, v_k_1970_);
v___x_1977_ = lean_unsigned_to_nat(1u);
v___x_1978_ = lean_nat_add(v_i_1969_, v___x_1977_);
lean_dec(v_i_1969_);
v___x_1979_ = lean_nat_add(v_k_1970_, v___x_1977_);
lean_dec(v_k_1970_);
v_as_1968_ = v___x_1976_;
v_i_1969_ = v___x_1978_;
v_k_1970_ = v___x_1979_;
goto _start;
}
else
{
lean_object* v___x_1981_; lean_object* v___x_1982_; 
v___x_1981_ = lean_unsigned_to_nat(1u);
v___x_1982_ = lean_nat_add(v_k_1970_, v___x_1981_);
lean_dec(v_k_1970_);
v_k_1970_ = v___x_1982_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg___boxed(lean_object* v_hi_1984_, lean_object* v_pivot_1985_, lean_object* v_as_1986_, lean_object* v_i_1987_, lean_object* v_k_1988_){
_start:
{
lean_object* v_res_1989_; 
v_res_1989_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg(v_hi_1984_, v_pivot_1985_, v_as_1986_, v_i_1987_, v_k_1988_);
lean_dec_ref(v_pivot_1985_);
lean_dec(v_hi_1984_);
return v_res_1989_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(uint8_t v___x_1990_, lean_object* v_x_1991_, lean_object* v_y_1992_){
_start:
{
uint8_t v___x_1993_; 
v___x_1993_ = lean_string_compare(v_x_1991_, v_y_1992_);
if (v___x_1993_ == 0)
{
return v___x_1990_;
}
else
{
uint8_t v___x_1994_; 
v___x_1994_ = 0;
return v___x_1994_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0___boxed(lean_object* v___x_1995_, lean_object* v_x_1996_, lean_object* v_y_1997_){
_start:
{
uint8_t v___x_3687__boxed_1998_; uint8_t v_res_1999_; lean_object* v_r_2000_; 
v___x_3687__boxed_1998_ = lean_unbox(v___x_1995_);
v_res_1999_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(v___x_3687__boxed_1998_, v_x_1996_, v_y_1997_);
lean_dec_ref(v_y_1997_);
lean_dec_ref(v_x_1996_);
v_r_2000_ = lean_box(v_res_1999_);
return v_r_2000_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(lean_object* v_n_2001_, lean_object* v_as_2002_, lean_object* v_lo_2003_, lean_object* v_hi_2004_){
_start:
{
lean_object* v___y_2006_; uint8_t v___x_2016_; 
v___x_2016_ = lean_nat_dec_lt(v_lo_2003_, v_hi_2004_);
if (v___x_2016_ == 0)
{
lean_dec(v_lo_2003_);
return v_as_2002_;
}
else
{
lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v_mid_2019_; lean_object* v___y_2021_; lean_object* v___y_2027_; lean_object* v___x_2032_; lean_object* v___x_2033_; uint8_t v___x_2034_; 
v___x_2017_ = lean_nat_add(v_lo_2003_, v_hi_2004_);
v___x_2018_ = lean_unsigned_to_nat(1u);
v_mid_2019_ = lean_nat_shiftr(v___x_2017_, v___x_2018_);
lean_dec(v___x_2017_);
v___x_2032_ = lean_array_fget_borrowed(v_as_2002_, v_mid_2019_);
v___x_2033_ = lean_array_fget_borrowed(v_as_2002_, v_lo_2003_);
v___x_2034_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(v___x_2016_, v___x_2032_, v___x_2033_);
if (v___x_2034_ == 0)
{
v___y_2027_ = v_as_2002_;
goto v___jp_2026_;
}
else
{
lean_object* v___x_2035_; 
v___x_2035_ = lean_array_fswap(v_as_2002_, v_lo_2003_, v_mid_2019_);
v___y_2027_ = v___x_2035_;
goto v___jp_2026_;
}
v___jp_2020_:
{
lean_object* v___x_2022_; lean_object* v___x_2023_; uint8_t v___x_2024_; 
v___x_2022_ = lean_array_fget_borrowed(v___y_2021_, v_mid_2019_);
v___x_2023_ = lean_array_fget_borrowed(v___y_2021_, v_hi_2004_);
v___x_2024_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(v___x_2016_, v___x_2022_, v___x_2023_);
if (v___x_2024_ == 0)
{
lean_dec(v_mid_2019_);
v___y_2006_ = v___y_2021_;
goto v___jp_2005_;
}
else
{
lean_object* v___x_2025_; 
v___x_2025_ = lean_array_fswap(v___y_2021_, v_mid_2019_, v_hi_2004_);
lean_dec(v_mid_2019_);
v___y_2006_ = v___x_2025_;
goto v___jp_2005_;
}
}
v___jp_2026_:
{
lean_object* v___x_2028_; lean_object* v___x_2029_; uint8_t v___x_2030_; 
v___x_2028_ = lean_array_fget_borrowed(v___y_2027_, v_hi_2004_);
v___x_2029_ = lean_array_fget_borrowed(v___y_2027_, v_lo_2003_);
v___x_2030_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___lam__0(v___x_2016_, v___x_2028_, v___x_2029_);
if (v___x_2030_ == 0)
{
v___y_2021_ = v___y_2027_;
goto v___jp_2020_;
}
else
{
lean_object* v___x_2031_; 
v___x_2031_ = lean_array_fswap(v___y_2027_, v_lo_2003_, v_hi_2004_);
v___y_2021_ = v___x_2031_;
goto v___jp_2020_;
}
}
}
v___jp_2005_:
{
lean_object* v_pivot_2007_; lean_object* v___x_2008_; lean_object* v_fst_2009_; lean_object* v_snd_2010_; uint8_t v___x_2011_; 
v_pivot_2007_ = lean_array_fget(v___y_2006_, v_hi_2004_);
lean_inc_n(v_lo_2003_, 2);
v___x_2008_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg(v_hi_2004_, v_pivot_2007_, v___y_2006_, v_lo_2003_, v_lo_2003_);
lean_dec(v_pivot_2007_);
v_fst_2009_ = lean_ctor_get(v___x_2008_, 0);
lean_inc(v_fst_2009_);
v_snd_2010_ = lean_ctor_get(v___x_2008_, 1);
lean_inc(v_snd_2010_);
lean_dec_ref(v___x_2008_);
v___x_2011_ = lean_nat_dec_le(v_hi_2004_, v_fst_2009_);
if (v___x_2011_ == 0)
{
lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; 
v___x_2012_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(v_n_2001_, v_snd_2010_, v_lo_2003_, v_fst_2009_);
v___x_2013_ = lean_unsigned_to_nat(1u);
v___x_2014_ = lean_nat_add(v_fst_2009_, v___x_2013_);
lean_dec(v_fst_2009_);
v_as_2002_ = v___x_2012_;
v_lo_2003_ = v___x_2014_;
goto _start;
}
else
{
lean_dec(v_fst_2009_);
lean_dec(v_lo_2003_);
return v_snd_2010_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg___boxed(lean_object* v_n_2036_, lean_object* v_as_2037_, lean_object* v_lo_2038_, lean_object* v_hi_2039_){
_start:
{
lean_object* v_res_2040_; 
v_res_2040_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(v_n_2036_, v_as_2037_, v_lo_2038_, v_hi_2039_);
lean_dec(v_hi_2039_);
lean_dec(v_n_2036_);
return v_res_2040_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(lean_object* v_xs_2041_){
_start:
{
lean_object* v___x_2042_; lean_object* v___x_2043_; uint8_t v___x_2044_; 
v___x_2042_ = lean_array_get_size(v_xs_2041_);
v___x_2043_ = lean_unsigned_to_nat(0u);
v___x_2044_ = lean_nat_dec_eq(v___x_2042_, v___x_2043_);
if (v___x_2044_ == 0)
{
lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___y_2048_; uint8_t v___x_2052_; 
v___x_2045_ = lean_unsigned_to_nat(1u);
v___x_2046_ = lean_nat_sub(v___x_2042_, v___x_2045_);
v___x_2052_ = lean_nat_dec_le(v___x_2043_, v___x_2046_);
if (v___x_2052_ == 0)
{
lean_inc(v___x_2046_);
v___y_2048_ = v___x_2046_;
goto v___jp_2047_;
}
else
{
v___y_2048_ = v___x_2043_;
goto v___jp_2047_;
}
v___jp_2047_:
{
uint8_t v___x_2049_; 
v___x_2049_ = lean_nat_dec_le(v___y_2048_, v___x_2046_);
if (v___x_2049_ == 0)
{
lean_object* v___x_2050_; 
lean_dec(v___x_2046_);
lean_inc(v___y_2048_);
v___x_2050_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(v___x_2042_, v_xs_2041_, v___y_2048_, v___y_2048_);
lean_dec(v___y_2048_);
return v___x_2050_;
}
else
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(v___x_2042_, v_xs_2041_, v___y_2048_, v___x_2046_);
lean_dec(v___x_2046_);
return v___x_2051_;
}
}
}
else
{
return v_xs_2041_;
}
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0(void){
_start:
{
lean_object* v___x_2053_; 
v___x_2053_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2053_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1(void){
_start:
{
lean_object* v___x_2054_; lean_object* v___x_2055_; 
v___x_2054_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__0);
v___x_2055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2055_, 0, v___x_2054_);
return v___x_2055_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2(void){
_start:
{
lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; 
v___x_2056_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1);
v___x_2057_ = lean_unsigned_to_nat(0u);
v___x_2058_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2058_, 0, v___x_2057_);
lean_ctor_set(v___x_2058_, 1, v___x_2057_);
lean_ctor_set(v___x_2058_, 2, v___x_2057_);
lean_ctor_set(v___x_2058_, 3, v___x_2057_);
lean_ctor_set(v___x_2058_, 4, v___x_2056_);
lean_ctor_set(v___x_2058_, 5, v___x_2056_);
lean_ctor_set(v___x_2058_, 6, v___x_2056_);
lean_ctor_set(v___x_2058_, 7, v___x_2056_);
lean_ctor_set(v___x_2058_, 8, v___x_2056_);
lean_ctor_set(v___x_2058_, 9, v___x_2056_);
return v___x_2058_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3(void){
_start:
{
lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; 
v___x_2059_ = lean_unsigned_to_nat(32u);
v___x_2060_ = lean_mk_empty_array_with_capacity(v___x_2059_);
v___x_2061_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2061_, 0, v___x_2060_);
return v___x_2061_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4(void){
_start:
{
size_t v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; 
v___x_2062_ = ((size_t)5ULL);
v___x_2063_ = lean_unsigned_to_nat(0u);
v___x_2064_ = lean_unsigned_to_nat(32u);
v___x_2065_ = lean_mk_empty_array_with_capacity(v___x_2064_);
v___x_2066_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__3);
v___x_2067_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2067_, 0, v___x_2066_);
lean_ctor_set(v___x_2067_, 1, v___x_2065_);
lean_ctor_set(v___x_2067_, 2, v___x_2063_);
lean_ctor_set(v___x_2067_, 3, v___x_2063_);
lean_ctor_set_usize(v___x_2067_, 4, v___x_2062_);
return v___x_2067_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5(void){
_start:
{
lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; 
v___x_2068_ = lean_box(1);
v___x_2069_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__4);
v___x_2070_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__1);
v___x_2071_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2071_, 0, v___x_2070_);
lean_ctor_set(v___x_2071_, 1, v___x_2069_);
lean_ctor_set(v___x_2071_, 2, v___x_2068_);
return v___x_2071_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10(lean_object* v_msgData_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_){
_start:
{
lean_object* v___x_2076_; lean_object* v_env_2077_; lean_object* v_options_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; 
v___x_2076_ = lean_st_ref_get(v___y_2074_);
v_env_2077_ = lean_ctor_get(v___x_2076_, 0);
lean_inc_ref(v_env_2077_);
lean_dec(v___x_2076_);
v_options_2078_ = lean_ctor_get(v___y_2073_, 2);
v___x_2079_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__2);
v___x_2080_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___closed__5);
lean_inc_ref(v_options_2078_);
v___x_2081_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2081_, 0, v_env_2077_);
lean_ctor_set(v___x_2081_, 1, v___x_2079_);
lean_ctor_set(v___x_2081_, 2, v___x_2080_);
lean_ctor_set(v___x_2081_, 3, v_options_2078_);
v___x_2082_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2082_, 0, v___x_2081_);
lean_ctor_set(v___x_2082_, 1, v_msgData_2072_);
v___x_2083_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2083_, 0, v___x_2082_);
return v___x_2083_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10___boxed(lean_object* v_msgData_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_){
_start:
{
lean_object* v_res_2088_; 
v_res_2088_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10(v_msgData_2084_, v___y_2085_, v___y_2086_);
lean_dec(v___y_2086_);
lean_dec_ref(v___y_2085_);
return v_res_2088_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0(void){
_start:
{
lean_object* v___x_2089_; double v___x_2090_; 
v___x_2089_ = lean_unsigned_to_nat(0u);
v___x_2090_ = lean_float_of_nat(v___x_2089_);
return v___x_2090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(lean_object* v_cls_2093_, lean_object* v_msg_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_){
_start:
{
lean_object* v_ref_2098_; lean_object* v___x_2099_; lean_object* v_a_2100_; lean_object* v___x_2102_; uint8_t v_isShared_2103_; uint8_t v_isSharedCheck_2144_; 
v_ref_2098_ = lean_ctor_get(v___y_2095_, 5);
v___x_2099_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10(v_msg_2094_, v___y_2095_, v___y_2096_);
v_a_2100_ = lean_ctor_get(v___x_2099_, 0);
v_isSharedCheck_2144_ = !lean_is_exclusive(v___x_2099_);
if (v_isSharedCheck_2144_ == 0)
{
v___x_2102_ = v___x_2099_;
v_isShared_2103_ = v_isSharedCheck_2144_;
goto v_resetjp_2101_;
}
else
{
lean_inc(v_a_2100_);
lean_dec(v___x_2099_);
v___x_2102_ = lean_box(0);
v_isShared_2103_ = v_isSharedCheck_2144_;
goto v_resetjp_2101_;
}
v_resetjp_2101_:
{
lean_object* v___x_2104_; lean_object* v_traceState_2105_; lean_object* v_env_2106_; lean_object* v_nextMacroScope_2107_; lean_object* v_ngen_2108_; lean_object* v_auxDeclNGen_2109_; lean_object* v_cache_2110_; lean_object* v_messages_2111_; lean_object* v_infoState_2112_; lean_object* v_snapshotTasks_2113_; lean_object* v___x_2115_; uint8_t v_isShared_2116_; uint8_t v_isSharedCheck_2143_; 
v___x_2104_ = lean_st_ref_take(v___y_2096_);
v_traceState_2105_ = lean_ctor_get(v___x_2104_, 4);
v_env_2106_ = lean_ctor_get(v___x_2104_, 0);
v_nextMacroScope_2107_ = lean_ctor_get(v___x_2104_, 1);
v_ngen_2108_ = lean_ctor_get(v___x_2104_, 2);
v_auxDeclNGen_2109_ = lean_ctor_get(v___x_2104_, 3);
v_cache_2110_ = lean_ctor_get(v___x_2104_, 5);
v_messages_2111_ = lean_ctor_get(v___x_2104_, 6);
v_infoState_2112_ = lean_ctor_get(v___x_2104_, 7);
v_snapshotTasks_2113_ = lean_ctor_get(v___x_2104_, 8);
v_isSharedCheck_2143_ = !lean_is_exclusive(v___x_2104_);
if (v_isSharedCheck_2143_ == 0)
{
v___x_2115_ = v___x_2104_;
v_isShared_2116_ = v_isSharedCheck_2143_;
goto v_resetjp_2114_;
}
else
{
lean_inc(v_snapshotTasks_2113_);
lean_inc(v_infoState_2112_);
lean_inc(v_messages_2111_);
lean_inc(v_cache_2110_);
lean_inc(v_traceState_2105_);
lean_inc(v_auxDeclNGen_2109_);
lean_inc(v_ngen_2108_);
lean_inc(v_nextMacroScope_2107_);
lean_inc(v_env_2106_);
lean_dec(v___x_2104_);
v___x_2115_ = lean_box(0);
v_isShared_2116_ = v_isSharedCheck_2143_;
goto v_resetjp_2114_;
}
v_resetjp_2114_:
{
uint64_t v_tid_2117_; lean_object* v_traces_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2142_; 
v_tid_2117_ = lean_ctor_get_uint64(v_traceState_2105_, sizeof(void*)*1);
v_traces_2118_ = lean_ctor_get(v_traceState_2105_, 0);
v_isSharedCheck_2142_ = !lean_is_exclusive(v_traceState_2105_);
if (v_isSharedCheck_2142_ == 0)
{
v___x_2120_ = v_traceState_2105_;
v_isShared_2121_ = v_isSharedCheck_2142_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_traces_2118_);
lean_dec(v_traceState_2105_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2142_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v___x_2122_; double v___x_2123_; uint8_t v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2132_; 
v___x_2122_ = lean_box(0);
v___x_2123_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0);
v___x_2124_ = 0;
v___x_2125_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_2126_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2126_, 0, v_cls_2093_);
lean_ctor_set(v___x_2126_, 1, v___x_2122_);
lean_ctor_set(v___x_2126_, 2, v___x_2125_);
lean_ctor_set_float(v___x_2126_, sizeof(void*)*3, v___x_2123_);
lean_ctor_set_float(v___x_2126_, sizeof(void*)*3 + 8, v___x_2123_);
lean_ctor_set_uint8(v___x_2126_, sizeof(void*)*3 + 16, v___x_2124_);
v___x_2127_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__1));
v___x_2128_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2128_, 0, v___x_2126_);
lean_ctor_set(v___x_2128_, 1, v_a_2100_);
lean_ctor_set(v___x_2128_, 2, v___x_2127_);
lean_inc(v_ref_2098_);
v___x_2129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2129_, 0, v_ref_2098_);
lean_ctor_set(v___x_2129_, 1, v___x_2128_);
v___x_2130_ = l_Lean_PersistentArray_push___redArg(v_traces_2118_, v___x_2129_);
if (v_isShared_2121_ == 0)
{
lean_ctor_set(v___x_2120_, 0, v___x_2130_);
v___x_2132_ = v___x_2120_;
goto v_reusejp_2131_;
}
else
{
lean_object* v_reuseFailAlloc_2141_; 
v_reuseFailAlloc_2141_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2141_, 0, v___x_2130_);
lean_ctor_set_uint64(v_reuseFailAlloc_2141_, sizeof(void*)*1, v_tid_2117_);
v___x_2132_ = v_reuseFailAlloc_2141_;
goto v_reusejp_2131_;
}
v_reusejp_2131_:
{
lean_object* v___x_2134_; 
if (v_isShared_2116_ == 0)
{
lean_ctor_set(v___x_2115_, 4, v___x_2132_);
v___x_2134_ = v___x_2115_;
goto v_reusejp_2133_;
}
else
{
lean_object* v_reuseFailAlloc_2140_; 
v_reuseFailAlloc_2140_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2140_, 0, v_env_2106_);
lean_ctor_set(v_reuseFailAlloc_2140_, 1, v_nextMacroScope_2107_);
lean_ctor_set(v_reuseFailAlloc_2140_, 2, v_ngen_2108_);
lean_ctor_set(v_reuseFailAlloc_2140_, 3, v_auxDeclNGen_2109_);
lean_ctor_set(v_reuseFailAlloc_2140_, 4, v___x_2132_);
lean_ctor_set(v_reuseFailAlloc_2140_, 5, v_cache_2110_);
lean_ctor_set(v_reuseFailAlloc_2140_, 6, v_messages_2111_);
lean_ctor_set(v_reuseFailAlloc_2140_, 7, v_infoState_2112_);
lean_ctor_set(v_reuseFailAlloc_2140_, 8, v_snapshotTasks_2113_);
v___x_2134_ = v_reuseFailAlloc_2140_;
goto v_reusejp_2133_;
}
v_reusejp_2133_:
{
lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2138_; 
v___x_2135_ = lean_st_ref_set(v___y_2096_, v___x_2134_);
v___x_2136_ = lean_box(0);
if (v_isShared_2103_ == 0)
{
lean_ctor_set(v___x_2102_, 0, v___x_2136_);
v___x_2138_ = v___x_2102_;
goto v_reusejp_2137_;
}
else
{
lean_object* v_reuseFailAlloc_2139_; 
v_reuseFailAlloc_2139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2139_, 0, v___x_2136_);
v___x_2138_ = v_reuseFailAlloc_2139_;
goto v_reusejp_2137_;
}
v_reusejp_2137_:
{
return v___x_2138_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___boxed(lean_object* v_cls_2145_, lean_object* v_msg_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_){
_start:
{
lean_object* v_res_2150_; 
v_res_2150_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_cls_2145_, v_msg_2146_, v___y_2147_, v___y_2148_);
lean_dec(v___y_2148_);
lean_dec_ref(v___y_2147_);
return v_res_2150_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6(lean_object* v_opt_2151_, lean_object* v_as_2152_, size_t v_sz_2153_, size_t v_i_2154_, lean_object* v_b_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_){
_start:
{
uint8_t v___x_2159_; 
v___x_2159_ = lean_usize_dec_lt(v_i_2154_, v_sz_2153_);
if (v___x_2159_ == 0)
{
lean_object* v___x_2160_; 
lean_dec_ref(v_opt_2151_);
v___x_2160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2160_, 0, v_b_2155_);
return v___x_2160_;
}
else
{
lean_object* v_traceClass_2161_; lean_object* v_a_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; 
v_traceClass_2161_ = lean_ctor_get(v_opt_2151_, 0);
v_a_2162_ = lean_array_uget_borrowed(v_as_2152_, v_i_2154_);
lean_inc(v_a_2162_);
v___x_2163_ = l_Lean_stringToMessageData(v_a_2162_);
lean_inc(v_traceClass_2161_);
v___x_2164_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_traceClass_2161_, v___x_2163_, v___y_2156_, v___y_2157_);
if (lean_obj_tag(v___x_2164_) == 0)
{
lean_object* v___x_2165_; size_t v___x_2166_; size_t v___x_2167_; 
lean_dec_ref_known(v___x_2164_, 1);
v___x_2165_ = lean_box(0);
v___x_2166_ = ((size_t)1ULL);
v___x_2167_ = lean_usize_add(v_i_2154_, v___x_2166_);
v_i_2154_ = v___x_2167_;
v_b_2155_ = v___x_2165_;
goto _start;
}
else
{
lean_dec_ref(v_opt_2151_);
return v___x_2164_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6___boxed(lean_object* v_opt_2169_, lean_object* v_as_2170_, lean_object* v_sz_2171_, lean_object* v_i_2172_, lean_object* v_b_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_){
_start:
{
size_t v_sz_boxed_2177_; size_t v_i_boxed_2178_; lean_object* v_res_2179_; 
v_sz_boxed_2177_ = lean_unbox_usize(v_sz_2171_);
lean_dec(v_sz_2171_);
v_i_boxed_2178_ = lean_unbox_usize(v_i_2172_);
lean_dec(v_i_2172_);
v_res_2179_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6(v_opt_2169_, v_as_2170_, v_sz_boxed_2177_, v_i_boxed_2178_, v_b_2173_, v___y_2174_, v___y_2175_);
lean_dec(v___y_2175_);
lean_dec_ref(v___y_2174_);
lean_dec_ref(v_as_2170_);
return v_res_2179_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg(lean_object* v_f_2180_, lean_object* v_keys_2181_, lean_object* v_vals_2182_, lean_object* v_i_2183_, lean_object* v_acc_2184_){
_start:
{
lean_object* v___x_2185_; uint8_t v___x_2186_; 
v___x_2185_ = lean_array_get_size(v_keys_2181_);
v___x_2186_ = lean_nat_dec_lt(v_i_2183_, v___x_2185_);
if (v___x_2186_ == 0)
{
lean_dec(v_i_2183_);
lean_dec(v_f_2180_);
return v_acc_2184_;
}
else
{
lean_object* v_k_2187_; lean_object* v_v_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; 
v_k_2187_ = lean_array_fget_borrowed(v_keys_2181_, v_i_2183_);
v_v_2188_ = lean_array_fget_borrowed(v_vals_2182_, v_i_2183_);
lean_inc(v_f_2180_);
lean_inc(v_v_2188_);
lean_inc(v_k_2187_);
v___x_2189_ = lean_apply_3(v_f_2180_, v_acc_2184_, v_k_2187_, v_v_2188_);
v___x_2190_ = lean_unsigned_to_nat(1u);
v___x_2191_ = lean_nat_add(v_i_2183_, v___x_2190_);
lean_dec(v_i_2183_);
v_i_2183_ = v___x_2191_;
v_acc_2184_ = v___x_2189_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg___boxed(lean_object* v_f_2193_, lean_object* v_keys_2194_, lean_object* v_vals_2195_, lean_object* v_i_2196_, lean_object* v_acc_2197_){
_start:
{
lean_object* v_res_2198_; 
v_res_2198_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg(v_f_2193_, v_keys_2194_, v_vals_2195_, v_i_2196_, v_acc_2197_);
lean_dec_ref(v_vals_2195_);
lean_dec_ref(v_keys_2194_);
return v_res_2198_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(lean_object* v_f_2199_, lean_object* v_x_2200_, lean_object* v_x_2201_){
_start:
{
if (lean_obj_tag(v_x_2200_) == 0)
{
lean_object* v_es_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; uint8_t v___x_2205_; 
v_es_2202_ = lean_ctor_get(v_x_2200_, 0);
v___x_2203_ = lean_unsigned_to_nat(0u);
v___x_2204_ = lean_array_get_size(v_es_2202_);
v___x_2205_ = lean_nat_dec_lt(v___x_2203_, v___x_2204_);
if (v___x_2205_ == 0)
{
lean_dec(v_f_2199_);
return v_x_2201_;
}
else
{
uint8_t v___x_2206_; 
v___x_2206_ = lean_nat_dec_le(v___x_2204_, v___x_2204_);
if (v___x_2206_ == 0)
{
if (v___x_2205_ == 0)
{
lean_dec(v_f_2199_);
return v_x_2201_;
}
else
{
size_t v___x_2207_; size_t v___x_2208_; lean_object* v___x_2209_; 
v___x_2207_ = ((size_t)0ULL);
v___x_2208_ = lean_usize_of_nat(v___x_2204_);
v___x_2209_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(v_f_2199_, v_es_2202_, v___x_2207_, v___x_2208_, v_x_2201_);
return v___x_2209_;
}
}
else
{
size_t v___x_2210_; size_t v___x_2211_; lean_object* v___x_2212_; 
v___x_2210_ = ((size_t)0ULL);
v___x_2211_ = lean_usize_of_nat(v___x_2204_);
v___x_2212_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(v_f_2199_, v_es_2202_, v___x_2210_, v___x_2211_, v_x_2201_);
return v___x_2212_;
}
}
}
else
{
lean_object* v_ks_2213_; lean_object* v_vs_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; 
v_ks_2213_ = lean_ctor_get(v_x_2200_, 0);
v_vs_2214_ = lean_ctor_get(v_x_2200_, 1);
v___x_2215_ = lean_unsigned_to_nat(0u);
v___x_2216_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg(v_f_2199_, v_ks_2213_, v_vs_2214_, v___x_2215_, v_x_2201_);
return v___x_2216_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(lean_object* v_f_2217_, lean_object* v_as_2218_, size_t v_i_2219_, size_t v_stop_2220_, lean_object* v_b_2221_){
_start:
{
lean_object* v___y_2223_; uint8_t v___x_2227_; 
v___x_2227_ = lean_usize_dec_eq(v_i_2219_, v_stop_2220_);
if (v___x_2227_ == 0)
{
lean_object* v___x_2228_; 
v___x_2228_ = lean_array_uget_borrowed(v_as_2218_, v_i_2219_);
switch(lean_obj_tag(v___x_2228_))
{
case 0:
{
lean_object* v_key_2229_; lean_object* v_val_2230_; lean_object* v___x_2231_; 
v_key_2229_ = lean_ctor_get(v___x_2228_, 0);
v_val_2230_ = lean_ctor_get(v___x_2228_, 1);
lean_inc(v_f_2217_);
lean_inc(v_val_2230_);
lean_inc(v_key_2229_);
v___x_2231_ = lean_apply_3(v_f_2217_, v_b_2221_, v_key_2229_, v_val_2230_);
v___y_2223_ = v___x_2231_;
goto v___jp_2222_;
}
case 1:
{
lean_object* v_node_2232_; lean_object* v___x_2233_; 
v_node_2232_ = lean_ctor_get(v___x_2228_, 0);
lean_inc(v_f_2217_);
v___x_2233_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_2217_, v_node_2232_, v_b_2221_);
v___y_2223_ = v___x_2233_;
goto v___jp_2222_;
}
default: 
{
v___y_2223_ = v_b_2221_;
goto v___jp_2222_;
}
}
}
else
{
lean_dec(v_f_2217_);
return v_b_2221_;
}
v___jp_2222_:
{
size_t v___x_2224_; size_t v___x_2225_; 
v___x_2224_ = ((size_t)1ULL);
v___x_2225_ = lean_usize_add(v_i_2219_, v___x_2224_);
v_i_2219_ = v___x_2225_;
v_b_2221_ = v___y_2223_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg___boxed(lean_object* v_f_2234_, lean_object* v_as_2235_, lean_object* v_i_2236_, lean_object* v_stop_2237_, lean_object* v_b_2238_){
_start:
{
size_t v_i_boxed_2239_; size_t v_stop_boxed_2240_; lean_object* v_res_2241_; 
v_i_boxed_2239_ = lean_unbox_usize(v_i_2236_);
lean_dec(v_i_2236_);
v_stop_boxed_2240_ = lean_unbox_usize(v_stop_2237_);
lean_dec(v_stop_2237_);
v_res_2241_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(v_f_2234_, v_as_2235_, v_i_boxed_2239_, v_stop_boxed_2240_, v_b_2238_);
lean_dec_ref(v_as_2235_);
return v_res_2241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg___boxed(lean_object* v_f_2242_, lean_object* v_x_2243_, lean_object* v_x_2244_){
_start:
{
lean_object* v_res_2245_; 
v_res_2245_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_2242_, v_x_2243_, v_x_2244_);
lean_dec_ref(v_x_2243_);
return v_res_2245_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3(uint8_t v_a_2246_, size_t v_sz_2247_, size_t v_i_2248_, lean_object* v_bs_2249_){
_start:
{
uint8_t v___x_2250_; 
v___x_2250_ = lean_usize_dec_lt(v_i_2248_, v_sz_2247_);
if (v___x_2250_ == 0)
{
return v_bs_2249_;
}
else
{
lean_object* v_v_2251_; lean_object* v_origin_2252_; lean_object* v___x_2253_; lean_object* v_bs_x27_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; size_t v___x_2257_; size_t v___x_2258_; lean_object* v___x_2259_; 
v_v_2251_ = lean_array_uget_borrowed(v_bs_2249_, v_i_2248_);
v_origin_2252_ = lean_ctor_get(v_v_2251_, 4);
lean_inc_ref(v_origin_2252_);
v___x_2253_ = lean_unsigned_to_nat(0u);
v_bs_x27_2254_ = lean_array_uset(v_bs_2249_, v_i_2248_, v___x_2253_);
v___x_2255_ = l_Lean_Meta_Origin_key(v_origin_2252_);
lean_dec_ref(v_origin_2252_);
v___x_2256_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_2255_, v_a_2246_);
v___x_2257_ = ((size_t)1ULL);
v___x_2258_ = lean_usize_add(v_i_2248_, v___x_2257_);
v___x_2259_ = lean_array_uset(v_bs_x27_2254_, v_i_2248_, v___x_2256_);
v_i_2248_ = v___x_2258_;
v_bs_2249_ = v___x_2259_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3___boxed(lean_object* v_a_2261_, lean_object* v_sz_2262_, lean_object* v_i_2263_, lean_object* v_bs_2264_){
_start:
{
uint8_t v_a_4062__boxed_2265_; size_t v_sz_boxed_2266_; size_t v_i_boxed_2267_; lean_object* v_res_2268_; 
v_a_4062__boxed_2265_ = lean_unbox(v_a_2261_);
v_sz_boxed_2266_ = lean_unbox_usize(v_sz_2262_);
lean_dec(v_sz_2262_);
v_i_boxed_2267_ = lean_unbox_usize(v_i_2263_);
lean_dec(v_i_2263_);
v_res_2268_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3(v_a_4062__boxed_2265_, v_sz_boxed_2266_, v_i_boxed_2267_, v_bs_2264_);
return v_res_2268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents(lean_object* v_t_2274_, lean_object* v_opt_2275_, lean_object* v_a_2276_, lean_object* v_a_2277_){
_start:
{
lean_object* v___x_2279_; lean_object* v_a_2280_; lean_object* v___x_2282_; uint8_t v_isShared_2283_; uint8_t v_isSharedCheck_2308_; 
v___x_2279_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(v_opt_2275_, v_a_2276_);
v_a_2280_ = lean_ctor_get(v___x_2279_, 0);
v_isSharedCheck_2308_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2308_ == 0)
{
v___x_2282_ = v___x_2279_;
v_isShared_2283_ = v_isSharedCheck_2308_;
goto v_resetjp_2281_;
}
else
{
lean_inc(v_a_2280_);
lean_dec(v___x_2279_);
v___x_2282_ = lean_box(0);
v_isShared_2283_ = v_isSharedCheck_2308_;
goto v_resetjp_2281_;
}
v_resetjp_2281_:
{
uint8_t v___x_2284_; 
v___x_2284_ = lean_unbox(v_a_2280_);
if (v___x_2284_ == 0)
{
lean_object* v___x_2285_; lean_object* v___x_2287_; 
lean_dec(v_a_2280_);
lean_dec_ref(v_opt_2275_);
v___x_2285_ = lean_box(0);
if (v_isShared_2283_ == 0)
{
lean_ctor_set(v___x_2282_, 0, v___x_2285_);
v___x_2287_ = v___x_2282_;
goto v_reusejp_2286_;
}
else
{
lean_object* v_reuseFailAlloc_2288_; 
v_reuseFailAlloc_2288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2288_, 0, v___x_2285_);
v___x_2287_ = v_reuseFailAlloc_2288_;
goto v_reusejp_2286_;
}
v_reusejp_2286_:
{
return v___x_2287_;
}
}
else
{
lean_object* v___f_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; size_t v_sz_2292_; size_t v___x_2293_; uint8_t v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; size_t v_sz_2298_; lean_object* v___x_2299_; 
lean_del_object(v___x_2282_);
v___f_2289_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__1));
v___x_2290_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheoremTreeContents___closed__2));
v___x_2291_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_2289_, v_t_2274_, v___x_2290_);
v_sz_2292_ = lean_array_size(v___x_2291_);
v___x_2293_ = ((size_t)0ULL);
v___x_2294_ = lean_unbox(v_a_2280_);
lean_dec(v_a_2280_);
v___x_2295_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheoremTreeContents_spec__3(v___x_2294_, v_sz_2292_, v___x_2293_, v___x_2291_);
v___x_2296_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_2295_);
v___x_2297_ = lean_box(0);
v_sz_2298_ = lean_array_size(v___x_2296_);
v___x_2299_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheoremTreeContents_spec__6(v_opt_2275_, v___x_2296_, v_sz_2298_, v___x_2293_, v___x_2297_, v_a_2276_, v_a_2277_);
lean_dec_ref(v___x_2296_);
if (lean_obj_tag(v___x_2299_) == 0)
{
lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2306_; 
v_isSharedCheck_2306_ = !lean_is_exclusive(v___x_2299_);
if (v_isSharedCheck_2306_ == 0)
{
lean_object* v_unused_2307_; 
v_unused_2307_ = lean_ctor_get(v___x_2299_, 0);
lean_dec(v_unused_2307_);
v___x_2301_ = v___x_2299_;
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
else
{
lean_dec(v___x_2299_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2304_; 
if (v_isShared_2302_ == 0)
{
lean_ctor_set(v___x_2301_, 0, v___x_2297_);
v___x_2304_ = v___x_2301_;
goto v_reusejp_2303_;
}
else
{
lean_object* v_reuseFailAlloc_2305_; 
v_reuseFailAlloc_2305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2305_, 0, v___x_2297_);
v___x_2304_ = v_reuseFailAlloc_2305_;
goto v_reusejp_2303_;
}
v_reusejp_2303_:
{
return v___x_2304_;
}
}
}
else
{
return v___x_2299_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheoremTreeContents___boxed(lean_object* v_t_2309_, lean_object* v_opt_2310_, lean_object* v_a_2311_, lean_object* v_a_2312_, lean_object* v_a_2313_){
_start:
{
lean_object* v_res_2314_; 
v_res_2314_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v_t_2309_, v_opt_2310_, v_a_2311_, v_a_2312_);
lean_dec(v_a_2312_);
lean_dec_ref(v_a_2311_);
lean_dec_ref(v_t_2309_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0(lean_object* v_opt_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_){
_start:
{
lean_object* v___x_2319_; 
v___x_2319_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(v_opt_2315_, v___y_2316_);
return v___x_2319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___boxed(lean_object* v_opt_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_){
_start:
{
lean_object* v_res_2324_; 
v_res_2324_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0(v_opt_2320_, v___y_2321_, v___y_2322_);
lean_dec(v___y_2322_);
lean_dec_ref(v___y_2321_);
lean_dec_ref(v_opt_2320_);
return v_res_2324_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1(lean_object* v_00_u03c3_2325_, lean_object* v_00_u03b1_2326_, lean_object* v_f_2327_, lean_object* v_x_2328_, lean_object* v_x_2329_){
_start:
{
lean_object* v___x_2330_; 
v___x_2330_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___redArg(v_f_2327_, v_x_2328_, v_x_2329_);
return v___x_2330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1___boxed(lean_object* v_00_u03c3_2331_, lean_object* v_00_u03b1_2332_, lean_object* v_f_2333_, lean_object* v_x_2334_, lean_object* v_x_2335_){
_start:
{
lean_object* v_res_2336_; 
v_res_2336_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1(v_00_u03c3_2331_, v_00_u03b1_2332_, v_f_2333_, v_x_2334_, v_x_2335_);
lean_dec_ref(v_x_2335_);
return v_res_2336_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___redArg(lean_object* v_map_2337_, lean_object* v_f_2338_, lean_object* v_init_2339_){
_start:
{
lean_object* v___x_2340_; 
v___x_2340_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_2338_, v_map_2337_, v_init_2339_);
return v___x_2340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___redArg___boxed(lean_object* v_map_2341_, lean_object* v_f_2342_, lean_object* v_init_2343_){
_start:
{
lean_object* v_res_2344_; 
v_res_2344_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___redArg(v_map_2341_, v_f_2342_, v_init_2343_);
lean_dec_ref(v_map_2341_);
return v_res_2344_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2(lean_object* v_00_u03c3_2345_, lean_object* v_00_u03b2_2346_, lean_object* v_map_2347_, lean_object* v_f_2348_, lean_object* v_init_2349_){
_start:
{
lean_object* v___x_2350_; 
v___x_2350_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_2348_, v_map_2347_, v_init_2349_);
return v___x_2350_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2___boxed(lean_object* v_00_u03c3_2351_, lean_object* v_00_u03b2_2352_, lean_object* v_map_2353_, lean_object* v_f_2354_, lean_object* v_init_2355_){
_start:
{
lean_object* v_res_2356_; 
v_res_2356_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2(v_00_u03c3_2351_, v_00_u03b2_2352_, v_map_2353_, v_f_2354_, v_init_2355_);
lean_dec_ref(v_map_2353_);
return v_res_2356_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2(lean_object* v_00_u03b1_2357_, lean_object* v_00_u03c3_2358_, lean_object* v_f_2359_, lean_object* v_as_2360_, size_t v_i_2361_, size_t v_stop_2362_, lean_object* v_b_2363_){
_start:
{
lean_object* v___x_2364_; 
v___x_2364_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___redArg(v_f_2359_, v_as_2360_, v_i_2361_, v_stop_2362_, v_b_2363_);
return v___x_2364_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2___boxed(lean_object* v_00_u03b1_2365_, lean_object* v_00_u03c3_2366_, lean_object* v_f_2367_, lean_object* v_as_2368_, lean_object* v_i_2369_, lean_object* v_stop_2370_, lean_object* v_b_2371_){
_start:
{
size_t v_i_boxed_2372_; size_t v_stop_boxed_2373_; lean_object* v_res_2374_; 
v_i_boxed_2372_ = lean_unbox_usize(v_i_2369_);
lean_dec(v_i_2369_);
v_stop_boxed_2373_ = lean_unbox_usize(v_stop_2370_);
lean_dec(v_stop_2370_);
v_res_2374_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__2(v_00_u03b1_2365_, v_00_u03c3_2366_, v_f_2367_, v_as_2368_, v_i_boxed_2372_, v_stop_boxed_2373_, v_b_2371_);
lean_dec_ref(v_as_2368_);
return v_res_2374_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3(lean_object* v_00_u03b1_2375_, lean_object* v_00_u03c3_2376_, lean_object* v_f_2377_, lean_object* v_as_2378_, size_t v_i_2379_, size_t v_stop_2380_, lean_object* v_b_2381_){
_start:
{
lean_object* v___x_2382_; 
v___x_2382_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___redArg(v_f_2377_, v_as_2378_, v_i_2379_, v_stop_2380_, v_b_2381_);
return v___x_2382_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3___boxed(lean_object* v_00_u03b1_2383_, lean_object* v_00_u03c3_2384_, lean_object* v_f_2385_, lean_object* v_as_2386_, lean_object* v_i_2387_, lean_object* v_stop_2388_, lean_object* v_b_2389_){
_start:
{
size_t v_i_boxed_2390_; size_t v_stop_boxed_2391_; lean_object* v_res_2392_; 
v_i_boxed_2390_ = lean_unbox_usize(v_i_2387_);
lean_dec(v_i_2387_);
v_stop_boxed_2391_ = lean_unbox_usize(v_stop_2388_);
lean_dec(v_stop_2388_);
v_res_2392_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_traceSimpTheoremTreeContents_spec__1_spec__3(v_00_u03b1_2383_, v_00_u03c3_2384_, v_f_2385_, v_as_2386_, v_i_boxed_2390_, v_stop_boxed_2391_, v_b_2389_);
lean_dec_ref(v_as_2386_);
return v_res_2392_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5(lean_object* v_00_u03c3_2393_, lean_object* v_00_u03b1_2394_, lean_object* v_00_u03b2_2395_, lean_object* v_f_2396_, lean_object* v_x_2397_, lean_object* v_x_2398_){
_start:
{
lean_object* v___x_2399_; 
v___x_2399_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_2396_, v_x_2397_, v_x_2398_);
return v___x_2399_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___boxed(lean_object* v_00_u03c3_2400_, lean_object* v_00_u03b1_2401_, lean_object* v_00_u03b2_2402_, lean_object* v_f_2403_, lean_object* v_x_2404_, lean_object* v_x_2405_){
_start:
{
lean_object* v_res_2406_; 
v_res_2406_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5(v_00_u03c3_2400_, v_00_u03b1_2401_, v_00_u03b2_2402_, v_f_2403_, v_x_2404_, v_x_2405_);
lean_dec_ref(v_x_2404_);
return v_res_2406_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8(lean_object* v_n_2407_, lean_object* v_as_2408_, lean_object* v_lo_2409_, lean_object* v_hi_2410_, lean_object* v_w_2411_, lean_object* v_hlo_2412_, lean_object* v_hhi_2413_){
_start:
{
lean_object* v___x_2414_; 
v___x_2414_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___redArg(v_n_2407_, v_as_2408_, v_lo_2409_, v_hi_2410_);
return v___x_2414_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8___boxed(lean_object* v_n_2415_, lean_object* v_as_2416_, lean_object* v_lo_2417_, lean_object* v_hi_2418_, lean_object* v_w_2419_, lean_object* v_hlo_2420_, lean_object* v_hhi_2421_){
_start:
{
lean_object* v_res_2422_; 
v_res_2422_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8(v_n_2415_, v_as_2416_, v_lo_2417_, v_hi_2418_, v_w_2419_, v_hlo_2420_, v_hhi_2421_);
lean_dec(v_hi_2418_);
lean_dec(v_n_2415_);
return v_res_2422_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6(lean_object* v_00_u03b1_2423_, lean_object* v_00_u03b2_2424_, lean_object* v_00_u03c3_2425_, lean_object* v_f_2426_, lean_object* v_as_2427_, size_t v_i_2428_, size_t v_stop_2429_, lean_object* v_b_2430_){
_start:
{
lean_object* v___x_2431_; 
v___x_2431_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___redArg(v_f_2426_, v_as_2427_, v_i_2428_, v_stop_2429_, v_b_2430_);
return v___x_2431_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6___boxed(lean_object* v_00_u03b1_2432_, lean_object* v_00_u03b2_2433_, lean_object* v_00_u03c3_2434_, lean_object* v_f_2435_, lean_object* v_as_2436_, lean_object* v_i_2437_, lean_object* v_stop_2438_, lean_object* v_b_2439_){
_start:
{
size_t v_i_boxed_2440_; size_t v_stop_boxed_2441_; lean_object* v_res_2442_; 
v_i_boxed_2440_ = lean_unbox_usize(v_i_2437_);
lean_dec(v_i_2437_);
v_stop_boxed_2441_ = lean_unbox_usize(v_stop_2438_);
lean_dec(v_stop_2438_);
v_res_2442_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__6(v_00_u03b1_2432_, v_00_u03b2_2433_, v_00_u03c3_2434_, v_f_2435_, v_as_2436_, v_i_boxed_2440_, v_stop_boxed_2441_, v_b_2439_);
lean_dec_ref(v_as_2436_);
return v_res_2442_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7(lean_object* v_00_u03c3_2443_, lean_object* v_00_u03b1_2444_, lean_object* v_00_u03b2_2445_, lean_object* v_f_2446_, lean_object* v_keys_2447_, lean_object* v_vals_2448_, lean_object* v_heq_2449_, lean_object* v_i_2450_, lean_object* v_acc_2451_){
_start:
{
lean_object* v___x_2452_; 
v___x_2452_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___redArg(v_f_2446_, v_keys_2447_, v_vals_2448_, v_i_2450_, v_acc_2451_);
return v___x_2452_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7___boxed(lean_object* v_00_u03c3_2453_, lean_object* v_00_u03b1_2454_, lean_object* v_00_u03b2_2455_, lean_object* v_f_2456_, lean_object* v_keys_2457_, lean_object* v_vals_2458_, lean_object* v_heq_2459_, lean_object* v_i_2460_, lean_object* v_acc_2461_){
_start:
{
lean_object* v_res_2462_; 
v_res_2462_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5_spec__7(v_00_u03c3_2453_, v_00_u03b1_2454_, v_00_u03b2_2455_, v_f_2456_, v_keys_2457_, v_vals_2458_, v_heq_2459_, v_i_2460_, v_acc_2461_);
lean_dec_ref(v_vals_2458_);
lean_dec_ref(v_keys_2457_);
return v_res_2462_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11(lean_object* v_n_2463_, lean_object* v_lo_2464_, lean_object* v_hi_2465_, lean_object* v_hhi_2466_, lean_object* v_pivot_2467_, lean_object* v_as_2468_, lean_object* v_i_2469_, lean_object* v_k_2470_, lean_object* v_ilo_2471_, lean_object* v_ik_2472_, lean_object* v_w_2473_){
_start:
{
lean_object* v___x_2474_; 
v___x_2474_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___redArg(v_hi_2465_, v_pivot_2467_, v_as_2468_, v_i_2469_, v_k_2470_);
return v___x_2474_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11___boxed(lean_object* v_n_2475_, lean_object* v_lo_2476_, lean_object* v_hi_2477_, lean_object* v_hhi_2478_, lean_object* v_pivot_2479_, lean_object* v_as_2480_, lean_object* v_i_2481_, lean_object* v_k_2482_, lean_object* v_ilo_2483_, lean_object* v_ik_2484_, lean_object* v_w_2485_){
_start:
{
lean_object* v_res_2486_; 
v_res_2486_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4_spec__8_spec__11(v_n_2475_, v_lo_2476_, v_hi_2477_, v_hhi_2478_, v_pivot_2479_, v_as_2480_, v_i_2481_, v_k_2482_, v_ilo_2483_, v_ik_2484_, v_w_2485_);
lean_dec_ref(v_pivot_2479_);
lean_dec(v_hi_2477_);
lean_dec(v_lo_2476_);
lean_dec(v_n_2475_);
return v_res_2486_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2487_ = lean_unsigned_to_nat(32u);
v___x_2488_ = lean_mk_empty_array_with_capacity(v___x_2487_);
v___x_2489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2489_, 0, v___x_2488_);
return v___x_2489_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1(void){
_start:
{
size_t v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; 
v___x_2490_ = ((size_t)5ULL);
v___x_2491_ = lean_unsigned_to_nat(0u);
v___x_2492_ = lean_unsigned_to_nat(32u);
v___x_2493_ = lean_mk_empty_array_with_capacity(v___x_2492_);
v___x_2494_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__0);
v___x_2495_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2495_, 0, v___x_2494_);
lean_ctor_set(v___x_2495_, 1, v___x_2493_);
lean_ctor_set(v___x_2495_, 2, v___x_2491_);
lean_ctor_set(v___x_2495_, 3, v___x_2491_);
lean_ctor_set_usize(v___x_2495_, 4, v___x_2490_);
return v___x_2495_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(lean_object* v___y_2496_){
_start:
{
lean_object* v___x_2498_; lean_object* v_traceState_2499_; lean_object* v_traces_2500_; lean_object* v___x_2501_; lean_object* v_traceState_2502_; lean_object* v_env_2503_; lean_object* v_nextMacroScope_2504_; lean_object* v_ngen_2505_; lean_object* v_auxDeclNGen_2506_; lean_object* v_cache_2507_; lean_object* v_messages_2508_; lean_object* v_infoState_2509_; lean_object* v_snapshotTasks_2510_; lean_object* v___x_2512_; uint8_t v_isShared_2513_; uint8_t v_isSharedCheck_2529_; 
v___x_2498_ = lean_st_ref_get(v___y_2496_);
v_traceState_2499_ = lean_ctor_get(v___x_2498_, 4);
lean_inc_ref(v_traceState_2499_);
lean_dec(v___x_2498_);
v_traces_2500_ = lean_ctor_get(v_traceState_2499_, 0);
lean_inc_ref(v_traces_2500_);
lean_dec_ref(v_traceState_2499_);
v___x_2501_ = lean_st_ref_take(v___y_2496_);
v_traceState_2502_ = lean_ctor_get(v___x_2501_, 4);
v_env_2503_ = lean_ctor_get(v___x_2501_, 0);
v_nextMacroScope_2504_ = lean_ctor_get(v___x_2501_, 1);
v_ngen_2505_ = lean_ctor_get(v___x_2501_, 2);
v_auxDeclNGen_2506_ = lean_ctor_get(v___x_2501_, 3);
v_cache_2507_ = lean_ctor_get(v___x_2501_, 5);
v_messages_2508_ = lean_ctor_get(v___x_2501_, 6);
v_infoState_2509_ = lean_ctor_get(v___x_2501_, 7);
v_snapshotTasks_2510_ = lean_ctor_get(v___x_2501_, 8);
v_isSharedCheck_2529_ = !lean_is_exclusive(v___x_2501_);
if (v_isSharedCheck_2529_ == 0)
{
v___x_2512_ = v___x_2501_;
v_isShared_2513_ = v_isSharedCheck_2529_;
goto v_resetjp_2511_;
}
else
{
lean_inc(v_snapshotTasks_2510_);
lean_inc(v_infoState_2509_);
lean_inc(v_messages_2508_);
lean_inc(v_cache_2507_);
lean_inc(v_traceState_2502_);
lean_inc(v_auxDeclNGen_2506_);
lean_inc(v_ngen_2505_);
lean_inc(v_nextMacroScope_2504_);
lean_inc(v_env_2503_);
lean_dec(v___x_2501_);
v___x_2512_ = lean_box(0);
v_isShared_2513_ = v_isSharedCheck_2529_;
goto v_resetjp_2511_;
}
v_resetjp_2511_:
{
uint64_t v_tid_2514_; lean_object* v___x_2516_; uint8_t v_isShared_2517_; uint8_t v_isSharedCheck_2527_; 
v_tid_2514_ = lean_ctor_get_uint64(v_traceState_2502_, sizeof(void*)*1);
v_isSharedCheck_2527_ = !lean_is_exclusive(v_traceState_2502_);
if (v_isSharedCheck_2527_ == 0)
{
lean_object* v_unused_2528_; 
v_unused_2528_ = lean_ctor_get(v_traceState_2502_, 0);
lean_dec(v_unused_2528_);
v___x_2516_ = v_traceState_2502_;
v_isShared_2517_ = v_isSharedCheck_2527_;
goto v_resetjp_2515_;
}
else
{
lean_dec(v_traceState_2502_);
v___x_2516_ = lean_box(0);
v_isShared_2517_ = v_isSharedCheck_2527_;
goto v_resetjp_2515_;
}
v_resetjp_2515_:
{
lean_object* v___x_2518_; lean_object* v___x_2520_; 
v___x_2518_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___closed__1);
if (v_isShared_2517_ == 0)
{
lean_ctor_set(v___x_2516_, 0, v___x_2518_);
v___x_2520_ = v___x_2516_;
goto v_reusejp_2519_;
}
else
{
lean_object* v_reuseFailAlloc_2526_; 
v_reuseFailAlloc_2526_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2526_, 0, v___x_2518_);
lean_ctor_set_uint64(v_reuseFailAlloc_2526_, sizeof(void*)*1, v_tid_2514_);
v___x_2520_ = v_reuseFailAlloc_2526_;
goto v_reusejp_2519_;
}
v_reusejp_2519_:
{
lean_object* v___x_2522_; 
if (v_isShared_2513_ == 0)
{
lean_ctor_set(v___x_2512_, 4, v___x_2520_);
v___x_2522_ = v___x_2512_;
goto v_reusejp_2521_;
}
else
{
lean_object* v_reuseFailAlloc_2525_; 
v_reuseFailAlloc_2525_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2525_, 0, v_env_2503_);
lean_ctor_set(v_reuseFailAlloc_2525_, 1, v_nextMacroScope_2504_);
lean_ctor_set(v_reuseFailAlloc_2525_, 2, v_ngen_2505_);
lean_ctor_set(v_reuseFailAlloc_2525_, 3, v_auxDeclNGen_2506_);
lean_ctor_set(v_reuseFailAlloc_2525_, 4, v___x_2520_);
lean_ctor_set(v_reuseFailAlloc_2525_, 5, v_cache_2507_);
lean_ctor_set(v_reuseFailAlloc_2525_, 6, v_messages_2508_);
lean_ctor_set(v_reuseFailAlloc_2525_, 7, v_infoState_2509_);
lean_ctor_set(v_reuseFailAlloc_2525_, 8, v_snapshotTasks_2510_);
v___x_2522_ = v_reuseFailAlloc_2525_;
goto v_reusejp_2521_;
}
v_reusejp_2521_:
{
lean_object* v___x_2523_; lean_object* v___x_2524_; 
v___x_2523_ = lean_st_ref_set(v___y_2496_, v___x_2522_);
v___x_2524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2524_, 0, v_traces_2500_);
return v___x_2524_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg___boxed(lean_object* v___y_2530_, lean_object* v___y_2531_){
_start:
{
lean_object* v_res_2532_; 
v_res_2532_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v___y_2530_);
lean_dec(v___y_2530_);
return v_res_2532_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3(lean_object* v___y_2533_, lean_object* v___y_2534_){
_start:
{
lean_object* v___x_2536_; 
v___x_2536_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v___y_2534_);
return v___x_2536_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___boxed(lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_){
_start:
{
lean_object* v_res_2540_; 
v_res_2540_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3(v___y_2537_, v___y_2538_);
lean_dec(v___y_2538_);
lean_dec_ref(v___y_2537_);
return v_res_2540_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__0(lean_object* v_d_2541_, lean_object* v_a_2542_, lean_object* v_x_2543_){
_start:
{
lean_object* v___x_2544_; 
v___x_2544_ = lean_array_push(v_d_2541_, v_a_2542_);
return v___x_2544_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__1(lean_object* v___x_2545_, lean_object* v_x_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_){
_start:
{
lean_object* v___x_2550_; 
v___x_2550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2550_, 0, v___x_2545_);
return v___x_2550_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed(lean_object* v___x_2551_, lean_object* v_x_2552_, lean_object* v___y_2553_, lean_object* v___y_2554_, lean_object* v___y_2555_){
_start:
{
lean_object* v_res_2556_; 
v_res_2556_ = lp_aesop_Aesop_traceSimpTheorems___lam__1(v___x_2551_, v_x_2552_, v___y_2553_, v___y_2554_);
lean_dec(v___y_2554_);
lean_dec_ref(v___y_2553_);
lean_dec_ref(v_x_2552_);
return v_res_2556_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___lam__4(lean_object* v_d_2557_, lean_object* v_a_2558_, lean_object* v_x_2559_){
_start:
{
lean_object* v___x_2560_; 
v___x_2560_ = lean_array_push(v_d_2557_, v_a_2558_);
return v___x_2560_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(lean_object* v___x_2561_, lean_object* v_as_2562_, size_t v_sz_2563_, size_t v_i_2564_, lean_object* v_b_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
uint8_t v___x_2569_; 
v___x_2569_ = lean_usize_dec_lt(v_i_2564_, v_sz_2563_);
if (v___x_2569_ == 0)
{
lean_object* v___x_2570_; 
lean_dec(v___x_2561_);
v___x_2570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2570_, 0, v_b_2565_);
return v___x_2570_;
}
else
{
lean_object* v_a_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; 
v_a_2571_ = lean_array_uget_borrowed(v_as_2562_, v_i_2564_);
lean_inc(v_a_2571_);
v___x_2572_ = l_Lean_stringToMessageData(v_a_2571_);
lean_inc(v___x_2561_);
v___x_2573_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v___x_2561_, v___x_2572_, v___y_2566_, v___y_2567_);
if (lean_obj_tag(v___x_2573_) == 0)
{
lean_object* v___x_2574_; size_t v___x_2575_; size_t v___x_2576_; 
lean_dec_ref_known(v___x_2573_, 1);
v___x_2574_ = lean_box(0);
v___x_2575_ = ((size_t)1ULL);
v___x_2576_ = lean_usize_add(v_i_2564_, v___x_2575_);
v_i_2564_ = v___x_2576_;
v_b_2565_ = v___x_2574_;
goto _start;
}
else
{
lean_dec(v___x_2561_);
return v___x_2573_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2___boxed(lean_object* v___x_2578_, lean_object* v_as_2579_, lean_object* v_sz_2580_, lean_object* v_i_2581_, lean_object* v_b_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_){
_start:
{
size_t v_sz_boxed_2586_; size_t v_i_boxed_2587_; lean_object* v_res_2588_; 
v_sz_boxed_2586_ = lean_unbox_usize(v_sz_2580_);
lean_dec(v_sz_2580_);
v_i_boxed_2587_ = lean_unbox_usize(v_i_2581_);
lean_dec(v_i_2581_);
v_res_2588_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v___x_2578_, v_as_2579_, v_sz_boxed_2586_, v_i_boxed_2587_, v_b_2582_, v___y_2583_, v___y_2584_);
lean_dec(v___y_2584_);
lean_dec_ref(v___y_2583_);
lean_dec_ref(v_as_2579_);
return v_res_2588_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(uint8_t v_a_2589_, size_t v_sz_2590_, size_t v_i_2591_, lean_object* v_bs_2592_){
_start:
{
uint8_t v___x_2593_; 
v___x_2593_ = lean_usize_dec_lt(v_i_2591_, v_sz_2590_);
if (v___x_2593_ == 0)
{
return v_bs_2592_;
}
else
{
lean_object* v_v_2594_; lean_object* v___x_2595_; lean_object* v_bs_x27_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; size_t v___x_2599_; size_t v___x_2600_; lean_object* v___x_2601_; 
v_v_2594_ = lean_array_uget(v_bs_2592_, v_i_2591_);
v___x_2595_ = lean_unsigned_to_nat(0u);
v_bs_x27_2596_ = lean_array_uset(v_bs_2592_, v_i_2591_, v___x_2595_);
v___x_2597_ = l_Lean_Meta_Origin_key(v_v_2594_);
lean_dec(v_v_2594_);
v___x_2598_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_2597_, v_a_2589_);
v___x_2599_ = ((size_t)1ULL);
v___x_2600_ = lean_usize_add(v_i_2591_, v___x_2599_);
v___x_2601_ = lean_array_uset(v_bs_x27_2596_, v_i_2591_, v___x_2598_);
v_i_2591_ = v___x_2600_;
v_bs_2592_ = v___x_2601_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6___boxed(lean_object* v_a_2603_, lean_object* v_sz_2604_, lean_object* v_i_2605_, lean_object* v_bs_2606_){
_start:
{
uint8_t v_a_24214__boxed_2607_; size_t v_sz_boxed_2608_; size_t v_i_boxed_2609_; lean_object* v_res_2610_; 
v_a_24214__boxed_2607_ = lean_unbox(v_a_2603_);
v_sz_boxed_2608_ = lean_unbox_usize(v_sz_2604_);
lean_dec(v_sz_2604_);
v_i_boxed_2609_ = lean_unbox_usize(v_i_2605_);
lean_dec(v_i_2605_);
v_res_2610_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(v_a_24214__boxed_2607_, v_sz_boxed_2608_, v_i_boxed_2609_, v_bs_2606_);
return v_res_2610_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1(uint8_t v_a_2611_, size_t v_sz_2612_, size_t v_i_2613_, lean_object* v_bs_2614_){
_start:
{
uint8_t v___x_2615_; 
v___x_2615_ = lean_usize_dec_lt(v_i_2613_, v_sz_2612_);
if (v___x_2615_ == 0)
{
return v_bs_2614_;
}
else
{
lean_object* v_v_2616_; lean_object* v___x_2617_; lean_object* v_bs_x27_2618_; lean_object* v___x_2619_; size_t v___x_2620_; size_t v___x_2621_; lean_object* v___x_2622_; 
v_v_2616_ = lean_array_uget(v_bs_2614_, v_i_2613_);
v___x_2617_ = lean_unsigned_to_nat(0u);
v_bs_x27_2618_ = lean_array_uset(v_bs_2614_, v_i_2613_, v___x_2617_);
v___x_2619_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_v_2616_, v_a_2611_);
v___x_2620_ = ((size_t)1ULL);
v___x_2621_ = lean_usize_add(v_i_2613_, v___x_2620_);
v___x_2622_ = lean_array_uset(v_bs_x27_2618_, v_i_2613_, v___x_2619_);
v_i_2613_ = v___x_2621_;
v_bs_2614_ = v___x_2622_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1___boxed(lean_object* v_a_2624_, lean_object* v_sz_2625_, lean_object* v_i_2626_, lean_object* v_bs_2627_){
_start:
{
uint8_t v_a_24233__boxed_2628_; size_t v_sz_boxed_2629_; size_t v_i_boxed_2630_; lean_object* v_res_2631_; 
v_a_24233__boxed_2628_ = lean_unbox(v_a_2624_);
v_sz_boxed_2629_ = lean_unbox_usize(v_sz_2625_);
lean_dec(v_sz_2625_);
v_i_boxed_2630_ = lean_unbox_usize(v_i_2626_);
lean_dec(v_i_2626_);
v_res_2631_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1(v_a_24233__boxed_2628_, v_sz_boxed_2629_, v_i_boxed_2630_, v_bs_2627_);
return v_res_2631_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5(size_t v_sz_2632_, size_t v_i_2633_, lean_object* v_bs_2634_){
_start:
{
uint8_t v___x_2635_; 
v___x_2635_ = lean_usize_dec_lt(v_i_2633_, v_sz_2632_);
if (v___x_2635_ == 0)
{
return v_bs_2634_;
}
else
{
lean_object* v_v_2636_; lean_object* v_msg_2637_; lean_object* v___x_2638_; lean_object* v_bs_x27_2639_; size_t v___x_2640_; size_t v___x_2641_; lean_object* v___x_2642_; 
v_v_2636_ = lean_array_uget_borrowed(v_bs_2634_, v_i_2633_);
v_msg_2637_ = lean_ctor_get(v_v_2636_, 1);
lean_inc_ref(v_msg_2637_);
v___x_2638_ = lean_unsigned_to_nat(0u);
v_bs_x27_2639_ = lean_array_uset(v_bs_2634_, v_i_2633_, v___x_2638_);
v___x_2640_ = ((size_t)1ULL);
v___x_2641_ = lean_usize_add(v_i_2633_, v___x_2640_);
v___x_2642_ = lean_array_uset(v_bs_x27_2639_, v_i_2633_, v_msg_2637_);
v_i_2633_ = v___x_2641_;
v_bs_2634_ = v___x_2642_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5___boxed(lean_object* v_sz_2644_, lean_object* v_i_2645_, lean_object* v_bs_2646_){
_start:
{
size_t v_sz_boxed_2647_; size_t v_i_boxed_2648_; lean_object* v_res_2649_; 
v_sz_boxed_2647_ = lean_unbox_usize(v_sz_2644_);
lean_dec(v_sz_2644_);
v_i_boxed_2648_ = lean_unbox_usize(v_i_2645_);
lean_dec(v_i_2645_);
v_res_2649_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5(v_sz_boxed_2647_, v_i_boxed_2648_, v_bs_2646_);
return v_res_2649_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4(lean_object* v_oldTraces_2650_, lean_object* v_data_2651_, lean_object* v_ref_2652_, lean_object* v_msg_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_){
_start:
{
lean_object* v_fileName_2657_; lean_object* v_fileMap_2658_; lean_object* v_options_2659_; lean_object* v_currRecDepth_2660_; lean_object* v_maxRecDepth_2661_; lean_object* v_ref_2662_; lean_object* v_currNamespace_2663_; lean_object* v_openDecls_2664_; lean_object* v_initHeartbeats_2665_; lean_object* v_maxHeartbeats_2666_; lean_object* v_quotContext_2667_; lean_object* v_currMacroScope_2668_; uint8_t v_diag_2669_; lean_object* v_cancelTk_x3f_2670_; uint8_t v_suppressElabErrors_2671_; lean_object* v_inheritedTraceOptions_2672_; lean_object* v___x_2673_; lean_object* v_traceState_2674_; lean_object* v_traces_2675_; lean_object* v_ref_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; size_t v_sz_2679_; size_t v___x_2680_; lean_object* v___x_2681_; lean_object* v_msg_2682_; lean_object* v___x_2683_; lean_object* v_a_2684_; lean_object* v___x_2686_; uint8_t v_isShared_2687_; uint8_t v_isSharedCheck_2721_; 
v_fileName_2657_ = lean_ctor_get(v___y_2654_, 0);
v_fileMap_2658_ = lean_ctor_get(v___y_2654_, 1);
v_options_2659_ = lean_ctor_get(v___y_2654_, 2);
v_currRecDepth_2660_ = lean_ctor_get(v___y_2654_, 3);
v_maxRecDepth_2661_ = lean_ctor_get(v___y_2654_, 4);
v_ref_2662_ = lean_ctor_get(v___y_2654_, 5);
v_currNamespace_2663_ = lean_ctor_get(v___y_2654_, 6);
v_openDecls_2664_ = lean_ctor_get(v___y_2654_, 7);
v_initHeartbeats_2665_ = lean_ctor_get(v___y_2654_, 8);
v_maxHeartbeats_2666_ = lean_ctor_get(v___y_2654_, 9);
v_quotContext_2667_ = lean_ctor_get(v___y_2654_, 10);
v_currMacroScope_2668_ = lean_ctor_get(v___y_2654_, 11);
v_diag_2669_ = lean_ctor_get_uint8(v___y_2654_, sizeof(void*)*14);
v_cancelTk_x3f_2670_ = lean_ctor_get(v___y_2654_, 12);
v_suppressElabErrors_2671_ = lean_ctor_get_uint8(v___y_2654_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2672_ = lean_ctor_get(v___y_2654_, 13);
v___x_2673_ = lean_st_ref_get(v___y_2655_);
v_traceState_2674_ = lean_ctor_get(v___x_2673_, 4);
lean_inc_ref(v_traceState_2674_);
lean_dec(v___x_2673_);
v_traces_2675_ = lean_ctor_get(v_traceState_2674_, 0);
lean_inc_ref(v_traces_2675_);
lean_dec_ref(v_traceState_2674_);
v_ref_2676_ = l_Lean_replaceRef(v_ref_2652_, v_ref_2662_);
lean_inc_ref(v_inheritedTraceOptions_2672_);
lean_inc(v_cancelTk_x3f_2670_);
lean_inc(v_currMacroScope_2668_);
lean_inc(v_quotContext_2667_);
lean_inc(v_maxHeartbeats_2666_);
lean_inc(v_initHeartbeats_2665_);
lean_inc(v_openDecls_2664_);
lean_inc(v_currNamespace_2663_);
lean_inc(v_maxRecDepth_2661_);
lean_inc(v_currRecDepth_2660_);
lean_inc_ref(v_options_2659_);
lean_inc_ref(v_fileMap_2658_);
lean_inc_ref(v_fileName_2657_);
v___x_2677_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2677_, 0, v_fileName_2657_);
lean_ctor_set(v___x_2677_, 1, v_fileMap_2658_);
lean_ctor_set(v___x_2677_, 2, v_options_2659_);
lean_ctor_set(v___x_2677_, 3, v_currRecDepth_2660_);
lean_ctor_set(v___x_2677_, 4, v_maxRecDepth_2661_);
lean_ctor_set(v___x_2677_, 5, v_ref_2676_);
lean_ctor_set(v___x_2677_, 6, v_currNamespace_2663_);
lean_ctor_set(v___x_2677_, 7, v_openDecls_2664_);
lean_ctor_set(v___x_2677_, 8, v_initHeartbeats_2665_);
lean_ctor_set(v___x_2677_, 9, v_maxHeartbeats_2666_);
lean_ctor_set(v___x_2677_, 10, v_quotContext_2667_);
lean_ctor_set(v___x_2677_, 11, v_currMacroScope_2668_);
lean_ctor_set(v___x_2677_, 12, v_cancelTk_x3f_2670_);
lean_ctor_set(v___x_2677_, 13, v_inheritedTraceOptions_2672_);
lean_ctor_set_uint8(v___x_2677_, sizeof(void*)*14, v_diag_2669_);
lean_ctor_set_uint8(v___x_2677_, sizeof(void*)*14 + 1, v_suppressElabErrors_2671_);
v___x_2678_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2675_);
lean_dec_ref(v_traces_2675_);
v_sz_2679_ = lean_array_size(v___x_2678_);
v___x_2680_ = ((size_t)0ULL);
v___x_2681_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4_spec__5(v_sz_2679_, v___x_2680_, v___x_2678_);
v_msg_2682_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2682_, 0, v_data_2651_);
lean_ctor_set(v_msg_2682_, 1, v_msg_2653_);
lean_ctor_set(v_msg_2682_, 2, v___x_2681_);
v___x_2683_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5_spec__10(v_msg_2682_, v___x_2677_, v___y_2655_);
lean_dec_ref_known(v___x_2677_, 14);
v_a_2684_ = lean_ctor_get(v___x_2683_, 0);
v_isSharedCheck_2721_ = !lean_is_exclusive(v___x_2683_);
if (v_isSharedCheck_2721_ == 0)
{
v___x_2686_ = v___x_2683_;
v_isShared_2687_ = v_isSharedCheck_2721_;
goto v_resetjp_2685_;
}
else
{
lean_inc(v_a_2684_);
lean_dec(v___x_2683_);
v___x_2686_ = lean_box(0);
v_isShared_2687_ = v_isSharedCheck_2721_;
goto v_resetjp_2685_;
}
v_resetjp_2685_:
{
lean_object* v___x_2688_; lean_object* v_traceState_2689_; lean_object* v_env_2690_; lean_object* v_nextMacroScope_2691_; lean_object* v_ngen_2692_; lean_object* v_auxDeclNGen_2693_; lean_object* v_cache_2694_; lean_object* v_messages_2695_; lean_object* v_infoState_2696_; lean_object* v_snapshotTasks_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2720_; 
v___x_2688_ = lean_st_ref_take(v___y_2655_);
v_traceState_2689_ = lean_ctor_get(v___x_2688_, 4);
v_env_2690_ = lean_ctor_get(v___x_2688_, 0);
v_nextMacroScope_2691_ = lean_ctor_get(v___x_2688_, 1);
v_ngen_2692_ = lean_ctor_get(v___x_2688_, 2);
v_auxDeclNGen_2693_ = lean_ctor_get(v___x_2688_, 3);
v_cache_2694_ = lean_ctor_get(v___x_2688_, 5);
v_messages_2695_ = lean_ctor_get(v___x_2688_, 6);
v_infoState_2696_ = lean_ctor_get(v___x_2688_, 7);
v_snapshotTasks_2697_ = lean_ctor_get(v___x_2688_, 8);
v_isSharedCheck_2720_ = !lean_is_exclusive(v___x_2688_);
if (v_isSharedCheck_2720_ == 0)
{
v___x_2699_ = v___x_2688_;
v_isShared_2700_ = v_isSharedCheck_2720_;
goto v_resetjp_2698_;
}
else
{
lean_inc(v_snapshotTasks_2697_);
lean_inc(v_infoState_2696_);
lean_inc(v_messages_2695_);
lean_inc(v_cache_2694_);
lean_inc(v_traceState_2689_);
lean_inc(v_auxDeclNGen_2693_);
lean_inc(v_ngen_2692_);
lean_inc(v_nextMacroScope_2691_);
lean_inc(v_env_2690_);
lean_dec(v___x_2688_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2720_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
uint64_t v_tid_2701_; lean_object* v___x_2703_; uint8_t v_isShared_2704_; uint8_t v_isSharedCheck_2718_; 
v_tid_2701_ = lean_ctor_get_uint64(v_traceState_2689_, sizeof(void*)*1);
v_isSharedCheck_2718_ = !lean_is_exclusive(v_traceState_2689_);
if (v_isSharedCheck_2718_ == 0)
{
lean_object* v_unused_2719_; 
v_unused_2719_ = lean_ctor_get(v_traceState_2689_, 0);
lean_dec(v_unused_2719_);
v___x_2703_ = v_traceState_2689_;
v_isShared_2704_ = v_isSharedCheck_2718_;
goto v_resetjp_2702_;
}
else
{
lean_dec(v_traceState_2689_);
v___x_2703_ = lean_box(0);
v_isShared_2704_ = v_isSharedCheck_2718_;
goto v_resetjp_2702_;
}
v_resetjp_2702_:
{
lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2708_; 
v___x_2705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2705_, 0, v_ref_2652_);
lean_ctor_set(v___x_2705_, 1, v_a_2684_);
v___x_2706_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2650_, v___x_2705_);
if (v_isShared_2704_ == 0)
{
lean_ctor_set(v___x_2703_, 0, v___x_2706_);
v___x_2708_ = v___x_2703_;
goto v_reusejp_2707_;
}
else
{
lean_object* v_reuseFailAlloc_2717_; 
v_reuseFailAlloc_2717_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2717_, 0, v___x_2706_);
lean_ctor_set_uint64(v_reuseFailAlloc_2717_, sizeof(void*)*1, v_tid_2701_);
v___x_2708_ = v_reuseFailAlloc_2717_;
goto v_reusejp_2707_;
}
v_reusejp_2707_:
{
lean_object* v___x_2710_; 
if (v_isShared_2700_ == 0)
{
lean_ctor_set(v___x_2699_, 4, v___x_2708_);
v___x_2710_ = v___x_2699_;
goto v_reusejp_2709_;
}
else
{
lean_object* v_reuseFailAlloc_2716_; 
v_reuseFailAlloc_2716_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2716_, 0, v_env_2690_);
lean_ctor_set(v_reuseFailAlloc_2716_, 1, v_nextMacroScope_2691_);
lean_ctor_set(v_reuseFailAlloc_2716_, 2, v_ngen_2692_);
lean_ctor_set(v_reuseFailAlloc_2716_, 3, v_auxDeclNGen_2693_);
lean_ctor_set(v_reuseFailAlloc_2716_, 4, v___x_2708_);
lean_ctor_set(v_reuseFailAlloc_2716_, 5, v_cache_2694_);
lean_ctor_set(v_reuseFailAlloc_2716_, 6, v_messages_2695_);
lean_ctor_set(v_reuseFailAlloc_2716_, 7, v_infoState_2696_);
lean_ctor_set(v_reuseFailAlloc_2716_, 8, v_snapshotTasks_2697_);
v___x_2710_ = v_reuseFailAlloc_2716_;
goto v_reusejp_2709_;
}
v_reusejp_2709_:
{
lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2714_; 
v___x_2711_ = lean_st_ref_set(v___y_2655_, v___x_2710_);
v___x_2712_ = lean_box(0);
if (v_isShared_2687_ == 0)
{
lean_ctor_set(v___x_2686_, 0, v___x_2712_);
v___x_2714_ = v___x_2686_;
goto v_reusejp_2713_;
}
else
{
lean_object* v_reuseFailAlloc_2715_; 
v_reuseFailAlloc_2715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2715_, 0, v___x_2712_);
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
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4___boxed(lean_object* v_oldTraces_2722_, lean_object* v_data_2723_, lean_object* v_ref_2724_, lean_object* v_msg_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_){
_start:
{
lean_object* v_res_2729_; 
v_res_2729_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4(v_oldTraces_2722_, v_data_2723_, v_ref_2724_, v_msg_2725_, v___y_2726_, v___y_2727_);
lean_dec(v___y_2727_);
lean_dec_ref(v___y_2726_);
return v_res_2729_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6(lean_object* v_e_2730_){
_start:
{
if (lean_obj_tag(v_e_2730_) == 0)
{
uint8_t v___x_2731_; 
v___x_2731_ = 2;
return v___x_2731_;
}
else
{
uint8_t v___x_2732_; 
v___x_2732_ = 0;
return v___x_2732_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6___boxed(lean_object* v_e_2733_){
_start:
{
uint8_t v_res_2734_; lean_object* v_r_2735_; 
v_res_2734_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6(v_e_2733_);
lean_dec_ref(v_e_2733_);
v_r_2735_ = lean_box(v_res_2734_);
return v_r_2735_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7(lean_object* v_opts_2736_, lean_object* v_opt_2737_){
_start:
{
lean_object* v_name_2738_; lean_object* v_defValue_2739_; lean_object* v_map_2740_; lean_object* v___x_2741_; 
v_name_2738_ = lean_ctor_get(v_opt_2737_, 0);
v_defValue_2739_ = lean_ctor_get(v_opt_2737_, 1);
v_map_2740_ = lean_ctor_get(v_opts_2736_, 0);
v___x_2741_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2740_, v_name_2738_);
if (lean_obj_tag(v___x_2741_) == 0)
{
lean_inc(v_defValue_2739_);
return v_defValue_2739_;
}
else
{
lean_object* v_val_2742_; 
v_val_2742_ = lean_ctor_get(v___x_2741_, 0);
lean_inc(v_val_2742_);
lean_dec_ref_known(v___x_2741_, 1);
if (lean_obj_tag(v_val_2742_) == 3)
{
lean_object* v_v_2743_; 
v_v_2743_ = lean_ctor_get(v_val_2742_, 0);
lean_inc(v_v_2743_);
lean_dec_ref_known(v_val_2742_, 1);
return v_v_2743_;
}
else
{
lean_dec(v_val_2742_);
lean_inc(v_defValue_2739_);
return v_defValue_2739_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7___boxed(lean_object* v_opts_2744_, lean_object* v_opt_2745_){
_start:
{
lean_object* v_res_2746_; 
v_res_2746_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7(v_opts_2744_, v_opt_2745_);
lean_dec_ref(v_opt_2745_);
lean_dec_ref(v_opts_2744_);
return v_res_2746_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(lean_object* v_x_2747_){
_start:
{
if (lean_obj_tag(v_x_2747_) == 0)
{
lean_object* v_a_2749_; lean_object* v___x_2751_; uint8_t v_isShared_2752_; uint8_t v_isSharedCheck_2756_; 
v_a_2749_ = lean_ctor_get(v_x_2747_, 0);
v_isSharedCheck_2756_ = !lean_is_exclusive(v_x_2747_);
if (v_isSharedCheck_2756_ == 0)
{
v___x_2751_ = v_x_2747_;
v_isShared_2752_ = v_isSharedCheck_2756_;
goto v_resetjp_2750_;
}
else
{
lean_inc(v_a_2749_);
lean_dec(v_x_2747_);
v___x_2751_ = lean_box(0);
v_isShared_2752_ = v_isSharedCheck_2756_;
goto v_resetjp_2750_;
}
v_resetjp_2750_:
{
lean_object* v___x_2754_; 
if (v_isShared_2752_ == 0)
{
lean_ctor_set_tag(v___x_2751_, 1);
v___x_2754_ = v___x_2751_;
goto v_reusejp_2753_;
}
else
{
lean_object* v_reuseFailAlloc_2755_; 
v_reuseFailAlloc_2755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2755_, 0, v_a_2749_);
v___x_2754_ = v_reuseFailAlloc_2755_;
goto v_reusejp_2753_;
}
v_reusejp_2753_:
{
return v___x_2754_;
}
}
}
else
{
lean_object* v_a_2757_; lean_object* v___x_2759_; uint8_t v_isShared_2760_; uint8_t v_isSharedCheck_2764_; 
v_a_2757_ = lean_ctor_get(v_x_2747_, 0);
v_isSharedCheck_2764_ = !lean_is_exclusive(v_x_2747_);
if (v_isSharedCheck_2764_ == 0)
{
v___x_2759_ = v_x_2747_;
v_isShared_2760_ = v_isSharedCheck_2764_;
goto v_resetjp_2758_;
}
else
{
lean_inc(v_a_2757_);
lean_dec(v_x_2747_);
v___x_2759_ = lean_box(0);
v_isShared_2760_ = v_isSharedCheck_2764_;
goto v_resetjp_2758_;
}
v_resetjp_2758_:
{
lean_object* v___x_2762_; 
if (v_isShared_2760_ == 0)
{
lean_ctor_set_tag(v___x_2759_, 0);
v___x_2762_ = v___x_2759_;
goto v_reusejp_2761_;
}
else
{
lean_object* v_reuseFailAlloc_2763_; 
v_reuseFailAlloc_2763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2763_, 0, v_a_2757_);
v___x_2762_ = v_reuseFailAlloc_2763_;
goto v_reusejp_2761_;
}
v_reusejp_2761_:
{
return v___x_2762_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg___boxed(lean_object* v_x_2765_, lean_object* v___y_2766_){
_start:
{
lean_object* v_res_2767_; 
v_res_2767_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(v_x_2765_);
return v_res_2767_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1(void){
_start:
{
lean_object* v___x_2769_; lean_object* v___x_2770_; 
v___x_2769_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__0));
v___x_2770_ = l_Lean_stringToMessageData(v___x_2769_);
return v___x_2770_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2(void){
_start:
{
lean_object* v___x_2771_; double v___x_2772_; 
v___x_2771_ = lean_unsigned_to_nat(1000u);
v___x_2772_ = lean_float_of_nat(v___x_2771_);
return v___x_2772_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(lean_object* v_cls_2773_, uint8_t v_collapsed_2774_, lean_object* v_tag_2775_, lean_object* v_opts_2776_, uint8_t v_clsEnabled_2777_, lean_object* v_oldTraces_2778_, lean_object* v_msg_2779_, lean_object* v_resStartStop_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_){
_start:
{
lean_object* v_fst_2784_; lean_object* v_snd_2785_; lean_object* v___y_2787_; lean_object* v___y_2788_; lean_object* v_data_2789_; lean_object* v_fst_2792_; lean_object* v_snd_2793_; lean_object* v___x_2794_; uint8_t v___x_2795_; lean_object* v___y_2797_; lean_object* v_a_2798_; uint8_t v___y_2813_; double v___y_2844_; 
v_fst_2784_ = lean_ctor_get(v_resStartStop_2780_, 0);
lean_inc(v_fst_2784_);
v_snd_2785_ = lean_ctor_get(v_resStartStop_2780_, 1);
lean_inc(v_snd_2785_);
lean_dec_ref(v_resStartStop_2780_);
v_fst_2792_ = lean_ctor_get(v_snd_2785_, 0);
lean_inc(v_fst_2792_);
v_snd_2793_ = lean_ctor_get(v_snd_2785_, 1);
lean_inc(v_snd_2793_);
lean_dec(v_snd_2785_);
v___x_2794_ = l_Lean_trace_profiler;
v___x_2795_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_opts_2776_, v___x_2794_);
if (v___x_2795_ == 0)
{
v___y_2813_ = v___x_2795_;
goto v___jp_2812_;
}
else
{
lean_object* v___x_2849_; uint8_t v___x_2850_; 
v___x_2849_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2850_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_opts_2776_, v___x_2849_);
if (v___x_2850_ == 0)
{
lean_object* v___x_2851_; lean_object* v___x_2852_; double v___x_2853_; double v___x_2854_; double v___x_2855_; 
v___x_2851_ = l_Lean_trace_profiler_threshold;
v___x_2852_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7(v_opts_2776_, v___x_2851_);
v___x_2853_ = lean_float_of_nat(v___x_2852_);
v___x_2854_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__2);
v___x_2855_ = lean_float_div(v___x_2853_, v___x_2854_);
v___y_2844_ = v___x_2855_;
goto v___jp_2843_;
}
else
{
lean_object* v___x_2856_; lean_object* v___x_2857_; double v___x_2858_; 
v___x_2856_ = l_Lean_trace_profiler_threshold;
v___x_2857_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__7(v_opts_2776_, v___x_2856_);
v___x_2858_ = lean_float_of_nat(v___x_2857_);
v___y_2844_ = v___x_2858_;
goto v___jp_2843_;
}
}
v___jp_2786_:
{
lean_object* v___x_2790_; 
lean_inc(v___y_2787_);
v___x_2790_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__4(v_oldTraces_2778_, v_data_2789_, v___y_2787_, v___y_2788_, v___y_2781_, v___y_2782_);
if (lean_obj_tag(v___x_2790_) == 0)
{
lean_object* v___x_2791_; 
lean_dec_ref_known(v___x_2790_, 1);
v___x_2791_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(v_fst_2784_);
return v___x_2791_;
}
else
{
lean_dec(v_fst_2784_);
return v___x_2790_;
}
}
v___jp_2796_:
{
uint8_t v_result_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; double v___x_2802_; lean_object* v_data_2803_; 
v_result_2799_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__6(v_fst_2784_);
v___x_2800_ = lean_box(v_result_2799_);
v___x_2801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2801_, 0, v___x_2800_);
v___x_2802_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5___closed__0);
lean_inc_ref(v_tag_2775_);
lean_inc_ref(v___x_2801_);
lean_inc(v_cls_2773_);
v_data_2803_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2803_, 0, v_cls_2773_);
lean_ctor_set(v_data_2803_, 1, v___x_2801_);
lean_ctor_set(v_data_2803_, 2, v_tag_2775_);
lean_ctor_set_float(v_data_2803_, sizeof(void*)*3, v___x_2802_);
lean_ctor_set_float(v_data_2803_, sizeof(void*)*3 + 8, v___x_2802_);
lean_ctor_set_uint8(v_data_2803_, sizeof(void*)*3 + 16, v_collapsed_2774_);
if (v___x_2795_ == 0)
{
lean_dec_ref_known(v___x_2801_, 1);
lean_dec(v_snd_2793_);
lean_dec(v_fst_2792_);
lean_dec_ref(v_tag_2775_);
lean_dec(v_cls_2773_);
v___y_2787_ = v___y_2797_;
v___y_2788_ = v_a_2798_;
v_data_2789_ = v_data_2803_;
goto v___jp_2786_;
}
else
{
lean_object* v_data_2804_; double v___x_2805_; double v___x_2806_; 
lean_dec_ref_known(v_data_2803_, 3);
v_data_2804_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2804_, 0, v_cls_2773_);
lean_ctor_set(v_data_2804_, 1, v___x_2801_);
lean_ctor_set(v_data_2804_, 2, v_tag_2775_);
v___x_2805_ = lean_unbox_float(v_fst_2792_);
lean_dec(v_fst_2792_);
lean_ctor_set_float(v_data_2804_, sizeof(void*)*3, v___x_2805_);
v___x_2806_ = lean_unbox_float(v_snd_2793_);
lean_dec(v_snd_2793_);
lean_ctor_set_float(v_data_2804_, sizeof(void*)*3 + 8, v___x_2806_);
lean_ctor_set_uint8(v_data_2804_, sizeof(void*)*3 + 16, v_collapsed_2774_);
v___y_2787_ = v___y_2797_;
v___y_2788_ = v_a_2798_;
v_data_2789_ = v_data_2804_;
goto v___jp_2786_;
}
}
v___jp_2807_:
{
lean_object* v_ref_2808_; lean_object* v___x_2809_; 
v_ref_2808_ = lean_ctor_get(v___y_2781_, 5);
lean_inc(v___y_2782_);
lean_inc_ref(v___y_2781_);
lean_inc(v_fst_2784_);
v___x_2809_ = lean_apply_4(v_msg_2779_, v_fst_2784_, v___y_2781_, v___y_2782_, lean_box(0));
if (lean_obj_tag(v___x_2809_) == 0)
{
lean_object* v_a_2810_; 
v_a_2810_ = lean_ctor_get(v___x_2809_, 0);
lean_inc(v_a_2810_);
lean_dec_ref_known(v___x_2809_, 1);
v___y_2797_ = v_ref_2808_;
v_a_2798_ = v_a_2810_;
goto v___jp_2796_;
}
else
{
lean_object* v___x_2811_; 
lean_dec_ref_known(v___x_2809_, 1);
v___x_2811_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___closed__1);
v___y_2797_ = v_ref_2808_;
v_a_2798_ = v___x_2811_;
goto v___jp_2796_;
}
}
v___jp_2812_:
{
if (v_clsEnabled_2777_ == 0)
{
if (v___y_2813_ == 0)
{
lean_object* v___x_2814_; lean_object* v_traceState_2815_; lean_object* v_env_2816_; lean_object* v_nextMacroScope_2817_; lean_object* v_ngen_2818_; lean_object* v_auxDeclNGen_2819_; lean_object* v_cache_2820_; lean_object* v_messages_2821_; lean_object* v_infoState_2822_; lean_object* v_snapshotTasks_2823_; lean_object* v___x_2825_; uint8_t v_isShared_2826_; uint8_t v_isSharedCheck_2842_; 
lean_dec(v_snd_2793_);
lean_dec(v_fst_2792_);
lean_dec_ref(v_msg_2779_);
lean_dec_ref(v_tag_2775_);
lean_dec(v_cls_2773_);
v___x_2814_ = lean_st_ref_take(v___y_2782_);
v_traceState_2815_ = lean_ctor_get(v___x_2814_, 4);
v_env_2816_ = lean_ctor_get(v___x_2814_, 0);
v_nextMacroScope_2817_ = lean_ctor_get(v___x_2814_, 1);
v_ngen_2818_ = lean_ctor_get(v___x_2814_, 2);
v_auxDeclNGen_2819_ = lean_ctor_get(v___x_2814_, 3);
v_cache_2820_ = lean_ctor_get(v___x_2814_, 5);
v_messages_2821_ = lean_ctor_get(v___x_2814_, 6);
v_infoState_2822_ = lean_ctor_get(v___x_2814_, 7);
v_snapshotTasks_2823_ = lean_ctor_get(v___x_2814_, 8);
v_isSharedCheck_2842_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2842_ == 0)
{
v___x_2825_ = v___x_2814_;
v_isShared_2826_ = v_isSharedCheck_2842_;
goto v_resetjp_2824_;
}
else
{
lean_inc(v_snapshotTasks_2823_);
lean_inc(v_infoState_2822_);
lean_inc(v_messages_2821_);
lean_inc(v_cache_2820_);
lean_inc(v_traceState_2815_);
lean_inc(v_auxDeclNGen_2819_);
lean_inc(v_ngen_2818_);
lean_inc(v_nextMacroScope_2817_);
lean_inc(v_env_2816_);
lean_dec(v___x_2814_);
v___x_2825_ = lean_box(0);
v_isShared_2826_ = v_isSharedCheck_2842_;
goto v_resetjp_2824_;
}
v_resetjp_2824_:
{
uint64_t v_tid_2827_; lean_object* v_traces_2828_; lean_object* v___x_2830_; uint8_t v_isShared_2831_; uint8_t v_isSharedCheck_2841_; 
v_tid_2827_ = lean_ctor_get_uint64(v_traceState_2815_, sizeof(void*)*1);
v_traces_2828_ = lean_ctor_get(v_traceState_2815_, 0);
v_isSharedCheck_2841_ = !lean_is_exclusive(v_traceState_2815_);
if (v_isSharedCheck_2841_ == 0)
{
v___x_2830_ = v_traceState_2815_;
v_isShared_2831_ = v_isSharedCheck_2841_;
goto v_resetjp_2829_;
}
else
{
lean_inc(v_traces_2828_);
lean_dec(v_traceState_2815_);
v___x_2830_ = lean_box(0);
v_isShared_2831_ = v_isSharedCheck_2841_;
goto v_resetjp_2829_;
}
v_resetjp_2829_:
{
lean_object* v___x_2832_; lean_object* v___x_2834_; 
v___x_2832_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2778_, v_traces_2828_);
lean_dec_ref(v_traces_2828_);
if (v_isShared_2831_ == 0)
{
lean_ctor_set(v___x_2830_, 0, v___x_2832_);
v___x_2834_ = v___x_2830_;
goto v_reusejp_2833_;
}
else
{
lean_object* v_reuseFailAlloc_2840_; 
v_reuseFailAlloc_2840_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2840_, 0, v___x_2832_);
lean_ctor_set_uint64(v_reuseFailAlloc_2840_, sizeof(void*)*1, v_tid_2827_);
v___x_2834_ = v_reuseFailAlloc_2840_;
goto v_reusejp_2833_;
}
v_reusejp_2833_:
{
lean_object* v___x_2836_; 
if (v_isShared_2826_ == 0)
{
lean_ctor_set(v___x_2825_, 4, v___x_2834_);
v___x_2836_ = v___x_2825_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v_env_2816_);
lean_ctor_set(v_reuseFailAlloc_2839_, 1, v_nextMacroScope_2817_);
lean_ctor_set(v_reuseFailAlloc_2839_, 2, v_ngen_2818_);
lean_ctor_set(v_reuseFailAlloc_2839_, 3, v_auxDeclNGen_2819_);
lean_ctor_set(v_reuseFailAlloc_2839_, 4, v___x_2834_);
lean_ctor_set(v_reuseFailAlloc_2839_, 5, v_cache_2820_);
lean_ctor_set(v_reuseFailAlloc_2839_, 6, v_messages_2821_);
lean_ctor_set(v_reuseFailAlloc_2839_, 7, v_infoState_2822_);
lean_ctor_set(v_reuseFailAlloc_2839_, 8, v_snapshotTasks_2823_);
v___x_2836_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
lean_object* v___x_2837_; lean_object* v___x_2838_; 
v___x_2837_ = lean_st_ref_set(v___y_2782_, v___x_2836_);
v___x_2838_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(v_fst_2784_);
return v___x_2838_;
}
}
}
}
}
else
{
goto v___jp_2807_;
}
}
else
{
goto v___jp_2807_;
}
}
v___jp_2843_:
{
double v___x_2845_; double v___x_2846_; double v___x_2847_; uint8_t v___x_2848_; 
v___x_2845_ = lean_unbox_float(v_snd_2793_);
v___x_2846_ = lean_unbox_float(v_fst_2792_);
v___x_2847_ = lean_float_sub(v___x_2845_, v___x_2846_);
v___x_2848_ = lean_float_decLt(v___y_2844_, v___x_2847_);
v___y_2813_ = v___x_2848_;
goto v___jp_2812_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4___boxed(lean_object* v_cls_2859_, lean_object* v_collapsed_2860_, lean_object* v_tag_2861_, lean_object* v_opts_2862_, lean_object* v_clsEnabled_2863_, lean_object* v_oldTraces_2864_, lean_object* v_msg_2865_, lean_object* v_resStartStop_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_){
_start:
{
uint8_t v_collapsed_boxed_2870_; uint8_t v_clsEnabled_boxed_2871_; lean_object* v_res_2872_; 
v_collapsed_boxed_2870_ = lean_unbox(v_collapsed_2860_);
v_clsEnabled_boxed_2871_ = lean_unbox(v_clsEnabled_2863_);
v_res_2872_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_cls_2859_, v_collapsed_boxed_2870_, v_tag_2861_, v_opts_2862_, v_clsEnabled_boxed_2871_, v_oldTraces_2864_, v_msg_2865_, v_resStartStop_2866_, v___y_2867_, v___y_2868_);
lean_dec(v___y_2868_);
lean_dec_ref(v___y_2867_);
lean_dec_ref(v_opts_2862_);
return v_res_2872_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__4(void){
_start:
{
lean_object* v___x_2879_; lean_object* v___x_2880_; 
v___x_2879_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__3));
v___x_2880_ = l_Lean_MessageData_ofFormat(v___x_2879_);
return v___x_2880_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__5(void){
_start:
{
lean_object* v___x_2881_; lean_object* v___f_2882_; 
v___x_2881_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__4, &lp_aesop_Aesop_traceSimpTheorems___closed__4_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__4);
v___f_2882_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed), 5, 1);
lean_closure_set(v___f_2882_, 0, v___x_2881_);
return v___f_2882_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__8(void){
_start:
{
lean_object* v___x_2886_; lean_object* v___x_2887_; 
v___x_2886_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__7));
v___x_2887_ = l_Lean_MessageData_ofFormat(v___x_2886_);
return v___x_2887_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__9(void){
_start:
{
lean_object* v___x_2888_; lean_object* v___f_2889_; 
v___x_2888_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__8, &lp_aesop_Aesop_traceSimpTheorems___closed__8_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__8);
v___f_2889_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed), 5, 1);
lean_closure_set(v___f_2889_, 0, v___x_2888_);
return v___f_2889_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__12(void){
_start:
{
lean_object* v___x_2893_; lean_object* v___x_2894_; 
v___x_2893_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__11));
v___x_2894_ = l_Lean_MessageData_ofFormat(v___x_2893_);
return v___x_2894_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__13(void){
_start:
{
lean_object* v___x_2895_; lean_object* v___f_2896_; 
v___x_2895_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__12, &lp_aesop_Aesop_traceSimpTheorems___closed__12_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__12);
v___f_2896_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed), 5, 1);
lean_closure_set(v___f_2896_, 0, v___x_2895_);
return v___f_2896_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__15(void){
_start:
{
lean_object* v___x_2898_; lean_object* v___x_2899_; 
v___x_2898_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__14));
v___x_2899_ = l_Lean_stringToMessageData(v___x_2898_);
return v___x_2899_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__20(void){
_start:
{
lean_object* v___x_2906_; lean_object* v___x_2907_; 
v___x_2906_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__19));
v___x_2907_ = l_Lean_MessageData_ofFormat(v___x_2906_);
return v___x_2907_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceSimpTheorems___closed__21(void){
_start:
{
lean_object* v___x_2908_; lean_object* v___f_2909_; 
v___x_2908_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__20, &lp_aesop_Aesop_traceSimpTheorems___closed__20_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__20);
v___f_2909_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traceSimpTheorems___lam__1___boxed), 5, 1);
lean_closure_set(v___f_2909_, 0, v___x_2908_);
return v___f_2909_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems(lean_object* v_s_2910_, lean_object* v_opt_2911_, lean_object* v_a_2912_, lean_object* v_a_2913_){
_start:
{
lean_object* v___x_2915_; lean_object* v_a_2916_; lean_object* v___x_2918_; uint8_t v_isShared_2919_; uint8_t v_isSharedCheck_3377_; 
v___x_2915_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0___redArg(v_opt_2911_, v_a_2912_);
v_a_2916_ = lean_ctor_get(v___x_2915_, 0);
v_isSharedCheck_3377_ = !lean_is_exclusive(v___x_2915_);
if (v_isSharedCheck_3377_ == 0)
{
v___x_2918_ = v___x_2915_;
v_isShared_2919_ = v_isSharedCheck_3377_;
goto v_resetjp_2917_;
}
else
{
lean_inc(v_a_2916_);
lean_dec(v___x_2915_);
v___x_2918_ = lean_box(0);
v_isShared_2919_ = v_isSharedCheck_3377_;
goto v_resetjp_2917_;
}
v_resetjp_2917_:
{
uint8_t v___x_2920_; 
v___x_2920_ = lean_unbox(v_a_2916_);
if (v___x_2920_ == 0)
{
lean_object* v___x_2921_; lean_object* v___x_2923_; 
lean_dec(v_a_2916_);
lean_dec_ref(v_opt_2911_);
v___x_2921_ = lean_box(0);
if (v_isShared_2919_ == 0)
{
lean_ctor_set(v___x_2918_, 0, v___x_2921_);
v___x_2923_ = v___x_2918_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_2924_; 
v_reuseFailAlloc_2924_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2924_, 0, v___x_2921_);
v___x_2923_ = v_reuseFailAlloc_2924_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
return v___x_2923_;
}
}
else
{
lean_object* v_traceClass_2925_; lean_object* v___y_2927_; lean_object* v___y_2928_; uint8_t v___y_2929_; lean_object* v___y_2930_; lean_object* v___y_2931_; lean_object* v___y_2932_; lean_object* v_a_2933_; lean_object* v___y_2947_; lean_object* v___y_2948_; uint8_t v___y_2949_; lean_object* v___y_2950_; lean_object* v___y_2951_; lean_object* v___y_2952_; lean_object* v_a_2953_; lean_object* v___y_2956_; lean_object* v___y_2957_; uint8_t v___y_2958_; lean_object* v___y_2959_; lean_object* v___y_2960_; lean_object* v___y_2961_; lean_object* v_a_2962_; lean_object* v___y_2973_; lean_object* v___y_2974_; uint8_t v___y_2975_; lean_object* v___y_2976_; lean_object* v___y_2977_; lean_object* v___y_2978_; lean_object* v_a_2979_; size_t v___y_2982_; size_t v___y_2983_; lean_object* v___y_2984_; lean_object* v___y_2985_; uint8_t v___y_2986_; lean_object* v___y_2987_; lean_object* v___y_2988_; lean_object* v___y_2989_; lean_object* v_options_3016_; lean_object* v_inheritedTraceOptions_3017_; uint8_t v_hasTrace_3018_; lean_object* v___f_3019_; lean_object* v___y_3021_; uint8_t v___y_3058_; lean_object* v___y_3059_; lean_object* v___y_3060_; lean_object* v___y_3061_; lean_object* v___y_3062_; lean_object* v___y_3063_; lean_object* v_a_3064_; uint8_t v___y_3075_; lean_object* v___y_3076_; lean_object* v___y_3077_; lean_object* v___y_3078_; lean_object* v___y_3079_; lean_object* v___y_3080_; lean_object* v_a_3081_; lean_object* v___y_3095_; uint8_t v___y_3096_; lean_object* v___y_3097_; lean_object* v___y_3098_; lean_object* v___y_3099_; lean_object* v___y_3141_; lean_object* v___y_3154_; uint8_t v___y_3155_; lean_object* v___y_3156_; lean_object* v___y_3157_; lean_object* v___y_3158_; lean_object* v___y_3159_; lean_object* v_a_3160_; lean_object* v___y_3174_; uint8_t v___y_3175_; lean_object* v___y_3176_; lean_object* v___y_3177_; lean_object* v___y_3178_; lean_object* v___y_3179_; lean_object* v_a_3180_; uint8_t v___y_3191_; lean_object* v___y_3192_; lean_object* v___y_3193_; lean_object* v___y_3194_; lean_object* v___y_3195_; lean_object* v___y_3249_; lean_object* v___x_3250_; 
lean_del_object(v___x_2918_);
v_traceClass_2925_ = lean_ctor_get(v_opt_2911_, 0);
lean_inc(v_traceClass_2925_);
v_options_3016_ = lean_ctor_get(v_a_2912_, 2);
v_inheritedTraceOptions_3017_ = lean_ctor_get(v_a_2912_, 13);
v_hasTrace_3018_ = lean_ctor_get_uint8(v_options_3016_, sizeof(void*)*1);
v___f_3019_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__0));
v___x_3250_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__15, &lp_aesop_Aesop_traceSimpTheorems___closed__15_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__15);
if (v_hasTrace_3018_ == 0)
{
lean_object* v___x_3251_; 
lean_inc(v_traceClass_2925_);
v___x_3251_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_traceClass_2925_, v___x_3250_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3251_) == 0)
{
lean_object* v_erased_3252_; lean_object* v___f_3253_; lean_object* v___x_3254_; lean_object* v___x_3255_; size_t v_sz_3256_; size_t v___x_3257_; uint8_t v___x_3258_; lean_object* v___x_3259_; lean_object* v___x_3260_; lean_object* v___x_3261_; size_t v_sz_3262_; lean_object* v___x_3263_; 
lean_dec_ref_known(v___x_3251_, 1);
v_erased_3252_ = lean_ctor_get(v_s_2910_, 4);
v___f_3253_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__16));
v___x_3254_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__17));
v___x_3255_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_3253_, v_erased_3252_, v___x_3254_);
v_sz_3256_ = lean_array_size(v___x_3255_);
v___x_3257_ = ((size_t)0ULL);
v___x_3258_ = lean_unbox(v_a_2916_);
v___x_3259_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(v___x_3258_, v_sz_3256_, v___x_3257_, v___x_3255_);
v___x_3260_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_3259_);
v___x_3261_ = lean_box(0);
v_sz_3262_ = lean_array_size(v___x_3260_);
lean_inc(v_traceClass_2925_);
v___x_3263_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3260_, v_sz_3262_, v___x_3257_, v___x_3261_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3260_);
if (lean_obj_tag(v___x_3263_) == 0)
{
lean_dec_ref_known(v___x_3263_, 1);
goto v___jp_3236_;
}
else
{
v___y_3249_ = v___x_3263_;
goto v___jp_3248_;
}
}
else
{
v___y_3249_ = v___x_3251_;
goto v___jp_3248_;
}
}
else
{
lean_object* v___f_3264_; lean_object* v___f_3265_; lean_object* v___x_3266_; lean_object* v___x_3267_; lean_object* v___x_3268_; uint8_t v___x_3269_; lean_object* v___y_3271_; lean_object* v___y_3272_; lean_object* v_a_3273_; lean_object* v___y_3287_; lean_object* v___y_3288_; lean_object* v_a_3289_; lean_object* v___y_3292_; lean_object* v___y_3293_; lean_object* v___y_3294_; lean_object* v___y_3305_; lean_object* v___y_3306_; lean_object* v_a_3307_; lean_object* v___y_3318_; lean_object* v___y_3319_; lean_object* v_a_3320_; lean_object* v___y_3323_; lean_object* v___y_3324_; lean_object* v___y_3325_; 
v___f_3264_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__16));
v___f_3265_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__21, &lp_aesop_Aesop_traceSimpTheorems___closed__21_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__21);
v___x_3266_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_3267_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0));
lean_inc(v_traceClass_2925_);
v___x_3268_ = l_Lean_Name_append(v___x_3267_, v_traceClass_2925_);
v___x_3269_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3017_, v_options_3016_, v___x_3268_);
lean_dec(v___x_3268_);
if (v___x_3269_ == 0)
{
lean_object* v___x_3364_; uint8_t v___x_3365_; 
v___x_3364_ = l_Lean_trace_profiler;
v___x_3365_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_3016_, v___x_3364_);
if (v___x_3365_ == 0)
{
lean_object* v___x_3366_; 
lean_inc(v_traceClass_2925_);
v___x_3366_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_traceClass_2925_, v___x_3250_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3366_) == 0)
{
lean_object* v_erased_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; size_t v_sz_3370_; size_t v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; lean_object* v___x_3374_; size_t v_sz_3375_; lean_object* v___x_3376_; 
lean_dec_ref_known(v___x_3366_, 1);
v_erased_3367_ = lean_ctor_get(v_s_2910_, 4);
v___x_3368_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__17));
v___x_3369_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_3264_, v_erased_3367_, v___x_3368_);
v_sz_3370_ = lean_array_size(v___x_3369_);
v___x_3371_ = ((size_t)0ULL);
v___x_3372_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(v_hasTrace_3018_, v_sz_3370_, v___x_3371_, v___x_3369_);
v___x_3373_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_3372_);
v___x_3374_ = lean_box(0);
v_sz_3375_ = lean_array_size(v___x_3373_);
lean_inc(v_traceClass_2925_);
v___x_3376_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3373_, v_sz_3375_, v___x_3371_, v___x_3374_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3373_);
if (lean_obj_tag(v___x_3376_) == 0)
{
lean_dec_ref_known(v___x_3376_, 1);
goto v___jp_3236_;
}
else
{
v___y_3249_ = v___x_3376_;
goto v___jp_3248_;
}
}
else
{
v___y_3249_ = v___x_3366_;
goto v___jp_3248_;
}
}
else
{
goto v___jp_3335_;
}
}
else
{
goto v___jp_3335_;
}
v___jp_3270_:
{
lean_object* v___x_3274_; double v___x_3275_; double v___x_3276_; double v___x_3277_; double v___x_3278_; double v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; lean_object* v___x_3283_; uint8_t v___x_3284_; lean_object* v___x_3285_; 
v___x_3274_ = lean_io_mono_nanos_now();
v___x_3275_ = lean_float_of_nat(v___y_3272_);
v___x_3276_ = lean_float_once(&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0, &lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once, _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0);
v___x_3277_ = lean_float_div(v___x_3275_, v___x_3276_);
v___x_3278_ = lean_float_of_nat(v___x_3274_);
v___x_3279_ = lean_float_div(v___x_3278_, v___x_3276_);
v___x_3280_ = lean_box_float(v___x_3277_);
v___x_3281_ = lean_box_float(v___x_3279_);
v___x_3282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3282_, 0, v___x_3280_);
lean_ctor_set(v___x_3282_, 1, v___x_3281_);
v___x_3283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3283_, 0, v_a_3273_);
lean_ctor_set(v___x_3283_, 1, v___x_3282_);
v___x_3284_ = lean_unbox(v_a_2916_);
lean_inc(v_traceClass_2925_);
v___x_3285_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3284_, v___x_3266_, v_options_3016_, v___x_3269_, v___y_3271_, v___f_3265_, v___x_3283_, v_a_2912_, v_a_2913_);
v___y_3249_ = v___x_3285_;
goto v___jp_3248_;
}
v___jp_3286_:
{
lean_object* v___x_3290_; 
v___x_3290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3290_, 0, v_a_3289_);
v___y_3271_ = v___y_3287_;
v___y_3272_ = v___y_3288_;
v_a_3273_ = v___x_3290_;
goto v___jp_3270_;
}
v___jp_3291_:
{
if (lean_obj_tag(v___y_3294_) == 0)
{
lean_object* v_a_3295_; 
v_a_3295_ = lean_ctor_get(v___y_3294_, 0);
lean_inc(v_a_3295_);
lean_dec_ref_known(v___y_3294_, 1);
v___y_3287_ = v___y_3292_;
v___y_3288_ = v___y_3293_;
v_a_3289_ = v_a_3295_;
goto v___jp_3286_;
}
else
{
lean_object* v_a_3296_; lean_object* v___x_3298_; uint8_t v_isShared_3299_; uint8_t v_isSharedCheck_3303_; 
v_a_3296_ = lean_ctor_get(v___y_3294_, 0);
v_isSharedCheck_3303_ = !lean_is_exclusive(v___y_3294_);
if (v_isSharedCheck_3303_ == 0)
{
v___x_3298_ = v___y_3294_;
v_isShared_3299_ = v_isSharedCheck_3303_;
goto v_resetjp_3297_;
}
else
{
lean_inc(v_a_3296_);
lean_dec(v___y_3294_);
v___x_3298_ = lean_box(0);
v_isShared_3299_ = v_isSharedCheck_3303_;
goto v_resetjp_3297_;
}
v_resetjp_3297_:
{
lean_object* v___x_3301_; 
if (v_isShared_3299_ == 0)
{
lean_ctor_set_tag(v___x_3298_, 0);
v___x_3301_ = v___x_3298_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3302_; 
v_reuseFailAlloc_3302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3302_, 0, v_a_3296_);
v___x_3301_ = v_reuseFailAlloc_3302_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
v___y_3271_ = v___y_3292_;
v___y_3272_ = v___y_3293_;
v_a_3273_ = v___x_3301_;
goto v___jp_3270_;
}
}
}
}
v___jp_3304_:
{
lean_object* v___x_3308_; double v___x_3309_; double v___x_3310_; lean_object* v___x_3311_; lean_object* v___x_3312_; lean_object* v___x_3313_; lean_object* v___x_3314_; uint8_t v___x_3315_; lean_object* v___x_3316_; 
v___x_3308_ = lean_io_get_num_heartbeats();
v___x_3309_ = lean_float_of_nat(v___y_3306_);
v___x_3310_ = lean_float_of_nat(v___x_3308_);
v___x_3311_ = lean_box_float(v___x_3309_);
v___x_3312_ = lean_box_float(v___x_3310_);
v___x_3313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3313_, 0, v___x_3311_);
lean_ctor_set(v___x_3313_, 1, v___x_3312_);
v___x_3314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3314_, 0, v_a_3307_);
lean_ctor_set(v___x_3314_, 1, v___x_3313_);
v___x_3315_ = lean_unbox(v_a_2916_);
lean_inc(v_traceClass_2925_);
v___x_3316_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3315_, v___x_3266_, v_options_3016_, v___x_3269_, v___y_3305_, v___f_3265_, v___x_3314_, v_a_2912_, v_a_2913_);
v___y_3249_ = v___x_3316_;
goto v___jp_3248_;
}
v___jp_3317_:
{
lean_object* v___x_3321_; 
v___x_3321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3321_, 0, v_a_3320_);
v___y_3305_ = v___y_3318_;
v___y_3306_ = v___y_3319_;
v_a_3307_ = v___x_3321_;
goto v___jp_3304_;
}
v___jp_3322_:
{
if (lean_obj_tag(v___y_3325_) == 0)
{
lean_object* v_a_3326_; 
v_a_3326_ = lean_ctor_get(v___y_3325_, 0);
lean_inc(v_a_3326_);
lean_dec_ref_known(v___y_3325_, 1);
v___y_3318_ = v___y_3323_;
v___y_3319_ = v___y_3324_;
v_a_3320_ = v_a_3326_;
goto v___jp_3317_;
}
else
{
lean_object* v_a_3327_; lean_object* v___x_3329_; uint8_t v_isShared_3330_; uint8_t v_isSharedCheck_3334_; 
v_a_3327_ = lean_ctor_get(v___y_3325_, 0);
v_isSharedCheck_3334_ = !lean_is_exclusive(v___y_3325_);
if (v_isSharedCheck_3334_ == 0)
{
v___x_3329_ = v___y_3325_;
v_isShared_3330_ = v_isSharedCheck_3334_;
goto v_resetjp_3328_;
}
else
{
lean_inc(v_a_3327_);
lean_dec(v___y_3325_);
v___x_3329_ = lean_box(0);
v_isShared_3330_ = v_isSharedCheck_3334_;
goto v_resetjp_3328_;
}
v_resetjp_3328_:
{
lean_object* v___x_3332_; 
if (v_isShared_3330_ == 0)
{
lean_ctor_set_tag(v___x_3329_, 0);
v___x_3332_ = v___x_3329_;
goto v_reusejp_3331_;
}
else
{
lean_object* v_reuseFailAlloc_3333_; 
v_reuseFailAlloc_3333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3333_, 0, v_a_3327_);
v___x_3332_ = v_reuseFailAlloc_3333_;
goto v_reusejp_3331_;
}
v_reusejp_3331_:
{
v___y_3305_ = v___y_3323_;
v___y_3306_ = v___y_3324_;
v_a_3307_ = v___x_3332_;
goto v___jp_3304_;
}
}
}
}
v___jp_3335_:
{
lean_object* v___x_3336_; lean_object* v_a_3337_; lean_object* v___x_3338_; uint8_t v___x_3339_; 
v___x_3336_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v_a_2913_);
v_a_3337_ = lean_ctor_get(v___x_3336_, 0);
lean_inc(v_a_3337_);
lean_dec_ref(v___x_3336_);
v___x_3338_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3339_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_3016_, v___x_3338_);
if (v___x_3339_ == 0)
{
lean_object* v___x_3340_; lean_object* v___x_3341_; 
v___x_3340_ = lean_io_mono_nanos_now();
lean_inc(v_traceClass_2925_);
v___x_3341_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_traceClass_2925_, v___x_3250_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3341_) == 0)
{
lean_object* v_erased_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; size_t v_sz_3345_; size_t v___x_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3349_; size_t v_sz_3350_; lean_object* v___x_3351_; 
lean_dec_ref_known(v___x_3341_, 1);
v_erased_3342_ = lean_ctor_get(v_s_2910_, 4);
v___x_3343_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__17));
v___x_3344_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_3264_, v_erased_3342_, v___x_3343_);
v_sz_3345_ = lean_array_size(v___x_3344_);
v___x_3346_ = ((size_t)0ULL);
v___x_3347_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(v_hasTrace_3018_, v_sz_3345_, v___x_3346_, v___x_3344_);
v___x_3348_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_3347_);
v___x_3349_ = lean_box(0);
v_sz_3350_ = lean_array_size(v___x_3348_);
lean_inc(v_traceClass_2925_);
v___x_3351_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3348_, v_sz_3350_, v___x_3346_, v___x_3349_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3348_);
if (lean_obj_tag(v___x_3351_) == 0)
{
lean_dec_ref_known(v___x_3351_, 1);
v___y_3287_ = v_a_3337_;
v___y_3288_ = v___x_3340_;
v_a_3289_ = v___x_3349_;
goto v___jp_3286_;
}
else
{
v___y_3292_ = v_a_3337_;
v___y_3293_ = v___x_3340_;
v___y_3294_ = v___x_3351_;
goto v___jp_3291_;
}
}
else
{
v___y_3292_ = v_a_3337_;
v___y_3293_ = v___x_3340_;
v___y_3294_ = v___x_3341_;
goto v___jp_3291_;
}
}
else
{
lean_object* v___x_3352_; lean_object* v___x_3353_; 
v___x_3352_ = lean_io_get_num_heartbeats();
lean_inc(v_traceClass_2925_);
v___x_3353_ = lp_aesop_Lean_addTrace___at___00Aesop_traceSimpTheoremTreeContents_spec__5(v_traceClass_2925_, v___x_3250_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3353_) == 0)
{
lean_object* v_erased_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; size_t v_sz_3357_; size_t v___x_3358_; lean_object* v___x_3359_; lean_object* v___x_3360_; lean_object* v___x_3361_; size_t v_sz_3362_; lean_object* v___x_3363_; 
lean_dec_ref_known(v___x_3353_, 1);
v_erased_3354_ = lean_ctor_get(v_s_2910_, 4);
v___x_3355_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__17));
v___x_3356_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_3264_, v_erased_3354_, v___x_3355_);
v_sz_3357_ = lean_array_size(v___x_3356_);
v___x_3358_ = ((size_t)0ULL);
v___x_3359_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__6(v___x_3339_, v_sz_3357_, v___x_3358_, v___x_3356_);
v___x_3360_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_3359_);
v___x_3361_ = lean_box(0);
v_sz_3362_ = lean_array_size(v___x_3360_);
lean_inc(v_traceClass_2925_);
v___x_3363_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3360_, v_sz_3362_, v___x_3358_, v___x_3361_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3360_);
if (lean_obj_tag(v___x_3363_) == 0)
{
lean_dec_ref_known(v___x_3363_, 1);
v___y_3318_ = v_a_3337_;
v___y_3319_ = v___x_3352_;
v_a_3320_ = v___x_3361_;
goto v___jp_3317_;
}
else
{
v___y_3323_ = v_a_3337_;
v___y_3324_ = v___x_3352_;
v___y_3325_ = v___x_3363_;
goto v___jp_3322_;
}
}
else
{
v___y_3323_ = v_a_3337_;
v___y_3324_ = v___x_3352_;
v___y_3325_ = v___x_3353_;
goto v___jp_3322_;
}
}
}
}
v___jp_2926_:
{
lean_object* v___x_2934_; double v___x_2935_; double v___x_2936_; double v___x_2937_; double v___x_2938_; double v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; uint8_t v___x_2944_; lean_object* v___x_2945_; 
v___x_2934_ = lean_io_mono_nanos_now();
v___x_2935_ = lean_float_of_nat(v___y_2928_);
v___x_2936_ = lean_float_once(&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0, &lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once, _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0);
v___x_2937_ = lean_float_div(v___x_2935_, v___x_2936_);
v___x_2938_ = lean_float_of_nat(v___x_2934_);
v___x_2939_ = lean_float_div(v___x_2938_, v___x_2936_);
v___x_2940_ = lean_box_float(v___x_2937_);
v___x_2941_ = lean_box_float(v___x_2939_);
v___x_2942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2942_, 0, v___x_2940_);
lean_ctor_set(v___x_2942_, 1, v___x_2941_);
v___x_2943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2943_, 0, v_a_2933_);
lean_ctor_set(v___x_2943_, 1, v___x_2942_);
v___x_2944_ = lean_unbox(v_a_2916_);
lean_dec(v_a_2916_);
lean_inc_ref(v___y_2932_);
lean_inc_ref(v___y_2927_);
v___x_2945_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_2944_, v___y_2927_, v___y_2930_, v___y_2929_, v___y_2931_, v___y_2932_, v___x_2943_, v_a_2912_, v_a_2913_);
return v___x_2945_;
}
v___jp_2946_:
{
lean_object* v___x_2954_; 
v___x_2954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2954_, 0, v_a_2953_);
v___y_2927_ = v___y_2947_;
v___y_2928_ = v___y_2948_;
v___y_2929_ = v___y_2949_;
v___y_2930_ = v___y_2950_;
v___y_2931_ = v___y_2951_;
v___y_2932_ = v___y_2952_;
v_a_2933_ = v___x_2954_;
goto v___jp_2926_;
}
v___jp_2955_:
{
lean_object* v___x_2963_; double v___x_2964_; double v___x_2965_; lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; uint8_t v___x_2970_; lean_object* v___x_2971_; 
v___x_2963_ = lean_io_get_num_heartbeats();
v___x_2964_ = lean_float_of_nat(v___y_2957_);
v___x_2965_ = lean_float_of_nat(v___x_2963_);
v___x_2966_ = lean_box_float(v___x_2964_);
v___x_2967_ = lean_box_float(v___x_2965_);
v___x_2968_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2968_, 0, v___x_2966_);
lean_ctor_set(v___x_2968_, 1, v___x_2967_);
v___x_2969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2969_, 0, v_a_2962_);
lean_ctor_set(v___x_2969_, 1, v___x_2968_);
v___x_2970_ = lean_unbox(v_a_2916_);
lean_dec(v_a_2916_);
lean_inc_ref(v___y_2961_);
lean_inc_ref(v___y_2956_);
v___x_2971_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_2970_, v___y_2956_, v___y_2959_, v___y_2958_, v___y_2960_, v___y_2961_, v___x_2969_, v_a_2912_, v_a_2913_);
return v___x_2971_;
}
v___jp_2972_:
{
lean_object* v___x_2980_; 
v___x_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2980_, 0, v_a_2979_);
v___y_2956_ = v___y_2973_;
v___y_2957_ = v___y_2974_;
v___y_2958_ = v___y_2975_;
v___y_2959_ = v___y_2976_;
v___y_2960_ = v___y_2977_;
v___y_2961_ = v___y_2978_;
v_a_2962_ = v___x_2980_;
goto v___jp_2955_;
}
v___jp_2981_:
{
lean_object* v___x_2990_; lean_object* v_a_2991_; lean_object* v___x_2992_; uint8_t v___x_2993_; 
v___x_2990_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v_a_2913_);
v_a_2991_ = lean_ctor_get(v___x_2990_, 0);
lean_inc(v_a_2991_);
lean_dec_ref(v___x_2990_);
v___x_2992_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2993_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v___y_2987_, v___x_2992_);
if (v___x_2993_ == 0)
{
lean_object* v___x_2994_; lean_object* v___x_2995_; 
v___x_2994_ = lean_io_mono_nanos_now();
lean_inc(v_traceClass_2925_);
v___x_2995_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___y_2985_, v___y_2983_, v___y_2982_, v___y_2988_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___y_2985_);
if (lean_obj_tag(v___x_2995_) == 0)
{
lean_dec_ref_known(v___x_2995_, 1);
v___y_2947_ = v___y_2984_;
v___y_2948_ = v___x_2994_;
v___y_2949_ = v___y_2986_;
v___y_2950_ = v___y_2987_;
v___y_2951_ = v_a_2991_;
v___y_2952_ = v___y_2989_;
v_a_2953_ = v___y_2988_;
goto v___jp_2946_;
}
else
{
if (lean_obj_tag(v___x_2995_) == 0)
{
lean_object* v_a_2996_; 
v_a_2996_ = lean_ctor_get(v___x_2995_, 0);
lean_inc(v_a_2996_);
lean_dec_ref_known(v___x_2995_, 1);
v___y_2947_ = v___y_2984_;
v___y_2948_ = v___x_2994_;
v___y_2949_ = v___y_2986_;
v___y_2950_ = v___y_2987_;
v___y_2951_ = v_a_2991_;
v___y_2952_ = v___y_2989_;
v_a_2953_ = v_a_2996_;
goto v___jp_2946_;
}
else
{
lean_object* v_a_2997_; lean_object* v___x_2999_; uint8_t v_isShared_3000_; uint8_t v_isSharedCheck_3004_; 
v_a_2997_ = lean_ctor_get(v___x_2995_, 0);
v_isSharedCheck_3004_ = !lean_is_exclusive(v___x_2995_);
if (v_isSharedCheck_3004_ == 0)
{
v___x_2999_ = v___x_2995_;
v_isShared_3000_ = v_isSharedCheck_3004_;
goto v_resetjp_2998_;
}
else
{
lean_inc(v_a_2997_);
lean_dec(v___x_2995_);
v___x_2999_ = lean_box(0);
v_isShared_3000_ = v_isSharedCheck_3004_;
goto v_resetjp_2998_;
}
v_resetjp_2998_:
{
lean_object* v___x_3002_; 
if (v_isShared_3000_ == 0)
{
lean_ctor_set_tag(v___x_2999_, 0);
v___x_3002_ = v___x_2999_;
goto v_reusejp_3001_;
}
else
{
lean_object* v_reuseFailAlloc_3003_; 
v_reuseFailAlloc_3003_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3003_, 0, v_a_2997_);
v___x_3002_ = v_reuseFailAlloc_3003_;
goto v_reusejp_3001_;
}
v_reusejp_3001_:
{
v___y_2927_ = v___y_2984_;
v___y_2928_ = v___x_2994_;
v___y_2929_ = v___y_2986_;
v___y_2930_ = v___y_2987_;
v___y_2931_ = v_a_2991_;
v___y_2932_ = v___y_2989_;
v_a_2933_ = v___x_3002_;
goto v___jp_2926_;
}
}
}
}
}
else
{
lean_object* v___x_3005_; lean_object* v___x_3006_; 
v___x_3005_ = lean_io_get_num_heartbeats();
lean_inc(v_traceClass_2925_);
v___x_3006_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___y_2985_, v___y_2983_, v___y_2982_, v___y_2988_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___y_2985_);
if (lean_obj_tag(v___x_3006_) == 0)
{
lean_dec_ref_known(v___x_3006_, 1);
v___y_2973_ = v___y_2984_;
v___y_2974_ = v___x_3005_;
v___y_2975_ = v___y_2986_;
v___y_2976_ = v___y_2987_;
v___y_2977_ = v_a_2991_;
v___y_2978_ = v___y_2989_;
v_a_2979_ = v___y_2988_;
goto v___jp_2972_;
}
else
{
if (lean_obj_tag(v___x_3006_) == 0)
{
lean_object* v_a_3007_; 
v_a_3007_ = lean_ctor_get(v___x_3006_, 0);
lean_inc(v_a_3007_);
lean_dec_ref_known(v___x_3006_, 1);
v___y_2973_ = v___y_2984_;
v___y_2974_ = v___x_3005_;
v___y_2975_ = v___y_2986_;
v___y_2976_ = v___y_2987_;
v___y_2977_ = v_a_2991_;
v___y_2978_ = v___y_2989_;
v_a_2979_ = v_a_3007_;
goto v___jp_2972_;
}
else
{
lean_object* v_a_3008_; lean_object* v___x_3010_; uint8_t v_isShared_3011_; uint8_t v_isSharedCheck_3015_; 
v_a_3008_ = lean_ctor_get(v___x_3006_, 0);
v_isSharedCheck_3015_ = !lean_is_exclusive(v___x_3006_);
if (v_isSharedCheck_3015_ == 0)
{
v___x_3010_ = v___x_3006_;
v_isShared_3011_ = v_isSharedCheck_3015_;
goto v_resetjp_3009_;
}
else
{
lean_inc(v_a_3008_);
lean_dec(v___x_3006_);
v___x_3010_ = lean_box(0);
v_isShared_3011_ = v_isSharedCheck_3015_;
goto v_resetjp_3009_;
}
v_resetjp_3009_:
{
lean_object* v___x_3013_; 
if (v_isShared_3011_ == 0)
{
lean_ctor_set_tag(v___x_3010_, 0);
v___x_3013_ = v___x_3010_;
goto v_reusejp_3012_;
}
else
{
lean_object* v_reuseFailAlloc_3014_; 
v_reuseFailAlloc_3014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3014_, 0, v_a_3008_);
v___x_3013_ = v_reuseFailAlloc_3014_;
goto v_reusejp_3012_;
}
v_reusejp_3012_:
{
v___y_2956_ = v___y_2984_;
v___y_2957_ = v___x_3005_;
v___y_2958_ = v___y_2986_;
v___y_2959_ = v___y_2987_;
v___y_2960_ = v_a_2991_;
v___y_2961_ = v___y_2989_;
v_a_2962_ = v___x_3013_;
goto v___jp_2955_;
}
}
}
}
}
}
v___jp_3020_:
{
if (lean_obj_tag(v___y_3021_) == 0)
{
lean_object* v_toUnfold_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; size_t v_sz_3025_; size_t v___x_3026_; uint8_t v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v___x_3030_; size_t v_sz_3031_; 
lean_dec_ref_known(v___y_3021_, 1);
v_toUnfold_3022_ = lean_ctor_get(v_s_2910_, 3);
v___x_3023_ = ((lean_object*)(lp_aesop_Aesop_traceSimpTheorems___closed__1));
v___x_3024_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v___f_3019_, v_toUnfold_3022_, v___x_3023_);
v_sz_3025_ = lean_array_size(v___x_3024_);
v___x_3026_ = ((size_t)0ULL);
v___x_3027_ = lean_unbox(v_a_2916_);
v___x_3028_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_traceSimpTheorems_spec__1(v___x_3027_, v_sz_3025_, v___x_3026_, v___x_3024_);
v___x_3029_ = lp_aesop_Array_qsortOrd___at___00Aesop_traceSimpTheoremTreeContents_spec__4(v___x_3028_);
v___x_3030_ = lean_box(0);
v_sz_3031_ = lean_array_size(v___x_3029_);
if (v_hasTrace_3018_ == 0)
{
lean_object* v___x_3032_; 
lean_dec(v_a_2916_);
v___x_3032_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3029_, v_sz_3031_, v___x_3026_, v___x_3030_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3029_);
if (lean_obj_tag(v___x_3032_) == 0)
{
lean_object* v___x_3034_; uint8_t v_isShared_3035_; uint8_t v_isSharedCheck_3039_; 
v_isSharedCheck_3039_ = !lean_is_exclusive(v___x_3032_);
if (v_isSharedCheck_3039_ == 0)
{
lean_object* v_unused_3040_; 
v_unused_3040_ = lean_ctor_get(v___x_3032_, 0);
lean_dec(v_unused_3040_);
v___x_3034_ = v___x_3032_;
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
else
{
lean_dec(v___x_3032_);
v___x_3034_ = lean_box(0);
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
v_resetjp_3033_:
{
lean_object* v___x_3037_; 
if (v_isShared_3035_ == 0)
{
lean_ctor_set(v___x_3034_, 0, v___x_3030_);
v___x_3037_ = v___x_3034_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v___x_3030_);
v___x_3037_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
return v___x_3037_;
}
}
}
else
{
return v___x_3032_;
}
}
else
{
lean_object* v___f_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; uint8_t v___x_3045_; 
v___f_3041_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__5, &lp_aesop_Aesop_traceSimpTheorems___closed__5_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__5);
v___x_3042_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_3043_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0));
lean_inc(v_traceClass_2925_);
v___x_3044_ = l_Lean_Name_append(v___x_3043_, v_traceClass_2925_);
v___x_3045_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3017_, v_options_3016_, v___x_3044_);
lean_dec(v___x_3044_);
if (v___x_3045_ == 0)
{
lean_object* v___x_3046_; uint8_t v___x_3047_; 
v___x_3046_ = l_Lean_trace_profiler;
v___x_3047_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_3016_, v___x_3046_);
if (v___x_3047_ == 0)
{
lean_object* v___x_3048_; 
lean_dec(v_a_2916_);
v___x_3048_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_traceSimpTheorems_spec__2(v_traceClass_2925_, v___x_3029_, v_sz_3031_, v___x_3026_, v___x_3030_, v_a_2912_, v_a_2913_);
lean_dec_ref(v___x_3029_);
if (lean_obj_tag(v___x_3048_) == 0)
{
lean_object* v___x_3050_; uint8_t v_isShared_3051_; uint8_t v_isSharedCheck_3055_; 
v_isSharedCheck_3055_ = !lean_is_exclusive(v___x_3048_);
if (v_isSharedCheck_3055_ == 0)
{
lean_object* v_unused_3056_; 
v_unused_3056_ = lean_ctor_get(v___x_3048_, 0);
lean_dec(v_unused_3056_);
v___x_3050_ = v___x_3048_;
v_isShared_3051_ = v_isSharedCheck_3055_;
goto v_resetjp_3049_;
}
else
{
lean_dec(v___x_3048_);
v___x_3050_ = lean_box(0);
v_isShared_3051_ = v_isSharedCheck_3055_;
goto v_resetjp_3049_;
}
v_resetjp_3049_:
{
lean_object* v___x_3053_; 
if (v_isShared_3051_ == 0)
{
lean_ctor_set(v___x_3050_, 0, v___x_3030_);
v___x_3053_ = v___x_3050_;
goto v_reusejp_3052_;
}
else
{
lean_object* v_reuseFailAlloc_3054_; 
v_reuseFailAlloc_3054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3054_, 0, v___x_3030_);
v___x_3053_ = v_reuseFailAlloc_3054_;
goto v_reusejp_3052_;
}
v_reusejp_3052_:
{
return v___x_3053_;
}
}
}
else
{
return v___x_3048_;
}
}
else
{
v___y_2982_ = v___x_3026_;
v___y_2983_ = v_sz_3031_;
v___y_2984_ = v___x_3042_;
v___y_2985_ = v___x_3029_;
v___y_2986_ = v___x_3045_;
v___y_2987_ = v_options_3016_;
v___y_2988_ = v___x_3030_;
v___y_2989_ = v___f_3041_;
goto v___jp_2981_;
}
}
else
{
v___y_2982_ = v___x_3026_;
v___y_2983_ = v_sz_3031_;
v___y_2984_ = v___x_3042_;
v___y_2985_ = v___x_3029_;
v___y_2986_ = v___x_3045_;
v___y_2987_ = v_options_3016_;
v___y_2988_ = v___x_3030_;
v___y_2989_ = v___f_3041_;
goto v___jp_2981_;
}
}
}
else
{
lean_dec(v_traceClass_2925_);
lean_dec(v_a_2916_);
return v___y_3021_;
}
}
v___jp_3057_:
{
lean_object* v___x_3065_; double v___x_3066_; double v___x_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3070_; lean_object* v___x_3071_; uint8_t v___x_3072_; lean_object* v___x_3073_; 
v___x_3065_ = lean_io_get_num_heartbeats();
v___x_3066_ = lean_float_of_nat(v___y_3061_);
v___x_3067_ = lean_float_of_nat(v___x_3065_);
v___x_3068_ = lean_box_float(v___x_3066_);
v___x_3069_ = lean_box_float(v___x_3067_);
v___x_3070_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3070_, 0, v___x_3068_);
lean_ctor_set(v___x_3070_, 1, v___x_3069_);
v___x_3071_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3071_, 0, v_a_3064_);
lean_ctor_set(v___x_3071_, 1, v___x_3070_);
v___x_3072_ = lean_unbox(v_a_2916_);
lean_inc_ref(v___y_3062_);
lean_inc_ref(v___y_3063_);
lean_inc(v_traceClass_2925_);
v___x_3073_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3072_, v___y_3063_, v___y_3059_, v___y_3058_, v___y_3060_, v___y_3062_, v___x_3071_, v_a_2912_, v_a_2913_);
v___y_3021_ = v___x_3073_;
goto v___jp_3020_;
}
v___jp_3074_:
{
lean_object* v___x_3082_; double v___x_3083_; double v___x_3084_; double v___x_3085_; double v___x_3086_; double v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; uint8_t v___x_3092_; lean_object* v___x_3093_; 
v___x_3082_ = lean_io_mono_nanos_now();
v___x_3083_ = lean_float_of_nat(v___y_3077_);
v___x_3084_ = lean_float_once(&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0, &lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once, _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0);
v___x_3085_ = lean_float_div(v___x_3083_, v___x_3084_);
v___x_3086_ = lean_float_of_nat(v___x_3082_);
v___x_3087_ = lean_float_div(v___x_3086_, v___x_3084_);
v___x_3088_ = lean_box_float(v___x_3085_);
v___x_3089_ = lean_box_float(v___x_3087_);
v___x_3090_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3090_, 0, v___x_3088_);
lean_ctor_set(v___x_3090_, 1, v___x_3089_);
v___x_3091_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3091_, 0, v_a_3081_);
lean_ctor_set(v___x_3091_, 1, v___x_3090_);
v___x_3092_ = lean_unbox(v_a_2916_);
lean_inc_ref(v___y_3079_);
lean_inc_ref(v___y_3080_);
lean_inc(v_traceClass_2925_);
v___x_3093_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3092_, v___y_3080_, v___y_3076_, v___y_3075_, v___y_3078_, v___y_3079_, v___x_3091_, v_a_2912_, v_a_2913_);
v___y_3021_ = v___x_3093_;
goto v___jp_3020_;
}
v___jp_3094_:
{
lean_object* v___x_3100_; lean_object* v_a_3101_; lean_object* v___x_3102_; uint8_t v___x_3103_; 
v___x_3100_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v_a_2913_);
v_a_3101_ = lean_ctor_get(v___x_3100_, 0);
lean_inc(v_a_3101_);
lean_dec_ref(v___x_3100_);
v___x_3102_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3103_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v___y_3097_, v___x_3102_);
if (v___x_3103_ == 0)
{
lean_object* v___x_3104_; lean_object* v___x_3105_; 
v___x_3104_ = lean_io_mono_nanos_now();
v___x_3105_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v___y_3095_, v_opt_2911_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3105_) == 0)
{
lean_object* v_a_3106_; lean_object* v___x_3108_; uint8_t v_isShared_3109_; uint8_t v_isSharedCheck_3113_; 
v_a_3106_ = lean_ctor_get(v___x_3105_, 0);
v_isSharedCheck_3113_ = !lean_is_exclusive(v___x_3105_);
if (v_isSharedCheck_3113_ == 0)
{
v___x_3108_ = v___x_3105_;
v_isShared_3109_ = v_isSharedCheck_3113_;
goto v_resetjp_3107_;
}
else
{
lean_inc(v_a_3106_);
lean_dec(v___x_3105_);
v___x_3108_ = lean_box(0);
v_isShared_3109_ = v_isSharedCheck_3113_;
goto v_resetjp_3107_;
}
v_resetjp_3107_:
{
lean_object* v___x_3111_; 
if (v_isShared_3109_ == 0)
{
lean_ctor_set_tag(v___x_3108_, 1);
v___x_3111_ = v___x_3108_;
goto v_reusejp_3110_;
}
else
{
lean_object* v_reuseFailAlloc_3112_; 
v_reuseFailAlloc_3112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3112_, 0, v_a_3106_);
v___x_3111_ = v_reuseFailAlloc_3112_;
goto v_reusejp_3110_;
}
v_reusejp_3110_:
{
v___y_3075_ = v___y_3096_;
v___y_3076_ = v___y_3097_;
v___y_3077_ = v___x_3104_;
v___y_3078_ = v_a_3101_;
v___y_3079_ = v___y_3098_;
v___y_3080_ = v___y_3099_;
v_a_3081_ = v___x_3111_;
goto v___jp_3074_;
}
}
}
else
{
lean_object* v_a_3114_; lean_object* v___x_3116_; uint8_t v_isShared_3117_; uint8_t v_isSharedCheck_3121_; 
v_a_3114_ = lean_ctor_get(v___x_3105_, 0);
v_isSharedCheck_3121_ = !lean_is_exclusive(v___x_3105_);
if (v_isSharedCheck_3121_ == 0)
{
v___x_3116_ = v___x_3105_;
v_isShared_3117_ = v_isSharedCheck_3121_;
goto v_resetjp_3115_;
}
else
{
lean_inc(v_a_3114_);
lean_dec(v___x_3105_);
v___x_3116_ = lean_box(0);
v_isShared_3117_ = v_isSharedCheck_3121_;
goto v_resetjp_3115_;
}
v_resetjp_3115_:
{
lean_object* v___x_3119_; 
if (v_isShared_3117_ == 0)
{
lean_ctor_set_tag(v___x_3116_, 0);
v___x_3119_ = v___x_3116_;
goto v_reusejp_3118_;
}
else
{
lean_object* v_reuseFailAlloc_3120_; 
v_reuseFailAlloc_3120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3120_, 0, v_a_3114_);
v___x_3119_ = v_reuseFailAlloc_3120_;
goto v_reusejp_3118_;
}
v_reusejp_3118_:
{
v___y_3075_ = v___y_3096_;
v___y_3076_ = v___y_3097_;
v___y_3077_ = v___x_3104_;
v___y_3078_ = v_a_3101_;
v___y_3079_ = v___y_3098_;
v___y_3080_ = v___y_3099_;
v_a_3081_ = v___x_3119_;
goto v___jp_3074_;
}
}
}
}
else
{
lean_object* v___x_3122_; lean_object* v___x_3123_; 
v___x_3122_ = lean_io_get_num_heartbeats();
v___x_3123_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v___y_3095_, v_opt_2911_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3123_) == 0)
{
lean_object* v_a_3124_; lean_object* v___x_3126_; uint8_t v_isShared_3127_; uint8_t v_isSharedCheck_3131_; 
v_a_3124_ = lean_ctor_get(v___x_3123_, 0);
v_isSharedCheck_3131_ = !lean_is_exclusive(v___x_3123_);
if (v_isSharedCheck_3131_ == 0)
{
v___x_3126_ = v___x_3123_;
v_isShared_3127_ = v_isSharedCheck_3131_;
goto v_resetjp_3125_;
}
else
{
lean_inc(v_a_3124_);
lean_dec(v___x_3123_);
v___x_3126_ = lean_box(0);
v_isShared_3127_ = v_isSharedCheck_3131_;
goto v_resetjp_3125_;
}
v_resetjp_3125_:
{
lean_object* v___x_3129_; 
if (v_isShared_3127_ == 0)
{
lean_ctor_set_tag(v___x_3126_, 1);
v___x_3129_ = v___x_3126_;
goto v_reusejp_3128_;
}
else
{
lean_object* v_reuseFailAlloc_3130_; 
v_reuseFailAlloc_3130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3130_, 0, v_a_3124_);
v___x_3129_ = v_reuseFailAlloc_3130_;
goto v_reusejp_3128_;
}
v_reusejp_3128_:
{
v___y_3058_ = v___y_3096_;
v___y_3059_ = v___y_3097_;
v___y_3060_ = v_a_3101_;
v___y_3061_ = v___x_3122_;
v___y_3062_ = v___y_3098_;
v___y_3063_ = v___y_3099_;
v_a_3064_ = v___x_3129_;
goto v___jp_3057_;
}
}
}
else
{
lean_object* v_a_3132_; lean_object* v___x_3134_; uint8_t v_isShared_3135_; uint8_t v_isSharedCheck_3139_; 
v_a_3132_ = lean_ctor_get(v___x_3123_, 0);
v_isSharedCheck_3139_ = !lean_is_exclusive(v___x_3123_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3134_ = v___x_3123_;
v_isShared_3135_ = v_isSharedCheck_3139_;
goto v_resetjp_3133_;
}
else
{
lean_inc(v_a_3132_);
lean_dec(v___x_3123_);
v___x_3134_ = lean_box(0);
v_isShared_3135_ = v_isSharedCheck_3139_;
goto v_resetjp_3133_;
}
v_resetjp_3133_:
{
lean_object* v___x_3137_; 
if (v_isShared_3135_ == 0)
{
lean_ctor_set_tag(v___x_3134_, 0);
v___x_3137_ = v___x_3134_;
goto v_reusejp_3136_;
}
else
{
lean_object* v_reuseFailAlloc_3138_; 
v_reuseFailAlloc_3138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3138_, 0, v_a_3132_);
v___x_3137_ = v_reuseFailAlloc_3138_;
goto v_reusejp_3136_;
}
v_reusejp_3136_:
{
v___y_3058_ = v___y_3096_;
v___y_3059_ = v___y_3097_;
v___y_3060_ = v_a_3101_;
v___y_3061_ = v___x_3122_;
v___y_3062_ = v___y_3098_;
v___y_3063_ = v___y_3099_;
v_a_3064_ = v___x_3137_;
goto v___jp_3057_;
}
}
}
}
}
v___jp_3140_:
{
if (lean_obj_tag(v___y_3141_) == 0)
{
lean_dec_ref_known(v___y_3141_, 1);
if (v_hasTrace_3018_ == 0)
{
lean_object* v_post_3142_; lean_object* v___x_3143_; 
v_post_3142_ = lean_ctor_get(v_s_2910_, 1);
v___x_3143_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v_post_3142_, v_opt_2911_, v_a_2912_, v_a_2913_);
v___y_3021_ = v___x_3143_;
goto v___jp_3020_;
}
else
{
lean_object* v_post_3144_; lean_object* v___f_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; uint8_t v___x_3149_; 
v_post_3144_ = lean_ctor_get(v_s_2910_, 1);
v___f_3145_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__9, &lp_aesop_Aesop_traceSimpTheorems___closed__9_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__9);
v___x_3146_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_3147_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0));
lean_inc(v_traceClass_2925_);
v___x_3148_ = l_Lean_Name_append(v___x_3147_, v_traceClass_2925_);
v___x_3149_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3017_, v_options_3016_, v___x_3148_);
lean_dec(v___x_3148_);
if (v___x_3149_ == 0)
{
lean_object* v___x_3150_; uint8_t v___x_3151_; 
v___x_3150_ = l_Lean_trace_profiler;
v___x_3151_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_3016_, v___x_3150_);
if (v___x_3151_ == 0)
{
lean_object* v___x_3152_; 
v___x_3152_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v_post_3144_, v_opt_2911_, v_a_2912_, v_a_2913_);
v___y_3021_ = v___x_3152_;
goto v___jp_3020_;
}
else
{
v___y_3095_ = v_post_3144_;
v___y_3096_ = v___x_3149_;
v___y_3097_ = v_options_3016_;
v___y_3098_ = v___f_3145_;
v___y_3099_ = v___x_3146_;
goto v___jp_3094_;
}
}
else
{
v___y_3095_ = v_post_3144_;
v___y_3096_ = v___x_3149_;
v___y_3097_ = v_options_3016_;
v___y_3098_ = v___f_3145_;
v___y_3099_ = v___x_3146_;
goto v___jp_3094_;
}
}
}
else
{
lean_dec(v_traceClass_2925_);
lean_dec(v_a_2916_);
lean_dec_ref(v_opt_2911_);
return v___y_3141_;
}
}
v___jp_3153_:
{
lean_object* v___x_3161_; double v___x_3162_; double v___x_3163_; double v___x_3164_; double v___x_3165_; double v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; uint8_t v___x_3171_; lean_object* v___x_3172_; 
v___x_3161_ = lean_io_mono_nanos_now();
v___x_3162_ = lean_float_of_nat(v___y_3158_);
v___x_3163_ = lean_float_once(&lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0, &lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0_once, _init_lp_aesop_Aesop_withAesopTraceNode___redArg___lam__3___closed__0);
v___x_3164_ = lean_float_div(v___x_3162_, v___x_3163_);
v___x_3165_ = lean_float_of_nat(v___x_3161_);
v___x_3166_ = lean_float_div(v___x_3165_, v___x_3163_);
v___x_3167_ = lean_box_float(v___x_3164_);
v___x_3168_ = lean_box_float(v___x_3166_);
v___x_3169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3169_, 0, v___x_3167_);
lean_ctor_set(v___x_3169_, 1, v___x_3168_);
v___x_3170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3170_, 0, v_a_3160_);
lean_ctor_set(v___x_3170_, 1, v___x_3169_);
v___x_3171_ = lean_unbox(v_a_2916_);
lean_inc_ref(v___y_3156_);
lean_inc_ref(v___y_3157_);
lean_inc(v_traceClass_2925_);
v___x_3172_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3171_, v___y_3157_, v___y_3159_, v___y_3155_, v___y_3154_, v___y_3156_, v___x_3170_, v_a_2912_, v_a_2913_);
v___y_3141_ = v___x_3172_;
goto v___jp_3140_;
}
v___jp_3173_:
{
lean_object* v___x_3181_; double v___x_3182_; double v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; uint8_t v___x_3188_; lean_object* v___x_3189_; 
v___x_3181_ = lean_io_get_num_heartbeats();
v___x_3182_ = lean_float_of_nat(v___y_3179_);
v___x_3183_ = lean_float_of_nat(v___x_3181_);
v___x_3184_ = lean_box_float(v___x_3182_);
v___x_3185_ = lean_box_float(v___x_3183_);
v___x_3186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3186_, 0, v___x_3184_);
lean_ctor_set(v___x_3186_, 1, v___x_3185_);
v___x_3187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3187_, 0, v_a_3180_);
lean_ctor_set(v___x_3187_, 1, v___x_3186_);
v___x_3188_ = lean_unbox(v_a_2916_);
lean_inc_ref(v___y_3176_);
lean_inc_ref(v___y_3177_);
lean_inc(v_traceClass_2925_);
v___x_3189_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4(v_traceClass_2925_, v___x_3188_, v___y_3177_, v___y_3178_, v___y_3175_, v___y_3174_, v___y_3176_, v___x_3187_, v_a_2912_, v_a_2913_);
v___y_3141_ = v___x_3189_;
goto v___jp_3140_;
}
v___jp_3190_:
{
lean_object* v___x_3196_; lean_object* v_a_3197_; lean_object* v___x_3198_; uint8_t v___x_3199_; 
v___x_3196_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_traceSimpTheorems_spec__3___redArg(v_a_2913_);
v_a_3197_ = lean_ctor_get(v___x_3196_, 0);
lean_inc(v_a_3197_);
lean_dec_ref(v___x_3196_);
v___x_3198_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3199_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v___y_3194_, v___x_3198_);
if (v___x_3199_ == 0)
{
lean_object* v___x_3200_; lean_object* v___x_3201_; 
v___x_3200_ = lean_io_mono_nanos_now();
lean_inc_ref(v_opt_2911_);
v___x_3201_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v___y_3195_, v_opt_2911_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3201_) == 0)
{
lean_object* v_a_3202_; lean_object* v___x_3204_; uint8_t v_isShared_3205_; uint8_t v_isSharedCheck_3209_; 
v_a_3202_ = lean_ctor_get(v___x_3201_, 0);
v_isSharedCheck_3209_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3209_ == 0)
{
v___x_3204_ = v___x_3201_;
v_isShared_3205_ = v_isSharedCheck_3209_;
goto v_resetjp_3203_;
}
else
{
lean_inc(v_a_3202_);
lean_dec(v___x_3201_);
v___x_3204_ = lean_box(0);
v_isShared_3205_ = v_isSharedCheck_3209_;
goto v_resetjp_3203_;
}
v_resetjp_3203_:
{
lean_object* v___x_3207_; 
if (v_isShared_3205_ == 0)
{
lean_ctor_set_tag(v___x_3204_, 1);
v___x_3207_ = v___x_3204_;
goto v_reusejp_3206_;
}
else
{
lean_object* v_reuseFailAlloc_3208_; 
v_reuseFailAlloc_3208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3208_, 0, v_a_3202_);
v___x_3207_ = v_reuseFailAlloc_3208_;
goto v_reusejp_3206_;
}
v_reusejp_3206_:
{
v___y_3154_ = v_a_3197_;
v___y_3155_ = v___y_3191_;
v___y_3156_ = v___y_3192_;
v___y_3157_ = v___y_3193_;
v___y_3158_ = v___x_3200_;
v___y_3159_ = v___y_3194_;
v_a_3160_ = v___x_3207_;
goto v___jp_3153_;
}
}
}
else
{
lean_object* v_a_3210_; lean_object* v___x_3212_; uint8_t v_isShared_3213_; uint8_t v_isSharedCheck_3217_; 
v_a_3210_ = lean_ctor_get(v___x_3201_, 0);
v_isSharedCheck_3217_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3217_ == 0)
{
v___x_3212_ = v___x_3201_;
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
else
{
lean_inc(v_a_3210_);
lean_dec(v___x_3201_);
v___x_3212_ = lean_box(0);
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
v_resetjp_3211_:
{
lean_object* v___x_3215_; 
if (v_isShared_3213_ == 0)
{
lean_ctor_set_tag(v___x_3212_, 0);
v___x_3215_ = v___x_3212_;
goto v_reusejp_3214_;
}
else
{
lean_object* v_reuseFailAlloc_3216_; 
v_reuseFailAlloc_3216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3216_, 0, v_a_3210_);
v___x_3215_ = v_reuseFailAlloc_3216_;
goto v_reusejp_3214_;
}
v_reusejp_3214_:
{
v___y_3154_ = v_a_3197_;
v___y_3155_ = v___y_3191_;
v___y_3156_ = v___y_3192_;
v___y_3157_ = v___y_3193_;
v___y_3158_ = v___x_3200_;
v___y_3159_ = v___y_3194_;
v_a_3160_ = v___x_3215_;
goto v___jp_3153_;
}
}
}
}
else
{
lean_object* v___x_3218_; lean_object* v___x_3219_; 
v___x_3218_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_opt_2911_);
v___x_3219_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v___y_3195_, v_opt_2911_, v_a_2912_, v_a_2913_);
if (lean_obj_tag(v___x_3219_) == 0)
{
lean_object* v_a_3220_; lean_object* v___x_3222_; uint8_t v_isShared_3223_; uint8_t v_isSharedCheck_3227_; 
v_a_3220_ = lean_ctor_get(v___x_3219_, 0);
v_isSharedCheck_3227_ = !lean_is_exclusive(v___x_3219_);
if (v_isSharedCheck_3227_ == 0)
{
v___x_3222_ = v___x_3219_;
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
else
{
lean_inc(v_a_3220_);
lean_dec(v___x_3219_);
v___x_3222_ = lean_box(0);
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
v_resetjp_3221_:
{
lean_object* v___x_3225_; 
if (v_isShared_3223_ == 0)
{
lean_ctor_set_tag(v___x_3222_, 1);
v___x_3225_ = v___x_3222_;
goto v_reusejp_3224_;
}
else
{
lean_object* v_reuseFailAlloc_3226_; 
v_reuseFailAlloc_3226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3226_, 0, v_a_3220_);
v___x_3225_ = v_reuseFailAlloc_3226_;
goto v_reusejp_3224_;
}
v_reusejp_3224_:
{
v___y_3174_ = v_a_3197_;
v___y_3175_ = v___y_3191_;
v___y_3176_ = v___y_3192_;
v___y_3177_ = v___y_3193_;
v___y_3178_ = v___y_3194_;
v___y_3179_ = v___x_3218_;
v_a_3180_ = v___x_3225_;
goto v___jp_3173_;
}
}
}
else
{
lean_object* v_a_3228_; lean_object* v___x_3230_; uint8_t v_isShared_3231_; uint8_t v_isSharedCheck_3235_; 
v_a_3228_ = lean_ctor_get(v___x_3219_, 0);
v_isSharedCheck_3235_ = !lean_is_exclusive(v___x_3219_);
if (v_isSharedCheck_3235_ == 0)
{
v___x_3230_ = v___x_3219_;
v_isShared_3231_ = v_isSharedCheck_3235_;
goto v_resetjp_3229_;
}
else
{
lean_inc(v_a_3228_);
lean_dec(v___x_3219_);
v___x_3230_ = lean_box(0);
v_isShared_3231_ = v_isSharedCheck_3235_;
goto v_resetjp_3229_;
}
v_resetjp_3229_:
{
lean_object* v___x_3233_; 
if (v_isShared_3231_ == 0)
{
lean_ctor_set_tag(v___x_3230_, 0);
v___x_3233_ = v___x_3230_;
goto v_reusejp_3232_;
}
else
{
lean_object* v_reuseFailAlloc_3234_; 
v_reuseFailAlloc_3234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3234_, 0, v_a_3228_);
v___x_3233_ = v_reuseFailAlloc_3234_;
goto v_reusejp_3232_;
}
v_reusejp_3232_:
{
v___y_3174_ = v_a_3197_;
v___y_3175_ = v___y_3191_;
v___y_3176_ = v___y_3192_;
v___y_3177_ = v___y_3193_;
v___y_3178_ = v___y_3194_;
v___y_3179_ = v___x_3218_;
v_a_3180_ = v___x_3233_;
goto v___jp_3173_;
}
}
}
}
}
v___jp_3236_:
{
if (v_hasTrace_3018_ == 0)
{
lean_object* v_pre_3237_; lean_object* v___x_3238_; 
v_pre_3237_ = lean_ctor_get(v_s_2910_, 0);
lean_inc_ref(v_opt_2911_);
v___x_3238_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v_pre_3237_, v_opt_2911_, v_a_2912_, v_a_2913_);
v___y_3141_ = v___x_3238_;
goto v___jp_3140_;
}
else
{
lean_object* v_pre_3239_; lean_object* v___f_3240_; lean_object* v___x_3241_; lean_object* v___x_3242_; lean_object* v___x_3243_; uint8_t v___x_3244_; 
v_pre_3239_ = lean_ctor_get(v_s_2910_, 0);
v___f_3240_ = lean_obj_once(&lp_aesop_Aesop_traceSimpTheorems___closed__13, &lp_aesop_Aesop_traceSimpTheorems___closed__13_once, _init_lp_aesop_Aesop_traceSimpTheorems___closed__13);
v___x_3241_ = ((lean_object*)(lp_aesop_Aesop___aux__Aesop__Tracing______macroRules__Aesop__doElemAesop__trace_x21_x5b___x5d______1___lam__0___closed__22));
v___x_3242_ = ((lean_object*)(lp_aesop_Aesop_withAesopTraceNode___redArg___lam__11___closed__0));
lean_inc(v_traceClass_2925_);
v___x_3243_ = l_Lean_Name_append(v___x_3242_, v_traceClass_2925_);
v___x_3244_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3017_, v_options_3016_, v___x_3243_);
lean_dec(v___x_3243_);
if (v___x_3244_ == 0)
{
lean_object* v___x_3245_; uint8_t v___x_3246_; 
v___x_3245_ = l_Lean_trace_profiler;
v___x_3246_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_traceSimpTheoremTreeContents_spec__0_spec__0(v_options_3016_, v___x_3245_);
if (v___x_3246_ == 0)
{
lean_object* v___x_3247_; 
lean_inc_ref(v_opt_2911_);
v___x_3247_ = lp_aesop_Aesop_traceSimpTheoremTreeContents(v_pre_3239_, v_opt_2911_, v_a_2912_, v_a_2913_);
v___y_3141_ = v___x_3247_;
goto v___jp_3140_;
}
else
{
v___y_3191_ = v___x_3244_;
v___y_3192_ = v___f_3240_;
v___y_3193_ = v___x_3241_;
v___y_3194_ = v_options_3016_;
v___y_3195_ = v_pre_3239_;
goto v___jp_3190_;
}
}
else
{
v___y_3191_ = v___x_3244_;
v___y_3192_ = v___f_3240_;
v___y_3193_ = v___x_3241_;
v___y_3194_ = v_options_3016_;
v___y_3195_ = v_pre_3239_;
goto v___jp_3190_;
}
}
}
v___jp_3248_:
{
if (lean_obj_tag(v___y_3249_) == 0)
{
lean_dec_ref_known(v___y_3249_, 1);
goto v___jp_3236_;
}
else
{
lean_dec(v_traceClass_2925_);
lean_dec(v_a_2916_);
lean_dec_ref(v_opt_2911_);
return v___y_3249_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceSimpTheorems___boxed(lean_object* v_s_3378_, lean_object* v_opt_3379_, lean_object* v_a_3380_, lean_object* v_a_3381_, lean_object* v_a_3382_){
_start:
{
lean_object* v_res_3383_; 
v_res_3383_ = lp_aesop_Aesop_traceSimpTheorems(v_s_3378_, v_opt_3379_, v_a_3380_, v_a_3381_);
lean_dec(v_a_3381_);
lean_dec_ref(v_a_3380_);
lean_dec_ref(v_s_3378_);
return v_res_3383_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___redArg(lean_object* v_map_3384_, lean_object* v_f_3385_, lean_object* v_init_3386_){
_start:
{
lean_object* v___x_3387_; 
v___x_3387_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_3385_, v_map_3384_, v_init_3386_);
return v___x_3387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___redArg___boxed(lean_object* v_map_3388_, lean_object* v_f_3389_, lean_object* v_init_3390_){
_start:
{
lean_object* v_res_3391_; 
v_res_3391_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___redArg(v_map_3388_, v_f_3389_, v_init_3390_);
lean_dec_ref(v_map_3388_);
return v_res_3391_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0(lean_object* v_00_u03c3_3392_, lean_object* v_00_u03b2_3393_, lean_object* v_map_3394_, lean_object* v_f_3395_, lean_object* v_init_3396_){
_start:
{
lean_object* v___x_3397_; 
v___x_3397_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_3395_, v_map_3394_, v_init_3396_);
return v___x_3397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0___boxed(lean_object* v_00_u03c3_3398_, lean_object* v_00_u03b2_3399_, lean_object* v_map_3400_, lean_object* v_f_3401_, lean_object* v_init_3402_){
_start:
{
lean_object* v_res_3403_; 
v_res_3403_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__0(v_00_u03c3_3398_, v_00_u03b2_3399_, v_map_3400_, v_f_3401_, v_init_3402_);
lean_dec_ref(v_map_3400_);
return v_res_3403_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5(lean_object* v_00_u03b1_3404_, lean_object* v_x_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_){
_start:
{
lean_object* v___x_3409_; 
v___x_3409_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___redArg(v_x_3405_);
return v___x_3409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5___boxed(lean_object* v_00_u03b1_3410_, lean_object* v_x_3411_, lean_object* v___y_3412_, lean_object* v___y_3413_, lean_object* v___y_3414_){
_start:
{
lean_object* v_res_3415_; 
v_res_3415_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_traceSimpTheorems_spec__4_spec__5(v_00_u03b1_3410_, v_x_3411_, v___y_3412_, v___y_3413_);
lean_dec(v___y_3413_);
lean_dec_ref(v___y_3412_);
return v_res_3415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___redArg(lean_object* v_map_3416_, lean_object* v_f_3417_, lean_object* v_init_3418_){
_start:
{
lean_object* v___x_3419_; 
v___x_3419_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_3417_, v_map_3416_, v_init_3418_);
return v___x_3419_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___redArg___boxed(lean_object* v_map_3420_, lean_object* v_f_3421_, lean_object* v_init_3422_){
_start:
{
lean_object* v_res_3423_; 
v_res_3423_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___redArg(v_map_3420_, v_f_3421_, v_init_3422_);
lean_dec_ref(v_map_3420_);
return v_res_3423_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5(lean_object* v_00_u03c3_3424_, lean_object* v_00_u03b2_3425_, lean_object* v_map_3426_, lean_object* v_f_3427_, lean_object* v_init_3428_){
_start:
{
lean_object* v___x_3429_; 
v___x_3429_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheoremTreeContents_spec__2_spec__5___redArg(v_f_3427_, v_map_3426_, v_init_3428_);
return v___x_3429_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5___boxed(lean_object* v_00_u03c3_3430_, lean_object* v_00_u03b2_3431_, lean_object* v_map_3432_, lean_object* v_f_3433_, lean_object* v_init_3434_){
_start:
{
lean_object* v_res_3435_; 
v_res_3435_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_traceSimpTheorems_spec__5(v_00_u03c3_3430_, v_00_u03b2_3431_, v_map_3432_, v_f_3433_, v_init_3434_);
lean_dec_ref(v_map_3432_);
return v_res_3435_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tracing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedTraceOption_default = _init_lp_aesop_Aesop_instInhabitedTraceOption_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedTraceOption_default);
lp_aesop_Aesop_instInhabitedTraceOption = _init_lp_aesop_Aesop_instInhabitedTraceOption();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedTraceOption);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3555991590____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_steps = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_steps);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3845851586____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_ruleSet = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_ruleSet);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1349429452____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_proof = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_proof);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1915903599____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_tree = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_tree);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_465462901____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_extraction = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_extraction);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2063951775____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_stats = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_stats);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3298134486____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_debug = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_debug);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1896596885____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_script = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_script);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_2168098131____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_forward = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_forward);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_3366025166____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_forwardDebug = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_forwardDebug);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Tracing_0__Aesop_TraceOption_initFn_00___x40_Aesop_Tracing_1550913053____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_TraceOption_rpinf = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_TraceOption_rpinf);
lean_dec_ref(res);
lp_aesop_Aesop_ruleSuccessEmoji = _init_lp_aesop_Aesop_ruleSuccessEmoji();
lean_mark_persistent(lp_aesop_Aesop_ruleSuccessEmoji);
lp_aesop_Aesop_ruleFailureEmoji = _init_lp_aesop_Aesop_ruleFailureEmoji();
lean_mark_persistent(lp_aesop_Aesop_ruleFailureEmoji);
lp_aesop_Aesop_ruleErrorEmoji = _init_lp_aesop_Aesop_ruleErrorEmoji();
lean_mark_persistent(lp_aesop_Aesop_ruleErrorEmoji);
lp_aesop_Aesop_nodeUnprovableEmoji = _init_lp_aesop_Aesop_nodeUnprovableEmoji();
lean_mark_persistent(lp_aesop_Aesop_nodeUnprovableEmoji);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tracing(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tracing(builtin);
}
#ifdef __cplusplus
}
#endif
